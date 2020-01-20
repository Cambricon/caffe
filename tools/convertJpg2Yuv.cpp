/*
All modification made by Cambricon Corporation: © 2018--2019 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2019, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/shared_ptr.hpp>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include "glog/logging.h"

using std::string;
using std::stringstream;
using std::ofstream;
using std::ifstream;
using std::endl;

DEFINE_bool(yuv2jpg, false,
    "The function will change to convert yuv to JPG, and the default is jpg2yuv");

// Specify the width and height to determine the size of the image to use.
// jpg2yuv: If JPG is converted to yuv,
//          JPG will be resize before conversion after setting size.
// yuv2jpg: Convert yuv data to JPG, requires providing the size of yuvdata.
DEFINE_int32(width, 0, "The width of the picture.");
DEFINE_int32(height, 0, "The height of the picture.");

void readYUV(string name, cv::Mat img, int h, int w) {
  std::ifstream fin(name);
  unsigned char a;
  for (int i = 0; i < h; i++)
    for (int j = 0; j < w; j++) {
      a = fin.get();
      img.at<char>(i, j) = a;
      fin.get();
    }
  fin.close();
}

cv::Mat yuv420sp2Bgr24(cv::Mat yuv_image) {
  cv::Mat bgr_image(yuv_image.rows / 3 * 2, yuv_image.cols, CV_8UC3);
  cvtColor(yuv_image, bgr_image, CV_YUV420sp2BGR);
  return bgr_image;
}

cv::Mat convertYuv2Mat(string img_name, cv::Size inGeometry) {
  cv::Mat img = cv::Mat(inGeometry, CV_8UC1);
  readYUV(img_name, img, inGeometry.height, inGeometry.width);
  return img;
}

cv::Mat convertYuv2Mat(string img_name, int width, int height) {
  cv::Size inGeometry_(width, height);
  return convertYuv2Mat(img_name, inGeometry_);
}

void jpg2yuv(string image_path) {
  ifstream stream(image_path.c_str());
  if (!stream.good())
    LOG(FATAL) << "file does not exist.";

  cv::Mat src = cv::imread(image_path);
  cv::Mat srcImg;

  if (FLAGS_width > 0 && FLAGS_height > 0) {
    cv::resize(src, srcImg, cv::Size(FLAGS_width, FLAGS_height), CV_INTER_CUBIC);
    LOG(INFO) << "The image has been reiszed: " <<
      FLAGS_width << "x" << FLAGS_height;
  } else {
    srcImg = src;
  }

  cv::Mat yuv_image;
  cvtColor(srcImg, yuv_image, CV_BGR2YUV_I420);

  LOG(INFO) << "srcImg -> height " << srcImg.rows
            << ", width: " << srcImg.cols;
  LOG(INFO) << "yuvImg -> height: " << yuv_image.rows
            << ", width: " << yuv_image.cols;

  // yuvdata
  int yuv_height = srcImg.rows * 3 / 2;
  char* yuvdata = new char[yuv_height*srcImg.cols];
  int index = 0;
  for (int i = 0; i < yuv_height; i++) {
    for (int j = 0; j < srcImg.cols; j++) {
      yuvdata[index] = yuv_image.at<char>(i, j);
      index++;
    }
  }

  // YUV_I420 -> yuv420sp
  int width = srcImg.cols;
  int height = srcImg.rows;
  int frameSize = width * height;
  int qFrameSize = frameSize / 4;
  int tempFrameSize = frameSize * 5 / 4;
  char* output = new char[frameSize * 3 / 2];
  memcpy(output, yuvdata, frameSize);
  for (int i = 0; i < qFrameSize; i++) {
    output[frameSize + i * 2] = yuvdata[tempFrameSize + i];
    output[frameSize + i * 2 + 1] = yuvdata[frameSize + i];
  }

  stringstream ss;
  string file_name;
  int position = image_path.find_last_of('/');
  string fileName(image_path.substr(position + 1));
  ss << image_path << ".yuv";
  ss >> file_name;
  ofstream fout(file_name);
  for (int i = 0; i < yuv_height; i++) {
    for (int j = 0; j < srcImg.cols; j++) {
      fout << output[i*srcImg.cols+j] << endl;
    }
  }
  fout.close();
  LOG(INFO) << "JPG had been converted to yuv data: " << file_name;
  delete[] yuvdata;
  delete[] output;

  // yuv2jpg
  cv::Mat yuv = convertYuv2Mat(file_name, yuv_image.cols, yuv_image.rows);
  cv::Mat image = yuv420sp2Bgr24(yuv);
  cv::imwrite("output.jpg", image);
  LOG(INFO) << "yuvData converted to JPG image: output.jgp, " <<
    "verify whether correct.";
}

void yuv2jpg(string yuv_path) {
  if (FLAGS_width == 0 || FLAGS_height == 0) {
    LOG(ERROR) << "width = 0, or height = 0";
  }
  int height = FLAGS_height / 2 * 3;
  cv::Mat yuv = convertYuv2Mat(yuv_path, FLAGS_width, height);
  cv::Mat image = yuv420sp2Bgr24(yuv);
  stringstream ss;
  string file_name;
  int position = yuv_path.find_last_of('/');
  string fileName(yuv_path.substr(position + 1));
  ss << yuv_path << ".jpg";
  ss >> file_name;
  cv::imwrite(file_name, image);
  LOG(INFO) << "yuvData had been converted to JPG: " << file_name;
}

int main(int argc, char** argv) {
#ifdef USE_OPENCV
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;
#ifndef GFLAGS_GFLAGS_H_
    namespace gflags = google;
#endif
  gflags::SetUsageMessage("jgp2yuv");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (argc != 2) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/jpg2yuv");
    return 1;
  }

  string image_path = argv[1];
  string type = image_path.substr(image_path.find_last_of('.') + 1);
  // check
  if (FLAGS_yuv2jpg)
    CHECK(type == "yuv") << "yuv";
  else
    CHECK(type == "jpg") << "jpg";
  // convert
  if (FLAGS_yuv2jpg)
    yuv2jpg(argv[1]);
  else
    jpg2yuv(argv[1]);

  return 0;
}
#endif  // USE_OPENCV
