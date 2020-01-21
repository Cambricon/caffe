/*
All modification made by Cambricon Corporation: © 2019 Cambricon Corporation
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
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#include <boost/property_tree/json_parser.hpp>
#pragma GCC diagnostic pop
#include <boost/property_tree/ptree.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <glog/logging.h> // NOLINT
#include "include/command_option.hpp"
#include "include/common_functions.hpp"

using std::vector;
using std::string;
using namespace boost::property_tree;  // NOLINT(build/namespaces)

void printfMluTime(float mluTime) {
  LOG(INFO) << " execution time: " << mluTime;
}

void printfAccuracy(int imageNum, float acc1, float acc5) {
  LOG(INFO) << "Global accuracy : ";
  LOG(INFO) << "top1: " << 1.0 * acc1 / imageNum << " ("
    << acc1 << "/" << imageNum << ")";
  LOG(INFO) << "top5: " << 1.0 * acc5 / imageNum << " ("
    << acc5 << "/" << imageNum << ")";
}

// Usage:
//      if thread is not needed, pass 1
//      batchsize is needed to calculate latency
void printPerf(int imageNum, float execTime, float mluTime, int threads, int batchsize) {
  int parallel_num = (threads > 32) ? 32 : threads;
  float hardwareFps = imageNum / mluTime * parallel_num * 1e6;
  float latency = -1;
  latency = (float)(float(mluTime)/float(imageNum))*batchsize;   // us
  LOG(INFO) << "throughput: " << hardwareFps;
  if (getenv("OUTPUT_E2E") != NULL && getenv("OUTPUT_E2E")[0] == 'O' && getenv("OUTPUT_E2E")[1] == 'N'){
      LOG(INFO) << "End2end throughput fps: " << imageNum / execTime * 1e6;
  }
  LOG(INFO) << "Latency: " << float2string(latency);
}
string float2string(float value) {
  std::stringstream strstream;
  strstream.setf(std::ios::fixed);
  strstream.precision(2);
  strstream << value;
  return strstream.str();
}

void saveResult(int imageNum, float top1, float top5, float meanAp,
    float hardwaretime, float endToEndTime, int threads, int batchsize) {
  if (getenv("OUTPUT_JSON_FILE") == NULL) return;
  string file = getenv("OUTPUT_JSON_FILE");
  float hardwareFps = (-1);
  float endToEndFps = (-1);
  int parallel_num;
  if (hardwaretime != (-1)) {
    parallel_num = (threads > 32) ? 32 : threads;
    hardwaretime = hardwaretime * batchsize / imageNum;
    hardwareFps = batchsize / hardwaretime * parallel_num * 1e6;
  } else {
    hardwareFps = (-1);
  }

  if (endToEndTime != (-1)) {
    endToEndFps = imageNum /endToEndTime * 1e6;
  } else {
    endToEndFps = (-1);
  }

  if (top1 != (-1)) {
    top1 = (1.0 * top1 / imageNum) * 100;
  } else {
    top1 = (-1);
  }

  if (top5 != (-1)) {
    top5 = (1.0 * top5 / imageNum) * 100;
  } else {
    top5 = (-1);
  }

  ptree output, output_list, accuracy, performance;
  accuracy.put("top1", float2string(top1));
  accuracy.put("top5", float2string(top5));
  accuracy.put("meanAp", float2string(meanAp));

  performance.put("hardwaretime", float2string(hardwaretime));
  performance.put("hardwareFps", float2string(hardwareFps));
  performance.put("endToEndTime", float2string(endToEndTime));
  performance.put("endToEndFps", float2string(endToEndFps));

  output_list.put_child("accuracy", accuracy);
  output_list.put_child("performance", performance);
  output.put_child("output", output_list);
  std::stringstream ss;
  write_json(ss, output);
  std::ofstream fout;
  fout.open(file);
  if (fout) {
    fout << ss.str() <<std::endl;
  } else {
    LOG(INFO) << "file open failed!";
  }
  fout.close();
}

vector<int> getTop5(vector<string> labels, string image, float* data, int count) {
  vector<int> index(5, 0);
  vector<float> value(5, 0);
  for (int i = 0; i < count; i++) {
    float tmp_data = data[i];
    int tmp_index = i;
    for (int j = 0; j < 5; j++) {
      if (data[i] > value[j]) {
        std::swap(value[j], tmp_data);
        std::swap(index[j], tmp_index);
      }
    }
  }
  std::stringstream stream;
  stream << "\n----- top5 for " << image << std::endl;
  for (int i = 0; i < 5; i++) {
    stream  << std::fixed << std::setprecision(4) << value[i] << " - "
            << labels[index[i]] << std::endl;
  }
  LOG(INFO) << stream.str();
  return index;
}

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

cv::Mat readImage(string name, cv::Size size, bool yuvImg) {
  cv::Mat image;
  if (yuvImg) {
    image = convertYuv2Mat(name, size);
  } else {
    image = cv::imread(name, -1);
  }
  return image;
}

cv::Mat yuv420sp2Bgr24(cv::Mat yuv_image) {
    cv::Mat bgr_image(yuv_image.rows / 3 * 2,
        yuv_image.cols, CV_8UC3);
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

vector<int> to_cpu_shape(const vector<int>& mlu_shape) {
  // shape: N(D)HWC --> NC(D)HW
  vector<int> cpu_shape(mlu_shape.size(), 1);
  int channel = mlu_shape[mlu_shape.size() - 1];
  for (int i = 2; i < mlu_shape.size(); i++) {
    cpu_shape[i] = mlu_shape[i - 1];
  }
  cpu_shape[0] = mlu_shape[0];
  cpu_shape[1] = channel;
  return cpu_shape;
}

vector<int> to_mlu_shape(const vector<int>& cpu_shape) {
  // shape : NC(D)HW --> N(D)HWC
  vector<int> mlu_shape(cpu_shape.size(), 1);
  int channel = cpu_shape[1];
  for (int i = 1; i < cpu_shape.size() - 1; i++) {
    mlu_shape[i] = cpu_shape[i + 1];
  }
  mlu_shape[0] = cpu_shape[0];
  mlu_shape[mlu_shape.size() - 1] = channel;
  return mlu_shape;
}
