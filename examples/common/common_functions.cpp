/*
All modification made by Cambricon Corporation: © 2018 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2018, the respective contributors
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

void printfMluTime(float mluTime) {
    LOG(INFO) << " execution time: " << mluTime;
}

void printfAccuracy(int imageNum, float acc1, float acc5) {
  LOG(INFO) << "Global accuracy : ";
  LOG(INFO) << "accuracy1: " << 1.0 * acc1 / imageNum << " ("
    << acc1 << "/" << imageNum << ")";
  LOG(INFO) << "accuracy5: " << 1.0 * acc5 / imageNum << " ("
    << acc5 << "/" << imageNum << ")";
}

void printPerf(int imageNum, float execTime, float mluTime, int threads) {
  float hardwareFps = imageNum / mluTime * threads * 1e6;
  LOG(INFO) << "Hardware fps: " << hardwareFps;
  LOG(INFO) << "End2end throughput fps: " << imageNum / execTime * 1e6;
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
