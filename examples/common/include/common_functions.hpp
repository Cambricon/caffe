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

#ifndef EXAMPLES_COMMON_INCLUDE_COMMON_FUNCTIONS_HPP_
#define EXAMPLES_COMMON_INCLUDE_COMMON_FUNCTIONS_HPP_
#include <glog/logging.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sys/time.h>
#include <fstream>
#include <vector>
#include <string>
#include <queue>
#include <chrono>  // NOLINT
#include <iostream>

using std::vector;
using std::string;
using std::queue;
using std::chrono::time_point;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::high_resolution_clock;

#define INSTANTIATE_OFF_CLASS(classname) \
  template class classname<void*, BlockingQueue>; \
  template class classname<void*, Queue>;

#define INSTANTIATE_ON_CLASS(classname) \
  template class classname<float, BlockingQueue>; \
  template class classname<float, Queue>;

#define INSTANTIATE_ALL_CLASS(classname) \
  INSTANTIATE_ON_CLASS(classname) \
  INSTANTIATE_OFF_CLASS(classname)

void setupConfig(int threadID, int deviceID, int deviceSize);

void setDeviceId(int deviceID);

vector<int> getTop5(vector<string> labels, string image, float* data, int count);

void readYUV(string name, cv::Mat img, int h, int w);

cv::Mat readImage(string name, cv::Size size, bool yuvImg);

cv::Mat yuv420sp2Bgr24(cv::Mat yuv_image);

cv::Mat convertYuv2Mat(string img_name, cv::Size inGeometry);

cv::Mat convertYuv2Mat(string img_name, int widht, int height);

void printfMluTime(string message, float mluTime);

void printfMluTime(float mluTime);

void printfAccuracy(int imageNum, float acc1, float acc5);

void printPerf(int imageNum, float execTime, float mluTime , int threads = 1, int batchsize = 1);

void saveResult(int imageNum, float top1, float top5, float meanAp,
    float hardwaretime, float endToEndTime, int threads = 1, int batchsize = 1);

string float2string(float value);

vector<int> to_cpu_shape(const vector<int>& mlu_shape);

vector<int> to_mlu_shape(const vector<int>& cpu_shape);

class Timer {
  public:
  Timer() : time_interval_(0.) {
    start_ = high_resolution_clock::now();
  }
  void log(const char* msg) {
    if (time_interval_ == 0.) {
      auto end = high_resolution_clock::now();
      std::chrono::duration<float> diff = end - start_;
      time_interval_ =
        duration_cast<microseconds>(diff).count();
    }
    LOG(INFO) << msg << ": " << time_interval_ << " us";
  }
  float getDuration() {
    if (time_interval_ == 0.) {
      auto end = high_resolution_clock::now();
      duration<float> diff = end - start_;
      time_interval_ = duration_cast<microseconds>(diff).count();
    }
    return time_interval_;
  }

  protected:
  float time_interval_;
  time_point<high_resolution_clock> start_;
};

class ImageReader {
  public:
  ImageReader(const string& file_list_path,
              int thread_num = 1) : image_num_(0) {
    image_list_.resize(thread_num);
    std::ifstream file_list(file_list_path, std::ios::in);
    string line_tmp;
    CHECK(!file_list.fail()) << "Image file is invalid!";
    while (getline(file_list, line_tmp)) {
      image_list_[image_num_ % thread_num].push(line_tmp);
      image_num_++;
    }
    file_list.close();
    LOG(INFO) << "there are " << image_num_ << " figures in "
              << file_list_path;
  }
  inline vector<queue<string>> getImageList() { return image_list_; }
  inline int getImageNum() { return image_num_; }
  private:
    vector<queue<string>> image_list_;
    int image_num_;
};

#endif  //   EXAMPLES_COMMON_INCLUDE_COMMON_FUNCTIONS_HPP_
