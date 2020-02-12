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

#ifndef EXAMPLES_COMMON_INCLUDE_DATA_PROVIDER_HPP_
#define EXAMPLES_COMMON_INCLUDE_DATA_PROVIDER_HPP_
#if defined(USE_OPENCV)
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <queue>
#include <string>
#include <vector>
#include <sstream>
#include <thread> // NOLINT
#include <utility>
#include <iomanip>
#include <fstream>
#include <map>
#include "common_functions.hpp"
#include "runner.hpp"

using std::string;
using std::queue;
using std::vector;

// #define PRE_READ
template<typename Dtype, template <typename> class Qtype>
class DataProvider {
  public:
  explicit DataProvider(const string& meanfile,
                        const string& meanvalue,
                        const queue<string>& images):
                        threadId_(0), deviceId_(0),
                        meanFile_(meanfile), meanValue_(meanvalue),
                        imageList(images), initSerialMode(false) {}
  virtual ~DataProvider() {
  }
  void readOneBatch();
  bool imageIsEmpty();
  void preRead();
  virtual void SetMeanFile() {}
  virtual void SetMeanValue();
  void SetMean();
  void WrapInputLayer(vector<vector<cv::Mat>>* wrappedImages, float* inputData);
  void Preprocess(const vector<cv::Mat>& srcImages, vector<vector<cv::Mat> >* dstImages);
  inline void setThreadId(int id) { threadId_ = id; }
  inline void setRunner(Runner<Dtype, Qtype> *p) {
    runner_ = p;
    inNum_ = p->n();  // make preRead happy
  }
  virtual void runParallel() {}
  virtual void runSerial() {}

  cv::Mat ResizeMethod(cv::Mat sample, int inputDim, int mode);

  protected:
  int inNum_;
  int inChannel_;
  int inHeight_;
  int inWidth_;
  cv::Size inGeometry_;

  int threadId_;
  int deviceId_;

  string meanFile_;
  string meanValue_;
  cv::Mat mean_;

  queue<string> imageList;
  vector<vector<cv::Mat>> inImages_;
  vector<vector<string>> imageName_;

  bool initSerialMode;

  Runner<Dtype, Qtype> *runner_;
};
#endif  // USE_OPENCV
#endif  // EXAMPLES_COMMON_INCLUDE_DATA_PROVIDER_HPP_
