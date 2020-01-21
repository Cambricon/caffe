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

#include "glog/logging.h"
#include "clas_processor.hpp"
#include "runner.hpp"
#include "common_functions.hpp"
#include "command_option.hpp"

template <typename Dtype, template <typename> class Qtype>
void ClassPostProcessor<Dtype, Qtype>::readLabels(vector<string>* labels) {
  if (!FLAGS_labels.empty()) {
    std::ifstream file(FLAGS_labels);
    if (file.fail())
      LOG(FATAL) << "failed to open labels file!";

    std::string line;
    while (getline(file, line)) {
      labels->push_back(line);
    }
    file.close();
    CHECK_EQ(this->outCount_ / this->outN_, labels->size())
        << "the number of classified objects is not equal to output of net";
  }
}

template <typename Dtype, template <typename> class Qtype>
void ClassPostProcessor<Dtype, Qtype>::updateResult(const vector<string>& origin_img,
                                    const vector<string>& labels,
                                    float* outCpuPtr) {
  for (int i = 0; i < this->outN_; i++) {
    string image = origin_img[i];
    if (image == "null") break;

    this->total_++;
    if (image.find_last_of(" ") != -1) {
      image = image.substr(0, image.find(" "));
    }
    vector<int> vtrTop5 = getTop5(labels,
                                  image,
                                  outCpuPtr + i * this->outCount_ / this->outN_,
                                  this->outCount_ / this->outN_);
    image = origin_img[i];
    if (image.find(" ") != string::npos) {
      image = image.substr(image.find(" "));
    }

    int labelID = atoi(image.c_str());
    for (int i = 0; i < 5; i++) {
      if (vtrTop5[i] == labelID) {
        this->top5_++;
        if (i == 0)
          this->top1_++;
        break;
      }
    }
  }
}

template <typename Dtype, template <typename> class Qtype>
void ClassPostProcessor<Dtype, Qtype>::printClassResult() {
  LOG(INFO) << "Accuracy thread id : " << this->runner_->threadId();
  LOG(INFO) << "accuracy1: " << 1.0 * this->top1_ / this->total_ << " ("
            << this->top1_ << "/" << this->total_ << ")";
  LOG(INFO) << "accuracy5: " << 1.0 * this->top5_ / this->total_ << " ("
            << this->top5_ << "/" << this->total_ << ")";
}

INSTANTIATE_ALL_CLASS(ClassPostProcessor);
