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

#ifndef EXAMPLES_COMMON_INCLUDE_ON_RUNNER_HPP_
#define EXAMPLES_COMMON_INCLUDE_ON_RUNNER_HPP_
#include <vector>
#include <string>
#ifdef USE_MLU
#include "cnrt.h" //NOLINT
#endif
#include "blocking_queue.hpp"
#include "queue.hpp"
#include "caffe/caffe.hpp"
#include "post_processor.hpp"
#include "runner.hpp"

using std::vector;
using std::string;

// #define PRE_READ
template <typename Dtype, template <typename> class Qtype>
class OnRunner : public Runner<Dtype, Qtype>{
  public:
  OnRunner(const string& onlinemodel,
           const string& onlineweights,
           const int& id,
           const int& deviceId,
           const int& deviceSize);
  OnRunner(const string& onlinemodel,
           const string& onlineweights,
           const int& deviceId);
  ~OnRunner();
  inline caffe::Net<Dtype>* net() {return net_;}

  virtual void runParallel();
  virtual void runSerial();

  private:
  caffe::Net<Dtype>* net_;
  vector<Dtype*> allocatedMLUPtrs_;
  vector<Dtype*> allocatedCpuPtrs_;
  vector<Dtype*> allocatedSyncPtrs_;
  vector<Dtype*> allocatedSyncTmpPtrs_;
};
#endif  // EXAMPLES_COMMON_INCLUDE_ON_RUNNER_HPP_
