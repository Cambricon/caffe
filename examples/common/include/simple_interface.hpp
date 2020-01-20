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

#ifndef EXAMPLES_COMMON_INCLUDE_SIMPLE_INTERFACE_HPP_
#define EXAMPLES_COMMON_INCLUDE_SIMPLE_INTERFACE_HPP_
#include <vector>
#include <string>
#include <queue>
#include <chrono>  // NOLINT
#include <iostream>
#include "cnrt.h" // NOLINT

using std::vector;
using std::string;

class SimpleInterface {
  public:
  static SimpleInterface& getInstance();
  void setFlag(bool b) {flag_ = b;}
  bool getFlag() {return flag_;}
  void loadOfflinemodel(const string& offlinemodel,
          const vector<int>& deviceIds, const bool& channel_dup);
  vector<cnrtRuntimeContext_t> get_runtime_contexts() {return dev_runtime_contexts_;}
  void destroyRuntimeContext();

  public:
  // 4 threads of data providers
  static const int data_provider_num_ = 4;
  // 4 threads of thread pool
  static const int thread_num = 4;

  private:
  SimpleInterface();
  ~SimpleInterface() = default;

  SimpleInterface(const SimpleInterface&)            = delete;
  SimpleInterface& operator=(const SimpleInterface&) = delete;
  SimpleInterface(SimpleInterface&&)                 = delete;
  SimpleInterface& operator=(SimpleInterface&&)      = delete;
  void prepareRuntimeContext(cnrtRuntimeContext_t *ctx, cnrtFunction_t function,
          int deviceId, const bool& channel_dup);

  private:
  bool flag_;
  uint64_t stack_size_;
  cnrtModel_t model_;
  vector<cnrtRuntimeContext_t> dev_runtime_contexts_;
};

#endif  //  EXAMPLES_COMMON_INCLUDE_SIMPLE_INTERFACE_HPP_
