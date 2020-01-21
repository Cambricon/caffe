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

#include "include/simple_interface.hpp"
#include <glog/logging.h> // NOLINT


SimpleInterface& SimpleInterface::getInstance() {
  static SimpleInterface instance;
  return instance;
}

SimpleInterface::SimpleInterface(): flag_(false),
  stack_size_(0) {
}

// load model and get function. init runtime context for each device
void SimpleInterface::loadOfflinemodel(const string& offlinemodel,
       const vector<int>& deviceIds, const bool& channel_dup) {
  cnrtFunction_t function;

  LOG(INFO)<< "load file: " << offlinemodel.c_str();
  cnrtLoadModel(&model_, offlinemodel.c_str());
  cnrtQueryModelStackSize(model_, &stack_size_);

  const string name = "subnet0";
  cnrtCreateFunction(&function);
  cnrtExtractFunction(&function, model_, name.c_str());
  for (auto device : deviceIds) {
    LOG(INFO)<< "Init runtime context for device" << device;
    cnrtRuntimeContext_t ctx;
    prepareRuntimeContext(&ctx, function, device, channel_dup);
    dev_runtime_contexts_.push_back(ctx);
  }

  cnrtDestroyFunction(function);
}

void SimpleInterface::prepareRuntimeContext(cnrtRuntimeContext_t *ctx,
       cnrtFunction_t function, int deviceId, const bool& channel_dup) {
  cnrtRuntimeContext_t rt_ctx;
  // cnrtRet_t ret;

  if (cnrtCreateRuntimeContext(&rt_ctx, function, nullptr) != CNRT_RET_SUCCESS) {
    LOG(FATAL)<< "Failed to create runtime context";
  }

  // set device ordinal. if not set, a random device will be used
  cnrtSetRuntimeContextDeviceId(rt_ctx, deviceId);

  // Instantiate the runtime context on actual MLU device
  // All cnrtSetRuntimeContext* interfaces must be caller prior to cnrtInitRuntimeContext
  if (cnrtInitRuntimeContext(rt_ctx, nullptr) != CNRT_RET_SUCCESS) {
    LOG(FATAL)<< "Failed to initialize runtime context";
  }

  *ctx = rt_ctx;
}

void SimpleInterface:: destroyRuntimeContext() {
  for (auto ctx : dev_runtime_contexts_) {
    cnrtDestroyRuntimeContext(ctx);
  }

  cnrtUnloadModel(model_);
}
