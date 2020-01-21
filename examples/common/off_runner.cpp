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

#if defined(USE_MLU) && defined(USE_OPENCV)
#include <algorithm>
#include <atomic>
#include <condition_variable> // NOLINT
#include <iomanip>
#include <iosfwd>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <thread> // NOLINT
#include <utility>
#include <vector>
#include "include/runner.hpp"
#include "include/off_runner.hpp"
#include "include/command_option.hpp"
#include "include/common_functions.hpp"
#include "include/runner_strategy.hpp"


using std::map;
using std::max;
using std::min;
using std::queue;
using std::thread;
using std::stringstream;

template<typename Dtype, template <typename> class Qtype>
OffRunner<Dtype, Qtype>::OffRunner(const cnrtRuntimeContext_t rt_ctx,
                                   const int& id) {
  this->rt_ctx_ = rt_ctx;
  this->threadId_ = id;
  this->runTime_ = 0;
  this->simple_flag_ = true;

  cnrtGetRuntimeContextInfo(rt_ctx, CNRT_RT_CTX_FUNCTION,
         reinterpret_cast<void **>(&this->function_));
  cnrtGetRuntimeContextInfo(rt_ctx, CNRT_RT_CTX_MODEL_PARALLEL,
         reinterpret_cast<void **>(&this->Parallel_));
  cnrtGetRuntimeContextInfo(rt_ctx, CNRT_RT_CTX_DEV_ORDINAL,
         reinterpret_cast<void **>(&this->deviceId_));

  getIODataDesc();
  runnerStrategy_ = new SimpleStrategy<Dtype, Qtype>();
}

template<typename Dtype, template <typename> class Qtype>
OffRunner<Dtype, Qtype>::OffRunner(const string& offlinemodel,
                                   const int& id,
                                   const int& parallel,
                                   const int& deviceId,
                                   const int& devicesize) {
  this->threadId_ = id;
  this->deviceId_ = deviceId;
  this->deviceSize_ = devicesize;
  this->runTime_ = 0;
  this->simple_flag_ = false;
  setDeviceId(deviceId);
  loadOfflinemodel(offlinemodel);
  getIODataDesc();
  runnerStrategy_ = new FlexibleStrategy<Dtype, Qtype>();
}

template<typename Dtype, template <typename> class Qtype>
OffRunner<Dtype, Qtype>::~OffRunner() {
  setDeviceId(this->deviceId_);
  delete runnerStrategy_;
}


// get function's I/O DataDesc
template<typename Dtype, template <typename> class Qtype>
void OffRunner<Dtype, Qtype>::getIODataDesc() {
  CNRT_CHECK(cnrtGetInputDataSize(&this->inputSizeArray_,
        &this->inBlobNum_, this->function_));
  CNRT_CHECK(cnrtGetOutputDataSize(&this->outputSizeArray_,
        &this->outBlobNum_, this->function_));
  CNRT_CHECK(cnrtGetInputDataType(&this->mluInputDtype_,
        &this->inBlobNum_, this->function_));
  CNRT_CHECK(cnrtGetOutputDataType(&this->mluOutputDtype_,
        &this->outBlobNum_, this->function_));
  LOG(INFO) << "input blob num is " << this->inBlobNum_;

  this->inCounts_.resize(this->inBlobNum_, 1);
  this->outCounts_.resize(this->outBlobNum_, 1);
  this->inDimNums_.resize(this->inBlobNum_, 0);
  this->outDimNums_.resize(this->outBlobNum_, 0);
  this->inDimValues_.resize(this->inBlobNum_, nullptr);
  this->outDimValues_.resize(this->outBlobNum_, nullptr);

  for (int i = 0; i < this->inBlobNum_; i++) {
    CNRT_CHECK(cnrtGetInputDataShape(&(this->inDimValues_[i]),
        &(this->inDimNums_[i]), i, this->function_));
    for (int j = 0; j < this->inDimNums_[i]; ++j) {
      this->inCounts_[i] *= this->inDimValues_[i][j];
      LOG(INFO) << "shape " << this->inDimValues_[i][j];
    }
    if (i == 0) {
      this->inNum_ = this->inDimValues_[i][0];
      this->inChannel_ = this->inDimValues_[i][3];
      this->inHeight_ = this->inDimValues_[i][1];
      this->inWidth_ = this->inDimValues_[i][2];
    }
  }

  for (int i = 0; i < this->outBlobNum_; i++) {
    CNRT_CHECK(cnrtGetOutputDataShape(&(this->outDimValues_[i]),
        &(this->outDimNums_[i]), i, this->function_));
    for (int j = 0; j < this->outDimNums_[i]; ++j) {
      this->outCounts_[i] *= this->outDimValues_[i][j];
      LOG(INFO) << "shape " << this->outDimValues_[i][j];
    }
    if (i == 0) {
      this->outNum_ = this->outDimValues_[i][0];
      this->outChannel_ = this->outDimValues_[i][3];
      this->outHeight_ = this->outDimValues_[i][1];
      this->outWidth_ = this->outDimValues_[i][2];
    }
  }
}

// load model and get function
template<typename Dtype, template <typename> class Qtype>
void OffRunner<Dtype, Qtype>::loadOfflinemodel(const string& offlinemodel) {
  LOG(INFO) << "load file: " << offlinemodel.c_str();
  cnrtLoadModel(&model_, offlinemodel.c_str());
  int parallel;
  cnrtQueryModelParallelism(model_, &parallel);

  CHECK_LE(FLAGS_apiversion, 2) << "The version number should be 1 or 2";
  CHECK_GE(FLAGS_apiversion, 1) << "The version number should be 1 or 2";
  const string name = "subnet0";
  cnrtCreateFunction(&function_);
  cnrtExtractFunction(&function_, model_, name.c_str());
}

template<typename Dtype, template <typename> class Qtype>
void OffRunner<Dtype, Qtype>::runParallel() {
  runnerStrategy_->runParallel(this);
}

template<typename Dtype, template <typename> class Qtype>
void OffRunner<Dtype, Qtype>::runSerial() {
  if (!this->initSerialMode) {
    CHECK(cnrtCreateQueue(&queue_) == CNRT_RET_SUCCESS) << "CNRT create queue error";
    if (cnrtCreateRuntimeContext(&this->rt_ctx_, function(), nullptr) != CNRT_RET_SUCCESS) {
      LOG(FATAL)<< "Failed to create runtime context";
    }
    cnrtSetRuntimeContextDeviceId(this->rt_ctx_, this->deviceId_);
    cnrtInitRuntimeContext(this->rt_ctx_, NULL);
    this->initSerialMode = true;
  }

  cnrtNotifier_t notifierBeginning, notifierEnd;
  cnrtCreateNotifier(&notifierBeginning);
  cnrtCreateNotifier(&notifierEnd);
  float eventInterval;

  Dtype* mluInData = this->popValidInputData();
  Dtype* mluOutData = this->popFreeOutputData();

  void* param[this->inBlobNum_ + this->outBlobNum_];
  for (int i = 0; i < this->inBlobNum_; i++) {
    param[i] = mluInData[i];
  }
  for (int i = 0; i < this->outBlobNum_; i++) {
    param[this->inBlobNum_ + i] = mluOutData[i];
  }

  cnrtPlaceNotifier(notifierBeginning, queue_);
  CNRT_CHECK(cnrtInvokeRuntimeContext(this->rt_ctx_, param, queue_, nullptr));
  cnrtPlaceNotifier(notifierEnd, queue_);
  if (cnrtSyncQueue(queue_) == CNRT_RET_SUCCESS) {
    cnrtNotifierDuration(notifierBeginning, notifierEnd, &eventInterval);
    this->runTime_ += eventInterval;
    printfMluTime(eventInterval);
  } else {
    LOG(ERROR) << " SyncQueue error";
  }

  this->pushValidOutputData(mluOutData);
  this->pushFreeInputData(mluInData);

  cnrtDestroyNotifier(&notifierBeginning);
  cnrtDestroyNotifier(&notifierEnd);
}

INSTANTIATE_OFF_CLASS(OffRunner);
#endif  // defined(USE_MLU) && defined(USE_OPENCV)
