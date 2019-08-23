/*
All modification made by Cambricon Corporation: © 2018--2019 Cambricon Corporation
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


using std::map;
using std::max;
using std::min;
using std::queue;
using std::thread;
using std::stringstream;

template<typename Dtype, template <typename> class Qtype>
OffRunner<Dtype, Qtype>::OffRunner(const string& offlinemodel,
                                   const int& id,
                                   const int& dp,
                                   const int& deviceId,
                                   const int& devicesize) {
  this->threadId_ = id;
  this->deviceId_ = deviceId;
  this->deviceSize_ = devicesize;
  this->runTime_ = 0;

  // 1. set current device
  setDeviceId(deviceId);

  // 2. load model and get function
  LOG(INFO) << "load file: " << offlinemodel.c_str();
  cnrtLoadModel(&model_, offlinemodel.c_str());
  int mp;
  cnrtQueryModelParallelism(model_, &mp);
  if (FLAGS_dataparallel * mp <= 32) {
    this->dataParallel_ = dp;
    this->modelParallel_ = mp;
  } else {
    this->dataParallel_ = 1;
    this->modelParallel_ = 1;
    LOG(ERROR) << "dataparallel * modelparallel should <= 32, changed them to 1";
  }
  const string name = "subnet0";
  cnrtCreateFunction(&function_);
  cnrtExtractFunction(&function_, model_, name.c_str());

  // 3. get function's I/O DataDesc
  cnrtGetInputDataDesc(&inDescs_, &this->inBlobNum_, function_);
  cnrtGetOutputDataDesc(&outDescs_, &this->outBlobNum_, function_);

#if !defined(CROSS_COMPILE) && !defined(CROSS_COMPILE_ARM64)
  uint64_t stack_size;
  cnrtQueryModelStackSize(model_, &stack_size);
  unsigned int current_device_size;
  cnrtGetStackMem(&current_device_size);
  if (stack_size > current_device_size) {
    cnrtSetStackMem(stack_size + 50);
  }
#endif  // CROSS_COMPILE && CROSS_COMPILE_ARM64

  LOG(INFO) << "input blob num is " << this->inBlobNum_;
  int in_count;
  for (int i = 0; i < this->inBlobNum_; i++) {
    unsigned int inN, inC, inH, inW;
    cnrtDataDesc_t desc = inDescs_[i];
    cnrtGetHostDataCount(desc, &in_count);
    if (FLAGS_yuv) {
      cnrtSetHostDataLayout(desc, CNRT_UINT8, CNRT_NCHW);
    } else {
      cnrtSetHostDataLayout(desc, CNRT_FLOAT32, CNRT_NCHW);
    }
    cnrtGetDataShape(desc, &inN, &inC, &inH, &inW);
    in_count *= this->dataParallel_;
    inN *= this->dataParallel_;
    LOG(INFO) << "shape " << inN;
    LOG(INFO) << "shape " << inC;
    LOG(INFO) << "shape " << inH;
    LOG(INFO) << "shape " << inW;
    if (i == 0) {
      this->inNum_ = inN;
      this->inChannel_ = inC;
      this->inWidth_ = inW;
      this->inHeight_ = inH;
      this->inCount_ = in_count;
    } else {
      cnrtGetHostDataCount(desc, &in_count);
    }
  }

  for (int i = 0; i < this->outBlobNum_; i++) {
    cnrtDataDesc_t desc = outDescs_[i];
    cnrtSetHostDataLayout(desc, CNRT_FLOAT32, CNRT_NCHW);
    cnrtGetHostDataCount(desc, &this->outCount_);
    cnrtGetDataShape(desc, &this->outNum_, &this->outChannel_,
                      &this->outHeight_, &this->outWidth_);
    this->outCount_ *= this->dataParallel_;
    this->outNum_ *= this->dataParallel_;
    LOG(INFO) << "output shape " << this->outNum_;
    LOG(INFO) << "output shape " << this->outChannel_;
    LOG(INFO) << "output shape " << this->outHeight_;
    LOG(INFO) << "output shape " << this->outWidth_;
  }
}

template<typename Dtype, template <typename> class Qtype>
OffRunner<Dtype, Qtype>::~OffRunner() {
  setDeviceId(this->deviceId_);
  cnrtDestroyQueue(queue_);
  cnrtDestroyFunction(function_);
  cnrtUnloadModel(model_);
  for (auto ptr : inPtrVector_)
    cnrtFreeArray(ptr, this->inBlobNum_);
  for (auto ptr : outPtrVector_)
    cnrtFreeArray(ptr, this->outBlobNum_);
}

template<typename Dtype, template <typename> class Qtype>
void OffRunner<Dtype, Qtype>::runParallel() {
  setDeviceId(this->deviceId_);
  int channel = this->threadId_ / this->deviceSize_ % 4;
  cnrtSetCurrentChannel((cnrtChannelType_t)channel);
  CHECK(cnrtCreateQueue(&queue_) == CNRT_RET_SUCCESS)
        << "CNRT create queue error, thread_id " << this->threadId_;
  // initialize function memory
  cnrtInitFuncParam_t initFuncParam;
  bool muta = false;
  unsigned int affinity = 0x01;
  int dp = this->dataParallel_;
  initFuncParam.muta = &muta;
  initFuncParam.affinity = &affinity;
  initFuncParam.data_parallelism = &dp;
  initFuncParam.end = CNRT_PARAM_END;
  cnrtInitFunctionMemory_V2(function_, &initFuncParam);
  cnrtNotifier_t notifierBeginning, notifierEnd;
  cnrtCreateNotifier(&notifierBeginning);
  cnrtCreateNotifier(&notifierEnd);
  float eventInterval;

  while (true) {
    Dtype* mluInData = this->popValidInputData();
    if ( mluInData == nullptr ) break;  // no more images

    Dtype* mluOutData = this->popFreeOutputData();
    void* param[this->inBlobNum_ + this->outBlobNum_];
    for (int i = 0; i < this->inBlobNum_; i++) {
      param[i] = mluInData[i];
    }
    for (int i = 0; i < this->outBlobNum_; i++) {
      param[this->inBlobNum_ + i] = mluOutData[i];
    }
    cnrtDim3_t dim = {1, 1, 1};
    cnrtInvokeFuncParam_t invokeFuncParam;
    invokeFuncParam.data_parallelism = &dp;
    invokeFuncParam.affinity = &affinity;
    invokeFuncParam.end = CNRT_PARAM_END;
#if defined(CROSS_COMPILE) || defined(CROSS_COMPILE_ARM64)
    struct timeval tpend, tpstart;
    gettimeofday(&tpstart, NULL);
#endif
    cnrtPlaceNotifier(notifierBeginning, queue_);
    CNRT_CHECK(cnrtInvokeFunction_V2(function_, dim, param, (cnrtFunctionType_t)0,
        queue_, &invokeFuncParam));
    cnrtPlaceNotifier(notifierEnd, queue_);
#if defined(CROSS_COMPILE) || defined(CROSS_COMPILE_ARM64)
    gettimeofday(&tpend, NULL);
#endif
    if (cnrtSyncQueue(queue_) == CNRT_RET_SUCCESS) {
      cnrtNotifierDuration(notifierBeginning, notifierEnd, &eventInterval);
#if defined(CROSS_COMPILE) || defined(CROSS_COMPILE_ARM64)
      eventInterval = 1000000 * (tpend.tv_sec - tpstart.tv_sec) +
          tpend.tv_usec - tpstart.tv_usec;
#endif
      this->runTime_ += eventInterval;
      printfMluTime(eventInterval);
    } else {
      LOG(ERROR) << " SyncQueue error";
    }

    this->pushValidOutputData(mluOutData);
    this->pushFreeInputData(mluInData);
  }

  cnrtDestroyNotifier(&notifierBeginning);
  cnrtDestroyNotifier(&notifierEnd);
  this->pushValidOutputData(static_cast<Dtype*>(nullptr));  // tell postprocessor to exit
}

template<typename Dtype, template <typename> class Qtype>
void OffRunner<Dtype, Qtype>::runSerial() {
  if (!this->initSerialMode) {
    CHECK(cnrtCreateQueue(&queue_) == CNRT_RET_SUCCESS) << "CNRT create queue error";

    // initliaz function memory
    cnrtInitFuncParam_t initFuncParam;
    bool muta = false;
    unsigned int affinity = 0x01;
    int dp = 1;
    initFuncParam.muta = &muta;
    initFuncParam.affinity = &affinity;
    initFuncParam.data_parallelism = &dp;
    initFuncParam.end = CNRT_PARAM_END;
    cnrtInitFunctionMemory_V2(function_, &initFuncParam);
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
  unsigned int affinity = 0x01;
  int dp = 1;
  cnrtDim3_t dim = {1, 1, 1};
  cnrtInvokeFuncParam_t invokeFuncParam;
  invokeFuncParam.data_parallelism = &dp;
  invokeFuncParam.affinity = &affinity;
  invokeFuncParam.end = CNRT_PARAM_END;

#if defined(CROSS_COMPILE) || defined(CROSS_COMPILE_ARM64)
  struct timeval tpend, tpstart;
  gettimeofday(&tpstart, NULL);
#endif
  cnrtPlaceNotifier(notifierBeginning, queue_);
  CNRT_CHECK(cnrtInvokeFunction_V2(function_, dim, param, (cnrtFunctionType_t)0,
      queue_, &invokeFuncParam));
  cnrtPlaceNotifier(notifierEnd, queue_);
#if defined(CROSS_COMPILE) || defined(CROSS_COMPILE_ARM64)
  gettimeofday(&tpend, NULL);
#endif
  if (cnrtSyncQueue(queue_) == CNRT_RET_SUCCESS) {
    cnrtNotifierDuration(notifierBeginning, notifierEnd, &eventInterval);
#if defined(CROSS_COMPILE) || defined(CROSS_COMPILE_ARM64)
    eventInterval = 1000000 * (tpend.tv_sec - tpstart.tv_sec) +
        tpend.tv_usec - tpstart.tv_usec;
#endif
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
