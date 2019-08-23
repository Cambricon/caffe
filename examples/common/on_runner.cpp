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

#if defined(USE_OPENCV)
#include <string>
#include <vector>
#include "caffe/proto/caffe.pb.h"
#include "caffe/blob.hpp"
#include "caffe/util/io.hpp"

#include "include/command_option.hpp"

#include "include/on_data_provider.hpp"
#include "include/on_runner.hpp"
#include "include/pipeline.hpp"
#include "include/common_functions.hpp"

using std::string;
using std::vector;

template <typename Dtype, template <typename> class Qtype>
OnRunner<Dtype, Qtype>::OnRunner(const string& onlinemodel,
                      const string& onlineweights,
                      const int& ThrID,
                      const int& data_parallel,
                      const int& deviceId,
                      const int& deviceSize) {
#ifdef USE_MLU
  this->threadId_ = ThrID;
  this->deviceId_ = deviceId;
  this->deviceSize_ = deviceSize;
  this->runTime_ = 0;
  //  config like mmode must be set before net initialization
  setupConfig(this->threadId_, deviceId, deviceSize);
  this->dataParallel_ = data_parallel;
  net_ = new caffe::Net<float>(onlinemodel, caffe::TEST);
  net_->CopyTrainedLayersFrom(onlineweights);
  CHECK_EQ(net_->input_blobs().size(), 1) << "doesn't support multiple input";
  // for mmode = MLU, Reshape will compile op
  // so that we can use net's tensor for memcpy
  net_->Reshape();
  // for mmode = MFUS, op will be compiled during forward
  if (FLAGS_mmode == std::string("MFUS"))
    net_->ForwardPrefilled();
  auto inputBlob = net_->input_blobs()[0];
  auto outputBlob = net_->output_blobs()[0];
  this->inNum_ = inputBlob->num();
  this->inChannel_ = inputBlob->channels();
  this->inHeight_ = inputBlob->height();
  this->inWidth_ = inputBlob->width();

  this->outNum_ = outputBlob->num();
  this->outChannel_ = outputBlob->channels();
  this->outHeight_ = outputBlob->height();
  this->outWidth_ = outputBlob->width();
  this->inCount_ = inputBlob->count();
  this->outCount_ = outputBlob->count();
  for (int i = 0; i < FLAGS_fifosize; i++) {
    auto inputMluTensorPtr = net_->input_blobs()[0]->mlu_tensor();
    auto outputMluTensorPtr = net_->output_blobs()[0]->mlu_tensor();

    Dtype* inputMluPtr = reinterpret_cast<Dtype*>
                     (cnmlMallocBatchBuffer(inputMluTensorPtr, this->dataParallel_));
    Dtype* outputMluPtr = reinterpret_cast<Dtype*>
                      (cnmlMallocBatchBuffer(outputMluTensorPtr, this->dataParallel_));
    // save the Malloced memory to delete later
    allocatedMLUPtrs_.push_back(inputMluPtr);
    allocatedMLUPtrs_.push_back(outputMluPtr);
    this->pushFreeInputData(inputMluPtr);
    this->pushFreeOutputData(outputMluPtr);
  }
  if (FLAGS_mmode == std::string("MFUS"))
    caffe::Caffe::freeQueue();
#endif
}

template <typename Dtype, template <typename> class Qtype>
OnRunner<Dtype, Qtype>::OnRunner(const string& onlinemodel,
                            const string& onlineweights,
                            const int& deviceId) {
  this->deviceId_ = deviceId;
  this->runTime_ = 0;

  if (FLAGS_mmode == "CPU") {
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
  } else {
    //  config like mmode must be set before net initialization
    setupConfig(0, deviceId, 1);
  }

  //  config like mmode must be set before net initialization
  net_ = new caffe::Net<float>(onlinemodel, caffe::TEST);
  net_->CopyTrainedLayersFrom(onlineweights);
  CHECK_EQ(net_->input_blobs().size(), 1) << "doesn't support multiple input";

  auto inputBlob = net_->input_blobs()[0];
  auto outputBlob = net_->output_blobs()[0];
  this->inNum_ = inputBlob->num();
  this->inChannel_ = inputBlob->channels();
  this->inHeight_ = inputBlob->height();
  this->inWidth_ = inputBlob->width();

  this->outNum_ = outputBlob->num();
  this->outChannel_ = outputBlob->channels();
  this->outHeight_ = outputBlob->height();
  this->outWidth_ = outputBlob->width();
  this->inCount_ = inputBlob->count();
  this->outCount_ = outputBlob->count();
  for (int i = 0; i < 1; i++) {
    Dtype* inputCpuPtr = new Dtype[ this->inCount_ ];
    Dtype* outputCpuPtr = new Dtype[ this->outCount_ ];
    // save the Malloced memory to delete later
    allocatedCpuPtrs_.push_back(inputCpuPtr);
    allocatedCpuPtrs_.push_back(outputCpuPtr);
    this->pushFreeInputData(inputCpuPtr);
    this->pushFreeOutputData(outputCpuPtr);
  }
}

template <typename Dtype, template <typename> class Qtype>
OnRunner<Dtype, Qtype>::~OnRunner() {
  if (FLAGS_mmode != "CPU") {
    setupConfig(this->threadId_, this->deviceId_, this->deviceSize_);
  }
  delete net_;
  for (auto ptr : allocatedMLUPtrs_) {
#ifdef USE_MLU
    if (ptr != nullptr)
      cnmlFreeBuffer(ptr);
#endif
  }
  for (auto ptr : allocatedCpuPtrs_) {
      delete [] ptr;
  }
}

template <typename Dtype, template <typename> class Qtype>
void OnRunner<Dtype, Qtype>::runParallel() {
#ifdef USE_MLU
  setupConfig(this->threadId_, this->deviceId_, this->deviceSize_);

  cnrtNotifier_t notifierBeginning, notifierEnd;
  cnrtCreateNotifier(&notifierBeginning);
  cnrtCreateNotifier(&notifierEnd);

  while (true) {
    Dtype* inputMluPtr = this->popValidInputData();
    if (nullptr == inputMluPtr) break;  // no more work to do, exit

    Dtype* outputMluPtr = this->popFreeOutputData();
    auto inputBlob = net_->input_blobs()[0];
    auto outputBlob = net_->output_blobs()[0];
    inputBlob->set_mlu_data(inputMluPtr);
    outputBlob->set_mlu_data(outputMluPtr);

    float eventTimeUse;
    cnrtPlaceNotifier(notifierBeginning, caffe::Caffe::queue());

    net_->ForwardPrefilled();

    cnrtPlaceNotifier(notifierEnd, caffe::Caffe::queue());
    cnrtNotifierDuration(notifierBeginning, notifierEnd, &eventTimeUse);
    this->runTime_ +=  eventTimeUse;
    printfMluTime(eventTimeUse);

    this->pushValidOutputData(outputMluPtr);
    this->pushFreeInputData(inputMluPtr);
  }

  cnrtDestroyNotifier(&notifierBeginning);
  cnrtDestroyNotifier(&notifierEnd);
#endif
  this->pushValidOutputData(static_cast<Dtype*>(nullptr));
#ifdef USE_MLU
  caffe::Caffe::freeQueue();
#endif
}

template <typename Dtype, template <typename> class Qtype>
void OnRunner<Dtype, Qtype>::runSerial() {
  Dtype* inputCpuPtr = this->popValidInputData();
  auto input_blob = net_->input_blobs()[0];
  input_blob->set_cpu_data(reinterpret_cast<float*>(inputCpuPtr));

#ifdef DEBUG
  Timer timer;
#endif

  net_->ForwardPrefilled();

#ifdef USE_MLU
  if (caffe::Caffe::mode() != caffe::Caffe::CPU)
    cnrtSyncQueue(caffe::Caffe::queue());
#endif

#ifdef DEBUG
  timer.log("net forward execution time");
#endif

  this->pushFreeInputData(inputCpuPtr);
}

INSTANTIATE_ON_CLASS(OnRunner);

#endif  // USE_OPENCV
