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
#include "glog/logging.h"
#include "cnrt.h"  // NOLINT
#include "post_processor.hpp"
#include "clas_on_post.hpp"
#include "runner.hpp"
#include "on_runner.hpp"
#include "command_option.hpp"
#include "common_functions.hpp"
#include "simple_interface.hpp"

using std::vector;
using std::string;

template <typename Dtype, template <typename> class Qtype>
void ClassOnPostProcessor<Dtype, Qtype>::runParallel() {
  OnRunner<Dtype, Qtype> *runner = static_cast<OnRunner<Dtype, Qtype>*>(this->runner_);
  ::setupConfig(this->threadId_, runner->deviceId(), runner->deviceSize());

  this->outCount_ = runner->outCounts()[0];
  this->outN_ = runner->outNum();

  this->readLabels(&this->labels);

  caffe::Net<float>* netBuff = runner->net();
  int outputCount = netBuff->output_blobs()[0]->count();
  outputCpuPtr_ = new Dtype[outputCount];
  size_t tensorSize;
  MLU_CHECK(cnmlGetTensorSize_V2(netBuff->output_blobs()[0]->mlu_tensor(), &tensorSize));
  while (true) {
    Dtype* outputMluPtr = runner->popValidOutputData();
    Dtype* outputSyncPtr = runner->popValidOutputSyncData();
    if (nullptr == outputMluPtr) break;  // no more work, exit

    auto outputBlob = netBuff->output_blobs()[0];
    Timer timer;
    CNRT_CHECK(cnrtMemcpy(outputSyncPtr, outputMluPtr,
          tensorSize, CNRT_MEM_TRANS_DIR_DEV2HOST));
    cnrtDataType_t cpuDtype = to_cnrt_dtype(outputBlob->cpu_type());
    cnrtDataType_t mluDtype = to_cnrt_dtype(outputBlob->mlu_type());
    int dim_values[4] = {outputBlob->mlu_shape()[0], outputBlob->mlu_shape()[1],
                         outputBlob->mlu_shape()[2], outputBlob->mlu_shape()[3]};
    int dim_order[4] = {0, 3, 1, 2};
    if (mluDtype != cpuDtype) {
      CNRT_CHECK(cnrtTransOrderAndCast(reinterpret_cast<void*>(outputSyncPtr),
                                       mluDtype,
                                       reinterpret_cast<void*>(outputCpuPtr_),
                                       cpuDtype,
                                       nullptr,
                                       outputBlob->mlu_shape().size(),
                                       dim_values,
                                       dim_order));
    } else {
      CNRT_CHECK(cnrtTransDataOrder(reinterpret_cast<void*>(outputSyncPtr),
                                    cpuDtype,
                                    reinterpret_cast<void*>(outputCpuPtr_),
                                    outputBlob->mlu_shape().size(),
                                    dim_values,
                                    dim_order));
    }
    timer.log("copy out time");
    vector<string> origin_img = runner->popValidInputNames();
    this->updateResult(origin_img, this->labels, outputCpuPtr_);
    runner->pushFreeOutputData(outputMluPtr);
    runner->pushFreeOutputSyncData(outputSyncPtr);
  }
  this->printClassResult();
}

INSTANTIATE_ON_CLASS(ClassOnPostProcessor);
#endif  // defined(USE_MLU) && defined(USE_OPENCV)
