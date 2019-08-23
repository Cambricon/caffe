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
#include "glog/logging.h"
#include "cnrt.h"  // NOLINT
#include "post_processor.hpp"
#include "clas_on_post.hpp"
#include "runner.hpp"
#include "on_runner.hpp"
#include "command_option.hpp"
#include "common_functions.hpp"

using std::vector;
using std::string;

template <typename Dtype, template <typename> class Qtype>
void ClassOnPostProcessor<Dtype, Qtype>::runParallel() {
  OnRunner<Dtype, Qtype> *runner = static_cast<OnRunner<Dtype, Qtype>*>(this->runner_);
  ::setupConfig(this->threadId_, runner->deviceId(), runner->deviceSize());

  this->outCount_ = runner->outCount();
  this->outN_ = runner->outNum();

  this->readLabels(&this->labels);

  caffe::Net<float>* netBuff = runner->net();
  int outputCount = netBuff->output_blobs()[0]->count();
  outputCpuPtr_ = new Dtype[outputCount];
  while (true) {
    Dtype* outputMluPtr = runner->popValidOutputData();
    if (nullptr == outputMluPtr) break;  // no more work, exit

    auto outputBlob = netBuff->output_blobs()[0];
    auto outputMluTensorPtr = outputBlob->mlu_tensor();
    auto outputCpuTensorPtr = outputBlob->cpu_tensor();
    Timer timer;
    cnmlMemcpyBatchTensorToHost(outputMluTensorPtr,
                                outputMluPtr,
                                outputCpuTensorPtr,
                                outputCpuPtr_,
                                runner->dataParallel());
    timer.log("copy out time");
    vector<string> origin_img = runner->popValidInputNames();
    this->updateResult(origin_img, this->labels, outputCpuPtr_);
    runner->pushFreeOutputData(outputMluPtr);
  }
  this->printClassResult();
}

INSTANTIATE_ON_CLASS(ClassOnPostProcessor);
#endif  // defined(USE_MLU) && defined(USE_OPENCV)
