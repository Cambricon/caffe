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

#if defined(USE_MLU) && defined(USE_OPENCV)
#include "glog/logging.h"
#include "cnrt.h" // NOLINT
#include "clas_off_post.hpp"
#include "post_processor.hpp"
#include "runner.hpp"
#include "off_runner.hpp"
#include "command_option.hpp"
#include "common_functions.hpp"

using std::vector;
using std::string;

template<typename Dtype, template <typename> class Qtype>
void ClassOffPostProcessor<Dtype, Qtype>::runParallel() {
  OffRunner<Dtype, Qtype> *infr = static_cast<OffRunner<Dtype, Qtype>*>(this->runner_);
  this->outCount_ = infr->outCount();
  this->outN_ = infr->outNum();

  setDeviceId(infr->deviceId());
  int channel = this->threadId_ / infr->deviceSize() % 4;
  cnrtSetCurrentChannel((cnrtChannelType_t)channel);

  this->readLabels(&this->labels);

  outCpuPtrs_ = new(Dtype);
  outCpuPtrs_[0] = new float[infr->outCount()];

  while (true) {
    Dtype* mluOutData = infr->popValidOutputData();
    if (mluOutData == nullptr) break;  // no more work

    Timer copyout;
    CNRT_CHECK(cnrtMemcpyBatchByDescArray(outCpuPtrs_,
                                          mluOutData,
                                          infr->outDescs(),
                                          1,
                                          infr->dataParallel(),
                                          CNRT_MEM_TRANS_DIR_DEV2HOST));
    copyout.log("copyout time ...");

    Timer postProcess;
    infr->pushFreeOutputData(mluOutData);
    float* data = reinterpret_cast<float*>(outCpuPtrs_[0]);

    vector<string> origin_img = infr->popValidInputNames();
    this->updateResult(origin_img, this->labels, data);
    postProcess.log("post process time ...");
  }
  this->printClassResult();
}

INSTANTIATE_OFF_CLASS(ClassOffPostProcessor);
#endif  // defined(USE_MLU) && defined(USE_OPENCV)
