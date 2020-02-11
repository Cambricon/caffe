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

#ifndef EXAMPLES_COMMON_INCLUDE_OFF_DATA_PROVIDER_HPP_
#define EXAMPLES_COMMON_INCLUDE_OFF_DATA_PROVIDER_HPP_
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <queue>
#include <string>
#include <vector>
#include "runner.hpp"
#include "data_provider.hpp"


using std::string;
using std::queue;
using std::vector;

// #define PRE_READ
template <typename Dtype, template <typename> class Qtype>
class OffDataProvider: public DataProvider<Dtype, Qtype>{
  public:
  explicit OffDataProvider(const string& meanfile,
                        const string& meanvalue,
                        const queue<string>& images):
            DataProvider<Dtype, Qtype>(meanfile, meanvalue, images) {}
  virtual ~OffDataProvider() {
    setDeviceId(this->deviceId_);
    delete [] reinterpret_cast<float*>(cpuData_[0]);
    delete cpuData_;
    delete [] reinterpret_cast<char*>(syncCpuData_[0]);
    delete syncCpuData_;
    for (auto ptr : inPtrVector_) {
      for (int i = 0; i < this->runner_->inBlobNum(); i++) {
        cnrtFree(ptr[i]);
      }
      if (ptr != nullptr) free(ptr);
    }
    for (auto ptr : outPtrVector_) {
      for (int i = 0; i < this->runner_->outBlobNum(); i++) {
        cnrtFree(ptr[i]);
      }
      if (ptr != nullptr) free(ptr);
    }
  }
  inline void pushInPtrVector(Dtype* data) { inPtrVector_.push_back(data); }
  inline void pushOutPtrVector(Dtype* data) { outPtrVector_.push_back(data); }
  virtual void runParallel();
  virtual void runSerial();

  protected:
  void allocateMemory(int queueLength);

  private:
  Dtype* cpuData_;
  Dtype* syncCpuData_;
  vector<Dtype*> inPtrVector_;
  vector<Dtype*> outPtrVector_;
};
#endif  // EXAMPLES_COMMON_INCLUDE_OFF_DATA_PROVIDER_HPP_
