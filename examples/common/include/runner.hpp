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

#ifndef EXAMPLES_COMMON_INCLUDE_RUNNER_HPP_
#define EXAMPLES_COMMON_INCLUDE_RUNNER_HPP_
#include <vector>
#include <string>
#include <queue>
#include <sstream>
#include <thread> // NOLINT
#include <utility>
#include <iomanip>
#include <map>
#include <fstream>
#ifdef USE_MLU
#include "cnrt.h" // NOLINT
#endif
#include "blocking_queue.hpp"
#include "queue.hpp"
#include "post_processor.hpp"

using std::vector;
using std::string;

template <typename Dtype, template <typename> class Qtype>
class Runner {
  public:
  Runner():initSerialMode(false) {}
  virtual ~Runner() {}
  int n() {return inNum_;}
  int c() {return inChannel_;}
  int h() {return inHeight_;}
  int w() {return inWidth_;}

  void pushValidInputData(Dtype* data) { validInputFifo_.push(data); }
  void pushFreeInputData(Dtype* data) { freeInputFifo_.push(data); }
  Dtype* popValidInputData() { return validInputFifo_.pop(); }
  Dtype* popFreeInputData() { return freeInputFifo_.pop(); }
  void pushValidOutputData(Dtype* data) { validOutputFifo_.push(data);}
  void pushFreeOutputData(Dtype* data) { freeOutputFifo_.push(data);}
  Dtype* popValidOutputData() { return validOutputFifo_.pop(); }
  Dtype* popFreeOutputData() { return freeOutputFifo_.pop(); }
  void pushValidInputNames(vector<string> images) { imagesFifo_.push(images); }
  vector<string> popValidInputNames() { return imagesFifo_.pop(); }

  virtual void runParallel() {}
  virtual void runSerial() {}

  inline int modelParallel() { return modelParallel_; }
  inline int dataParallel() { return dataParallel_; }
  inline int inBlobNum() { return inBlobNum_; }
  inline int inCount() { return inCount_; }
  inline int outBlobNum() { return outBlobNum_; }
  inline int outCount() { return outCount_; }
  inline int outNum() { return outNum_; }
  inline int outChannel() { return outChannel_; }
  inline int outHeight() { return outHeight_; }
  inline int outWidth() { return outWidth_; }
  inline int threadId() { return threadId_; }
  inline int deviceId() { return deviceId_; }
  inline int deviceSize() { return deviceSize_; }
  inline float runTime() { return runTime_; }
  inline void setThreadId(int id) { threadId_ = id; }
  inline void setPostProcessor(PostProcessor<Dtype, Qtype> *p ) { postProcessor_ = p; }

  private:
  Qtype<Dtype*> validInputFifo_;
  Qtype<Dtype*> freeInputFifo_;
  Qtype<Dtype*> validOutputFifo_;
  Qtype<Dtype*> freeOutputFifo_;
  Qtype<vector<string> > imagesFifo_;

  protected:
  int inBlobNum_, outBlobNum_;
  unsigned int inNum_, inChannel_, inHeight_, inWidth_;
  unsigned int outNum_, outChannel_, outHeight_, outWidth_;
  int inCount_, outCount_;
  int threadId_ = 0;
  int deviceId_;
  int deviceSize_ = 1;
  int dataParallel_;
  int modelParallel_ = 1;
  float runTime_;

  bool initSerialMode;

  PostProcessor<Dtype, Qtype> *postProcessor_;
};
#endif  // EXAMPLES_COMMON_INCLUDE_RUNNER_HPP_
