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
#include <mutex>  // NOLINT
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
  Runner():initSerialMode(false), simple_flag_(false) {}
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
  void pushValidInputDataAndNames(Dtype* data, const vector<string>& names) {
    std::lock_guard<std::mutex> lk(runner_mutex_);
    pushValidInputData(data);
    pushValidInputNames(names);
  }
  void pushValidOutputSyncData(Dtype* data) { validOutputSyncFifo_.push(data);}
  void pushFreeOutputSyncData(Dtype* data) { freeOutputSyncFifo_.push(data);}
  Dtype* popValidOutputSyncData() { return validOutputSyncFifo_.pop(); }
  Dtype* popFreeOutputSyncData() { return freeOutputSyncFifo_.pop(); }
  void pushValidInputSyncData(Dtype* data) { validInputSyncFifo_.push(data); }
  void pushFreeInputSyncData(Dtype* data) { freeInputSyncFifo_.push(data); }
  Dtype* popValidInputSyncData() { return validInputSyncFifo_.pop(); }
  Dtype* popFreeInputSyncData() { return freeInputSyncFifo_.pop(); }
  void pushValidInputSyncTmpData(Dtype* data) { validInputSyncTmpFifo_.push(data); }
  void pushFreeInputSyncTmpData(Dtype* data) { freeInputSyncTmpFifo_.push(data); }
  Dtype* popValidInputSyncTmpData() { return validInputSyncTmpFifo_.pop(); }
  Dtype* popFreeInputSyncTmpData() { return freeInputSyncTmpFifo_.pop(); }

  virtual void runParallel() {}
  virtual void runSerial() {}

  inline int inBlobNum() { return inBlobNum_; }
  inline int outBlobNum() { return outBlobNum_; }
  inline int outNum() { return outNum_; }
  inline int outChannel() { return outChannel_; }
  inline int outHeight() { return outHeight_; }
  inline int outWidth() { return outWidth_; }
  inline int threadId() { return threadId_; }
  inline int deviceId() { return deviceId_; }
  inline int deviceSize() { return deviceSize_; }
  inline float runTime() { return runTime_; }
  inline void setRunTime(const float& time) {runTime_ = time;}
  inline void setThreadId(int id) { threadId_ = id; }
  inline void setPostProcessor(PostProcessor<Dtype, Qtype> *p ) { postProcessor_ = p; }
  inline bool simpleFlag() {return simple_flag_;}

  inline int64_t* inputSizeArray() { return inputSizeArray_; }
  inline int64_t* outputSizeArray() { return outputSizeArray_; }
  inline cnrtDataType_t* mluInputDtype() { return mluInputDtype_; }
  inline cnrtDataType_t* mluOutputDtype() { return mluOutputDtype_; }
  inline vector<int> inCounts() { return inCounts_; }
  inline vector<int> outCounts() { return outCounts_; }
  inline vector<int> inDimNums() { return inDimNums_; }
  inline vector<int> outDimNums() { return outDimNums_; }
  inline vector<int*> inDimValues() { return inDimValues_; }
  inline vector<int*> outDimValues() { return outDimValues_; }

  private:
  Qtype<Dtype*> validInputFifo_;
  Qtype<Dtype*> freeInputFifo_;
  Qtype<Dtype*> validInputSyncFifo_;
  Qtype<Dtype*> freeInputSyncFifo_;
  Qtype<Dtype*> validInputSyncTmpFifo_;
  Qtype<Dtype*> freeInputSyncTmpFifo_;
  Qtype<Dtype*> validOutputFifo_;
  Qtype<Dtype*> freeOutputFifo_;
  Qtype<Dtype*> validOutputSyncFifo_;
  Qtype<Dtype*> freeOutputSyncFifo_;
  Qtype<vector<string> > imagesFifo_;
  std::mutex runner_mutex_;

  protected:
  int inBlobNum_, outBlobNum_;
  unsigned int inNum_, inChannel_, inHeight_, inWidth_;
  unsigned int outNum_, outChannel_, outHeight_, outWidth_;
  int threadId_ = 0;
  int deviceId_;
  int deviceSize_ = 1;
  int Parallel_ = 1;
  float runTime_;
  bool initSerialMode;
  bool simple_flag_;

  int64_t *inputSizeArray_;
  int64_t* outputSizeArray_;
  cnrtDataType_t* mluInputDtype_;
  cnrtDataType_t* mluOutputDtype_;
  vector<int> inCounts_, outCounts_;
  vector<int> inDimNums_, outDimNums_;
  vector<int*> inDimValues_, outDimValues_;

  PostProcessor<Dtype, Qtype> *postProcessor_;
};
#endif  // EXAMPLES_COMMON_INCLUDE_RUNNER_HPP_
