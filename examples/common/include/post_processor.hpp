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

#ifndef EXAMPLES_COMMON_INCLUDE_POST_PROCESSOR_HPP_
#define EXAMPLES_COMMON_INCLUDE_POST_PROCESSOR_HPP_
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <thread> // NOLINT
#include <utility>
#include <queue>
#include <map>
#include <fstream>
#include "glog/logging.h"

using std::map;
using std::vector;
using std::string;
using std::stringstream;

template<typename Dtype, template <typename> class Qtype> class Runner;

template<typename Dtype, template <typename> class Qtype>
class PostProcessor {
  public:
  PostProcessor() : threadId_(0), deviceId_(0),
                    initSerialMode(false) {}
  virtual ~PostProcessor() {}
  virtual void runParallel() {}
  virtual void runSerial() {}
  inline void setThreadId(int id) { threadId_ = id; }
  inline void setRunner(Runner<Dtype, Qtype> *p) { runner_ = p; }
  virtual int top1() { return 0;}
  virtual int top5() { return 0;}

  protected:
  int threadId_;
  int deviceId_;
  int total_ = 0;
  int outCount_ = 0;
  int outN_ = 0;

  bool initSerialMode;

  Runner<Dtype, Qtype> *runner_;
};

class NormalizedBBox {
  public:
  NormalizedBBox() {has_set_size = false;}
  ~NormalizedBBox() {}

  float xmin() const {return xmin_;}
  float xmax() const {return xmax_;}
  float ymin() const {return ymin_;}
  float ymax() const {return ymax_;}
  float size() const {return size_;}
  float score() const {return score_;}
  bool has_size() const {
    if (has_set_size)
      return true;
    else
      return false;
  }

  void set_xmin(float value) {xmin_ = value;}
  void set_xmax(float value) {xmax_ = value;}
  void set_ymin(float value) {ymin_ = value;}
  void set_ymax(float value) {ymax_ = value;}
  void set_size(float value) {
    size_ = value;
    has_set_size = true;
  }
  void set_score(float value) {score_ = value;}

  private:
  float xmin_;
  float ymin_;
  float xmax_;
  float ymax_;
  int label_;
  bool difficult_;
  float score_;
  float size_;
  bool has_set_size;
};

typedef map<int, vector<NormalizedBBox> > LabelBBox;

#endif  // EXAMPLES_COMMON_INCLUDE_POST_PROCESSOR_HPP_
