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

#ifndef INCLUDE_CAFFE_MLU_FUSION_HPP_
#define INCLUDE_CAFFE_MLU_FUSION_HPP_
#ifdef USE_MLU

#include <algorithm>
#include <string>
#include <unordered_set>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/mlu/util.hpp"
#include "cnml.h"  // NOLINT

namespace caffe {

template <typename Dtype>
class MFusion {
  public:
  MFusion()
      : fuse_op_(nullptr),
        compiled_(false),
        tensor_converted_(false),
        memory_converted_(false) {}

  void reset();
  void fuse(cnmlBaseOp_t op);
  void compile();
  void addInput(Blob<Dtype>* input);
  void addOutput(Blob<Dtype>* output);
  void addInputs(vector<Blob<Dtype>*> inputs);
  void addOutputs(vector<Blob<Dtype>*> outputs);

  void sortIO(const unordered_map<Blob<Dtype>*, int>& blob2index);
  string inputBlobs(const unordered_map<Blob<Dtype>*, string>& blob2name);
  string outputBlobs(const unordered_map<Blob<Dtype>*, string>& blob2name);

  vector<string> inputBlobNames(
      const unordered_map<Blob<Dtype>*, string>& blob2name);
  vector<string> outputBlobNames(
      const unordered_map<Blob<Dtype>*, string>& blob2name);

  // for layers declaring input but doesn't use blob *as* input
  // Proposal layer of Faster R-CNN is an example...
  void rmInput(Blob<Dtype>* input);
  void rmOutput(Blob<Dtype>* output);

  void forward();
  void backward();

  cnmlFusionOp_t op() const { return fuse_op_; }
  float get_event_time() { return event_time_; }

  ~MFusion() { destroy(); }

  private:
  cnmlFusionOp_t fuse_op_;
  bool compiled_;
  vector<Blob<Dtype>*> inputs_;
  vector<Blob<Dtype>*> outputs_;

  // the sequence in these vectors are fixed.
  // these tensor and memory pointers are used after compiling.
  bool tensor_converted_;
  vector<cnmlTensor_t> input_tensors_;
  vector<cnmlTensor_t> output_tensors_;
  bool memory_converted_;
  vector<void*> input_mem_;
  vector<void*> output_mem_;
  float event_time_ = 0;

  void create();
  void destroy();

  void setFusionIO();
  void resetFusionIO();
  void blob2Tensors();
  void blob2Memory();

  MFusion(const MFusion&) = delete;
  MFusion& operator=(const MFusion&) = delete;
  void* operator new(const size_t) = delete;
  void operator delete(void* ptr) = delete;
};

}  // namespace caffe

#endif  // USE_MLU
#endif  // INCLUDE_CAFFE_MLU_FUSION_HPP_
