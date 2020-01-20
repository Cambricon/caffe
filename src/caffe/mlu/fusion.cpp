/*
All modification made by Cambricon Corporation: © 2018-2019 Cambricon Corporation
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

#include <algorithm>
#include <mutex>  //  NOLINT
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "caffe/mlu/fusion.hpp"

#ifdef USE_MLU
namespace caffe {

template <typename Dtype>
void MFusion<Dtype>::reset() {
  CHECK(Caffe::mode() == Caffe::MFUS);
  LOG(INFO) << "[Fusion] reset...";
  destroy();
  resetFusionIO();
  compiled_ = false;
  create();
}

template <typename Dtype>
void MFusion<Dtype>::create() {
  CHECK(Caffe::mode() == Caffe::MFUS);
  CHECK(fuse_op_ == nullptr);
  MLU_CHECK(cnmlCreateFusionOp(&fuse_op_));
}

template <typename Dtype>
void MFusion<Dtype>::fuse(cnmlBaseOp_t op) {
  CHECK(Caffe::mode() == Caffe::MFUS);
  CHECK_NOTNULL(fuse_op_);
  MLU_CHECK(cnmlFuseOp(op, fuse_op_));
}

template <typename Dtype>
void MFusion<Dtype>::compile() {
  CHECK(Caffe::mode() == Caffe::MFUS);
  CHECK_NOTNULL(fuse_op_);
  if (compiled_) {
    return;
  }

  setFusionIO();
  LOG(INFO) << "[Fusion] compiling..." << fuse_op_;
  MLU_CHECK(cnmlSetFusionOpCorenum(fuse_op_, Caffe::core_number()));
  LOG(INFO) << "Core Number is " << Caffe::core_number();
  MLU_CHECK(cnmlSetFusionOpCoreVersion(fuse_op_, Caffe::rt_core()));
  MLU_CHECK(cnmlCompileFusionOp_V2(fuse_op_));
  compiled_ = true;
}

template <typename Dtype>
void MFusion<Dtype>::setFusionIO() {
  CHECK(Caffe::mode() == Caffe::MFUS);
  CHECK_NOTNULL(fuse_op_);
  CHECK(!tensor_converted_);
  CHECK(!memory_converted_);

  blob2Tensors();
  LOG(INFO) << "[Fusion] setFusionIO (size: " << input_tensors_.size()
            << ", " << output_tensors_.size() << ")...";
  MLU_CHECK(cnmlSetFusionIO(fuse_op_,
                           input_tensors_.data(), input_tensors_.size(),
                           output_tensors_.data(), output_tensors_.size()));
}

template <typename Dtype>
void MFusion<Dtype>::resetFusionIO() {
  CHECK(Caffe::mode() == Caffe::MFUS);
  inputs_.clear();
  outputs_.clear();
  tensor_converted_ = false;
  input_tensors_.clear();
  output_tensors_.clear();
  memory_converted_ = false;
  input_mem_.clear();
  output_mem_.clear();
}

template <typename Dtype>
void MFusion<Dtype>::blob2Tensors() {
  CHECK(Caffe::mode() == Caffe::MFUS);
  if (tensor_converted_) {
    return;
  }

  for (auto blob : inputs_) {
    input_tensors_.push_back(blob->mlu_tensor());
  }
  for (auto blob : outputs_) {
    output_tensors_.push_back(blob->mlu_tensor());
  }
  tensor_converted_ = true;
}

template <typename Dtype>
void MFusion<Dtype>::blob2Memory() {
  CHECK(Caffe::mode() == Caffe::MFUS);
  CHECK(compiled_);
  // we must take it into consideration that input and output addr may be
  // changed in a situtation when we use set_mlu_data().
  input_mem_.clear();
  output_mem_.clear();
  for (auto blob : inputs_) {
    input_mem_.push_back(blob->mutable_mlu_data());
  }
  for (auto blob : outputs_) {
    output_mem_.push_back(blob->mutable_mlu_data());
  }
}

template <typename Dtype>
void MFusion<Dtype>::addInput(Blob<Dtype>* input) {
  CHECK(Caffe::mode() == Caffe::MFUS);
  CHECK_NOTNULL(fuse_op_);
  CHECK(!tensor_converted_);
  CHECK(!memory_converted_);
  CHECK(!compiled_);
  uniquePushBack(&inputs_, input);
}

template <typename Dtype>
void MFusion<Dtype>::addOutput(Blob<Dtype>* output) {
  CHECK(Caffe::mode() == Caffe::MFUS);
  CHECK_NOTNULL(fuse_op_);
  CHECK(!tensor_converted_);
  CHECK(!memory_converted_);
  CHECK(!compiled_);
  uniquePushBack(&outputs_, output);
}

template <typename Dtype>
void MFusion<Dtype>::addInputs(vector<Blob<Dtype>*> inputs) {
  CHECK(Caffe::mode() == Caffe::MFUS);
  CHECK_NOTNULL(fuse_op_);
  CHECK(!tensor_converted_);
  CHECK(!memory_converted_);
  CHECK(!compiled_);
  for (auto blob : inputs) {
    uniquePushBack(&inputs_, blob);
  }
}

template <typename Dtype>
void MFusion<Dtype>::addOutputs(vector<Blob<Dtype>*> outputs) {
  CHECK(Caffe::mode() == Caffe::MFUS);
  CHECK_NOTNULL(fuse_op_);
  CHECK(!tensor_converted_);
  CHECK(!memory_converted_);
  CHECK(!compiled_);
  for (auto blob : outputs) {
    uniquePushBack(&outputs_, blob);
  }
}

template <typename Dtype>
void MFusion<Dtype>::sortIO(
    const unordered_map<Blob<Dtype>*, int>& blob2index) {
  CHECK(!tensor_converted_);
  CHECK(!memory_converted_);
  CHECK(!compiled_);

  // input blobs may not all come from net input,
  // additional input blobs may come from other layers.
  // from now on, there are no additional input blobs anymore.
  // codes left here only for safety.
  auto sorting = [blob2index] (vector<Blob<Dtype>*>& vecs) {
    sort(vecs.begin(), vecs.end(),
         [blob2index] (Blob<Dtype>* a, Blob<Dtype>* b) -> bool {
            if (blob2index.find(a) != blob2index.end() &&
                blob2index.find(b) != blob2index.end()) {
              return blob2index.at(a) < blob2index.at(b);
            } else if (blob2index.find(a) == blob2index.end() &&
                       blob2index.find(b) != blob2index.end()) {
              return false;
            } else {
              return true;
            }});
  };
  sorting(inputs_);
  sorting(outputs_);
}

template <typename Dtype>
string MFusion<Dtype>::inputBlobs(const unordered_map<Blob<Dtype>*,
                                  string>& blob2name) {
  // input blobs may not all come from net input,
  // additional input blobs may come from other layers.
  // from now on, there are no additional input blobs anymore.
  // codes left here only for safety.
  std::stringstream ss;
  for (auto& blob : inputs_) {
    if (blob2name.find(blob) != blob2name.end())
      ss << " " << blob2name.at(blob) << ",";
    else
      ss << " additional blob" << ",";
  }
  return ss.str();
}

template <typename Dtype>
string MFusion<Dtype>::outputBlobs(const unordered_map<Blob<Dtype>*,
                                   string>& blob2name) {
  std::stringstream ss;
  for (auto& blob : outputs_) {
    ss << " " << blob2name.at(blob) << ",";
  }
  return ss.str();
}

template <typename Dtype>
vector<string> MFusion<Dtype>::inputBlobNames
              (const unordered_map<Blob<Dtype>*, string>& blob2name) {
  vector<string> inputs_vec;
  for (auto& blob : inputs_) {
    if (blob2name.find(blob) != blob2name.end())
      inputs_vec.push_back(blob2name.at(blob));
    else
      inputs_vec.push_back("additional blob");
  }
  return inputs_vec;
}

template <typename Dtype>
vector<string> MFusion<Dtype>::outputBlobNames
               (const unordered_map<Blob<Dtype>*, string>& blob2name) {
  vector<string> outputs_vec;
  for (auto& blob : outputs_) {
    outputs_vec.push_back(blob2name.at(blob));
  }
  return outputs_vec;
}

template <typename Dtype>
void MFusion<Dtype>::rmInput(Blob<Dtype>* blob) {
  CHECK(Caffe::mode() == Caffe::MFUS);
  CHECK_NOTNULL(fuse_op_);
  CHECK(!tensor_converted_);
  CHECK(!memory_converted_);
  CHECK(!compiled_);
  rmOneFromVector(&inputs_, blob);
}

template <typename Dtype>
void MFusion<Dtype>::rmOutput(Blob<Dtype>* blob) {
  CHECK(Caffe::mode() == Caffe::MFUS);
  CHECK_NOTNULL(fuse_op_);
  CHECK(!tensor_converted_);
  CHECK(!memory_converted_);
  CHECK(!compiled_);

  rmOneFromVector(&outputs_, blob);
}

template <typename Dtype>
void MFusion<Dtype>::forward() {
  CHECK(Caffe::mode() == Caffe::MFUS);
  CHECK_NOTNULL(fuse_op_);
  blob2Memory();
  event_time_ = 0;
  LOG(INFO) << "[Fusion] forwarding...";
  cnrtNotifier_t notifierBeginning, notifierEnd;
  cnrtCreateNotifier(&notifierBeginning);
  cnrtCreateNotifier(&notifierEnd);

  cnrtPlaceNotifier(notifierBeginning, Caffe::queue());
  MLU_CHECK(cnmlComputeFusionOpForward_V3(fuse_op_,
                                      input_mem_.data(), input_mem_.size(),
                                      output_mem_.data(), output_mem_.size(),
                                      Caffe::forward_param(), Caffe::queue()));
  cnrtPlaceNotifier(notifierEnd, Caffe::queue());
  CNRT_CHECK(cnrtSyncQueue(Caffe::queue()));
  cnrtNotifierDuration(notifierBeginning, notifierEnd, &event_time_);
  LOG(INFO) << "Hardware execution time: "<< event_time_
    << "(" << event_time_ /1000 << "ms)";

  cnrtDestroyNotifier(&notifierBeginning);
  cnrtDestroyNotifier(&notifierEnd);
}

template <typename Dtype>
void MFusion<Dtype>::backward() {
  CHECK(Caffe::mode() == Caffe::MFUS);
  CHECK_NOTNULL(fuse_op_);
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void MFusion<Dtype>::destroy() {
  if (fuse_op_ != nullptr) {
    MLU_CHECK(cnmlDestroyFusionOp(&fuse_op_));
    fuse_op_ = nullptr;
  }
}

INSTANTIATE_CLASS(MFusion);
}  // namespace caffe

#endif  // USE_MLU
