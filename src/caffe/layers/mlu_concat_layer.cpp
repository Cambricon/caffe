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

#ifdef USE_MLU
#include <vector>

#include "caffe/layers/mlu_concat_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MLUConcatLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
  ConcatLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void MLUConcatLayer<Dtype>::Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {
  ConcatLayer<Dtype>::Reshape(bottom, top);
}

template <typename Dtype>
void MLUConcatLayer<Dtype>::MLUCreateOpBindData(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  if (bottom.size() == 1) {
    MLU_CHECK(cnmlCreateDeviceMemcpyOp(&concat_op_ptr_,
                                      bottom[0]->mlu_tensor(),
                                      top[0]->mlu_tensor()));
    return;
  }
  cnmlConcatMode_t concatModes[4] {cnmlConcatMode_t::CNML_CONCAT_BATCH,
                                cnmlConcatMode_t::CNML_CONCAT_FEAT,
                                cnmlConcatMode_t::CNML_CONCAT_HIGHT,
                                cnmlConcatMode_t::CNML_CONCAT_WIDTH};
  int axis = this->concat_axis_;
  MLU_CHECK(cnmlCreateConcatOpParam(&concat_param_ptr_,
                                 bottom.size(),
                                 top.size(),
                                 concatModes[axis]));
  int kBottomSize = bottom.size();
  cnmlTensor_t mlutensor_inputs[kBottomSize];
  for (int i = 0; i < bottom.size(); i++)
    mlutensor_inputs[i] = bottom[i]->mlu_tensor();

  int kTopSize = top.size();
  cnmlTensor_t mlutensor_outputs[kTopSize];
  for (int i = 0; i < top.size(); i++)
    mlutensor_outputs[i] = top[i]->mlu_tensor();

  MLU_CHECK(cnmlCreateConcatOp(&concat_op_ptr_,
                              concat_param_ptr_,
                              mlutensor_inputs,
                              kBottomSize,
                              mlutensor_outputs,
                              kTopSize));
}

template <typename Dtype>
void MLUConcatLayer<Dtype>::MLUDestroyOp() {
  if (concat_param_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyConcatOpParam(&concat_param_ptr_));
    concat_param_ptr_ = nullptr;
  }

  if (concat_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&concat_op_ptr_));
    concat_op_ptr_ = nullptr;
  }
}

template <typename Dtype>
MLUConcatLayer<Dtype>::~MLUConcatLayer() {
  MLUDestroyOp();
}

template <typename Dtype>
void MLUConcatLayer<Dtype>::Forward_mlu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (bottom.size() == 1) {
    MLU_CHECK(cnmlComputeDeviceMemcpyOpForward_V3(concat_op_ptr_,
        bottom[0]->mutable_mlu_data(),
        top[0]->mutable_mlu_data(),
        Caffe::forward_param(), Caffe::queue()));
    return;
  }

  void* mlutensor_input_ptrs[bottom.size()];
  void* mlutensor_output_ptrs[top.size()];
  for (int i = 0; i < top.size(); i++) {
    mlutensor_output_ptrs[i] = top[i]->mutable_mlu_data();
  }
  if (bottom[0]->tensor_type() == CNML_CONST) {
    MLU_CHECK(cnmlComputeConcatOpForward_V3(concat_op_ptr_,
                                        nullptr,
                                        0,
                                        mlutensor_output_ptrs,
                                        top.size(),
                                        Caffe::forward_param(),
                                        Caffe::queue()));
  } else {
    for (int i = 0; i < bottom.size(); i++)
      mlutensor_input_ptrs[i] = bottom[i]->mutable_mlu_data();
      MLU_CHECK(cnmlComputeConcatOpForward_V3(concat_op_ptr_,
                                           mlutensor_input_ptrs,
                                           bottom.size(),
                                           mlutensor_output_ptrs,
                                           top.size(),
                                           Caffe::forward_param(),
                                           Caffe::queue()));
  }
}

INSTANTIATE_CLASS(MLUConcatLayer);
}  // namespace caffe
#endif
