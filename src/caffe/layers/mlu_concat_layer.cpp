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
  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
  BaseDataType mlu_dtype = bottom[0]->mlu_type();
  top[0]->Reshape(top[0]->shape(), cpu_dtype, mlu_dtype, CNML_TENSOR);
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

  /* adapte axis order NCHW to NHWC */
  int axis;
  if (this->concat_axis_ == 0) {
    axis = 0;
  } else if (this->concat_axis_ == 2) {
    axis = 1;
  } else if (this->concat_axis_ == 3) {
    axis = 2;
  } else {
    axis = 3;
  }
  int kBottomSize = bottom.size();
  cnmlTensor_t mlutensor_inputs[kBottomSize];
  for (int i = 0; i < bottom.size(); i++)
    mlutensor_inputs[i] = bottom[i]->mlu_tensor();

  int kTopSize = top.size();
  cnmlTensor_t mlutensor_outputs[kTopSize];
  for (int i = 0; i < top.size(); i++)
    mlutensor_outputs[i] = top[i]->mlu_tensor();

  MLU_CHECK(cnmlCreateNdConcatOp(&concat_op_ptr_,
                                 axis,
                                 mlutensor_inputs,
                                 kBottomSize,
                                 mlutensor_outputs,
                                 kTopSize));
}

template <typename Dtype>
void MLUConcatLayer<Dtype>::MLUDestroyOp() {
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
