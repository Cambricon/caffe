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
#include "caffe/layers/mlu_crelu_layer.hpp"

namespace caffe {

template <typename Dtype>
void MLUCReLULayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //  slope is aligned against channel
  CHECK_GE(bottom[0]->num_axes(), 2) << "Number of axes of bottom blob must be >= 2";
  CReLULayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void MLUCReLULayer<Dtype>::Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32:DT_DOUBLE;
  BaseDataType mlu_dtype = bottom[0]->mlu_type();
  vector<int> top_shape(bottom[0]->shape());
  top_shape[this->concat_axis_] = bottom[0]->shape(this->concat_axis_) * 2;
  top[0]->Reshape(top_shape, cpu_dtype, mlu_dtype, CNML_TENSOR);
  negative_input_.Reshape(bottom[0]->shape(), cpu_dtype, mlu_dtype, CNML_TENSOR);
  concated_data_.Reshape(top_shape, cpu_dtype, mlu_dtype, CNML_TENSOR);

  //  if axis_ is 1, top_shape's channel has been doubbled already,
  //  otherwise it is channel dimension of bottom 0
  vector<int> slope_shape(4, 1);
  slope_shape[1] = top_shape[1];
  negative_slope_b_.Reshape(slope_shape, cpu_dtype, mlu_dtype, CNML_CONST);
  for (int i = 0; i < negative_slope_b_.count(); i++) {
    negative_slope_b_.mutable_cpu_data()[i] = this->negative_slope_;
  }
}

template <typename Dtype>
void MLUCReLULayer<Dtype>::MLUDestroyOp() {
  if (minus_op_ptr_ != NULL) {
    MLU_CHECK(cnmlDestroyBaseOp(&minus_op_ptr_));
    minus_op_ptr_ = NULL;
  }
  if (concat_op_ptr_ != NULL) {
    MLU_CHECK(cnmlDestroyBaseOp(&concat_op_ptr_));
    concat_op_ptr_ = NULL;
  }
  if (prelu_op_ptr_ != NULL) {
    MLU_CHECK(cnmlDestroyBaseOp(&prelu_op_ptr_));
    prelu_op_ptr_ = NULL;
  }
}

template <typename Dtype>
void MLUCReLULayer<Dtype>::MLUCreateOpBindData(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  //  Please refer to comments in mlu_crelu_layer.hpp
  MLU_CHECK(cnmlCreateMinusOp(&minus_op_ptr_,
                              bottom[0]->mlu_tensor(),
                              negative_input_.mlu_tensor()));
  cnmlTensor_t input_tensors[2] = {bottom[0]->mlu_tensor(),
                                  negative_input_.mlu_tensor()};
  cnmlTensor_t output_tensor[1] = {concated_data_.mlu_tensor()};
  int length = bottom[0]->shape().size();
  vector<int> dim_order(length, 1);  // = {0, 3, 1, 2};
  dim_order[0] = 0;
  dim_order[1] = length - 1;
  for (int i = 2; i < length; i++) {
      dim_order[i] = i-1;
  }
  int concat_axis = dim_order[this->concat_axis_];
  MLU_CHECK(cnmlCreateNdConcatOp(&concat_op_ptr_,
                              concat_axis,
                              input_tensors,
                              2,
                              output_tensor,
                              1));
  MLU_CHECK(cnmlCreatePreluOp(&prelu_op_ptr_,
                              concated_data_.mlu_tensor(),
                              top[0]->mlu_tensor(),
                              negative_slope_b_.mlu_tensor()));
  MLU_CHECK(cnmlBindConstData_V2(negative_slope_b_.mlu_tensor(),
                              reinterpret_cast<float*>
                              (negative_slope_b_.sync_data()),
                              false));
}

template <typename Dtype>
void MLUCReLULayer<Dtype>::Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
  MLU_CHECK(cnmlComputeMinusOpForward_V3(minus_op_ptr_,
            bottom[0]->mutable_mlu_data(),
            negative_input_.mutable_mlu_data(),
            Caffe::forward_param(),
            Caffe::queue()));
  void* input_ptrs[2] = { bottom[0]->mutable_mlu_data(),
                          negative_input_.mutable_mlu_data() };
  void* output_ptrs[1] = { concated_data_.mutable_mlu_data() };
  MLU_CHECK(cnmlComputeConcatOpForward_V3(concat_op_ptr_,
            input_ptrs,
            2,
            output_ptrs,
            1,
            Caffe::forward_param(),
            Caffe::queue()));
  MLU_CHECK(cnmlComputePreluOpForward_V3(prelu_op_ptr_,
            concated_data_.mutable_mlu_data(),
            top[0]->mutable_mlu_data(),
            Caffe::forward_param(), Caffe::queue()));
}

template <typename Dtype>
MLUCReLULayer<Dtype>::~MLUCReLULayer() {
  MLUDestroyOp();
}

INSTANTIATE_CLASS(MLUCReLULayer);

}  // namespace caffe
#endif
