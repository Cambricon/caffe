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
#include "caffe/layers/mlu_elu_layer.hpp"

namespace caffe {

template <typename Dtype>
void MLUELULayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top) {
  ELULayer<Dtype>::LayerSetUp(bottom, top);
  alpha = this->layer_param_.elu_param().alpha();
}

template <typename Dtype>
void MLUELULayer<Dtype>::Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
  BaseDataType mlu_dtype = bottom[0]->mlu_type();

  top[0]->Reshape(bottom[0]->shape(), cpu_dtype, mlu_dtype, CNML_TENSOR);
  relu_tensor_.Reshape(bottom[0]->shape(), cpu_dtype, mlu_dtype, CNML_TENSOR);
  sub_tensor_.Reshape(bottom[0]->shape(), cpu_dtype, mlu_dtype, CNML_TENSOR);
  exp_tensor_.Reshape(bottom[0]->shape(), cpu_dtype, mlu_dtype, CNML_TENSOR);
  scale_tensor_.Reshape(bottom[0]->shape(), cpu_dtype, mlu_dtype, CNML_TENSOR);

  vector<int> param_shape(4, 1);
  param_shape[1] = bottom[0]->channels();
  alpha_tensor_.Reshape(param_shape, cpu_dtype, mlu_dtype, CNML_CONST);
  beta_tensor_.Reshape(param_shape, cpu_dtype, mlu_dtype, CNML_CONST);
}

template <typename Dtype>
void MLUELULayer<Dtype>::MLUCreateOpBindData(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
  if (alpha == Dtype(0.0)) {
    /* activeOp: y = max(x, 0) */
    MLU_CHECK(cnmlCreateActiveOp(&relu_active_op_ptr_,
                                cnmlActiveFunction_t::CNML_ACTIVE_RELU,
                                bottom[0]->mlu_tensor(),
                                top[0]->mlu_tensor()));
  } else {
    /* ActiveOp: max(x, 0) */
    MLU_CHECK(cnmlCreateActiveOp(&relu_active_op_ptr_,
                                cnmlActiveFunction_t::CNML_ACTIVE_RELU,
                                bottom[0]->mlu_tensor(),
                                relu_tensor_.mlu_tensor()));
    /* SubOp: min(x, 0) */
    MLU_CHECK(cnmlCreateSubOp(&sub_op_ptr_,
                              bottom[0]->mlu_tensor(),
                              relu_tensor_.mlu_tensor(),
                              sub_tensor_.mlu_tensor()));

    /* ExpOp: exp(min(x, 0)) */
    MLU_CHECK(cnmlCreateExpOp(&exp_op_ptr_,
                              sub_tensor_.mlu_tensor(),
                              exp_tensor_.mlu_tensor()));

    /* ScaleOp: alpha*(exp(min(x, 0))-1) */
    caffe_set(alpha_tensor_.count(), alpha, alpha_tensor_.mutable_cpu_data());
    caffe_set(beta_tensor_.count(), -alpha, beta_tensor_.mutable_cpu_data());

    MLU_CHECK(cnmlCreateScaleOp(&scale_op_ptr_,
                                exp_tensor_.mlu_tensor(),
                                scale_tensor_.mlu_tensor(),
                                alpha_tensor_.mlu_tensor(),
                                beta_tensor_.mlu_tensor()));

    MLU_CHECK(cnmlBindConstData_V2(alpha_tensor_.mlu_tensor(),
                                   alpha_tensor_.sync_data(),
                                   false));

    MLU_CHECK(cnmlBindConstData_V2(beta_tensor_.mlu_tensor(),
                                   beta_tensor_.sync_data(),
                                   false));

    /* AddOp: f(x) = max(x,0) + alpha * (exp(min(x, 0)) - 1) */
    MLU_CHECK(cnmlCreateAddOp(&add_op_ptr_,
                                scale_tensor_.mlu_tensor(),
                                relu_tensor_.mlu_tensor(),
                                top[0]->mlu_tensor()));
  }
}

template <typename Dtype>
void MLUELULayer<Dtype>::MLUCompileOp() {
  if (alpha == Dtype(0.0)) {
    MLU_CHECK(cnmlCompileBaseOp(relu_active_op_ptr_,
                                Caffe::rt_core(),
                                Caffe::core_number()));
  } else {
    MLU_CHECK(cnmlCompileBaseOp(relu_active_op_ptr_,
                                Caffe::rt_core(),
                                Caffe::core_number()));
    MLU_CHECK(cnmlCompileBaseOp(sub_op_ptr_,
                                Caffe::rt_core(),
                                Caffe::core_number()));
    MLU_CHECK(cnmlCompileBaseOp(exp_op_ptr_,
                                Caffe::rt_core(),
                                Caffe::core_number()));
    MLU_CHECK(cnmlCompileBaseOp(scale_op_ptr_,
                                Caffe::rt_core(),
                                Caffe::core_number()));
    MLU_CHECK(cnmlCompileBaseOp(add_op_ptr_,
                                Caffe::rt_core(),
                                Caffe::core_number()));
  }
}

template <typename Dtype>
void MLUELULayer<Dtype>::Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top) {
  if (alpha == Dtype(0.0)) {
    MLU_CHECK(cnmlComputeActiveOpForward_V3(relu_active_op_ptr_,
                                bottom[0]->mutable_mlu_data(),
                                top[0]->mutable_mlu_data(),
                                Caffe::forward_param(),
                                Caffe::queue()));
  } else {
    MLU_CHECK(cnmlComputeActiveOpForward_V3(relu_active_op_ptr_,
                                bottom[0]->mutable_mlu_data(),
                                relu_tensor_.mutable_mlu_data(),
                                Caffe::forward_param(),
                                Caffe::queue()));
    MLU_CHECK(cnmlComputeSubOpForward_V3(sub_op_ptr_,
                                bottom[0]->mutable_mlu_data(),
                                relu_tensor_.mutable_mlu_data(),
                                sub_tensor_.mutable_mlu_data(),
                                Caffe::forward_param(),
                                Caffe::queue()));
    MLU_CHECK(cnmlComputeActiveOpForward_V3(exp_op_ptr_,
                                sub_tensor_.mutable_mlu_data(),
                                exp_tensor_.mutable_mlu_data(),
                                Caffe::forward_param(),
                                Caffe::queue()));
    MLU_CHECK(cnmlComputeScaleOpForward_V3(scale_op_ptr_,
                                exp_tensor_.mutable_mlu_data(),
                                scale_tensor_.mutable_mlu_data(),
                                Caffe::forward_param(),
                                Caffe::queue()));
    MLU_CHECK(cnmlComputeAddOpForward_V3(add_op_ptr_,
                                scale_tensor_.mutable_mlu_data(),
                                relu_tensor_.mutable_mlu_data(),
                                top[0]->mutable_mlu_data(),
                                Caffe::forward_param(),
                                Caffe::queue()));
  }
}

template <typename Dtype>
void MLUELULayer<Dtype>::fuse(MFusion<Dtype>* fuser) {
  if (alpha == Dtype(0.0)) {
    fuser->fuse(relu_active_op_ptr_);
  } else {
    fuser->fuse(relu_active_op_ptr_);
    fuser->fuse(sub_op_ptr_);
    fuser->fuse(exp_op_ptr_);
    fuser->fuse(scale_op_ptr_);
    fuser->fuse(add_op_ptr_);
  }
}

template <typename Dtype>
void MLUELULayer<Dtype>::MLUDestroyOp() {
  if (relu_active_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&relu_active_op_ptr_));
    relu_active_op_ptr_ = nullptr;
  }
  if (sub_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&sub_op_ptr_));
    sub_op_ptr_ = nullptr;
  }
  if (exp_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&exp_op_ptr_));
    exp_op_ptr_ = nullptr;
  }
  if (scale_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&scale_op_ptr_));
    scale_op_ptr_ = nullptr;
  }
  if (add_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&add_op_ptr_));
    add_op_ptr_ = nullptr;
  }
}

INSTANTIATE_CLASS(MLUELULayer);

}  // namespace caffe
#endif
