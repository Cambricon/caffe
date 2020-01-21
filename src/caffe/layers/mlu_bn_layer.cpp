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
#include <algorithm>
#include <sstream>
#include <string>
#include <vector>
#include "caffe/layers/mlu_bn_layer.hpp"
#include "caffe/mlu/fusion.hpp"

namespace caffe {
template <typename Dtype>
void MLUBNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK(this->phase_ == TEST) << "The phase of the model should be TEST.";
  BNLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void MLUBNLayer<Dtype>::Reshape_tensor(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  BNLayer<Dtype>::Reshape(bottom, top);
  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
  BaseDataType mlu_dtype = bottom[0]->mlu_type();
  if (bottom[0] != top[0]) {
    top[0]->Reshape(bottom[0]->shape(), cpu_dtype, mlu_dtype, CNML_TENSOR);
  }
  vector<int> sz(4, 1);
  sz[1] = this->channels_;
  // CNML_CNHW : convert (sz, 1, 1, 1) to (1, sz, 1, 1);
  this->mean_.Reshape(sz, cpu_dtype, mlu_dtype, CNML_CONST);
  this->variance_.Reshape(sz, cpu_dtype, mlu_dtype, CNML_CONST);
  this->temp_bn_.Reshape(bottom[0]->shape(), cpu_dtype, mlu_dtype, CNML_TENSOR);
  this->temp_mult_.Reshape(bottom[0]->shape(), cpu_dtype, mlu_dtype, CNML_TENSOR);
  this->alpha_expand_.Reshape(bottom[0]->shape(), cpu_dtype, mlu_dtype, CNML_CONST);
  this->beta_expand_.Reshape(bottom[0]->shape(), cpu_dtype, mlu_dtype, CNML_CONST);
}

template <typename Dtype>
void MLUBNLayer<Dtype>::MLUDestroyOp() {
  if (batch_norm_op_ptr_ != NULL) {
    MLU_CHECK(cnmlDestroyBaseOp(&batch_norm_op_ptr_));
    batch_norm_op_ptr_ = NULL;
  }
  if (mult_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&mult_op_ptr_));
    mult_op_ptr_ = nullptr;
  }
  if (add_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&add_op_ptr_));
    add_op_ptr_ = nullptr;
  }
}

template <typename Dtype>
void MLUBNLayer<Dtype>::MLUCreateOpBindData(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // Use the moving average mean
  caffe_copy(mean_.count(), this->blobs_[2]->cpu_data(),
      mean_.mutable_cpu_data());
  // Use the moving average variance
  caffe_copy(variance_.count(), this->blobs_[3]->cpu_data(),
      variance_.mutable_cpu_data());
  // normalize variance
  caffe_add_scalar(this->variance_.count(), this->bn_eps_,
      this->variance_.mutable_cpu_data());
  caffe_powx(this->variance_.count(), this->variance_.cpu_data(), Dtype(-0.5),
      this->variance_.mutable_cpu_data());
  MLU_CHECK(cnmlCreateBatchNormOp(&batch_norm_op_ptr_,
                                 bottom[0]->mlu_tensor(),
                                 temp_bn_.mlu_tensor(),
                                 this->mean_.mlu_tensor(),
                                 this->variance_.mlu_tensor()));
  MLU_CHECK(cnmlBindConstData_V2(this->mean_.mlu_tensor(),
                                 this->mean_.sync_data(),
                                 false));
  MLU_CHECK(cnmlBindConstData_V2(this->variance_.mlu_tensor(),
                                 this->variance_.sync_data(),
                                 false));
  /* mult */
  caffe_cpu_gemm<Dtype>(CblasNoTrans,
      CblasNoTrans, this->num_, this->channels_, 1, 1,
      this->batch_sum_multiplier_.cpu_data(),
      this->blobs_[0]->cpu_data(), 0.,
      this->spatial_statistic_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
      this->channels_ * this->num_, this->height_ * this->width_,
      1, Dtype(1), this->spatial_statistic_.cpu_data(),
      this->spatial_sum_multiplier_.cpu_data(), Dtype(0),
      this->alpha_expand_.mutable_cpu_data());
  MLU_CHECK(cnmlCreateMultOp(&mult_op_ptr_,
                            this->alpha_expand_.mlu_tensor(),
                            temp_bn_.mlu_tensor(),
                            temp_mult_.mlu_tensor()));
  MLU_CHECK(cnmlBindConstData_V2(this->alpha_expand_.mlu_tensor(),
                                 this->alpha_expand_.sync_data(),
                                 false));
  /* add */
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
                        this->num_, this->channels_, 1, Dtype(1),
                        this->batch_sum_multiplier_.cpu_data(),
                        this->blobs_[1]->cpu_data(), Dtype(0),
                        this->spatial_statistic_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
                        this->channels_ * this->num_, this->height_ * this->width_,
                        1, Dtype(1), this->spatial_statistic_.cpu_data(),
                        this->spatial_sum_multiplier_.cpu_data(), Dtype(0),
                        this->beta_expand_.mutable_cpu_data());
  MLU_CHECK(cnmlCreateAddOp(&add_op_ptr_,
                            this->beta_expand_.mlu_tensor(),
                            temp_mult_.mlu_tensor(),
                            top[0]->mlu_tensor()));
  MLU_CHECK(cnmlBindConstData_V2(this->beta_expand_.mlu_tensor(),
                                 this->beta_expand_.sync_data(),
                                 false));
}

template <typename Dtype>
void MLUBNLayer<Dtype>::MLUCompileOp() {
  MLU_CHECK(cnmlCompileBaseOp(batch_norm_op_ptr_,
                              Caffe::rt_core(), Caffe::core_number()));
  MLU_CHECK(cnmlCompileBaseOp(mult_op_ptr_,
                              Caffe::rt_core(), Caffe::core_number()));
  MLU_CHECK(cnmlCompileBaseOp(add_op_ptr_,
                              Caffe::rt_core(), Caffe::core_number()));
}

template <typename Dtype>
void MLUBNLayer<Dtype>::fuse(MFusion<Dtype>* fuser) {
  fuser->fuse(batch_norm_op_ptr_);
  fuser->fuse(mult_op_ptr_);
  fuser->fuse(add_op_ptr_);
}

template <typename Dtype>
MLUBNLayer<Dtype>::~MLUBNLayer() {
  MLUDestroyOp();
}

template <typename Dtype>
void MLUBNLayer<Dtype>::Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {
  MLU_CHECK(cnmlComputeBatchNormOpForward_V3(batch_norm_op_ptr_,
                                         bottom[0]->mutable_mlu_data(),
                                         temp_bn_.mutable_mlu_data(),
                                         Caffe::forward_param(),
                                         Caffe::queue()));
  MLU_CHECK(cnmlComputeMultOpForward_V3(mult_op_ptr_,
                                 nullptr,
                                 temp_bn_.mutable_mlu_data(),
                                 temp_mult_.mutable_mlu_data(),
                                 Caffe::forward_param(),
                                 Caffe::queue()));
  MLU_CHECK(cnmlComputeAddOpForward_V3(add_op_ptr_,
                                 nullptr,
                                 temp_mult_.mutable_mlu_data(),
                                 top[0]->mutable_mlu_data(),
                                 Caffe::forward_param(),
                                 Caffe::queue()));
}

INSTANTIATE_CLASS(MLUBNLayer);

}  // namespace caffe
#endif  // USE_MLU
