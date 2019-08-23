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

#ifdef USE_MLU

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

#include "caffe/layers/mlu_batch_norm_layer.hpp"
#include "caffe/mlu/fusion.hpp"

namespace caffe {

template <typename Dtype>
void MLUBatchNormLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  BatchNormLayer<Dtype>::LayerSetUp(bottom, top);
  this->use_global_stats_ = this->phase_ == TEST;
}

template <typename Dtype>
void MLUBatchNormLayer<Dtype>::Reshape_tensor(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  BatchNormLayer<Dtype>::Reshape(bottom, top);
  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
  BaseDataType mlu_dtype = DT_FLOAT16;
  vector<int> sz;
  sz.push_back(this->channels_);
  // CNML_CNHW : convert (sz, 1, 1, 1) to (1, sz, 1, 1);
  this->mean_.Reshape(sz, cpu_dtype, mlu_dtype, CNML_CONST, CNML_CNHW);
  this->variance_.Reshape(sz, cpu_dtype, mlu_dtype, CNML_CONST, CNML_CNHW);
  if (this->use_alpha_beta_) {
    this->temp_bn_.Reshape(bottom[0]->shape(),
                                cpu_dtype,
                                mlu_dtype,
                                CNML_TENSOR);
    this->temp_.Reshape(bottom[0]->shape(),
                                cpu_dtype,
                                mlu_dtype,
                                CNML_TENSOR);
    this->alpha_expand_.Reshape(bottom[0]->shape(),
                                cpu_dtype,
                                mlu_dtype,
                                CNML_CONST);
    this->beta_expand_.Reshape(bottom[0]->shape(),
                               cpu_dtype,
                               mlu_dtype,
                               CNML_CONST);
  }
}

template <typename Dtype>
void MLUBatchNormLayer<Dtype>::MLUDestroyOp() {
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
void MLUBatchNormLayer<Dtype>::MLUCreateOpBindData(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int num = bottom[0]->shape(0);
  int spatial_dim = bottom[0]->count()/(bottom[0]->shape(0)*this->channels_);

  // use the stored mean/variance estimates.
  const Dtype scale_factor = this->blobs_[2]->cpu_data()[0] == 0 ?
      0 : 1 / this->blobs_[2]->cpu_data()[0];
  caffe_cpu_scale(this->mean_.count(), scale_factor,
        this->blobs_[0]->cpu_data(), this->mean_.mutable_cpu_data());
  caffe_cpu_scale(this->variance_.count(), scale_factor,
        this->blobs_[1]->cpu_data(), this->variance_.mutable_cpu_data());

  // normalize variance
  caffe_add_scalar(this->variance_.count(), this->eps_,
      this->variance_.mutable_cpu_data());
  caffe_powx(this->variance_.count(), this->variance_.cpu_data(), Dtype(-0.5),
      this->variance_.mutable_cpu_data());

  if (this->use_global_stats_) {
    MLU_CHECK(cnmlCreateBatchNormOp(&batch_norm_op_ptr_,
                                   bottom[0]->mlu_tensor(),
                                   this->use_alpha_beta_  ?
                                   this->temp_bn_.mlu_tensor() :
                                   top[0]->mlu_tensor(),
                                   this->mean_.mlu_tensor(),
                                   this->variance_.mlu_tensor()));
    MLU_CHECK(cnmlBindConstData(this->mean_.mlu_tensor(),
                               this->mean_.cpu_tensor(),
                               this->mean_.mutable_cpu_data()));
    MLU_CHECK(cnmlBindConstData(this->variance_.mlu_tensor(),
                               this->variance_.cpu_tensor(),
                               this->variance_.mutable_cpu_data()));

    if (this->use_alpha_beta_) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans,
          CblasNoTrans, num, this->channels_, 1, 1,
          this->batch_sum_multiplier_.cpu_data(),
          this->blobs_[3]->cpu_data(), 0.,
          this->num_by_chans_.mutable_cpu_data());
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          this->channels_ * num,
          spatial_dim, 1, 1, this->num_by_chans_.cpu_data(),
          this->spatial_sum_multiplier_.cpu_data(), 0,
          this->alpha_expand_.mutable_cpu_data());
      /* mult */
      MLU_CHECK(cnmlCreateMultOp(&mult_op_ptr_,
                                this->alpha_expand_.mlu_tensor(),
                                this->temp_bn_.mlu_tensor(),
                                this->temp_.mlu_tensor()));
      MLU_CHECK(cnmlBindConstData(this->alpha_expand_.mlu_tensor(),
                                  this->alpha_expand_.cpu_tensor(),
                                  this->alpha_expand_.mutable_cpu_data()));
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num,
                            this->channels_, 1, 1,
                            this->batch_sum_multiplier_.cpu_data(),
                            this->blobs_[4]->cpu_data(), 0.,
                            this->num_by_chans_.mutable_cpu_data());
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, this->channels_ * num,
                            spatial_dim, 1, 1, this->num_by_chans_.cpu_data(),
                            this->spatial_sum_multiplier_.cpu_data(), 0,
                            this->beta_expand_.mutable_cpu_data());
      /* add */
      MLU_CHECK(cnmlCreateAddOp(&add_op_ptr_,
                               this->beta_expand_.mlu_tensor(),
                               this->temp_.mlu_tensor(),
                               top[0]->mlu_tensor()));
      MLU_CHECK(cnmlBindConstData(this->beta_expand_.mlu_tensor(),
                                  this->beta_expand_.cpu_tensor(),
                                  this->beta_expand_.mutable_cpu_data()));
    }
  }
}

template <typename Dtype>
void MLUBatchNormLayer<Dtype>::MLUCompileOp() {
  if (this->use_global_stats_) {
    MLU_CHECK(cnmlCompileBaseOp(batch_norm_op_ptr_,
                                Caffe::rt_core(), Caffe::model_parallel()));
    if (this->use_alpha_beta_) {
      MLU_CHECK(cnmlCompileBaseOp(mult_op_ptr_,
                                  Caffe::rt_core(), Caffe::model_parallel()));
      MLU_CHECK(cnmlCompileBaseOp(add_op_ptr_,
                                  Caffe::rt_core(), Caffe::model_parallel()));
    }
  }
}

template <typename Dtype>
void MLUBatchNormLayer<Dtype>::fuse(MFusion<Dtype>* fuser) {
  if (this->use_global_stats_) {
    fuser->fuse(batch_norm_op_ptr_);
    if (this->use_alpha_beta_) {
      fuser->fuse(mult_op_ptr_);
      fuser->fuse(add_op_ptr_);
    }
  }
}

template <typename Dtype>
MLUBatchNormLayer<Dtype>::~MLUBatchNormLayer() {
  MLUDestroyOp();
}

template <typename Dtype>
void MLUBatchNormLayer<Dtype>::Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {
  if (this->use_global_stats_) {
    MLU_CHECK(cnmlComputeBatchNormOpForward_V3(batch_norm_op_ptr_,
                                           bottom[0]->mutable_mlu_data(),
                                           this->use_alpha_beta_  ?
                                           this->temp_bn_.mutable_mlu_data() :
                                           top[0]->mutable_mlu_data(),
                                           Caffe::forward_param(),
                                           Caffe::queue()));
    if (this->use_alpha_beta_) {
      MLU_CHECK(cnmlComputeMultOpForward_V3(mult_op_ptr_,
                                     nullptr,
                                     this->temp_bn_.mutable_mlu_data(),
                                     this->temp_.mutable_mlu_data(),
                                     Caffe::forward_param(),
                                     Caffe::queue()));
      MLU_CHECK(cnmlComputeAddOpForward_V3(add_op_ptr_,
                                     nullptr,
                                     this->temp_.mutable_mlu_data(),
                                     top[0]->mutable_mlu_data(),
                                     Caffe::forward_param(),
                                     Caffe::queue()));
    }
  }
}

INSTANTIATE_CLASS(MLUBatchNormLayer);
}  // namespace caffe
#endif  // USE_MLU
