/*
All modification made by Cambricon Corporation: © 2018--2019 Cambricon Corporation
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

#include <memory>
#include <vector>
#include "caffe/filler.hpp"
#include "caffe/layers/mlu_normalize_layer.hpp"

namespace caffe {

template <typename Dtype>
void MLUNormalizeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "Number of axes of bottom blob must be >=2.";
  NormalizeParameter norm_param = this->layer_param().norm_param();
  this->across_spatial_ = norm_param.across_spatial();
  this->eps_ = norm_param.eps();
  this->channel_shared_ = norm_param.channel_shared();
  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
  int8_mode = this->layer_param_.bottom_mlu_dtype_size() > 0 ||
              this->layer_param_.blobs_dtype_size() > 0;
  // mlu_dtype should be used if the blob is the input of Mlp or Conv OP
  BaseDataType mlu_dtype = bottom[0]->mlu_type();
  if (int8_mode) {
    if (this->layer_param_.blobs_dtype_size()) {
       mlu_dtype = this->layer_param_.blobs_dtype(0).type();
    } else {
       mlu_dtype = this->layer_param_.bottom_mlu_dtype(0).type();
    }
  }
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    if (this->channel_shared_) {
      this->blobs_[0].reset(new Blob<Dtype>(vector<int>(0)));
    } else {
      this->blobs_[0].reset(
          new Blob<Dtype>(vector<int>(1, bottom[0]->channels()),
          cpu_dtype,
          mlu_dtype,
          CNML_FILTER));
    }
  }
  shared_ptr<Filler<Dtype> > scale_filler;
  if (norm_param.has_scale_filler()) {
    scale_filler.reset(GetFiller<Dtype>(norm_param.scale_filler()));
  } else {
    FillerParameter filler_param;
    filler_param.set_type("constant");
    filler_param.set_value(1.0);
    scale_filler.reset(GetFiller<Dtype>(filler_param));
  }
  scale_filler->Fill(this->blobs_[0].get());
  if (this->channel_shared_) {
    CHECK_EQ(this->blobs_[0]->count(), 1)
        << "Scale size is inconsistent with prototxt config";
  } else {
    CHECK_EQ(this->blobs_[0]->count(), bottom[0]->channels())
        << "Scale size is inconsistent with prototxt config";
  }
  BaseDataType mlu_datatype = bottom[0]->mlu_type();
  vector<int> shape1(4, 1);
  shape1[1] = bottom[0]->channels();
  gemm_matrix_ = new Blob<Dtype>(shape1,
      cpu_dtype,
      mlu_datatype,   // according to lib interface, this should always be fp16
      CNML_CONST);
  shape1[1] = 1;
  mult_blob_ = new Blob<Dtype>(shape1,
                               cpu_dtype,
                               mlu_datatype,
                               CNML_CONST);
  if (this->across_spatial_) {
    vector<int> mlp_shape(4, 1);
    mlp_shape[1] = bottom[0]->channels();
    mlp_shape[2] = bottom[0]->height();
    mlp_shape[3] = bottom[0]->width();
    mlp_weight_blob_ = new Blob<Dtype>(mlp_shape,
                                       cpu_dtype,
                                       mlu_dtype,
                                       CNML_FILTER);
    caffe_set(mlp_weight_blob_->count(), Dtype(1),
            mlp_weight_blob_->mutable_cpu_data());
    vector<int> shape(4, 1);
    mlp_bias_blob_ = new Blob<Dtype>(shape,
                                     cpu_dtype,
                                     mlu_datatype,
                                     CNML_CONST);
    caffe_set(mlp_bias_blob_->count(), Dtype(0),
            mlp_bias_blob_->mutable_cpu_data());
    shape[0] = bottom[0]->num();
    eps_blob_ = new Blob<Dtype>(shape,
                                cpu_dtype,
                                mlu_datatype,
                                CNML_CONST);
    caffe_set(eps_blob_->count(), Dtype(this->eps_),
            eps_blob_->mutable_cpu_data());
    div0_blob_ = new Blob<Dtype>(shape,
                                 cpu_dtype,
                                 mlu_datatype,
                                 CNML_CONST);
    caffe_set(div0_blob_->count(), Dtype(1.0),
            div0_blob_->mutable_cpu_data());
    quant_params.resize(1, nullptr);
  } else {
    vector<int> shape(4, 1);
    shape[1] = bottom[0]->channels();
    gemv_weight_ = new Blob<Dtype>(shape,
                                   cpu_dtype,
                                   mlu_dtype,
                                   CNML_FILTER);
    shape[0] = shape[1];
    shape[1] = 1;
    gemm_weight_ = new Blob<Dtype>(shape,
                                   cpu_dtype,
                                   mlu_dtype,
                                   CNML_FILTER);
    shape[0] = bottom[0]->num();
    shape[1] = bottom[0]->channels();
    shape[2] = bottom[0]->height();
    shape[3] = bottom[0]->width();
    caffe_set(gemv_weight_->count(),
              Dtype(1),
              gemv_weight_->mutable_cpu_data());
    caffe_set(gemm_weight_->count(),
              Dtype(1),
              gemm_weight_->mutable_cpu_data());
    quant_params.resize(2, nullptr);
  }
}

template <typename Dtype>
void MLUNormalizeLayer<Dtype>::Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {
  MLU_CHECK(cnmlComputeMaxOpForward_V3(max_op_ptr_,
                                       bottom[0]->mutable_mlu_data(),
                                       max_blob_.mutable_mlu_data(),
                                       index_blob_.mutable_mlu_data(),
                                       Caffe::forward_param(),
                                       Caffe::queue()));
  MLU_CHECK(cnmlComputePowerOpForward_V3(power_op_ptr_,
                                         max_blob_.mutable_mlu_data(),
                                         power_blob_.mutable_mlu_data(),
                                         Caffe::forward_param(),
                                         Caffe::queue()));
  MLU_CHECK(cnmlComputeBroadcastMultOpForward_V3(bMult_op_ptr_,
                                                 bottom[0]->mutable_mlu_data(),
                                                 power_blob_.mutable_mlu_data(),
                                                 bMult_blob_.mutable_mlu_data(),
                                                 Caffe::forward_param(),
                                                 Caffe::queue()));
  MLU_CHECK(cnmlComputeMultOpForward_V3(sqr_op_ptr_,
                                        bMult_blob_.mutable_mlu_data(),
                                        bMult_blob_.mutable_mlu_data(),
                                        sqr_blob_.mutable_mlu_data(),
                                        Caffe::forward_param(),
                                        Caffe::queue()));
  // normalization in whole bacth which square-sum number is channel*height*width.
  if (this->across_spatial_) {
    MLU_CHECK(cnmlComputeMlpOpForward_V3(mlp_op_ptr_,
                                         sqr_blob_.mutable_mlu_data(),
                                         mlp_blob_.mutable_mlu_data(),
                                         Caffe::forward_param(),
                                         Caffe::queue()));
    MLU_CHECK(cnmlComputeAddOpForward_V3(add_op_ptr_,
                                         mlp_blob_.mutable_mlu_data(),
                                         nullptr,
                                         add_blob_.mutable_mlu_data(),
                                         Caffe::forward_param(),
                                         Caffe::queue()));
    MLU_CHECK(cnmlComputeSqrtOpForward_V3(pow_op_ptr_,
                                          add_blob_.mutable_mlu_data(),
                                          gemv_blob_.mutable_mlu_data(),
                                          Caffe::forward_param(),
                                          Caffe::queue()));
    MLU_CHECK(cnmlComputeRealDivOpForward_V3(div1_op_ptr_,
                                             nullptr,
                                             gemv_blob_.mutable_mlu_data(),
                                             div1_blob_.mutable_mlu_data(),
                                             Caffe::forward_param(),
                                             Caffe::queue()));
    MLU_CHECK(cnmlComputeTransposeProOpForward_V3(trans_div1_d2h_layout_,
                                                  div1_blob_.mutable_mlu_data(),
                                                  div1_d2h_blob_.mutable_mlu_data(),
                                                  Caffe::forward_param(),
                                                  Caffe::queue()));
    MLU_CHECK(cnmlComputeReshapeOpForward_V3(reshape_op_ptr_,
                                             div1_d2h_blob_.mutable_mlu_data(),
                                             div1_h2d_blob_.mutable_mlu_data(),
                                             Caffe::forward_param(),
                                             Caffe::queue()));
    MLU_CHECK(cnmlComputeTransposeProOpForward_V3(trans_div1_h2d_layout_,
                                                  div1_h2d_blob_.mutable_mlu_data(),
                                                  reshape_blob_.mutable_mlu_data(),
                                                  Caffe::forward_param(),
                                                  Caffe::queue()));
    MLU_CHECK(cnmlComputeTransposeProOpForward_V3(trans_bMult_d2h_layout_,
                                                  bMult_blob_.mutable_mlu_data(),
                                                  bMult_d2h_blob_.mutable_mlu_data(),
                                                  Caffe::forward_param(),
                                                  Caffe::queue()));
    MLU_CHECK(cnmlComputeReshapeOpForward_V3(reshape1_op_ptr_,
                                             bMult_d2h_blob_.mutable_mlu_data(),
                                             bMult_h2d_blob_.mutable_mlu_data(),
                                             Caffe::forward_param(),
                                             Caffe::queue()));
    MLU_CHECK(cnmlComputeTransposeProOpForward_V3(trans_bMult_h2d_layout_,
                                                  bMult_h2d_blob_.mutable_mlu_data(),
                                                  reshape1_blob_.mutable_mlu_data(),
                                                  Caffe::forward_param(),
                                                  Caffe::queue()));
    MLU_CHECK(cnmlComputeCycleMultOpForward_V3(cyclemult_op_ptr_,
                                               reshape1_blob_.mutable_mlu_data(),
                                               reshape_blob_.mutable_mlu_data(),
                                               scale_blob_.mutable_mlu_data(),
                                               Caffe::forward_param(),
                                               Caffe::queue()));
    MLU_CHECK(cnmlComputeTransposeProOpForward_V3(trans_scale_d2h_layout_,
                                                  scale_blob_.mutable_mlu_data(),
                                                  scale_d2h_blob_.mutable_mlu_data(),
                                                  Caffe::forward_param(),
                                                  Caffe::queue()));
    MLU_CHECK(cnmlComputeReshapeOpForward_V3(reshape2_op_ptr_,
                                             scale_d2h_blob_.mutable_mlu_data(),
                                             scale_h2d_blob_.mutable_mlu_data(),
                                             Caffe::forward_param(),
                                             Caffe::queue()));
    MLU_CHECK(cnmlComputeTransposeProOpForward_V3(trans_scale_h2d_layout_,
                                                  scale_h2d_blob_.mutable_mlu_data(),
                                                  reshape2_blob_.mutable_mlu_data(),
                                                  Caffe::forward_param(),
                                                  Caffe::queue()));

  // normalization in channels which square-sum number is height*width.
  } else {
    MLU_CHECK(cnmlComputeConvOpForward_V3(gemv_op_ptr_,
                                          sqr_blob_.mutable_mlu_data(),
                                          gemv_blob_.mutable_mlu_data(),
                                          Caffe::forward_param(),
                                          Caffe::queue()));
    MLU_CHECK(cnmlComputeSqrtOpForward_V3(sqrt_op_ptr_,
                                          gemv_blob_.mutable_mlu_data(),
                                          powx_blob_.mutable_mlu_data(),
                                          Caffe::forward_param(),
                                          Caffe::queue()));
    MLU_CHECK(cnmlComputeConvOpForward_V3(gemm1_op_ptr_,
                                          powx_blob_.mutable_mlu_data(),
                                          gemm1_blob_.mutable_mlu_data(),
                                          Caffe::forward_param(),
                                          Caffe::queue()));
    MLU_CHECK(cnmlComputeRealDivOpForward_V3(div_op_ptr_,
                                             bMult_blob_.mutable_mlu_data(),
                                             gemm1_blob_.mutable_mlu_data(),
                                             reshape2_blob_.mutable_mlu_data(),
                                             Caffe::forward_param(),
                                             Caffe::queue()));
  }
  // scale the results with same value
  if (this->channel_shared_) {
     MLU_CHECK(cnmlComputeBroadcastMultOpForward_V3(mult1_op_ptr_,
                                                    reshape2_blob_.mutable_mlu_data(),
                                                    nullptr,
                                                    top[0]->mutable_mlu_data(),
                                                    Caffe::forward_param(),
                                                    Caffe::queue()));
  // scale the results with different values.
  } else {
    MLU_CHECK(cnmlComputeBroadcastMultOpForward_V3(mult_op_ptr_, nullptr,
                                                   reshape2_blob_.mutable_mlu_data(),
                                                   top[0]->mutable_mlu_data(),
                                                   Caffe::forward_param(),
                                                   Caffe::queue()));
  // LOG(INFO) << "after mult " << top[0]->cpu_data()[0];
  }
}

template <typename Dtype>
void MLUNormalizeLayer<Dtype>::MLUCompileOp() {
  MLU_CHECK(cnmlCompileBaseOp(max_op_ptr_,
                              Caffe::rt_core(),
                              Caffe::core_number()));
  MLU_CHECK(cnmlCompileBaseOp(power_op_ptr_,
                              Caffe::rt_core(),
                              Caffe::core_number()));
  MLU_CHECK(cnmlCompileBaseOp(bMult_op_ptr_,
                              Caffe::rt_core(),
                              Caffe::core_number()));
  MLU_CHECK(cnmlCompileBaseOp(sqr_op_ptr_,
                              Caffe::rt_core(),
                              Caffe::core_number()));
  if (this->across_spatial_) {
    MLU_CHECK(cnmlCompileBaseOp(mlp_op_ptr_,
                                Caffe::rt_core(),
                                Caffe::core_number()));
    MLU_CHECK(cnmlCompileBaseOp(add_op_ptr_,
                                Caffe::rt_core(),
                                Caffe::core_number()));
    MLU_CHECK(cnmlCompileBaseOp(pow_op_ptr_,
                                Caffe::rt_core(),
                                Caffe::core_number()));
    MLU_CHECK(cnmlCompileBaseOp(div1_op_ptr_,
                                Caffe::rt_core(),
                                Caffe::core_number()));
    MLU_CHECK(cnmlCompileBaseOp(trans_div1_d2h_layout_,
                                Caffe::rt_core(),
                                Caffe::core_number()));
    MLU_CHECK(cnmlCompileBaseOp(reshape_op_ptr_,
                                Caffe::rt_core(),
                                Caffe::core_number()));
    MLU_CHECK(cnmlCompileBaseOp(trans_div1_h2d_layout_,
                                Caffe::rt_core(),
                                Caffe::core_number()));
    MLU_CHECK(cnmlCompileBaseOp(trans_bMult_d2h_layout_,
                                Caffe::rt_core(),
                                Caffe::core_number()));
    MLU_CHECK(cnmlCompileBaseOp(reshape1_op_ptr_,
                                Caffe::rt_core(),
                                Caffe::core_number()));
    MLU_CHECK(cnmlCompileBaseOp(trans_bMult_h2d_layout_,
                                Caffe::rt_core(),
                                Caffe::core_number()));
    MLU_CHECK(cnmlCompileBaseOp(cyclemult_op_ptr_,
                                Caffe::rt_core(),
                                Caffe::core_number()));
    MLU_CHECK(cnmlCompileBaseOp(trans_scale_d2h_layout_,
                                Caffe::rt_core(),
                                Caffe::core_number()));
    MLU_CHECK(cnmlCompileBaseOp(reshape2_op_ptr_,
                                Caffe::rt_core(),
                                Caffe::core_number()));
    MLU_CHECK(cnmlCompileBaseOp(trans_scale_h2d_layout_,
                                Caffe::rt_core(),
                                Caffe::core_number()));
  } else {
    MLU_CHECK(cnmlCompileBaseOp(gemv_op_ptr_,
                                Caffe::rt_core(),
                                Caffe::core_number()));
    MLU_CHECK(cnmlCompileBaseOp(sqrt_op_ptr_,
                                Caffe::rt_core(),
                                Caffe::core_number()));
    MLU_CHECK(cnmlCompileBaseOp(gemm1_op_ptr_,
                                Caffe::rt_core(),
                                Caffe::core_number()));
    MLU_CHECK(cnmlCompileBaseOp(div_op_ptr_,
                                Caffe::rt_core(),
                                Caffe::core_number()));
  }
  if (this->channel_shared_) {
    MLU_CHECK(cnmlCompileBaseOp(mult1_op_ptr_,
                                Caffe::rt_core(),
                                Caffe::core_number()));
  } else {
    MLU_CHECK(cnmlCompileBaseOp(mult_op_ptr_,
                                Caffe::rt_core(),
                                Caffe::core_number()));
  }
}

template <typename Dtype>
void MLUNormalizeLayer<Dtype>::fuse(MFusion<Dtype>* fuser) {
  fuser->fuse(max_op_ptr_);
  fuser->fuse(power_op_ptr_);
  fuser->fuse(bMult_op_ptr_);
  fuser->fuse(sqr_op_ptr_);
  if (this->across_spatial_) {
    fuser->fuse(mlp_op_ptr_);
    fuser->fuse(add_op_ptr_);
    fuser->fuse(pow_op_ptr_);
    fuser->fuse(div1_op_ptr_);
    fuser->fuse(trans_div1_d2h_layout_);
    fuser->fuse(reshape_op_ptr_);
    fuser->fuse(trans_div1_h2d_layout_);
    fuser->fuse(trans_bMult_d2h_layout_);
    fuser->fuse(reshape1_op_ptr_);
    fuser->fuse(trans_bMult_h2d_layout_);
    fuser->fuse(cyclemult_op_ptr_);
    fuser->fuse(trans_scale_d2h_layout_);
    fuser->fuse(reshape2_op_ptr_);
    fuser->fuse(trans_scale_h2d_layout_);
  } else {
    fuser->fuse(gemv_op_ptr_);
    fuser->fuse(sqrt_op_ptr_);
    fuser->fuse(gemm1_op_ptr_);
    fuser->fuse(div_op_ptr_);
  }
  if (this->channel_shared_) {
    fuser->fuse(mult1_op_ptr_);
  } else {
    fuser->fuse(mult_op_ptr_);
  }
}

template <typename Dtype>
MLUNormalizeLayer<Dtype>::~MLUNormalizeLayer() {
  MLUDestroyOp();
  delete mlp_weight_blob_;
  delete mlp_bias_blob_;
  delete eps_blob_;
  delete div0_blob_;
  delete mult_blob_;
  delete gemv_weight_;
  delete gemm_weight_;
  delete gemm_matrix_;
  mlp_weight_blob_ = nullptr;
  mlp_bias_blob_ = nullptr;
  eps_blob_ = nullptr;
  div0_blob_ = nullptr;
  mult_blob_ = nullptr;
  gemv_weight_ = nullptr;
  gemm_weight_ = nullptr;
  gemm_matrix_ = nullptr;
}

template <typename Dtype>
void MLUNormalizeLayer<Dtype>::MLUDestroyOp() {
  if (max_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&max_op_ptr_));
    max_op_ptr_ = nullptr;
  }
  if (power_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&power_op_ptr_));
    power_op_ptr_ = nullptr;
  }
  if (bMult_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&bMult_op_ptr_));
    bMult_op_ptr_ = nullptr;
  }
  if (sqr_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&sqr_op_ptr_));
    sqr_op_ptr_ = nullptr;
  }
  if (gemv_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&gemv_op_ptr_));
    gemv_op_ptr_ = nullptr;
  }
  if (sqrt_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&sqrt_op_ptr_));
    sqrt_op_ptr_ = nullptr;
  }
  if (gemm1_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&gemm1_op_ptr_));
    gemm1_op_ptr_ = nullptr;
  }
  if (div_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&div_op_ptr_));
    div_op_ptr_ = nullptr;
  }
  if (mult_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&mult_op_ptr_));
    mult_op_ptr_ = nullptr;
  }
  if (gemv_param_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyConvOpParam(&gemv_param_ptr_));
    gemv_param_ptr_ = nullptr;
  }
  if (mlp_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&mlp_op_ptr_));
    mlp_op_ptr_ = nullptr;
  }
  if (add_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&add_op_ptr_));
    add_op_ptr_ = nullptr;
  }
  if (pow_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&pow_op_ptr_));
    pow_op_ptr_ = nullptr;
  }
  if (div1_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&div1_op_ptr_));
    div1_op_ptr_ = nullptr;
  }
  if (reshape_param_ != nullptr) {
     MLU_CHECK(cnmlDestroyReshapeOpParam(&reshape_param_));
     reshape_param_ = nullptr;
  }
  if (reshape_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&reshape_op_ptr_));
    reshape_op_ptr_ = nullptr;
  }
  if (reshape1_param_ != nullptr) {
     MLU_CHECK(cnmlDestroyReshapeOpParam(&reshape1_param_));
     reshape1_param_ = nullptr;
  }
  if (reshape1_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&reshape1_op_ptr_));
    reshape1_op_ptr_ = nullptr;
  }
  if (cyclemult_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&cyclemult_op_ptr_));
    cyclemult_op_ptr_ = nullptr;
  }
  if (reshape2_param_ != nullptr) {
     MLU_CHECK(cnmlDestroyReshapeOpParam(&reshape2_param_));
     reshape2_param_ = nullptr;
  }
  if (reshape2_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&reshape2_op_ptr_));
    reshape2_op_ptr_ = nullptr;
  }
  if (mult1_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&mult1_op_ptr_));
    mult1_op_ptr_ = nullptr;
  }
  if (trans_div1_d2h_layout_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&trans_div1_d2h_layout_));
    trans_div1_d2h_layout_ = nullptr;
  }
  if (trans_div1_d2h_param_ != nullptr) {
    MLU_CHECK(cnmlDestroyNdTransposeOpParam(&trans_div1_d2h_param_));
    trans_div1_d2h_param_ = nullptr;
  }
  if (trans_div1_h2d_layout_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&trans_div1_h2d_layout_));
    trans_div1_h2d_layout_ = nullptr;
  }
  if (trans_div1_h2d_param_ != nullptr) {
    MLU_CHECK(cnmlDestroyNdTransposeOpParam(&trans_div1_h2d_param_));
    trans_div1_h2d_param_ = nullptr;
  }
  if (trans_bMult_d2h_layout_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&trans_bMult_d2h_layout_));
    trans_bMult_d2h_layout_ = nullptr;
  }
  if (trans_bMult_d2h_param_ != nullptr) {
    MLU_CHECK(cnmlDestroyNdTransposeOpParam(&trans_bMult_d2h_param_));
    trans_bMult_d2h_param_ = nullptr;
  }
  if (trans_bMult_h2d_layout_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&trans_bMult_h2d_layout_));
    trans_bMult_h2d_layout_ = nullptr;
  }
  if (trans_bMult_h2d_param_ != nullptr) {
    MLU_CHECK(cnmlDestroyNdTransposeOpParam(&trans_bMult_h2d_param_));
    trans_bMult_h2d_param_ = nullptr;
  }
  if (trans_scale_d2h_layout_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&trans_scale_d2h_layout_));
    trans_scale_d2h_layout_ = nullptr;
  }
  if (trans_scale_d2h_param_ != nullptr) {
    MLU_CHECK(cnmlDestroyNdTransposeOpParam(&trans_scale_d2h_param_));
    trans_scale_d2h_param_ = nullptr;
  }
  if (trans_scale_h2d_layout_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&trans_scale_h2d_layout_));
    trans_scale_h2d_layout_ = nullptr;
  }
  if (trans_scale_h2d_param_ != nullptr) {
    MLU_CHECK(cnmlDestroyNdTransposeOpParam(&trans_scale_h2d_param_));
    trans_scale_h2d_param_ = nullptr;
  }
  for (int i = 0; i < quant_params.size(); i++) {
    if (quant_params[i] != nullptr) {
      MLU_CHECK(cnmlDestroyQuantizedParam(&quant_params[i]));
      quant_params[i] = nullptr;
    }
  }
}

template <typename Dtype>
void MLUNormalizeLayer<Dtype>::MLUCreateOpBindData(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  if (int8_mode) {
    if (this->layer_param_.bottom_mlu_dtype_size() > 0) {
      if (this->layer_param_.bottom_mlu_dtype(0).position_size())
        sqr_blob_.set_mlu_position(
            this->layer_param_.bottom_mlu_dtype(0).position(0));
      if (this->layer_param_.bottom_mlu_dtype(0).scale_size())
        sqr_blob_.set_mlu_scale(
            this->layer_param_.bottom_mlu_dtype(0).scale(0));
    } else {
      sqr_blob_.set_mlu_position(9);
      sqr_blob_.set_mlu_scale(1);
    }
    if (!this->across_spatial_) {
      gemv_weight_->set_mlu_position(-6);
      gemm_weight_->set_mlu_position(-6);
      if (this->layer_param_.bottom_mlu_dtype_size() == 2) {
        if (this->layer_param_.bottom_mlu_dtype(1).position_size())
          powx_blob_.set_mlu_position(
            this->layer_param_.bottom_mlu_dtype(1).position(0));
        if (this->layer_param_.bottom_mlu_dtype(1).scale_size())
          powx_blob_.set_mlu_scale(
            this->layer_param_.bottom_mlu_dtype(1).scale(0));
      } else {
        powx_blob_.set_mlu_position(-2);
        powx_blob_.set_mlu_scale(1);
      }
    } else {
      mlp_weight_blob_->set_mlu_position(-6);
    }
  }

  Dtype* data = gemm_matrix_->mutable_cpu_data();
  for (int j = 0; j < gemm_matrix_->channels(); j++) {
    data[j] = this->blobs_[0]->cpu_data()[j];
  }
  cnmlBindConstData_V2(gemm_matrix_->mlu_tensor(),
                   reinterpret_cast<void*>(gemm_matrix_->sync_data()),
                   false);

  MLU_CHECK(cnmlCreateMaxOp(&max_op_ptr_,
      bottom[0]->mlu_tensor(),
      max_blob_.mlu_tensor(),
      index_blob_.mlu_tensor()));
  MLU_CHECK(cnmlCreatePowerOp(&power_op_ptr_,
       max_blob_.mlu_tensor(),
       power_blob_.mlu_tensor(),
       -1));
  MLU_CHECK(cnmlCreateBroadcastMultOp(&bMult_op_ptr_,
      bottom[0]->mlu_tensor(),
      power_blob_.mlu_tensor(),
      bMult_blob_.mlu_tensor()));
  MLU_CHECK(cnmlCreateMultOp(&sqr_op_ptr_,
      bMult_blob_.mlu_tensor(),
      bMult_blob_.mlu_tensor(),
      sqr_blob_.mlu_tensor()));

  if (this->across_spatial_) {
    MLU_CHECK(cnmlCreateMlpOp(&mlp_op_ptr_,
                             sqr_blob_.mlu_tensor(),
                             mlp_blob_.mlu_tensor(),
                             mlp_weight_blob_->mlu_tensor(),
                             mlp_bias_blob_->mlu_tensor()));
    if (int8_mode) {
      MLU_CHECK(cnmlCreateQuantizedParam(&quant_params[0],
                    this->layer_param_.bottom_mlu_dtype(0).position(0),
                    this->layer_param_.bottom_mlu_dtype(0).scale(0),
                    0.0));
      MLU_CHECK(cnmlSetOperationComputingDataType(mlp_op_ptr_,
                               sqr_blob_.mlu_tensor(),
                               CNML_DATA_INT8,
                               quant_params[0]));
    }

    MLU_CHECK(cnmlBindConstData_V2(mlp_weight_blob_->mlu_tensor(),
                          mlp_weight_blob_->sync_data(),
                          false));
    MLU_CHECK(cnmlBindConstData_V2(mlp_bias_blob_->mlu_tensor(),
                          mlp_bias_blob_->sync_data(),
                          false));
    MLU_CHECK(cnmlCreateAddOp(&add_op_ptr_,
                             mlp_blob_.mlu_tensor(),
                             eps_blob_->mlu_tensor(),
                             add_blob_.mlu_tensor()));
    MLU_CHECK(cnmlBindConstData_V2(eps_blob_->mlu_tensor(),
                          eps_blob_->sync_data(),
                          false));
    MLU_CHECK(cnmlCreateSqrtOp(&pow_op_ptr_,
              add_blob_.mlu_tensor(),
              gemv_blob_.mlu_tensor()));

    MLU_CHECK(cnmlCreateRealDivOp(&div1_op_ptr_,
              div0_blob_->mlu_tensor(),
              gemv_blob_.mlu_tensor(),
              div1_blob_.mlu_tensor()));

    MLU_CHECK(cnmlBindConstData_V2(div0_blob_->mlu_tensor(),
             div0_blob_->sync_data(),
       false));
  // div1_blob_ n,h,w,c to n,c,h,w
  int input_len = div1_blob_.shape().size();
  vector<int> dim_order(input_len, 1);
  dim_order[0] = 0;
  dim_order[1] = input_len - 1;
  for (int i = 2; i < input_len; i++) {
      dim_order[i] = i-1;
  }
  MLU_CHECK(cnmlCreateNdTransposeOpParam(&trans_div1_d2h_param_,
              dim_order.data(), input_len));

  MLU_CHECK(cnmlCreateNdTransposeProOp(&trans_div1_d2h_layout_,
              div1_blob_.mlu_tensor(),
              div1_d2h_blob_.mlu_tensor(),
              trans_div1_d2h_param_));

  // reshape_blob tensor shape should be: 1, n, c*w, h
  vector<int> reshape_dim = div1_h2d_blob_.mlu_shape();
  MLU_CHECK(cnmlCreateNdReshapeOpParam(&reshape_param_,
                                      reshape_dim.data(),
                                      reshape_blob_.mlu_shape().size()));
  MLU_CHECK(cnmlCreateReshapeOp(&reshape_op_ptr_,
                              reshape_param_,
                              div1_d2h_blob_.mlu_tensor(),
                              div1_h2d_blob_.mlu_tensor()));
  //  div1_d2h_blob_ n,c,h,w to n,h,w,c
  input_len = div1_h2d_blob_.shape().size();
  vector<int> dim_order_last(input_len, 1);
  dim_order_last[0] = 0;
  dim_order_last[input_len-1] = 1;
  for (int i = 1; i < input_len - 1; i++) {
      dim_order_last[i] = i + 1;
  }
  MLU_CHECK(cnmlCreateNdTransposeOpParam(&trans_div1_h2d_param_,
                                       dim_order_last.data(), input_len));

  MLU_CHECK(cnmlCreateNdTransposeProOp(&trans_div1_h2d_layout_,
                                     div1_h2d_blob_.mlu_tensor(),
                                     reshape_blob_.mlu_tensor(),
                                     trans_div1_h2d_param_));
  // bMult_blob_ n,h,w,c to n,c,h,w
  input_len = bMult_blob_.shape().size();
  MLU_CHECK(cnmlCreateNdTransposeOpParam(&trans_bMult_d2h_param_,
                                       dim_order.data(), input_len));

  MLU_CHECK(cnmlCreateNdTransposeProOp(&trans_bMult_d2h_layout_,
                                     bMult_blob_.mlu_tensor(),
                                     bMult_d2h_blob_.mlu_tensor(),
                                     trans_bMult_d2h_param_));
  // reshape1_blob tensor shape should be: 1, n, c*w, h
  vector<int> reshape1_dim = bMult_h2d_blob_.mlu_shape();
  MLU_CHECK(cnmlCreateNdReshapeOpParam(&reshape1_param_,
                                      reshape1_dim.data(),
                                      reshape1_blob_.mlu_shape().size()));
  MLU_CHECK(cnmlCreateReshapeOp(&reshape1_op_ptr_,
                              reshape1_param_,
                              bMult_d2h_blob_.mlu_tensor(),
                              bMult_h2d_blob_.mlu_tensor()));
  // bMult_d2h_blob_ n,c,h,w to n,h,w,c
  input_len = bMult_h2d_blob_.shape().size();
  MLU_CHECK(cnmlCreateNdTransposeOpParam(&trans_bMult_h2d_param_,
                                       dim_order_last.data(), input_len));

  MLU_CHECK(cnmlCreateNdTransposeProOp(&trans_bMult_h2d_layout_,
                                     bMult_h2d_blob_.mlu_tensor(),
                                     reshape1_blob_.mlu_tensor(),
                                     trans_bMult_h2d_param_));
  MLU_CHECK(cnmlCreateCycleMultOp(&cyclemult_op_ptr_,
                                reshape1_blob_.mlu_tensor(),
                                reshape_blob_.mlu_tensor(),
                                scale_blob_.mlu_tensor()));
  // scale_blob_ n,h,w,c to n,c,h,w
  input_len = scale_blob_.shape().size();
  MLU_CHECK(cnmlCreateNdTransposeOpParam(&trans_scale_d2h_param_,
                                       dim_order.data(), input_len));

  MLU_CHECK(cnmlCreateNdTransposeProOp(&trans_scale_d2h_layout_,
              scale_blob_.mlu_tensor(),
              scale_d2h_blob_.mlu_tensor(),
              trans_scale_d2h_param_));
  // reshape2_blob shape: n, c, h, w
  vector<int> reshape2_dim = scale_h2d_blob_.mlu_shape();
  MLU_CHECK(cnmlCreateNdReshapeOpParam(&reshape2_param_,
                                      reshape2_dim.data(),
                                      bottom[0]->num_axes()));

  MLU_CHECK(cnmlCreateReshapeOp(&reshape2_op_ptr_,
                              reshape2_param_,
                              scale_d2h_blob_.mlu_tensor(),
                              scale_h2d_blob_.mlu_tensor()));
  // scale_d2h_blob_ n,c,h,w to n,h,w,c
  input_len = scale_h2d_blob_.shape().size();
  MLU_CHECK(cnmlCreateNdTransposeOpParam(&trans_scale_h2d_param_,
                                       dim_order_last.data(), input_len));

  MLU_CHECK(cnmlCreateNdTransposeProOp(&trans_scale_h2d_layout_,
                                       scale_h2d_blob_.mlu_tensor(),
                                       reshape2_blob_.mlu_tensor(),
                                       trans_scale_h2d_param_));
  } else {
    MLU_CHECK(cnmlCreateConvOpParam(&gemv_param_ptr_,
                                     1,
                                     1,
                                     1,
                                     1,
                                     0,
                                     0));
    MLU_CHECK(cnmlCreateConvOp(&gemv_op_ptr_,
                                gemv_param_ptr_,
                                sqr_blob_.mlu_tensor(),
                                gemv_blob_.mlu_tensor(),
                                gemv_weight_->mlu_tensor(),
                                nullptr));
    if (int8_mode) {
      if (this->layer_param_.bottom_mlu_dtype_size() > 0) {
        MLU_CHECK(cnmlCreateQuantizedParam(&quant_params[0],
                         this->layer_param_.bottom_mlu_dtype(0).position(0),
                         this->layer_param_.bottom_mlu_dtype(0).scale(0),
                         0.0));
        MLU_CHECK(cnmlSetOperationComputingDataType(gemv_op_ptr_,
                                                    sqr_blob_.mlu_tensor(),
                                                    CNML_DATA_INT8,
                                                    quant_params[0]));
      }
    }
    MLU_CHECK(cnmlBindConstData_V2(gemv_weight_->mlu_tensor(),
                                   reinterpret_cast<void *>(gemv_weight_->sync_data()),
                                   false));
    MLU_CHECK(cnmlCreateSqrtOp(&sqrt_op_ptr_,
                                gemv_blob_.mlu_tensor(),
                                powx_blob_.mlu_tensor()));
    MLU_CHECK(cnmlCreateConvOp(&gemm1_op_ptr_,
                                gemv_param_ptr_,
                                powx_blob_.mlu_tensor(),
                                gemm1_blob_.mlu_tensor(),
                                gemm_weight_->mlu_tensor(),
                                nullptr));
    if (int8_mode) {
      if (this->layer_param_.bottom_mlu_dtype_size() > 1) {
        MLU_CHECK(cnmlCreateQuantizedParam(&quant_params[1],
                                this->layer_param_.bottom_mlu_dtype(1).position(0),
                                this->layer_param_.bottom_mlu_dtype(1).scale(0),
                                0.0));
        MLU_CHECK(cnmlSetOperationComputingDataType(gemm1_op_ptr_,
                                                    powx_blob_.mlu_tensor(),
                                                    CNML_DATA_INT8,
                                                    quant_params[1]));
      }
    }
    MLU_CHECK(cnmlBindConstData_V2(gemm_weight_->mlu_tensor(),
                                   reinterpret_cast<void *>(gemm_weight_->sync_data()),
                                   false));
    MLU_CHECK(cnmlCreateRealDivOp(&div_op_ptr_,
                                   bMult_blob_.mlu_tensor(),
                                   gemm1_blob_.mlu_tensor(),
                                   reshape2_blob_.mlu_tensor()));
  }
  if (this->channel_shared_) {
    Dtype* data = mult_blob_->mutable_cpu_data();
    data[0] = this->blobs_[0]->cpu_data()[0];
    MLU_CHECK(cnmlCreateBroadcastMultOp(&mult1_op_ptr_,
                                         reshape2_blob_.mlu_tensor(),
                                         mult_blob_->mlu_tensor(),
                                         top[0]->mlu_tensor()));
    MLU_CHECK(cnmlBindConstData_V2(mult_blob_->mlu_tensor(),
                                   mult_blob_->sync_data(),
                                   false));
  } else {
    MLU_CHECK(cnmlCreateBroadcastMultOp(&mult_op_ptr_,
                                         gemm_matrix_->mlu_tensor(),
                                         reshape2_blob_.mlu_tensor(),
                                         top[0]->mlu_tensor()));
  }
}

template <typename Dtype>
void MLUNormalizeLayer<Dtype>::Reshape_tensor(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
  BaseDataType mlu_dtype = bottom[0]->mlu_type();
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "Number of axes of bottom blob must be >=2.";
  top[0]->Reshape(bottom[0]->shape(), cpu_dtype, mlu_dtype, CNML_TENSOR);
  vector<int> shape(4, 1);
  max_blob_.Reshape(shape, cpu_dtype, mlu_dtype, CNML_TENSOR);
  index_blob_.Reshape(shape, cpu_dtype, mlu_dtype, CNML_TENSOR);
  power_blob_.Reshape(shape, cpu_dtype, mlu_dtype, CNML_TENSOR);
  bMult_blob_.Reshape(bottom[0]->shape(), cpu_dtype, mlu_dtype, CNML_TENSOR);
  sqr_blob_.Reshape(bottom[0]->shape(), cpu_dtype, mlu_dtype, CNML_TENSOR);
  reshape2_blob_.Reshape(bottom[0]->shape(), cpu_dtype, mlu_dtype, CNML_TENSOR);
  if (this->across_spatial_) {
    vector<int> mlp_shape(4, 1);
    mlp_shape[0] = bottom[0]->num();
    mlp_blob_.Reshape(mlp_shape, cpu_dtype, mlu_dtype, CNML_TENSOR);
    add_blob_.Reshape(mlp_shape, cpu_dtype, mlu_dtype, CNML_TENSOR);
    gemv_blob_.Reshape(mlp_shape, cpu_dtype, mlu_dtype, CNML_TENSOR);
    div1_blob_.Reshape(mlp_shape, cpu_dtype, mlu_dtype, CNML_TENSOR);
    vector<int> scale_shape(4, 1);
    scale_shape[1] = bottom[0]->num();
    scale_shape[2] = bottom[0]->channels() * bottom[0]->height();
    scale_shape[3] = bottom[0]->width();
    scale_blob_.Reshape(scale_shape, cpu_dtype, mlu_dtype, CNML_TENSOR);
    vector<int> shape1(4, 1);
    shape1[1] =  bottom[0]->shape(0);
    reshape_blob_.Reshape(shape1, cpu_dtype, mlu_dtype, CNML_TENSOR);
    shape1[2] = bottom[0]->channels() * bottom[0]->height();
    shape1[3] = bottom[0]->width();
    reshape1_blob_.Reshape(shape1, cpu_dtype, mlu_dtype, CNML_TENSOR);
    div1_d2h_blob_.Reshape(mlp_shape, cpu_dtype, mlu_dtype, CNML_TENSOR, CNML_NHWC);
    div1_h2d_blob_.Reshape(reshape_blob_.shape(),
                      cpu_dtype, mlu_dtype, CNML_TENSOR, CNML_NHWC);
    bMult_d2h_blob_.Reshape(bMult_blob_.shape(),
                      cpu_dtype, mlu_dtype, CNML_TENSOR, CNML_NHWC);
    bMult_h2d_blob_.Reshape(shape1,
                      cpu_dtype, mlu_dtype, CNML_TENSOR, CNML_NHWC);
    scale_d2h_blob_.Reshape(scale_shape,
                      cpu_dtype, mlu_dtype, CNML_TENSOR, CNML_NHWC);
    scale_h2d_blob_.Reshape(reshape2_blob_.shape(),
                      cpu_dtype, mlu_dtype, CNML_TENSOR, CNML_NHWC);
  } else {
    vector<int> gemv_shape(bottom[0]->shape());
    gemv_shape[1] = 1;
    gemv_blob_.Reshape(gemv_shape, cpu_dtype, mlu_dtype, CNML_TENSOR);
    powx_blob_.Reshape(gemv_shape, cpu_dtype, mlu_dtype, CNML_TENSOR);
    gemm1_blob_.Reshape(bottom[0]->shape(), cpu_dtype, mlu_dtype, CNML_TENSOR);
  }
}

INSTANTIATE_CLASS(MLUNormalizeLayer);

}  // namespace caffe
#endif  // USE_MLU
