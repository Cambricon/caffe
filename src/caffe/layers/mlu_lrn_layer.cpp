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
#include <vector>
#include "caffe/layers/mlu_lrn_layer.hpp"

namespace caffe {

template <typename Dtype>
void MLULRNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top) {
  LRNLayer<Dtype>::LayerSetUp(bottom, top);
  int c = bottom[0]->channels();
  if (c > 8) {
    CHECK_LE(this->size_, 15)
      << "localsize must <= 15 when c > 8!";
  }
}

template <typename Dtype>
void MLULRNLayer<Dtype>::Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  this->num_ = bottom[0]->num();
  this->channels_ = bottom[0]->channels();
  this->height_ = bottom[0]->height();
  this->width_ = bottom[0]->width();

  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
  BaseDataType mlu_dtype = bottom[0]->mlu_type();
  switch (this->layer_param_.lrn_param().norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
    top[0]->Reshape(this->num_, this->channels_, this->height_,
                    this->width_, cpu_dtype, mlu_dtype, CNML_TENSOR);
    this->scale_.Reshape(this->num_, this->channels_,
                         this->height_, this->width_);
    break;
  case LRNParameter_NormRegion_WITHIN_CHANNEL:
    top[0]->Reshape(this->num_, this->channels_, this->height_,
                    this->width_, cpu_dtype, mlu_dtype, CNML_TENSOR);
    vector<int> blob_shape{this->num_, this->channels_,
                           this->height_, this->width_};

    scale_blob_.Reshape(blob_shape, cpu_dtype, mlu_dtype, CNML_TENSOR);

    vector<int> scale_shape(4, 1);
    scale_shape[1] = bottom[0]->shape()[1];
    alpha1_blob_.Reshape(scale_shape, cpu_dtype, mlu_dtype, CNML_CONST);
    caffe_set(alpha1_blob_.count(),
              Dtype(0.01),
              alpha1_blob_.mutable_cpu_data());

    beta1_blob_.Reshape(scale_shape, cpu_dtype, mlu_dtype, CNML_CONST);
    caffe_set(beta1_blob_.count(),
              Dtype(0.0),
              beta1_blob_.mutable_cpu_data());


    // square blob
    square_blob_.Reshape(blob_shape, cpu_dtype, mlu_dtype, CNML_TENSOR);
    // pool blob
    vector<int> addpad_shape;
    addpad_shape.push_back(bottom[0]->num());
    addpad_shape.push_back(bottom[0]->channels());
    addpad_shape.push_back(bottom[0]->height() + this->pre_pad_*2);
    addpad_shape.push_back(bottom[0]->width() + this->pre_pad_*2);
    addpad_blob_.Reshape(addpad_shape, cpu_dtype, mlu_dtype, CNML_TENSOR);
    pool_blob_.Reshape(blob_shape, cpu_dtype, mlu_dtype, CNML_TENSOR);
    // power blob
    vector<int> alpha_beta_shape(4, 1);
    alpha_beta_shape[1] = this->channels_;
    alpha_blob_.Reshape(alpha_beta_shape, cpu_dtype, mlu_dtype, CNML_CONST);
    beta_blob_.Reshape(alpha_beta_shape, cpu_dtype, mlu_dtype, CNML_CONST);
    temp_blob_.Reshape(blob_shape, cpu_dtype, mlu_dtype, CNML_TENSOR);
    power_blob_.Reshape(blob_shape, cpu_dtype, mlu_dtype, CNML_TENSOR);
    break;
  }
}

template <typename Dtype>
void MLULRNLayer<Dtype>::MLUDestroyOp() {
  switch (this->layer_param_.lrn_param().norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
    if (mlu_lrn_op_ptr_ != nullptr) {
      MLU_CHECK(cnmlDestroyBaseOp(&mlu_lrn_op_ptr_));
      mlu_lrn_op_ptr_ = nullptr;
    }
    if (mlu_lrn_param_ptr_ != nullptr) {
      MLU_CHECK(cnmlDestroyLrnOpParam(&mlu_lrn_param_ptr_));
      mlu_lrn_param_ptr_ = nullptr;
    }
    if (input_quant_params != nullptr) {
      MLU_CHECK(cnmlDestroyQuantizedParam(&input_quant_params));
      input_quant_params = nullptr;
    }
    if (output_quant_params != nullptr) {
      MLU_CHECK(cnmlDestroyQuantizedParam(&output_quant_params));
      output_quant_params = nullptr;
    }
    break;
  case LRNParameter_NormRegion_WITHIN_CHANNEL:
    if (mlu_scale1_op_ptr_ != nullptr) {
      MLU_CHECK(cnmlDestroyBaseOp(&mlu_scale1_op_ptr_));
      mlu_scale1_op_ptr_ = nullptr;
    }
    if (mlu_square_op_ptr_ != nullptr) {
      MLU_CHECK(cnmlDestroyBaseOp(&mlu_square_op_ptr_));
      mlu_square_op_ptr_ = nullptr;
    }
    if (mlu_pool_op_ptr_ != nullptr) {
      MLU_CHECK(cnmlDestroyBaseOp(&mlu_pool_op_ptr_));
      mlu_pool_op_ptr_ = nullptr;
    }
    if (mlu_addpad_op_ptr_ != nullptr) {
      MLU_CHECK(cnmlDestroyBaseOp(&mlu_addpad_op_ptr_));
      mlu_addpad_op_ptr_ = nullptr;
    }
    if (mlu_scale_op_ptr_ != nullptr) {
      MLU_CHECK(cnmlDestroyBaseOp(&mlu_scale_op_ptr_));
      mlu_scale_op_ptr_ = nullptr;
    }
    if (mlu_power_op_ptr_ != nullptr) {
      MLU_CHECK(cnmlDestroyBaseOp(&mlu_power_op_ptr_));
      mlu_power_op_ptr_ = nullptr;
    }
    if (mlu_product_op_ptr_ != nullptr) {
      MLU_CHECK(cnmlDestroyBaseOp(&mlu_product_op_ptr_));
      mlu_product_op_ptr_ = nullptr;
    }
    break;
  }
}

template <typename Dtype>
void MLULRNLayer<Dtype>::fuse(MFusion<Dtype>* fuser) {
  switch (this->layer_param_.lrn_param().norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
    fuser->fuse(mlu_lrn_op_ptr_);
    break;
  case LRNParameter_NormRegion_WITHIN_CHANNEL:
    fuser->fuse(mlu_scale1_op_ptr_);
    fuser->fuse(mlu_square_op_ptr_);
    if (this->pre_pad_) {
      fuser->fuse(mlu_addpad_op_ptr_);
    }
    fuser->fuse(mlu_pool_op_ptr_);
    fuser->fuse(mlu_scale_op_ptr_);
    if ((-this->beta_) != 1) {
      fuser->fuse(mlu_power_op_ptr_);
    }
    fuser->fuse(mlu_product_op_ptr_);
    break;
  }
}

template <typename Dtype>
void MLULRNLayer<Dtype>::MLUCompileOp() {
  switch (this->layer_param_.lrn_param().norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
    MLU_CHECK(cnmlCompileBaseOp(mlu_lrn_op_ptr_,
                                Caffe::rt_core(),
                                Caffe::core_number()));
    break;
  case LRNParameter_NormRegion_WITHIN_CHANNEL:
    MLU_CHECK(cnmlCompileBaseOp(mlu_scale1_op_ptr_,
                                Caffe::rt_core(),
                                Caffe::core_number()));
    MLU_CHECK(cnmlCompileBaseOp(mlu_square_op_ptr_,
                                Caffe::rt_core(),
                                Caffe::core_number()));
    if (this->pre_pad_) {
      MLU_CHECK(cnmlCompileBaseOp(mlu_addpad_op_ptr_,
                                  Caffe::rt_core(),
                                  Caffe::core_number()));
    }
    MLU_CHECK(cnmlCompileBaseOp(mlu_pool_op_ptr_,
                                Caffe::rt_core(),
                                Caffe::core_number()));
    MLU_CHECK(cnmlCompileBaseOp(mlu_scale_op_ptr_,
                                Caffe::rt_core(),
                                Caffe::core_number()));
    if ((-this->beta_) != 1) {
      MLU_CHECK(cnmlCompileBaseOp(mlu_power_op_ptr_,
                                  Caffe::rt_core(),
                                  Caffe::core_number()));
    }
    MLU_CHECK(cnmlCompileBaseOp(mlu_product_op_ptr_,
                                Caffe::rt_core(),
                                Caffe::core_number()));
    break;
  }
}

template <typename Dtype>
void MLULRNLayer<Dtype>::MLUCreateOpBindData(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
  switch (this->layer_param_.lrn_param().norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
    MLU_CHECK(cnmlCreateLrnOpParam(&mlu_lrn_param_ptr_,
                                CNML_LRN_V3,
                                this->size_,
                                this->alpha_,
                                this->beta_,
                                this->k_));
    MLU_CHECK(cnmlCreateLrnOp(&mlu_lrn_op_ptr_,
                              mlu_lrn_param_ptr_,
                              bottom[0]->mlu_tensor(),
                              top[0]->mlu_tensor()));
    if (this->layer_param_.bottom_mlu_dtype_size() > 0 &&
        this->layer_param_.bottom_mlu_dtype(0).position_size()) {
          bottom[0]->set_mlu_position(
              this->layer_param_.bottom_mlu_dtype(0).position(0));
          double scale = this->layer_param_.bottom_mlu_dtype(0).scale(0);
          bottom[0]->set_mlu_scale(scale);
          MLU_CHECK(cnmlCreateQuantizedParam(&input_quant_params,
                this->layer_param_.bottom_mlu_dtype(0).position(0),
                scale,
                0.0));
          MLU_CHECK(cnmlSetOperationComputingDataType(mlu_lrn_op_ptr_,
                  bottom[0]->mlu_tensor(),
                  to_cnml_dtype(this->layer_param_.bottom_mlu_dtype(0).type()),
                  input_quant_params));
    }

    MLU_CHECK(cnmlCreateQuantizedParam(&output_quant_params, 1, 1.0, 0.0));
    MLU_CHECK(cnmlSetOperationComputingDataType(mlu_lrn_op_ptr_,
            top[0]->mlu_tensor(),
            to_cnml_dtype(bottom[0]->mlu_type()),
            output_quant_params));
    break;
  case LRNParameter_NormRegion_WITHIN_CHANNEL:

     MLU_CHECK(cnmlCreateScaleOp(&mlu_scale1_op_ptr_,
                                 bottom[0]->mlu_tensor(),
                                 scale_blob_.mlu_tensor(),
                                 alpha1_blob_.mlu_tensor(),
                                 beta1_blob_.mlu_tensor()));
     MLU_CHECK(cnmlBindConstData_V2(alpha1_blob_.mlu_tensor(),
                                 alpha1_blob_.sync_data(),
                                 false));

     MLU_CHECK(cnmlBindConstData_V2(beta1_blob_.mlu_tensor(),
                                 beta1_blob_.sync_data(),
                                 false));


    // square op
    MLU_CHECK(cnmlCreateMultOp(&mlu_square_op_ptr_,
                               scale_blob_.mlu_tensor(),
                               scale_blob_.mlu_tensor(),
                               square_blob_.mlu_tensor()));
    // pool layer op
    if (this->pre_pad_) {
      MLU_CHECK(cnmlCreateAddPadOpParam(&addpad_param_ptr_,
                                        this->pre_pad_ * 2,
                                        this->pre_pad_ * 2,
                                        0));

      MLU_CHECK(cnmlCreateAddPadOp(&mlu_addpad_op_ptr_,
                                   addpad_param_ptr_,
                                   square_blob_.mlu_tensor(),
                                   addpad_blob_.mlu_tensor()));
    }
    cnmlPoolMode_t pool_mode = CNML_POOL_AVG;
    MLU_CHECK(cnmlCreatePoolOpParam(&pool_param_ptr_,
                                    this->size_,
                                    this->size_,
                                    1,
                                    1,
                                    0, /* origin pad_h ignored */
                                    0, /* origin pad_w ignored */
                                    1, /* dilation_h not set */
                                    1, /* dilation_w not set */
                                    pool_mode,
                                    CNML_POOL_KFULL,
                                    false));

    MLU_CHECK(cnmlCreatePoolOp(&mlu_pool_op_ptr_,
                               pool_param_ptr_,
                               mlu_addpad_op_ptr_?
                               addpad_blob_.mlu_tensor():
                               square_blob_.mlu_tensor(),
                               pool_blob_.mlu_tensor()));
    // power
    for (int i = 0; i < alpha_blob_.count(); i++) {
      alpha_blob_.mutable_cpu_data()[i] = this->alpha_ * 10000;
    }
    for (int i = 0; i < beta_blob_.count(); i++) {
      beta_blob_.mutable_cpu_data()[i] = 1;
    }
    MLU_CHECK(cnmlCreateScaleOp(&mlu_scale_op_ptr_,
                                pool_blob_.mlu_tensor(),
                                (-this->beta_) == 1 ?
                                power_blob_.mlu_tensor():
                                temp_blob_.mlu_tensor(),
                                alpha_blob_.mlu_tensor(),
                                beta_blob_.mlu_tensor()));
    MLU_CHECK(cnmlBindConstData_V2(alpha_blob_.mlu_tensor(),
                                alpha_blob_.sync_data(),
                                false));
    MLU_CHECK(cnmlBindConstData_V2(beta_blob_.mlu_tensor(),
                                beta_blob_.sync_data(),
                                false));
    if ((-this->beta_) != 1) {
      MLU_CHECK(cnmlCreatePowerOp(&mlu_power_op_ptr_,
                                  temp_blob_.mlu_tensor(),
                                  power_blob_.mlu_tensor(),
                                  (-this->beta_)));
    }

    // product
    MLU_CHECK(cnmlCreateMultOp(&mlu_product_op_ptr_,
                               bottom[0]->mlu_tensor(),
                               power_blob_.mlu_tensor(),
                               top[0]->mlu_tensor()));
    break;
  }
}

template <typename Dtype>
void MLULRNLayer<Dtype>::Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top) {
  switch (this->layer_param_.lrn_param().norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
    MLU_CHECK(cnmlComputeLrnOpForward_V3(mlu_lrn_op_ptr_,
                                         bottom[0]->mutable_mlu_data(),
                                         top[0]->mutable_mlu_data(),
                                         Caffe::forward_param(), Caffe::queue()));
    break;
  case LRNParameter_NormRegion_WITHIN_CHANNEL:
    // add to narrow data with mult 0.01
    MLU_CHECK(cnmlComputeScaleOpForward_V3(mlu_scale1_op_ptr_,
                                           bottom[0]->mutable_mlu_data(),
                                           scale_blob_.mutable_mlu_data(),
                                           Caffe::forward_param(),
                                           Caffe::queue()));


    // square:use mult to impelment square
    MLU_CHECK(cnmlComputeMultOpForward_V3(mlu_square_op_ptr_,
                                          scale_blob_.mutable_mlu_data(),
                                          scale_blob_.mutable_mlu_data(),
                                          square_blob_.mutable_mlu_data(),
                                          Caffe::forward_param(),
                                          Caffe::queue()));
    // pool
    if (this->pre_pad_) {
      MLU_CHECK(cnmlComputeAddPadOpForward_V3(mlu_addpad_op_ptr_,
                                              square_blob_.mutable_mlu_data(),
                                              addpad_blob_.mutable_mlu_data(),
                                              Caffe::forward_param(),
                                              Caffe::queue()));
    }
    auto pool_input = mlu_addpad_op_ptr_? addpad_blob_.mutable_mlu_data() :
                                          square_blob_.mutable_mlu_data();
    MLU_CHECK(cnmlComputePoolOpForward_V3(mlu_pool_op_ptr_,
                                          pool_input,
                                          pool_blob_.mutable_mlu_data(),
                                          Caffe::forward_param(),
                                          Caffe::queue()));

    // power
    MLU_CHECK(cnmlComputeScaleOpForward_V3(mlu_scale_op_ptr_,
                                           pool_blob_.mutable_mlu_data(),
                                           (-this->beta_) == 1 ?
                                           power_blob_.mutable_mlu_data() :
                                           temp_blob_.mutable_mlu_data(),
                                           Caffe::forward_param(),
                                           Caffe::queue()));

    if ((-this->beta_) != 1) {
      MLU_CHECK(cnmlComputePowerOpForward_V3(mlu_power_op_ptr_,
                                             temp_blob_.mutable_mlu_data(),
                                             power_blob_.mutable_mlu_data(),
                                             Caffe::forward_param(),
                                             Caffe::queue()));
    }
    // product
    MLU_CHECK(cnmlComputeMultOpForward_V3(mlu_product_op_ptr_,
                                          bottom[0]->mutable_mlu_data(),
                                          power_blob_.mutable_mlu_data(),
                                          top[0]->mutable_mlu_data(),
                                          Caffe::forward_param(),
                                          Caffe::queue()));
    break;
  }
}

template <typename Dtype>
MLULRNLayer<Dtype>::~MLULRNLayer() {
  MLUDestroyOp();
}

INSTANTIATE_CLASS(MLULRNLayer);

}  // namespace caffe
#endif  // USE_MLU
