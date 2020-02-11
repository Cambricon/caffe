/*
All modification made by Cambricon Corporation: © 2018--2019 Cambricon
Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2019, the respective contributors
All rights reserved.
For the list of contributors go to
https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md
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
#include <memory>
#include <string>
#include <vector>
#include "caffe/filler.hpp"
#include "caffe/layers/mlu_conv3d_layer.hpp"
#include "caffe/mlu/util.hpp"
#include "caffe/common.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

template <typename Dtype>
void MLUConvolution3DLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Convolution3DLayer<Dtype>::LayerSetUp(bottom, top);
  Convolution3DParameter conv3d_param =
      this->layer_param_.convolution3d_param();

  // const data
  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
  BaseDataType mlu_dtype = this->layer_param_.blobs_dtype_size() > 0 ?
    this->layer_param_.blobs_dtype(0).type(): DT_FLOAT16;

  if (conv3d_param.mean_value_size() || conv3d_param.has_std() ||
      conv3d_param.has_mean_file()) {
    convFirst_ = true;
    bottom[0]->set_mlu_type(DT_UINT8);
  }

  // weight
  vector<int> weight_shape(5);
  weight_shape[0] = this->num_output_;
  weight_shape[1] = this->channels_;
  weight_shape[2] = this->kernel_depth_;
  weight_shape[3] = this->kernel_size_;
  weight_shape[4] = this->kernel_size_;

  // blob_[0]
  vector<int> stride_value(5, 0);
  stride_value[1] = 1;
  this->blobs_[0].reset(
      new Blob<Dtype>(weight_shape, cpu_dtype, mlu_dtype, CNML_FILTER, CNML_NCHW,
      (convFirst_ && bottom[0]->channels() == 3) ? &stride_value: nullptr));
  if (this->layer_param_.blobs_dtype_size() > 0 &&
      (this->layer_param_.blobs_dtype(0).position_size() ||
       this->layer_param_.blobs_dtype(0).scale_size())) {
    blobs_position_ = this->layer_param_.blobs_dtype(0).position(0);
    blobs_scale_ = this->layer_param_.blobs_dtype(0).scale(0);
    this->blobs_[0]->set_mlu_position(blobs_position_);
    this->blobs_[0]->set_mlu_scale(blobs_scale_);
  }
  shared_ptr<Filler<Dtype>> weight_filler(
      GetFiller<Dtype>(conv3d_param.weight_filler()));
  weight_filler->Fill(this->blobs_[0].get());

  // bias
  vector<int> bias_shape(5, 1);
  bias_shape[1] = this->num_output_;
  BaseDataType const_dtype = this->layer_param_.has_top_mlu_dtype() ?
         this->layer_param_.top_mlu_dtype(): DT_FLOAT16;

  if (this->bias_term_) {
    this->blobs_[1].reset(
        new Blob<Dtype>(bias_shape, cpu_dtype, const_dtype, CNML_CONST));
    shared_ptr<Filler<Dtype>> bias_filler(
        GetFiller<Dtype>(conv3d_param.bias_filler()));
    bias_filler->Fill(this->blobs_[1].get());
  }
  if (convFirst_ && bottom[0]->channels() == 3) {
    bottom[0]->set_dim_strides(stride_value);
    this->mean_.set_dim_strides(stride_value);
  }

}

template <typename Dtype>
void MLUConvolution3DLayer<Dtype>::Reshape_tensor(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
  BaseDataType mlu_dtype = this->layer_param_.has_top_mlu_dtype() ?
             this->layer_param_.top_mlu_dtype(): DT_FLOAT16;
  Convolution3DLayer<Dtype>::Reshape(bottom, top);
  top[0]->Reshape(top[0]->shape(), cpu_dtype, mlu_dtype, CNML_TENSOR);
  vector<int> mean_shape(bottom[0]->shape());
  if (convFirst_) {
    this->mean_.Reshape(mean_shape, cpu_dtype, mlu_dtype, CNML_CONST);
  }
}

template <typename Dtype>
void MLUConvolution3DLayer<Dtype>::MLUCreateOpBindData(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (Caffe::core_number() != 4 &&
      Caffe::core_number() != 16 ){
    LOG(FATAL) << "core number is wrong, it should be equal to 4 or 16. "
               << "but now it is " << Caffe::core_number();
  }
  Convolution3DParameter conv3d_param =
          this->layer_param_.convolution3d_param();
  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
  BaseDataType mlu_dtype = this->layer_param_.has_top_mlu_dtype() ?
                 this->layer_param_.top_mlu_dtype(): DT_FLOAT16;
  vector<int> stride_value(5, 0);
  stride_value[1] = 1;

  if (conv3d_param.has_std()) {
    vector<int> std_shape(5, 1);
    std_shape[1] = bottom[0]->channels();
    if (bottom[0]->channels() == 3)
            std_.set_dim_strides(stride_value);
    std_.Reshape(std_shape, cpu_dtype, mlu_dtype, CNML_CONST);
    Dtype* std_ptr_ = std_.mutable_cpu_data();
    for (int i = 0; i < bottom[0]->channels(); i++) {
      std_ptr_[i] = 1 / this->layer_param().convolution3d_param().std();
    }
    MLU_CHECK(cnmlBindConstData_V2(std_.mlu_tensor(), std_.sync_data(), false));
  }

  cnmlConvMode_t conv_mode = CNML_CONV;

  if (this->layer_param_.bottom_mlu_dtype_size() > 0 &&(
      this->layer_param_.bottom_mlu_dtype(0).position_size() ||
       this->layer_param_.bottom_mlu_dtype(0).scale_size())) {
    bottom_position_ = this->layer_param_.bottom_mlu_dtype(0).position(0);
    bottom_scale_ = this->layer_param_.bottom_mlu_dtype(0).scale(0);
    bottom[0]->set_mlu_position(bottom_position_);
    bottom[0]->set_mlu_scale(bottom_scale_);
  }
  // createOp
  int dim_size = 5;
  int dilations[dim_size];
  for (int i = 0; i < dim_size; i++) {
    dilations[i] = 1;
  }
  int strides[5];
  strides[0] = strides[1] = 0;
  strides[2] = this->temporal_stride_;
  strides[3] = strides[4] = this->stride_;
  int paddings[5][2];
  paddings[0][0] = paddings[0][1] = 0;
  paddings[1][0] = paddings[1][1] = 0;
  paddings[2][0] = paddings[2][1] = this->temporal_pad_;
  paddings[3][0] = paddings[3][1] = this->pad_;
  paddings[4][0] = paddings[4][1] = this->pad_;

  MLU_CHECK(cnmlCreateNdConvParam(&param_ptr_, dim_size, dilations, strides, paddings));

  bool std = false, mean = false;
  if (conv3d_param.mean_value_size() > 0 || conv3d_param.has_mean_file()) {
    mean = true;
  }
  if (conv3d_param.has_std()) {
    std = true;
  }

  MLU_CHECK(cnmlCreateNdConvOp(
         &conv_op_, conv_mode, param_ptr_,
         bottom[0]->mlu_tensor(),
         top[0]->mlu_tensor(), this->blobs_[0]->mlu_tensor(),
         this->bias_term_ ? this->blobs_[1]->mlu_tensor() : nullptr,
         mean ? this->mean_.mlu_tensor() : nullptr,
         std ? std_.mlu_tensor(): nullptr));

  //bottom
  MLU_CHECK(cnmlSetOperationComputingLayout(conv_op_, CNML_NDHWC));
  cnmlQuantizedParam_t input_quant_param;
  MLU_CHECK(cnmlCreateQuantizedParam(&input_quant_param, bottom_position_, bottom_scale_, 0));
  MLU_CHECK(cnmlSetOperationComputingDataType(conv_op_, bottom[0]->mlu_tensor(),
          CNML_DATA_INT8, input_quant_param));
  //blobs
  cnmlQuantizedParam_t blobs_quant_param;
  MLU_CHECK(cnmlCreateQuantizedParam(&blobs_quant_param, blobs_position_, blobs_scale_, 0));
  MLU_CHECK(cnmlSetOperationComputingDataType(conv_op_, this->blobs_[0]->mlu_tensor(),
        CNML_DATA_INT8, blobs_quant_param));

  // BindConstData
  MLU_CHECK(cnmlBindConstData_V2(this->blobs_[0]->mlu_tensor(),
                                 this->blobs_[0]->sync_data(), false));
  if (this->bias_term_) {
    MLU_CHECK(cnmlBindConstData_V2(this->blobs_[1]->mlu_tensor(),
                                   this->blobs_[1]->sync_data(), false));
  }
  if (mean) {
    MLU_CHECK(cnmlBindConstData_V2(this->mean_.mlu_tensor(),
          this->mean_.sync_data(), false));
  }
}

template <typename Dtype>
void MLUConvolution3DLayer<Dtype>::MLUCompileOp() {
  MLU_CHECK(cnmlCompileBaseOp(conv_op_, Caffe::rt_core(), Caffe::core_number()));
}

template <typename Dtype>
void MLUConvolution3DLayer<Dtype>::Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                                               const vector<Blob<Dtype>*>& top) {
  MLU_CHECK(cnmlComputeNdConvOpForward_V2(conv_op_,
        bottom[0]->mlu_tensor(),
        bottom[0]->mutable_mlu_data(),
        top[0]->mlu_tensor(),
        top[0]->mutable_mlu_data(),
        Caffe::queue(),
        NULL));
}

template <typename Dtype>
void MLUConvolution3DLayer<Dtype>::fuse(MFusion<Dtype>* fuser) {
  fuser->fuse(conv_op_);
}

template <typename Dtype>
void MLUConvolution3DLayer<Dtype>::MLUDestroyOp() {
  if (conv_op_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&conv_op_));
    conv_op_ = nullptr;
  }
  if (param_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyNdConvParam(&param_ptr_));
    param_ptr_ = nullptr;
  }
}

INSTANTIATE_CLASS(MLUConvolution3DLayer);

}  // namespace caffe
#endif  // USE_MLU
