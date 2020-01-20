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

#include <algorithm>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include "caffe/filler.hpp"
#include "caffe/layers/mlu_inner_product_layer.hpp"
#include "caffe/mlu/fusion.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MLUInnerProductLayer<Dtype>::bindDataAndSetComputingDataType(
    shared_ptr<Blob<Dtype>> blob, cnmlBaseOp_t op, BaseDataType type) {
  MLU_CHECK(cnmlBindConstData_V2(blob->mlu_tensor(), blob->sync_data(), false));
  cnmlQuantizedParam_t param;
  if (blob->has_mlu_position()) {
    CNRT_CHECK(cnmlCreateQuantizedParam(&param, blob->mlu_position(),
          blob->mlu_scale(), 0));
    cnmlSetOperationComputingDataType(op, blob->mlu_tensor(),
        to_cnml_dtype(type), param);
    if (param != nullptr) {
      MLU_CHECK(cnmlDestroyQuantizedParam(&param));
      param = nullptr;
    }
  } else {
    LOG(FATAL) << "Quantized tensor should have position";
  }
}

template <typename Dtype>
void MLUInnerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  InnerProductLayer<Dtype>::LayerSetUp(bottom, top);
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  const int num_output = this->layer_param_.inner_product_param().num_output();
  CHECK(axis == 1 || (bottom[0]->channels() == 1 && num_output ==  1)) <<
     " MLU only support axis == 1";
  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
  vector<int> mlu_weight_shape;
  if (bottom[0]->shape().size() < 5) {
    mlu_weight_shape = bottom[0]->shape();
    mlu_weight_shape[0] = this->N_;
  } else {
    mlu_weight_shape.push_back(this->N_);
    mlu_weight_shape.push_back(bottom[0]->count(1));
    reshape_ = true;
  }
  this->blobs_[0].reset(new Blob<Dtype>(
      mlu_weight_shape, cpu_dtype, cpu_dtype, CNML_FILTER, CNML_NCHW));
  if (this->layer_param_.blobs_dtype_size() > 0 &&
      (this->layer_param_.blobs_dtype(0).position_size() ||
       this->layer_param_.blobs_dtype(0).scale_size())) {
    int pos_size = this->layer_param_.blobs_dtype(0).position_size();
    int scale_size = this->layer_param_.blobs_dtype(0).scale_size();
    vector<int> positions(pos_size);
    vector<float> scales(scale_size);
    for (int i = 0; i< pos_size; i++) {
      positions[i] = this->layer_param_.blobs_dtype(0).position(i);
    }
    for (int i = 0; i< scale_size; i++) {
      scales[i] = this->layer_param_.blobs_dtype(0).scale(i);
    }

    if (this->layer_param_.blobs_dtype(0).position_size()) {
      if (pos_size == 1)
        this->blobs_[0]->set_mlu_position(positions[0]);
      else
        this->blobs_[0]->set_mlu_positions(positions);
    }
    if (this->layer_param_.blobs_dtype(0).scale_size()) {
      if (scale_size == 1)
        this->blobs_[0]->set_mlu_scale(scales[0]);
      else
        this->blobs_[0]->set_mlu_scales(scales);
    }
  }

  // fill the weights
  shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
      this->layer_param_.inner_product_param().weight_filler()));
  weight_filler->Fill(this->blobs_[0].get());

  // If necessary, intiialize and fill the bias term
  BaseDataType const_dtype = this->layer_param_.has_top_mlu_dtype() ?
             this->layer_param_.top_mlu_dtype(): DT_FLOAT16;
  if (this->bias_term_) {
    vector<int> mlu_bias_shape(4, 1);
    mlu_bias_shape[1] = this->N_;
    this->blobs_[1].reset(new Blob<Dtype>(
        mlu_bias_shape, cpu_dtype, const_dtype, CNML_CONST));
    shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
        this->layer_param_.inner_product_param().bias_filler()));
    bias_filler->Fill(this->blobs_[1].get());
  }
}

template <typename Dtype>
void MLUInnerProductLayer<Dtype>::Reshape_tensor(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  InnerProductLayer<Dtype>::Reshape(bottom, top);
  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
  BaseDataType mlu_dtype = this->layer_param_.has_top_mlu_dtype() ?
             this->layer_param_.top_mlu_dtype(): DT_FLOAT16;
  if (reshape_) {
    vector<int> trans_shape;
    trans_shape.push_back(bottom[0]->shape()[0]);
    trans_shape.push_back(bottom[0]->shape()[4]);
    trans_shape.push_back(bottom[0]->shape()[1]);
    trans_shape.push_back(bottom[0]->shape()[2]);
    trans_shape.push_back(bottom[0]->shape()[3]);
    transpose.Reshape(trans_shape, cpu_dtype, mlu_dtype , CNML_TENSOR);
    vector<int> shape;
    shape.push_back(bottom[0]->num());
    shape.push_back(bottom[0]->count(1));
    reshape.Reshape(shape, cpu_dtype, mlu_dtype, CNML_TENSOR, CNML_NCHW);
  }
  top[0]->Reshape(top[0]->shape(), cpu_dtype, mlu_dtype, CNML_TENSOR);
}

template <typename Dtype>
void MLUInnerProductLayer<Dtype>::MLUDestroyOp() {
  if (mlp_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&mlp_op_ptr_));
    mlp_op_ptr_ = nullptr;
  }
  if (input_quant_param_ != nullptr) {
    MLU_CHECK(cnmlDestroyQuantizedParam(&input_quant_param_));
    input_quant_param_ = nullptr;
  }
  if (reshape_param_ != nullptr) {
    MLU_CHECK(cnmlDestroyReshapeOpParam(&reshape_param_));
    reshape_param_ = nullptr;
  }
  if (reshape_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&reshape_op_ptr_));
    reshape_op_ptr_ = nullptr;
  }
  if (transpose_pro_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&transpose_pro_op_ptr_));
    transpose_pro_op_ptr_ = nullptr;
  }
  if (transpose_op_param_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyNdTransposeOpParam(&transpose_op_param_ptr_));
    transpose_op_param_ptr_ = nullptr;
  }
}

template <typename Dtype>
void MLUInnerProductLayer<Dtype>::MLUCompileOp() {
  if (reshape_) {
    MLU_CHECK(cnmlCompileBaseOp(transpose_pro_op_ptr_,
                                Caffe::rt_core(),
                                Caffe::core_number()));
    MLU_CHECK(cnmlCompileBaseOp(reshape_op_ptr_,
                                Caffe::rt_core(),
                                Caffe::core_number()));
  }
  MLU_CHECK(cnmlCompileBaseOp(mlp_op_ptr_,
                              Caffe::rt_core(),
                              Caffe::core_number()));
}

template <typename Dtype>
void MLUInnerProductLayer<Dtype>::MLUCreateOpBindData(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  if (reshape_) {
    int dim_size = bottom[0]->num_axes();
    int order[dim_size];
    order[0] = 0;
    order[1] = 4;
    order[2] = 1;
    order[3] = 2;
    order[4] = 3;
    MLU_CHECK(cnmlCreateNdTransposeOpParam(&transpose_op_param_ptr_,
          order,
          dim_size));
    /* TransposeProOp */
    MLU_CHECK(cnmlCreateNdTransposeProOp(&transpose_pro_op_ptr_,
          bottom[0]->mlu_tensor(),
          transpose.mlu_tensor(),
          transpose_op_param_ptr_));

    int length = 4;
    int dim[length];
    dim[0] = bottom[0]->num();
    dim[3] = bottom[0]->count(1);
    dim[1] = dim[2] = 1;
    MLU_CHECK(cnmlCreateNdReshapeOpParam(&reshape_param_,
          dim,
          length));
    MLU_CHECK(cnmlCreateReshapeOp(&reshape_op_ptr_,
          reshape_param_,
          transpose.mlu_tensor(),
          reshape.mlu_tensor()));
  }
  if (this->transpose_) {
    vector<Dtype> data(this->blobs_[0]->count());
    for (int i = 0; i < this->N_; i++) {
      for (int j = 0; j < this->K_; j++) {
        data[i * this->K_ + j] = this->blobs_[0]->cpu_data()[j * this->N_ + i];
      }
    }
    caffe_copy(this->blobs_[0]->count(),
               data.data(),
               this->blobs_[0]->mutable_cpu_data());
  }

  MLU_CHECK(cnmlCreateMlpOp(&mlp_op_ptr_,
                           reshape_? reshape.mlu_tensor():
                           bottom[0]->mlu_tensor(),
                           top[0]->mlu_tensor(),
                           this->blobs_[0]->mlu_tensor(),
                           this->bias_term_ ?
                           this->blobs_[1]->mlu_tensor():nullptr));

  if (this->layer_param_.bottom_mlu_dtype_size() > 0) {
    bottom[0]->set_mlu_position(
        this->layer_param_.bottom_mlu_dtype(0).position(0));
    bottom[0]->set_mlu_scale(
        this->layer_param_.bottom_mlu_dtype(0).scale(0));
    MLU_CHECK(cnmlCreateQuantizedParam(&input_quant_param_,
                  this->layer_param_.bottom_mlu_dtype(0).position(0),
                  this->layer_param_.bottom_mlu_dtype(0).scale(0),
                  0.0));
    BaseDataType blobs_dtype = this->layer_param_.bottom_mlu_dtype(0).has_type() ?
      this->layer_param_.bottom_mlu_dtype(0).type() :
      this->layer_param_.blobs_dtype(0).type();
    MLU_CHECK(cnmlSetOperationComputingDataType(mlp_op_ptr_,
                  reshape_? reshape.mlu_tensor(): bottom[0]->mlu_tensor(),
                  to_cnml_dtype(blobs_dtype),
                  input_quant_param_));
  }
  BaseDataType mlu_dtype = this->layer_param_.blobs_dtype_size() > 0 ?
      this->layer_param_.blobs_dtype(0).type() : DT_INT8;
  bindDataAndSetComputingDataType(this->blobs_[0], mlp_op_ptr_, mlu_dtype);
  if (this->bias_term_) {
    MLU_CHECK(cnmlBindConstData_V2(this->blobs_[1]->mlu_tensor(),
                                  this->blobs_[1]->sync_data(),
                                  false));
  }
}

template <typename Dtype>
MLUInnerProductLayer<Dtype>::~MLUInnerProductLayer() {
  MLUDestroyOp();
}

template <typename Dtype>
void MLUInnerProductLayer<Dtype>::Forward_mlu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  if (reshape_) {
    MLU_CHECK(cnmlComputeNdTransposeProOpForward(transpose_pro_op_ptr_,
                                    bottom[0]->mutable_mlu_data(),
                                    transpose.mutable_mlu_data(),
                                    Caffe::forward_param(), Caffe::queue()));

    MLU_CHECK(cnmlComputeReshapeOpForward_V3(reshape_op_ptr_,
                                    transpose.mutable_mlu_data(),
                                    reshape.mutable_mlu_data(),
                                    Caffe::forward_param(), Caffe::queue()));
  }
  MLU_CHECK(cnmlComputeMlpOpForward_V3(mlp_op_ptr_,
                                   reshape_ ? reshape.mutable_mlu_data():
                                   bottom[0]->mutable_mlu_data(),
                                   top[0]->mutable_mlu_data(),
                                   Caffe::forward_param(), Caffe::queue()));
}

INSTANTIATE_CLASS(MLUInnerProductLayer);

}  // namespace caffe

#endif  // USE_MLU
