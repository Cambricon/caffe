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
#include "caffe/filler.hpp"
#include "caffe/layers/mlu_dropout_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MLUDropoutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
  DropoutLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void MLUDropoutLayer<Dtype>::Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
  BaseDataType cpu_dtype = sizeof(Dtype) ==4 ? DT_FLOAT32: DT_DOUBLE;
  BaseDataType mlu_dtype = bottom[0]->mlu_type();

  vector<int> true_dropout_shape(bottom[0]->num_axes(), 1);
  true_dropout_shape[1] = bottom[0]->shape()[1];

  this->weight.Reshape(true_dropout_shape, cpu_dtype, mlu_dtype, CNML_CONST);
  if (this->scale_train_) {
    caffe_set(this->weight.count(),
              Dtype(1),
              this->weight.mutable_cpu_data());
  } else {
    caffe_set(this->weight.count(),
              Dtype(1. / this->scale_),
              this->weight.mutable_cpu_data());
  }

  zero_bias_data_.Reshape(true_dropout_shape, cpu_dtype, mlu_dtype, CNML_CONST);
  caffe_set(this->zero_bias_data_.count(),
            Dtype(0),
            this->zero_bias_data_.mutable_cpu_data());

  top[0]->Reshape(bottom[0]->shape(), cpu_dtype, mlu_dtype, CNML_TENSOR);
}

template <typename Dtype>
void MLUDropoutLayer<Dtype>::MLUDestroyOp() {
  if (mlu_dropout_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&mlu_dropout_op_ptr_));
    mlu_dropout_op_ptr_ = nullptr;
  }
}

template <typename Dtype>
void MLUDropoutLayer<Dtype>::MLUCompileOp() {
  MLU_CHECK(cnmlCompileBaseOp(mlu_dropout_op_ptr_,
                              Caffe::rt_core(),
                              Caffe::core_number()));
}

template <typename Dtype>
void MLUDropoutLayer<Dtype>::fuse(MFusion<Dtype>* fuser) {
  fuser->fuse(mlu_dropout_op_ptr_);
}

template <typename Dtype>
void MLUDropoutLayer<Dtype>::MLUCreateOpBindData(const vector<Blob<Dtype>*>& bottom,
                                                 const vector<Blob<Dtype>*>& top) {

  MLU_CHECK(cnmlBindConstData_V2(weight.mlu_tensor(),
                              weight.sync_data(),
                              false));

  MLU_CHECK(cnmlBindConstData_V2(zero_bias_data_.mlu_tensor(),
                              zero_bias_data_.sync_data(),
                              false));

  MLU_CHECK(cnmlCreateNdScaleOp(&mlu_dropout_op_ptr_,
                              bottom[0]->mlu_shape().size() - 1,
                              bottom[0]->mlu_tensor(),
                              top[0]->mlu_tensor(),
                              weight.mlu_tensor(),
                              zero_bias_data_.mlu_tensor()));
}

template <typename Dtype>
void MLUDropoutLayer<Dtype>::Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  MLU_CHECK(cnmlComputeNdScaleOpForward(mlu_dropout_op_ptr_,
                                      NULL,
                                      bottom[0]->mutable_mlu_data(),
                                      NULL,
                                      top[0]->mutable_mlu_data(),
                                      Caffe::queue(),
                                      NULL));
}

template <typename Dtype>
MLUDropoutLayer<Dtype>::~MLUDropoutLayer() {
  MLUDestroyOp();
}

INSTANTIATE_CLASS(MLUDropoutLayer);

}  // namespace caffe
#endif  // USE_MLU
