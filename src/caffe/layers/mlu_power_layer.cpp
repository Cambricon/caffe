/*
All modification made by Cambricon Corporation: © 2018 Cambricon Corporation
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
#include <vector>
#include "caffe/layers/mlu_power_layer.hpp"
namespace caffe {

template <typename Dtype>
void MLUPowerLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  PowerLayer<Dtype>::LayerSetUp(bottom, top);
  this->alpha_ = new Blob<Dtype>();
  this->beta_ = new Blob<Dtype>();
  this->temp_ = new Blob<Dtype>();
}

template <typename Dtype>
void MLUPowerLayer<Dtype>::Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32:DT_DOUBLE;
  BaseDataType mlu_dtype = DT_FLOAT16;
  top[0]->Reshape(bottom[0]->shape(), cpu_dtype, mlu_dtype, CNML_TENSOR);
  this->temp_->Reshape(bottom[0]->shape(), cpu_dtype, mlu_dtype, CNML_TENSOR);
}

template <typename Dtype>
void MLUPowerLayer<Dtype>::fuse(MFusion<Dtype>* fuser) {
  fuser->fuse(scale_op_ptr_);
  if (this->power_ != 1)
    fuser->fuse(power_op_ptr_);
}

template <typename Dtype>
MLUPowerLayer<Dtype>::~MLUPowerLayer() {
  MLUDestroyOp();
}

template <typename Dtype>
void MLUPowerLayer<Dtype>::MLUDestroyOp() {
  if (power_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&power_op_ptr_));
    power_op_ptr_ = nullptr;
  }
  if (scale_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&scale_op_ptr_));
    scale_op_ptr_ = nullptr;
  }
}


template <typename Dtype>
void MLUPowerLayer<Dtype>::MLUCreateOpBindData(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32:DT_DOUBLE;
  vector <int> alpha_shape(4, 1);
  vector <int> beta_shape(4, 1);
  alpha_shape[1] = beta_shape[1] = bottom[0]->channels();
  this->alpha_->Reshape(alpha_shape,
                        cpu_dtype, DT_FLOAT16, CNML_CONST);
  for (int i = 0; i < this->alpha_->count(); i++) {
    this->alpha_->mutable_cpu_data()[i] = this->scale_;
  }
  this->beta_->Reshape(beta_shape,
  cpu_dtype, DT_FLOAT16, CNML_CONST);
  for (int i = 0; i < this->beta_->count(); i++) {
    this->beta_->mutable_cpu_data()[i] = this->shift_;
  }

/**
 *  Bottom -- ( * scale + shift )  --  [( ^ power )] --> Top
 *            <    Scale  Op    >       <Power  OP>
 *  NOTICE: Bottom(*scale + shift) should > 0.
 */

  MLU_CHECK(cnmlCreateScaleOp(&scale_op_ptr_,
                              bottom[0]->mlu_tensor(),
                              this->power_ == 1 ? top[0]->mlu_tensor()
                              : this->temp_->mlu_tensor(),
                              this->alpha_->mlu_tensor(),
                              this->beta_->mlu_tensor()));
  MLU_CHECK(cnmlBindConstData(this->alpha_->mlu_tensor(),
                              this->alpha_->cpu_tensor(),
                              this->alpha_->mutable_cpu_data()));
  MLU_CHECK(cnmlBindConstData(this->beta_->mlu_tensor(),
                              this->beta_->cpu_tensor(),
                              this->beta_->mutable_cpu_data()));
  if (this->power_ != 1) {
    MLU_CHECK(cnmlCreatePowerOp(&power_op_ptr_,
                                this->temp_->mlu_tensor(),
                                top[0]->mlu_tensor(),
                                this->power_));
  }
}

template <typename Dtype>
void MLUPowerLayer<Dtype>::MLUCompileOp() {
  MLU_CHECK(cnmlCompileBaseOp(scale_op_ptr_,
                              Caffe::rt_core(),
                              Caffe::model_parallel()));
  if (this->power_ != 1)
    MLU_CHECK(cnmlCompileBaseOp(power_op_ptr_,
                                Caffe::rt_core(),
                                Caffe::model_parallel()));
}

template <typename Dtype>
void MLUPowerLayer<Dtype>::Forward_mlu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  MLU_CHECK(cnmlComputeScaleOpForward_V3(scale_op_ptr_,
                                      bottom[0]->mutable_mlu_data(),
                                      this->power_ == 1 ?
                                      top[0]->mutable_mlu_data() :
                                      this->temp_->mutable_mlu_data(),
                                      Caffe::forward_param(),
                                      Caffe::queue()));
  if (this->power_ != 1) {
    MLU_CHECK(cnmlComputePowerOpForward_V3(power_op_ptr_,
                                        this->temp_->mutable_mlu_data(),
                                        top[0]->mutable_mlu_data(),
                                        Caffe::forward_param(),
                                        Caffe::queue()));
  }
}
INSTANTIATE_CLASS(MLUPowerLayer);
}  //  namespace caffe
#endif  //  USE_MLU
