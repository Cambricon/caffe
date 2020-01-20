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
#include <memory>
#include <string>
#include <vector>
#include "caffe/filler.hpp"
#include "caffe/layers/mlu_rnn_layer.hpp"

namespace caffe {

template <typename Dtype>
void MLURNNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "bottom[0] must have at least 2 axes -- (#timesteps, #streams, ...)";
  this->T_ = bottom[0]->shape(0);
  this->N_ = bottom[0]->shape(1);
  LOG(INFO) << "Initializing recurrent layer: assuming input batch contains "
            << this->T_ << " timesteps of "
            << this->N_ << " independent streams.";

  CHECK_EQ(bottom[1]->num_axes(), 2)
      << "bottom[1] must have exactly 2 axes -- (#timesteps, #streams)";
  CHECK_EQ(this->T_, bottom[1]->shape(0));
  CHECK_EQ(this->N_, bottom[1]->shape(1));

  // If expose_hidden is set, we take as input and produce as output
  // the hidden state blobs at the first and last timesteps.
  this->expose_hidden_ = this->layer_param_.recurrent_param().expose_hidden();

  // Get (recurrent) input/output names.
  vector<string> output_names;
  this->OutputBlobNames(&output_names);
  vector<string> recur_input_names;
  this->RecurrentInputBlobNames(&recur_input_names);
  vector<string> recur_output_names;
  this->RecurrentOutputBlobNames(&recur_output_names);
  const int num_recur_blobs = recur_input_names.size();
  CHECK_EQ(num_recur_blobs, recur_output_names.size());

  // If provided, bottom[2] is a static input to the recurrent net.
  const int num_hidden_exposed = this->expose_hidden_ * num_recur_blobs;
  this->static_input_ = (bottom.size() > 2 + num_hidden_exposed);
  if (this->static_input_) {
    CHECK_GE(bottom[2]->num_axes(), 1);
    CHECK_EQ(this->N_, bottom[2]->shape(0));
  }
  // CHECK_EQ(top.size() - this->expose_hidden_*2, 1);
  // num_output
  num_output_ = this->layer_param_.recurrent_param().num_output();

  h_conted_reshape_op_.resize(this->T_, nullptr);
  h_conted_reshape_param_ptr_.resize(this->T_, nullptr);

  cyclemult_op_.resize(this->T_, nullptr);
  add_op_.resize(this->T_, nullptr);
  add_1_op_.resize(this->T_, nullptr);
  h_mlp_op_.resize(this->T_, nullptr);
  h_conted_mlp_op_.resize(this->T_, nullptr);
  w_ho_tanh_op_.resize(this->T_, nullptr);
  h_neuron_input_tanh_op_.resize(this->T_, nullptr);


  if (this->expose_hidden_) {
    h_reshape_op_.resize(this->T_ - 1, nullptr);
    h_reshape_param_ptr_.resize(this->T_ - 1, nullptr);
  } else {
    h_reshape_op_.resize(this->T_ - 1, nullptr);
    h_reshape_param_ptr_.resize(this->T_ - 1, nullptr);
  }

  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
  BaseDataType mlu_dtype = this->layer_param_.blobs_dtype_size() > 0 ?
      this->layer_param_.blobs_dtype(0).type() : DT_FLOAT16;

  vector<int> input_shape1 = bottom[0]->shape();
  if (input_shape1.size() > 2)
    channel = input_shape1[2];
  if (input_shape1.size() > 3)
    height = input_shape1[3];
  if (input_shape1.size() > 4)
    width = input_shape1[4];

  // set weights and bias
  this->blobs_.clear();
  if (this->static_input_)
    this->blobs_.resize(6);
  else
    this->blobs_.resize(5);

  // w_xh
  vector<int> w_xh_shape(4, 1);
  w_xh_shape[0] = num_output_;
  w_xh_shape[1] = channel;
  w_xh_shape[2] = height;
  w_xh_shape[3] = width;
  this->blobs_[0].reset(new Blob<Dtype>(w_xh_shape,
                                       cpu_dtype,
                                       mlu_dtype,
                                       CNML_FILTER));
  // b_h
  BaseDataType mlu_dtype1 = bottom[0]->mlu_type();
  vector<int> b_h_shape(4, 1);
  b_h_shape[1] = num_output_;
  this->blobs_[1].reset(new Blob<Dtype>(b_h_shape,
                                       cpu_dtype,
                                       mlu_dtype1,
                                       CNML_CONST));
  if (this->static_input_) {
    // w_xh_static
    vector<int> w_xh_static_shape = bottom[2]->shape();
    w_xh_static_shape[0] = num_output_;
    this->blobs_[2].reset(new Blob<Dtype>(w_xh_static_shape,
                                        cpu_dtype,
                                        mlu_dtype,
                                        CNML_FILTER));
    // w_hh
    vector<int> w_hh_shape(4, 1);
    w_hh_shape[0] = num_output_;
    w_hh_shape[1] = num_output_;
    this->blobs_[3].reset(new Blob<Dtype>(w_hh_shape,
                                        cpu_dtype,
                                        mlu_dtype,
                                        CNML_FILTER));
    // w_ho
    this->blobs_[4].reset(new Blob<Dtype>(w_hh_shape,
                                        cpu_dtype,
                                        mlu_dtype,
                                        CNML_FILTER));
    // b_o
    this->blobs_[5].reset(new Blob<Dtype>(b_h_shape,
                                        cpu_dtype,
                                        mlu_dtype1,
                                        CNML_CONST));
  } else {
    // w_hh
    vector<int> w_hh_shape(4, 1);
    w_hh_shape[0] = num_output_;
    w_hh_shape[1] = num_output_;
    this->blobs_[2].reset(new Blob<Dtype>(w_hh_shape,
                                        cpu_dtype,
                                        mlu_dtype,
                                        CNML_FILTER));
    // w_ho
    this->blobs_[3].reset(new Blob<Dtype>(w_hh_shape,
                                        cpu_dtype,
                                        mlu_dtype,
                                        CNML_FILTER));
    // b_o
    this->blobs_[4].reset(new Blob<Dtype>(b_h_shape,
                                        cpu_dtype,
                                        mlu_dtype1,
                                        CNML_CONST));
  }
  shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
      this->layer_param_.recurrent_param().weight_filler()));
  weight_filler->Fill(this->blobs_[0].get());
  shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
      this->layer_param_.recurrent_param().bias_filler()));
  bias_filler->Fill(this->blobs_[1].get());
  if (this->static_input_) {
    weight_filler->Fill(this->blobs_[2].get());
    weight_filler->Fill(this->blobs_[3].get());
    weight_filler->Fill(this->blobs_[4].get());
    bias_filler->Fill(this->blobs_[5].get());
  } else {
    weight_filler->Fill(this->blobs_[2].get());
    weight_filler->Fill(this->blobs_[3].get());
    bias_filler->Fill(this->blobs_[4].get());
  }
  // int8 position
  if (this->static_input_) {
    if (this->layer_param_.blobs_dtype_size() > 3  &&
      (this->layer_param_.blobs_dtype(0).position_size() ||
       this->layer_param_.blobs_dtype(0).scale_size()) &&
      (this->layer_param_.blobs_dtype(1).position_size() ||
       this->layer_param_.blobs_dtype(1).scale_size()) &&
      (this->layer_param_.blobs_dtype(2).position_size() ||
       this->layer_param_.blobs_dtype(2).scale_size()) &&
      (this->layer_param_.blobs_dtype(3).position_size() ||
       this->layer_param_.blobs_dtype(3).scale_size())) {
      if (this->layer_param_.blobs_dtype(0).position_size() &&
          this->layer_param_.blobs_dtype(1).position_size() &&
          this->layer_param_.blobs_dtype(2).position_size() &&
          this->layer_param_.blobs_dtype(3).position_size()) {
        this->blobs_[0]->set_mlu_position(
            this->layer_param_.blobs_dtype(0).position(0));
        this->blobs_[2]->set_mlu_position(
            this->layer_param_.blobs_dtype(1).position(0));
        this->blobs_[3]->set_mlu_position(
            this->layer_param_.blobs_dtype(2).position(0));
        this->blobs_[4]->set_mlu_position(
            this->layer_param_.blobs_dtype(3).position(0));
      }
      if (this->layer_param_.blobs_dtype(0).scale_size() &&
          this->layer_param_.blobs_dtype(1).scale_size() &&
          this->layer_param_.blobs_dtype(2).scale_size() &&
          this->layer_param_.blobs_dtype(3).scale_size()) {
            this->blobs_[0]->set_mlu_scale(
                this->layer_param_.blobs_dtype(0).scale(0));
            this->blobs_[2]->set_mlu_scale(
                this->layer_param_.blobs_dtype(1).scale(0));
            this->blobs_[3]->set_mlu_scale(
                this->layer_param_.blobs_dtype(2).scale(0));
            this->blobs_[4]->set_mlu_scale(
                this->layer_param_.blobs_dtype(3).scale(0));
      }
    }
  } else {
    if (this->layer_param_.blobs_dtype_size() > 2 &&
      (this->layer_param_.blobs_dtype(0).position_size() ||
       this->layer_param_.blobs_dtype(0).scale_size()) &&
      (this->layer_param_.blobs_dtype(2).position_size() ||
       this->layer_param_.blobs_dtype(2).scale_size()) &&
      (this->layer_param_.blobs_dtype(3).position_size() ||
       this->layer_param_.blobs_dtype(3).scale_size())) {
      if (this->layer_param_.blobs_dtype(0).position_size() &&
          this->layer_param_.blobs_dtype(2).position_size() &&
          this->layer_param_.blobs_dtype(3).position_size()) {
        this->blobs_[0]->set_mlu_position(
            this->layer_param_.blobs_dtype(0).position(0));
        this->blobs_[2]->set_mlu_position(
            this->layer_param_.blobs_dtype(1).position(0));
        this->blobs_[3]->set_mlu_position(
            this->layer_param_.blobs_dtype(2).position(0));
      }
      if (this->layer_param_.blobs_dtype(0).scale_size() &&
          this->layer_param_.blobs_dtype(2).scale_size() &&
          this->layer_param_.blobs_dtype(3).scale_size()) {
            this->blobs_[0]->set_mlu_scale(
                this->layer_param_.blobs_dtype(0).scale(0));
            this->blobs_[2]->set_mlu_scale(
                this->layer_param_.blobs_dtype(1).scale(0));
            this->blobs_[3]->set_mlu_scale(
                this->layer_param_.blobs_dtype(2).scale(0));
      }
    }
  }
}

template <typename Dtype>
void MLURNNLayer<Dtype>::Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
  BaseDataType mlu_dtype = bottom[0]->mlu_type();

  // bottom0 reshape
  vector<int> bottom_shape = {this->T_ * this->N_, channel, height, width};
  bottom0_reshape_blob_.reset(new Blob<Dtype>(
                   bottom_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));

  if (this->layer_param_.bottom_mlu_dtype_size() > 0 &&
      (this->layer_param_.bottom_mlu_dtype(0).position_size() ||
       this->layer_param_.bottom_mlu_dtype(0).scale_size())) {
    if (this->layer_param_.bottom_mlu_dtype(0).position_size()) {
      bottom0_reshape_blob_->set_mlu_position(
          this->layer_param_.bottom_mlu_dtype(0).position(0));
    }
    if (this->layer_param_.bottom_mlu_dtype(0).scale_size()) {
      bottom0_reshape_blob_->set_mlu_scale(
          this->layer_param_.bottom_mlu_dtype(0).scale(0));
    }
  }

  // bottom1 reshape
  vector<int> bottom_shape1 = {this->T_, this->N_, 1, 1};
  bottom1_reshape_blob_.reset(new Blob<Dtype>(
                   bottom_shape1, cpu_dtype, mlu_dtype, CNML_TENSOR));

  // bottom2 reshape
  if (this->static_input_) {
    vector<int> static_shape(4, 1);
    static_shape[0] = this->N_;
    static_shape[1] = bottom[2]->shape(1);
    bottom2_reshape_blob_.reset(new Blob<Dtype>(
                   static_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));
    if (this->layer_param_.bottom_mlu_dtype_size() > 1 &&
        (this->layer_param_.bottom_mlu_dtype(1).position_size() ||
         this->layer_param_.bottom_mlu_dtype(1).scale_size())) {
      if (this->layer_param_.bottom_mlu_dtype(1).position_size()) {
        bottom2_reshape_blob_->set_mlu_position(
            this->layer_param_.bottom_mlu_dtype(1).position(0));
      }
      if (this->layer_param_.bottom_mlu_dtype(1).scale_size()) {
        bottom2_reshape_blob_->set_mlu_scale(
            this->layer_param_.bottom_mlu_dtype(1).scale(0));
      }
    }
  }

  // bottom3 reshape
  if (this->expose_hidden_) {
    vector<int> bottom_hidden_shape(4, 1);
    bottom_hidden_shape[1] = this->N_;
    bottom_hidden_shape[2] = num_output_;
    bottom3_reshape_blob_.reset(new Blob<Dtype>(
                   bottom_hidden_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));
  }

  // top[0]
  vector<int> top_shape(3, 1);
  top_shape[0] =  this->T_;
  top_shape[1] = this->N_;
  top_shape[2] = num_output_;
  top[0]->Reshape(top_shape, cpu_dtype, mlu_dtype, CNML_TENSOR);

  // top[1]
  if (this->expose_hidden_) {
    vector<int> hidden_shape(3, 1);
    hidden_shape[1] = this->N_;
    hidden_shape[2] = num_output_;
    top[1]->Reshape(hidden_shape, cpu_dtype, mlu_dtype, CNML_TENSOR);
  }

  // bak_weight1_blobs_
  BaseDataType bak_mlu_dtype = this->layer_param_.blobs_dtype_size() > 0 ?
      this->layer_param_.blobs_dtype(0).type() : DT_FLOAT16;
  vector<int> w_hh_shape(4, 1);
  w_hh_shape[0] = num_output_;
  w_hh_shape[1] = num_output_;
  bak_weight1_blobs_.resize(this->T_);
  for (int i = 0; i < this->T_; i++) {
    bak_weight1_blobs_[i].reset(new Blob<Dtype>(
                w_hh_shape, cpu_dtype, bak_mlu_dtype, CNML_FILTER));
  }
  // bak_weight2_blobs_
  bak_weight2_blobs_.resize(this->T_);
  for (int i = 0; i < this->T_; i++) {
    bak_weight2_blobs_[i].reset(new Blob<Dtype>(
                w_hh_shape, cpu_dtype, bak_mlu_dtype, CNML_FILTER));
  }
  // bak_beta_blobs_
  vector<int> b_h_shape(4, 1);
  b_h_shape[1] = num_output_;
  BaseDataType mlu_dtype1 = bottom[0]->mlu_type();
  bak_beta_blobs_.resize(this->T_);
  for (int i = 0; i < this->T_; i++) {
    bak_beta_blobs_[i].reset(new Blob<Dtype>(
                b_h_shape, cpu_dtype, mlu_dtype1, CNML_CONST));
  }


  // w_xh_blob MLP
  vector<int> w_xh_shape(4, 1);
  w_xh_shape[0] = this->T_ * this->N_;
  w_xh_shape[1] = num_output_;
  w_xh_x_blob_.reset(new Blob<Dtype>(
                   w_xh_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));

  // w_xh_x_blobs_  Slice
  w_xh_x_blobs_.clear();
  w_xh_x_blobs_.resize(this->T_);
  vector<int> w_xh_x_shape(4, 1);
  w_xh_x_shape[0] = this->N_;
  w_xh_x_shape[1] = num_output_;
  for (int i = 0; i < this->T_; i++) {
    w_xh_x_blobs_[i].reset(new Blob<Dtype>(
                   w_xh_x_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));
  }

  // h_neuron_input_blobs_ Eltwise
  h_neuron_input_blobs_.clear();
  h_neuron_input_blobs_.resize(this->T_);
  for (int i = 0; i < this->T_; i++) {
    h_neuron_input_blobs_[i].reset(new Blob<Dtype>(
                   w_xh_x_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));
  }

  // temp_input_blobs_ Eltwise,  if static_input_ is true
  temp_input_blobs_.clear();
  temp_input_blobs_.resize(this->T_);
  for (int i = 0; i < this->T_; i++) {
    temp_input_blobs_[i].reset(new Blob<Dtype>(
                    w_xh_x_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));
  }

  // w_xh_x_static MLP
  if (this->static_input_) {
    w_xh_x_static_blob_.reset(new Blob<Dtype>(
                    w_xh_x_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));
  }

  // cont_blobs Slice
  cont_blobs_.clear();
  cont_blobs_.resize(this->T_);
  vector<int> cont_shape(4, 1);
  cont_shape[1] = this->N_;
  for (int i = 0; i < this->T_; i++) {
    cont_blobs_[i].reset(new Blob<Dtype>(
                   cont_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));
  }

  // h_cont_blobs  Scale
  h_conted_blobs_.clear();
  h_conted_blobs_.resize(this->T_);
  vector<int> h_cont_shape(4, 1);
  h_cont_shape[1] = this->N_;
  h_cont_shape[2] = num_output_;
  for (int i = 0; i < this->T_; i++) {
    h_conted_blobs_[i].reset(new Blob<Dtype>(
                      h_cont_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));
  }

  // w_hh_h_blob MLP
  w_hh_h_blobs_.clear();
  w_hh_h_blobs_.resize(this->T_);
  vector<int> w_hh_h_shape(4, 1);
  w_hh_h_shape[0] = this->N_;
  w_hh_h_shape[1] = num_output_;
  for (int i = 0; i < this->T_; i++) {
    w_hh_h_blobs_[i].reset(new Blob<Dtype>(
                     w_hh_h_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));
  }

  // h_blobs_ Tanh
  h_blobs_.clear();
  h_blobs_.resize(this->T_ + 1);
  h_blobs_[0].reset(new Blob<Dtype>(
                    h_cont_shape, cpu_dtype, mlu_dtype, CNML_CONST));
  for (int i = 1; i < this->T_ + 1; i++) {
    h_blobs_[i].reset(new Blob<Dtype>(
                      w_xh_x_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));
  }
  if (this->static_input_) {
    for (int i = 0; i < this->T_; i++) {
      if (this->layer_param_.bottom_mlu_dtype_size() > 3 + 2 * i &&
          (this->layer_param_.bottom_mlu_dtype(3 + 2 * i).position_size() ||
           this->layer_param_.bottom_mlu_dtype(3 + 2 * i).scale_size())) {
        if (this->layer_param_.bottom_mlu_dtype(3 + 2 * i).position_size()) {
          h_blobs_[i + 1]->set_mlu_position(
              this->layer_param_.bottom_mlu_dtype(3 + 2 * i).position(0));
        }
        if (this->layer_param_.bottom_mlu_dtype(3 + 2 * i).scale_size()) {
          h_blobs_[i + 1]->set_mlu_scale(
              this->layer_param_.bottom_mlu_dtype(3 + 2 * i).scale(0));
        }
      }
    }
  } else {
    for (int i = 0; i < this->T_; i++) {
      if (this->layer_param_.bottom_mlu_dtype_size() > 2 + 2 * i &&
         (this->layer_param_.bottom_mlu_dtype(2 + 2 * i).position_size()||
          this->layer_param_.bottom_mlu_dtype(2 + 2 * i).scale_size())) {
        if (this->layer_param_.bottom_mlu_dtype(2 + 2 * i).position_size()) {
          h_blobs_[i + 1]->set_mlu_position(
              this->layer_param_.bottom_mlu_dtype(2 + 2 * i).position(0));
        }
        if (this->layer_param_.bottom_mlu_dtype(2 + 2 * i).scale_size()) {
          h_blobs_[i + 1]->set_mlu_scale(
              this->layer_param_.bottom_mlu_dtype(2 + 2 * i).scale(0));
        }
      }
    }
  }

  // h_reshape_blobs_
  // reshape 2,18,1,1 to 1,2,18,1
  h_reshape_blobs_.clear();
  h_reshape_blobs_.resize(this->T_-1);
  vector<int> h_reshape_shape(4, 1);
  h_reshape_shape[1] =  this->N_;
  h_reshape_shape[2] =  num_output_;
  for (int i = 0; i < this->T_-1; i++) {
    h_reshape_blobs_[i].reset(new Blob<Dtype>(
                      h_reshape_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));
  }

  // h_conted_reshape_blobs_
  // h_conted_reshape 1,2,18,1 to 2,18,1,1
  h_conted_reshape_blobs_.clear();
  h_conted_reshape_blobs_.resize(this->T_);
  vector<int> h_conted_reshape_shape(4, 1);
  h_conted_reshape_shape[0] =  this->N_;
  h_conted_reshape_shape[1] =  num_output_;
  for (int i = 0; i < this->T_; i++) {
    h_conted_reshape_blobs_[i].reset(new Blob<Dtype>(
            h_conted_reshape_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));
  }
  if (this->static_input_) {
    for (int i = 0; i < this->T_; i++) {
      if (this->layer_param_.bottom_mlu_dtype_size() > 2 + 2 * i &&
          (this->layer_param_.bottom_mlu_dtype(2 + 2 * i).position_size() ||
           this->layer_param_.bottom_mlu_dtype(2 + 2 * i).scale_size())) {
        if (this->layer_param_.bottom_mlu_dtype(2 + 2 * i).position_size()) {
          h_conted_reshape_blobs_[i]->set_mlu_position(
              this->layer_param_.bottom_mlu_dtype(2 + 2 * i).position(0));
        }
        if (this->layer_param_.bottom_mlu_dtype(2 + 2 * i).scale_size()) {
          h_conted_reshape_blobs_[i]->set_mlu_scale(
              this->layer_param_.bottom_mlu_dtype(2 + 2 * i).scale(0));
        }
      }
    }
  } else {
    for (int i = 0; i < this->T_; i++) {
      if (this->layer_param_.bottom_mlu_dtype_size() > 1 + 2 * i &&
          (this->layer_param_.bottom_mlu_dtype(1 + 2 * i).position_size() ||
           this->layer_param_.bottom_mlu_dtype(1 + 2 * i).scale_size())) {
        if (this->layer_param_.bottom_mlu_dtype(1 + 2 * i).position_size()) {
          h_conted_reshape_blobs_[i]->set_mlu_position(
              this->layer_param_.bottom_mlu_dtype(1 + 2 * i).position(0));
        }
        if (this->layer_param_.bottom_mlu_dtype(1 + 2 * i).scale_size()) {
          h_conted_reshape_blobs_[i]->set_mlu_scale(
              this->layer_param_.bottom_mlu_dtype(1 + 2 * i).scale(0));
        }
      }
    }
  }

  // o_blobs_ Tanh
  o_blobs_.clear();
  o_blobs_.resize(this->T_);
  for (int i = 0; i < this->T_; i++) {
    o_blobs_[i].reset(new Blob<Dtype>(
                      w_xh_x_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));
  }


  // w_ho_blobs MLP
  w_ho_blobs_.clear();
  w_ho_blobs_.resize(this->T_);
  vector<int> w_ho_shape(4, 1);
  w_ho_shape[0] = this->N_;
  w_ho_shape[1] = num_output_;
  for (int i = 0; i < this->T_; i++) {
    w_ho_blobs_[i].reset(new Blob<Dtype>(
                     w_ho_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));
  }

  // top0_reshape_blob_
  vector<int> top_temp_shape(4, 1);
  top_temp_shape[0] = this->T_ * this->N_;
  top_temp_shape[1] = num_output_;
  top0_reshape_blob_.reset(new Blob<Dtype>(
                     top_temp_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));

  // top1_reshape_blob_
  if (this->expose_hidden_) {
    vector<int> top_hidden_shape(4, 1);
    top_hidden_shape[0] = this->N_;
    top_hidden_shape[1] = num_output_;
    top1_reshape_blob_.reset(new Blob<Dtype>(
                     top_hidden_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));
  }
}

template <typename Dtype>
void MLURNNLayer<Dtype>::fuse(MFusion<Dtype>* fuser) {
  fuser->fuse(bottom0_reshape_op_);
  fuser->fuse(bottom1_reshape_op_);
  if (this->static_input_) {
    fuser->fuse(bottom2_reshape_op_);
  }
  if (this->expose_hidden_) {
    fuser->fuse(bottom3_reshape_op_);
  }
  fuser->fuse(cont_slice_op_);
  fuser->fuse(x_mlp_op_);
  fuser->fuse(w_xh_x_slice_op_);
  if (this->static_input_) {
    fuser->fuse(x_static_mlp_op_);
  }
  for (int i = 0; i < this->T_; i++) {
    if (i != 0)
      fuser->fuse(h_reshape_op_[i-1]);
    fuser->fuse(cyclemult_op_[i]);
    fuser->fuse(h_conted_reshape_op_[i]);
    fuser->fuse(h_conted_mlp_op_[i]);
    fuser->fuse(add_op_[i]);
    if (this->static_input_)
      fuser->fuse(add_1_op_[i]);
    fuser->fuse(h_neuron_input_tanh_op_[i]);
    fuser->fuse(h_mlp_op_[i]);
    fuser->fuse(w_ho_tanh_op_[i]);
  }
  fuser->fuse(concat_op_);
  fuser->fuse(top0_reshape_op_);
  if (this->expose_hidden_) {
    fuser->fuse(device_memcpy_op_);
    fuser->fuse(top1_reshape_op_);
  }
}  // fuse

template <typename Dtype>
void MLURNNLayer<Dtype>::MLUCreateOpBindData(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
  int bottom0_dim[4];
  bottom0_dim[0] = bottom0_reshape_blob_->shape(0);
  bottom0_dim[1] = bottom0_reshape_blob_->shape(1);
  bottom0_dim[2] = bottom0_reshape_blob_->shape(2);
  bottom0_dim[3] = bottom0_reshape_blob_->shape(3);
  MLU_CHECK(cnmlCreateNdReshapeOpParam(&bottom0_reshape_param_ptr_,
                                     bottom0_dim,
                                     4));
  MLU_CHECK(cnmlCreateReshapeOp(&bottom0_reshape_op_,
                         bottom0_reshape_param_ptr_,
                         bottom[0]->mlu_tensor(),
                         bottom0_reshape_blob_->mlu_tensor()));

  int bottom1_dim[4];
  bottom1_dim[0] = bottom1_reshape_blob_->shape(0);
  bottom1_dim[1] = bottom1_reshape_blob_->shape(1);
  bottom1_dim[2] = bottom1_reshape_blob_->shape(2);
  bottom1_dim[3] = bottom1_reshape_blob_->shape(3);
  MLU_CHECK(cnmlCreateNdReshapeOpParam(&bottom1_reshape_param_ptr_,
                                     bottom1_dim,
                                     4));
  MLU_CHECK(cnmlCreateReshapeOp(&bottom1_reshape_op_,
                         bottom1_reshape_param_ptr_,
                         bottom[1]->mlu_tensor(),
                         bottom1_reshape_blob_->mlu_tensor()));

  if (this->static_input_) {
    int bottom2_dim[4];
    bottom2_dim[0] = bottom2_reshape_blob_->shape(0);
    bottom2_dim[1] = bottom2_reshape_blob_->shape(1);
    bottom2_dim[2] = bottom2_reshape_blob_->shape(2);
    bottom2_dim[3] = bottom2_reshape_blob_->shape(3);
    MLU_CHECK(cnmlCreateNdReshapeOpParam(&bottom2_reshape_param_ptr_,
                                       bottom2_dim,
                                       4));
    MLU_CHECK(cnmlCreateReshapeOp(&bottom2_reshape_op_,
                           bottom2_reshape_param_ptr_,
                           bottom[2]->mlu_tensor(),
                           bottom2_reshape_blob_->mlu_tensor()));
  }

  if (this->expose_hidden_) {
    int bottom3_dim[4];
    bottom3_dim[0] = bottom3_reshape_blob_->shape(0);
    bottom3_dim[1] = bottom3_reshape_blob_->shape(1);
    bottom3_dim[2] = bottom3_reshape_blob_->shape(2);
    bottom3_dim[3] = bottom3_reshape_blob_->shape(3);
    MLU_CHECK(cnmlCreateNdReshapeOpParam(&bottom3_reshape_param_ptr_,
                                       bottom3_dim,
                                       4));
    MLU_CHECK(cnmlCreateReshapeOp(&bottom3_reshape_op_,
                           bottom3_reshape_param_ptr_,
                           bottom[3]->mlu_tensor(),
                           bottom3_reshape_blob_->mlu_tensor()));
  }

  // slice cont_{t}
  const size_t kContTensors = this->T_;
  cnmlTensor_t cont_tensors[kContTensors];
  for (int i = 0; i < this->T_; i++) {
    cont_tensors[i] = cont_blobs_[i]->mlu_tensor();
  }
  cnmlTensor_t bottom_cont_tensor = bottom1_reshape_blob_->mlu_tensor();
  MLU_CHECK(cnmlCreateNdSplitOp(&cont_slice_op_,
                             0,
                             &bottom_cont_tensor,
                             1,
                             cont_tensors,
                             this->T_));


  // W_xh_x = W_xh * x + b_h
  MLU_CHECK(cnmlCreateMlpOp(&x_mlp_op_,
                           bottom0_reshape_blob_->mlu_tensor(),
                           w_xh_x_blob_->mlu_tensor(),
                           this->blobs_[0]->mlu_tensor(),
                           this->blobs_[1]->mlu_tensor()));
  MLU_CHECK(cnmlBindConstData_V2(this->blobs_[0]->mlu_tensor(),
                           this->blobs_[0]->sync_data(),
                           false));
  MLU_CHECK(cnmlBindConstData_V2(this->blobs_[1]->mlu_tensor(),
                           this->blobs_[1]->sync_data(),
                           false));

  // slice w_xh_x_{t}
  const size_t kWxhxTensors = this->T_;
  cnmlTensor_t w_xh_x_tensors[kWxhxTensors];
  for (int i = 0; i < this->T_; i++) {
    w_xh_x_tensors[i] = w_xh_x_blobs_[i]->mlu_tensor();
  }
  cnmlTensor_t w_xh_tensor = w_xh_x_blob_->mlu_tensor();
  MLU_CHECK(cnmlCreateSplitOp(&w_xh_x_slice_op_,
                             0,
                             &w_xh_tensor,
                             1,
                             w_xh_x_tensors,
                             this->T_));

  if (this->static_input_) {
    // W_xh_x_static_preshape = W_xh_static * x_static
    MLU_CHECK(cnmlCreateMlpOp(&x_static_mlp_op_,
                             bottom2_reshape_blob_->mlu_tensor(),
                             w_xh_x_static_blob_->mlu_tensor(),
                             this->blobs_[2]->mlu_tensor(),
                             nullptr));
    MLU_CHECK(cnmlBindConstData_V2(this->blobs_[2]->mlu_tensor(),
                             this->blobs_[2]->sync_data(),
                             false));
  }

  // h_conted_{t-1} := cont_t * h_{t-1}
  for (int i = 0; i < this->T_; i++) {
    if (i == 0) {
      if (this->expose_hidden_) {
        MLU_CHECK(cnmlCreateCycleMultOp(&cyclemult_op_[i],
                               bottom3_reshape_blob_->mlu_tensor(),
                               cont_blobs_[i]->mlu_tensor(),
                               h_conted_blobs_[i]->mlu_tensor()));
      } else {
        caffe_set(h_blobs_[0]->count(), Dtype(0),
                h_blobs_[0]->mutable_cpu_data());
        MLU_CHECK(cnmlCreateCycleMultOp(&cyclemult_op_[i],
                               h_blobs_[i]->mlu_tensor(),
                               cont_blobs_[i]->mlu_tensor(),
                               h_conted_blobs_[i]->mlu_tensor()));

        MLU_CHECK(cnmlBindConstData_V2(h_blobs_[i]->mlu_tensor(),
                                 h_blobs_[i]->sync_data(),
                                 false));
      }

    } else {
      if (this->expose_hidden_) {
        int h_reshape_dim[4];
        h_reshape_dim[0] = h_reshape_blobs_[i-1]->shape(0);
        h_reshape_dim[1] = h_reshape_blobs_[i-1]->shape(1);
        h_reshape_dim[2] = h_reshape_blobs_[i-1]->shape(2);
        h_reshape_dim[3] = h_reshape_blobs_[i-1]->shape(3);
        MLU_CHECK(cnmlCreateNdReshapeOpParam(&h_reshape_param_ptr_[i-1],
                                           h_reshape_dim,
                                           4));
        MLU_CHECK(cnmlCreateReshapeOp(&h_reshape_op_[i-1],
                               h_reshape_param_ptr_[i-1],
                               h_blobs_[i]->mlu_tensor(),
                               h_reshape_blobs_[i-1]->mlu_tensor()));
        MLU_CHECK(cnmlCreateCycleMultOp(&cyclemult_op_[i],
                               h_reshape_blobs_[i-1]->mlu_tensor(),
                               cont_blobs_[i]->mlu_tensor(),
                               h_conted_blobs_[i]->mlu_tensor()));
      } else {
        int h_reshape_dim[4];
        h_reshape_dim[0] = h_reshape_blobs_[i-1]->shape(0);
        h_reshape_dim[1] = h_reshape_blobs_[i-1]->shape(1);
        h_reshape_dim[2] = h_reshape_blobs_[i-1]->shape(2);
        h_reshape_dim[3] = h_reshape_blobs_[i-1]->shape(3);
        MLU_CHECK(cnmlCreateNdReshapeOpParam(&h_reshape_param_ptr_[i-1],
                                           h_reshape_dim,
                                           4));
        MLU_CHECK(cnmlCreateReshapeOp(&h_reshape_op_[i-1],
                               h_reshape_param_ptr_[i-1],
                               h_blobs_[i]->mlu_tensor(),
                               h_reshape_blobs_[i-1]->mlu_tensor()));
        MLU_CHECK(cnmlCreateCycleMultOp(&cyclemult_op_[i],
                               h_reshape_blobs_[i-1]->mlu_tensor(),
                               cont_blobs_[i]->mlu_tensor(),
                               h_conted_blobs_[i]->mlu_tensor()));
      }
    }
    int h_conted_reshape_dim[4];
    h_conted_reshape_dim[0] = h_conted_reshape_blobs_[i]->shape(0);
    h_conted_reshape_dim[1] = h_conted_reshape_blobs_[i]->shape(1);
    h_conted_reshape_dim[2] = h_conted_reshape_blobs_[i]->shape(2);
    h_conted_reshape_dim[3] = h_conted_reshape_blobs_[i]->shape(3);
    MLU_CHECK(cnmlCreateNdReshapeOpParam(&h_conted_reshape_param_ptr_[i],
                                       h_conted_reshape_dim,
                                       4));

    MLU_CHECK(cnmlCreateReshapeOp(&h_conted_reshape_op_[i],
                           h_conted_reshape_param_ptr_[i],
                           h_conted_blobs_[i]->mlu_tensor(),
                           h_conted_reshape_blobs_[i]->mlu_tensor()));


  if (this->static_input_) {
      for (int j = 0; j < this->T_; j++) {
        for (int q = 0; q < this->blobs_[3]->count(); q++) {
          bak_weight1_blobs_[j]->mutable_cpu_data()[q] =
              this->blobs_[3]->mutable_cpu_data()[q];
        }
        if (this->blobs_[3]->mlu_type() == DT_INT8) {
            if (this->layer_param_.blobs_dtype(2).position_size())
              bak_weight1_blobs_[j]->set_mlu_position(
                    this->layer_param_.blobs_dtype(2).position(0));
            if (this->layer_param_.blobs_dtype(2).scale_size())
              bak_weight1_blobs_[j]->set_mlu_scale(
                    this->layer_param_.blobs_dtype(2).scale(0));
        }
      }
      for (int j = 0; j < this->T_; j++) {
        for (int q = 0; q < this->blobs_[4]->count(); q++) {
          bak_weight2_blobs_[j]->mutable_cpu_data()[q] =
              this->blobs_[4]->mutable_cpu_data()[q];
        }
        if (this->blobs_[4]->mlu_type() == DT_INT8) {
            if (this->layer_param_.blobs_dtype(3).position_size())
              bak_weight2_blobs_[j]->set_mlu_position(
                    this->layer_param_.blobs_dtype(3).position(0));
            if (this->layer_param_.blobs_dtype(3).scale_size())
              bak_weight2_blobs_[j]->set_mlu_scale(
                    this->layer_param_.blobs_dtype(3).scale(0));
        }
      }
      for (int j = 0; j < this->T_; j++) {
        for (int q = 0; q < this->blobs_[5]->count(); q++) {
          bak_beta_blobs_[j]->mutable_cpu_data()[q] =
              this->blobs_[5]->mutable_cpu_data()[q];
        }
      }
  } else {
      for (int j = 0; j < this->T_; j++) {
        for (int q = 0; q < this->blobs_[2]->count(); q++) {
          bak_weight1_blobs_[j]->mutable_cpu_data()[q] =
              this->blobs_[2]->mutable_cpu_data()[q];
        }
        if (this->blobs_[2]->mlu_type() == DT_INT8) {
            if (this->layer_param_.blobs_dtype(1).position_size())
              bak_weight1_blobs_[j]->set_mlu_position(
                    this->layer_param_.blobs_dtype(1).position(0));
            if (this->layer_param_.blobs_dtype(1).scale_size())
              bak_weight1_blobs_[j]->set_mlu_scale(
                    this->layer_param_.blobs_dtype(1).scale(0));
        }
      }
      for (int j = 0; j < this->T_; j++) {
        for (int q = 0; q < this->blobs_[3]->count(); q++) {
          bak_weight2_blobs_[j]->mutable_cpu_data()[q] =
              this->blobs_[3]->mutable_cpu_data()[q];
        }
        if (this->blobs_[3]->mlu_type() == DT_INT8) {
            if (this->layer_param_.blobs_dtype(2).position_size())
              bak_weight2_blobs_[j]->set_mlu_position(
                    this->layer_param_.blobs_dtype(2).position(0));
            if (this->layer_param_.blobs_dtype(2).scale_size())
              bak_weight2_blobs_[j]->set_mlu_scale(
                    this->layer_param_.blobs_dtype(2).scale(0));
        }
      }
      for (int j = 0; j < this->T_; j++) {
        for (int q = 0; q < this->blobs_[4]->count(); q++) {
          bak_beta_blobs_[j]->mutable_cpu_data()[q] =
              this->blobs_[4]->mutable_cpu_data()[q];
        }
      }
  }
    // W_hh_h_{t-1} := W_hh * h_conted_{t-1}
    if (this->static_input_) {
      MLU_CHECK(cnmlCreateMlpOp(&h_conted_mlp_op_[i],
                               h_conted_reshape_blobs_[i]->mlu_tensor(),
                               w_hh_h_blobs_[i]->mlu_tensor(),
                               bak_weight1_blobs_[i]->mlu_tensor(),
                               nullptr));

      MLU_CHECK(cnmlBindConstData_V2(bak_weight1_blobs_[i]->mlu_tensor(),
                               bak_weight1_blobs_[i]->sync_data(),
                               false));
    } else {
    // W_hh_h_{t-1} := W_hh * h_conted_{t-1}
      MLU_CHECK(cnmlCreateMlpOp(&h_conted_mlp_op_[i],
                               h_conted_reshape_blobs_[i]->mlu_tensor(),
                               w_hh_h_blobs_[i]->mlu_tensor(),
                               bak_weight1_blobs_[i]->mlu_tensor(),
                               nullptr));
      MLU_CHECK(cnmlBindConstData_V2(bak_weight1_blobs_[i]->mlu_tensor(),
                               bak_weight1_blobs_[i]->sync_data(),
                               false));
    }

    // Add the outputs of the linear transformations to compute the gate input.
    //     h_neuron_input_t := W_hh * h_conted_{t-1} + W_xh * x_t + b_h
    //                   = W_hh_h_{t-1} + W_xh_x_t + b_h
    if (!this->static_input_) {
      MLU_CHECK(cnmlCreateAddOp(&add_op_[i],
                               w_hh_h_blobs_[i]->mlu_tensor(),
                               w_xh_x_blobs_[i]->mlu_tensor(),
                               h_neuron_input_blobs_[i]->mlu_tensor()));
    } else {
      MLU_CHECK(cnmlCreateAddOp(&add_op_[i],
                               w_hh_h_blobs_[i]->mlu_tensor(),
                               w_xh_x_blobs_[i]->mlu_tensor(),
                               temp_input_blobs_[i]->mlu_tensor()));

      MLU_CHECK(cnmlCreateAddOp(&add_1_op_[i],
                               temp_input_blobs_[i]->mlu_tensor(),
                               w_xh_x_static_blob_->mlu_tensor(),
                               h_neuron_input_blobs_[i]->mlu_tensor()));
    }



    // h = tanh(h_neuron_input_t)
    MLU_CHECK(cnmlCreateActiveOp(&h_neuron_input_tanh_op_[i],
                                cnmlActiveFunction_t::CNML_ACTIVE_TANH,
                                h_neuron_input_blobs_[i]->mlu_tensor(),
                                h_blobs_[i+1]->mlu_tensor()));


    // w_ho = InnerProduct of h
    if (this->static_input_) {
        MLU_CHECK(cnmlCreateMlpOp(&h_mlp_op_[i],
                            h_blobs_[i+1]->mlu_tensor(),
                            w_ho_blobs_[i]->mlu_tensor(),
                            bak_weight2_blobs_[i]->mlu_tensor(),
                            bak_beta_blobs_[i]->mlu_tensor()));

        MLU_CHECK(cnmlBindConstData_V2(bak_weight2_blobs_[i]->mlu_tensor(),
                             bak_weight2_blobs_[i]->sync_data(),
                             false));
        MLU_CHECK(cnmlBindConstData_V2(bak_beta_blobs_[i]->mlu_tensor(),
                             bak_beta_blobs_[i]->sync_data(),
                             false));
    } else {
        MLU_CHECK(cnmlCreateMlpOp(&h_mlp_op_[i],
                            h_blobs_[i+1]->mlu_tensor(),
                            w_ho_blobs_[i]->mlu_tensor(),
                            bak_weight2_blobs_[i]->mlu_tensor(),
                            bak_beta_blobs_[i]->mlu_tensor()));

        MLU_CHECK(cnmlBindConstData_V2(bak_weight2_blobs_[i]->mlu_tensor(),
                             bak_weight2_blobs_[i]->sync_data(),
                             false));
        MLU_CHECK(cnmlBindConstData_V2(bak_beta_blobs_[i]->mlu_tensor(),
                             bak_beta_blobs_[i]->sync_data(),
                             false));
    }
    //  o = tanh(w_ho)
    MLU_CHECK(cnmlCreateActiveOp(&w_ho_tanh_op_[i],
                                cnmlActiveFunction_t::CNML_ACTIVE_TANH,
                                w_ho_blobs_[i]->mlu_tensor(),
                                o_blobs_[i]->mlu_tensor()));
  }   //  end of for loop

  // top0_reshape_blob_
  const size_t kOInputsTensor = this->T_;
  cnmlTensor_t o_inputs_tensor[kOInputsTensor];
  for (int i = 0; i < this->T_; i++) {
    o_inputs_tensor[i] = o_blobs_[i]->mlu_tensor();
  }

  cnmlTensor_t o_outputs_tensor = top0_reshape_blob_->mlu_tensor();
  MLU_CHECK(cnmlCreateNdConcatOp(&concat_op_,
                              0,
                              o_inputs_tensor,
                              this->T_,
                              &o_outputs_tensor,
                              1));
  int top0_reshape_dim[3];
  top0_reshape_dim[0] = top[0]->shape(0);
  top0_reshape_dim[1] = top[0]->shape(1);
  top0_reshape_dim[2] = top[0]->shape(2);
  MLU_CHECK(cnmlCreateNdReshapeOpParam(&top0_reshape_param_ptr_,
                                     top0_reshape_dim,
                                     3));
  MLU_CHECK(cnmlCreateReshapeOp(&top0_reshape_op_,
                         top0_reshape_param_ptr_,
                         top0_reshape_blob_->mlu_tensor(),
                         top[0]->mlu_tensor()));



  if (this->expose_hidden_) {
    MLU_CHECK(cnmlCreateDeviceMemcpyOp(&device_memcpy_op_,
                                      h_blobs_[this->T_]->mlu_tensor(),
                                      top1_reshape_blob_->mlu_tensor()));
    int top1_reshape_dim[3];
    top1_reshape_dim[0] = top[1]->shape(0);
    top1_reshape_dim[1] = top[1]->shape(1);
    top1_reshape_dim[2] = top[1]->shape(2);
    MLU_CHECK(cnmlCreateNdReshapeOpParam(&top1_reshape_param_ptr_,
                                       top1_reshape_dim,
                                       3));
    MLU_CHECK(cnmlCreateReshapeOp(&top1_reshape_op_,
                           top1_reshape_param_ptr_,
                           top1_reshape_blob_->mlu_tensor(),
                           top[1]->mlu_tensor()));
  }
}   //    MLUCreateOpBindData

template <typename Dtype>
void MLURNNLayer<Dtype>::MLUDestroyOp() {
  if (bottom0_reshape_op_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&bottom0_reshape_op_));
    bottom0_reshape_op_ = nullptr;
  }
  if (bottom0_reshape_param_ptr_ != nullptr) {
     MLU_CHECK(cnmlDestroyReshapeOpParam(&bottom0_reshape_param_ptr_));
     bottom0_reshape_param_ptr_ = nullptr;
  }
  if (bottom1_reshape_op_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&bottom1_reshape_op_));
    bottom1_reshape_op_ = nullptr;
  }
  if (bottom1_reshape_param_ptr_ != nullptr) {
     MLU_CHECK(cnmlDestroyReshapeOpParam(&bottom1_reshape_param_ptr_));
     bottom1_reshape_param_ptr_ = nullptr;
  }
  if (bottom2_reshape_op_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&bottom2_reshape_op_));
    bottom2_reshape_op_ = nullptr;
  }
  if (bottom2_reshape_param_ptr_ != nullptr) {
     MLU_CHECK(cnmlDestroyReshapeOpParam(&bottom2_reshape_param_ptr_));
     bottom2_reshape_param_ptr_ = nullptr;
  }
  if (bottom3_reshape_op_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&bottom3_reshape_op_));
    bottom3_reshape_op_ = nullptr;
  }
  if (bottom3_reshape_param_ptr_ != nullptr) {
     MLU_CHECK(cnmlDestroyReshapeOpParam(&bottom3_reshape_param_ptr_));
     bottom3_reshape_param_ptr_ = nullptr;
  }
  if (cont_slice_op_ != nullptr) {
     MLU_CHECK(cnmlDestroyBaseOp(&cont_slice_op_));
     cont_slice_op_ = nullptr;
  }
  if (w_xh_x_slice_op_ != nullptr) {
     MLU_CHECK(cnmlDestroyBaseOp(&w_xh_x_slice_op_));
     w_xh_x_slice_op_ = nullptr;
  }
  if (x_mlp_op_ != nullptr) {
     MLU_CHECK(cnmlDestroyBaseOp(&x_mlp_op_));
     x_mlp_op_ = nullptr;
  }
  if (x_static_mlp_op_ != nullptr) {
     MLU_CHECK(cnmlDestroyBaseOp(&x_static_mlp_op_));
     x_static_mlp_op_ = nullptr;
  }
  if (device_memcpy_op_ != nullptr) {
     MLU_CHECK(cnmlDestroyBaseOp(&device_memcpy_op_));
     device_memcpy_op_ = nullptr;
  }
  if (top0_reshape_op_ != nullptr) {
     MLU_CHECK(cnmlDestroyBaseOp(&top0_reshape_op_));
     top0_reshape_op_ = nullptr;
  }
  if (top1_reshape_op_ != nullptr) {
     MLU_CHECK(cnmlDestroyBaseOp(&top1_reshape_op_));
     top1_reshape_op_ = nullptr;
  }
  for (int i = 0; i < cyclemult_op_.size(); i++) {
    if ( cyclemult_op_[i] != nullptr ) {
      MLU_CHECK(cnmlDestroyBaseOp(&cyclemult_op_[i]));
      cyclemult_op_[i] = nullptr;
    }
  }
  for (int i = 0; i < h_conted_mlp_op_.size(); i++) {
    if ( h_conted_mlp_op_[i] != nullptr ) {
      MLU_CHECK(cnmlDestroyBaseOp(&h_conted_mlp_op_[i]));
      h_conted_mlp_op_[i] = nullptr;
    }
  }
  for (int i = 0; i < h_mlp_op_.size(); i++) {
    if ( h_mlp_op_[i] != nullptr ) {
      MLU_CHECK(cnmlDestroyBaseOp(&h_mlp_op_[i]));
      h_mlp_op_[i] = nullptr;
    }
  }
  for (int i = 0; i < w_ho_tanh_op_.size(); i++) {
    if ( w_ho_tanh_op_[i] != nullptr ) {
      MLU_CHECK(cnmlDestroyBaseOp(&w_ho_tanh_op_[i]));
      w_ho_tanh_op_[i] = nullptr;
    }
  }
  for (int i = 0; i < h_neuron_input_tanh_op_.size(); i++) {
    if (h_neuron_input_tanh_op_[i] != nullptr) {
      MLU_CHECK(cnmlDestroyBaseOp(&h_neuron_input_tanh_op_[i]));
      h_neuron_input_tanh_op_[i] = nullptr;
    }
  }
  for (int i = 0; i < add_op_.size(); i++) {
    if (add_op_[i] != nullptr) {
      MLU_CHECK(cnmlDestroyBaseOp(&add_op_[i]));
      add_op_[i] = nullptr;
    }
  }
  for (int i = 0; i < add_1_op_.size(); i++) {
    if (add_1_op_[i] != nullptr) {
      MLU_CHECK(cnmlDestroyBaseOp(&add_1_op_[i]));
      add_1_op_[i] = nullptr;
    }
  }
  if (concat_op_ != nullptr) {
     MLU_CHECK(cnmlDestroyBaseOp(&concat_op_));
     concat_op_ = nullptr;
  }
  for (int i = 0; i < h_reshape_op_.size(); i++) {
    if (h_reshape_op_[i] != nullptr) {
      MLU_CHECK(cnmlDestroyBaseOp(&h_reshape_op_[i]));
      h_reshape_op_[i] = nullptr;
    }
  }
  for (int i = 0; i < h_conted_reshape_op_.size(); i++) {
    if (h_conted_reshape_op_[i] != nullptr) {
      MLU_CHECK(cnmlDestroyBaseOp(&h_conted_reshape_op_[i]));
      h_conted_reshape_op_[i] = nullptr;
    }
  }
  if (top0_reshape_param_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyReshapeOpParam(&top0_reshape_param_ptr_));
    top0_reshape_param_ptr_ = nullptr;
  }
  if (top1_reshape_param_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyReshapeOpParam(&top1_reshape_param_ptr_));
    top1_reshape_param_ptr_ = nullptr;
  }
  for (int i = 0; i < h_reshape_param_ptr_.size(); i++) {
    if (h_reshape_param_ptr_[i] != nullptr) {
      MLU_CHECK(cnmlDestroyReshapeOpParam(&h_reshape_param_ptr_[i]));
      h_reshape_param_ptr_[i] = nullptr;
    }
  }
  for (int i = 0; i < h_conted_reshape_param_ptr_.size(); i++) {
    if (h_conted_reshape_param_ptr_[i] != nullptr) {
      MLU_CHECK(cnmlDestroyReshapeOpParam(&h_conted_reshape_param_ptr_[i]));
      h_conted_reshape_param_ptr_[i] = nullptr;
    }
  }
}

template <typename Dtype>
void MLURNNLayer<Dtype>::MLUCompileOp() {
  MLU_CHECK(cnmlCompileBaseOp(bottom0_reshape_op_,
                              Caffe::rt_core(),
                              Caffe::core_number()));
  MLU_CHECK(cnmlCompileBaseOp(bottom1_reshape_op_,
                              Caffe::rt_core(),
                              Caffe::core_number()));
  if (this->static_input_) {
    MLU_CHECK(cnmlCompileBaseOp(bottom2_reshape_op_,
                                Caffe::rt_core(),
                                Caffe::core_number()));
  }
  if (this->expose_hidden_) {
    MLU_CHECK(cnmlCompileBaseOp(bottom3_reshape_op_,
                                Caffe::rt_core(),
                                Caffe::core_number()));
  }
  MLU_CHECK(cnmlCompileBaseOp(cont_slice_op_,
                              Caffe::rt_core(),
                              Caffe::core_number()));
  MLU_CHECK(cnmlCompileBaseOp(x_mlp_op_,
                              Caffe::rt_core(),
                              Caffe::core_number()));
  MLU_CHECK(cnmlCompileBaseOp(w_xh_x_slice_op_,
                              Caffe::rt_core(),
                              Caffe::core_number()));
  if (this->static_input_) {
    MLU_CHECK(cnmlCompileBaseOp(x_static_mlp_op_,
                                Caffe::rt_core(),
                                Caffe::core_number()));
  }
  for (int i = 0; i < this->T_; i++) {
    if (i != 0) {
      MLU_CHECK(cnmlCompileBaseOp(h_reshape_op_[i-1],
                                  Caffe::rt_core(),
                                  Caffe::core_number()));
    }
    MLU_CHECK(cnmlCompileBaseOp(cyclemult_op_[i],
                                Caffe::rt_core(),
                                Caffe::core_number()));
    MLU_CHECK(cnmlCompileBaseOp(h_conted_reshape_op_[i],
                                Caffe::rt_core(),
                                Caffe::core_number()));
    MLU_CHECK(cnmlCompileBaseOp(h_conted_mlp_op_[i],
                                Caffe::rt_core(),
                                Caffe::core_number()));
    MLU_CHECK(cnmlCompileBaseOp(add_op_[i],
                                Caffe::rt_core(),
                                Caffe::core_number()));
    if (this->static_input_) {
      MLU_CHECK(cnmlCompileBaseOp(add_1_op_[i],
                                  Caffe::rt_core(),
                                  Caffe::core_number()));
    }
    MLU_CHECK(cnmlCompileBaseOp(h_neuron_input_tanh_op_[i],
                                Caffe::rt_core(),
                                Caffe::core_number()));
    MLU_CHECK(cnmlCompileBaseOp(h_mlp_op_[i],
                                Caffe::rt_core(),
                                Caffe::core_number()));
    MLU_CHECK(cnmlCompileBaseOp(w_ho_tanh_op_[i],
                                Caffe::rt_core(),
                                Caffe::core_number()));
  }  //   end of for loop
  MLU_CHECK(cnmlCompileBaseOp(concat_op_,
                              Caffe::rt_core(),
                              Caffe::core_number()));
  MLU_CHECK(cnmlCompileBaseOp(top0_reshape_op_,
                              Caffe::rt_core(),
                              Caffe::core_number()));
  if (this->expose_hidden_) {
    MLU_CHECK(cnmlCompileBaseOp(device_memcpy_op_,
                                Caffe::rt_core(),
                                Caffe::core_number()));
    MLU_CHECK(cnmlCompileBaseOp(top1_reshape_op_,
                                Caffe::rt_core(),
                                Caffe::core_number()));
  }
}  //  MLUCompileOp

template <typename Dtype>
MLURNNLayer<Dtype>::~MLURNNLayer() {
  MLUDestroyOp();
}

template <typename Dtype>
void MLURNNLayer<Dtype>::Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top) {
  // reshape bottom[0]_ to bottom0_reshape_blob_
  // 3, 2, 16 to 6, 16, 1, 1
  MLU_CHECK(cnmlComputeReshapeOpForward_V3(bottom0_reshape_op_,
                                 bottom[0]->mutable_mlu_data(),
                                 bottom0_reshape_blob_->mutable_mlu_data(),
                                 Caffe::forward_param(),
                                 Caffe::queue()));

  MLU_CHECK(cnmlComputeReshapeOpForward_V3(bottom1_reshape_op_,
                                 bottom[1]->mutable_mlu_data(),
                                 bottom1_reshape_blob_->mutable_mlu_data(),
                                 Caffe::forward_param(),
                                 Caffe::queue()));

  if (this->static_input_) {
    MLU_CHECK(cnmlComputeReshapeOpForward_V3(bottom2_reshape_op_,
                                 bottom[2]->mutable_mlu_data(),
                                 bottom2_reshape_blob_->mutable_mlu_data(),
                                 Caffe::forward_param(),
                                 Caffe::queue()));
  }
  if (this->expose_hidden_) {
    MLU_CHECK(cnmlComputeReshapeOpForward_V3(bottom3_reshape_op_,
                                   bottom[3]->mutable_mlu_data(),
                                   bottom3_reshape_blob_->mutable_mlu_data(),
                                   Caffe::forward_param(),
                                   Caffe::queue()));
  }

  // Slice cont_{t}
  void* cont_input_ptrs = bottom1_reshape_blob_->mutable_mlu_data();
  void* cont_output_ptrs[this->T_];
  for (int i = 0; i < this->T_; i++) {
    cont_output_ptrs[i] = cont_blobs_[i]->mutable_mlu_data();
  }
  MLU_CHECK(cnmlComputeSplitOpForward_V3(cont_slice_op_,
                                         &cont_input_ptrs,
                                         1,
                                         cont_output_ptrs,
                                         this->T_,
                                         Caffe::forward_param(),
                                         Caffe::queue()));

  // W_xh_x = W_xh * x + b_h
  MLU_CHECK(cnmlComputeMlpOpForward_V3(x_mlp_op_,
                           bottom0_reshape_blob_->mutable_mlu_data(),
                           w_xh_x_blob_->mutable_mlu_data(),
                           Caffe::forward_param(),
                           Caffe::queue()));
  // Slice w_xh_x_{t}
  void* w_xh_x_input_ptrs = w_xh_x_blob_->mutable_mlu_data();
  void* w_xh_x_output_ptrs[this->T_];
  for (int i = 0; i < this->T_; i++) {
    w_xh_x_output_ptrs[i] = w_xh_x_blobs_[i]->mutable_mlu_data();
  }
  MLU_CHECK(cnmlComputeSplitOpForward_V3(w_xh_x_slice_op_,
                                         &w_xh_x_input_ptrs,
                                         1,
                                         w_xh_x_output_ptrs,
                                         this->T_,
                                         Caffe::forward_param(),
                                         Caffe::queue()));
  // W_xh_x_static_preshape = W_xh_static * x_static
  if (this->static_input_) {
    MLU_CHECK(cnmlComputeMlpOpForward_V3(x_static_mlp_op_,
                            bottom2_reshape_blob_->mutable_mlu_data(),
                            w_xh_x_static_blob_->mutable_mlu_data(),
                            Caffe::forward_param(),
                            Caffe::queue()));
  }
  // h_conted_{t-1} := cont_t * h_{t-1}
  for (int i = 0; i < this->T_; i++) {
    if (i == 0) {
      if (this->expose_hidden_) {
        MLU_CHECK(cnmlComputeCycleMultOpForward_V3(cyclemult_op_[i],
                          bottom3_reshape_blob_->mutable_mlu_data(),
                          cont_blobs_[i]->mutable_mlu_data(),
                          h_conted_blobs_[i]->mutable_mlu_data(),
                          Caffe::forward_param(),
                          Caffe::queue()));
      } else {
        MLU_CHECK(cnmlComputeCycleMultOpForward_V3(cyclemult_op_[i],
                          nullptr,
                          cont_blobs_[i]->mutable_mlu_data(),
                          h_conted_blobs_[i]->mutable_mlu_data(),
                          Caffe::forward_param(),
                          Caffe::queue()));
      }
    } else {
      // reshape h_blobs_ to h_reshape_blobs_
      // 2,18,1,1, to 1,2,18,1
      if (this->expose_hidden_) {
        MLU_CHECK(cnmlComputeReshapeOpForward_V3(h_reshape_op_[i-1],
                                   h_blobs_[i]->mutable_mlu_data(),
                                   h_reshape_blobs_[i-1]->mutable_mlu_data(),
                                   Caffe::forward_param(),
                                   Caffe::queue()));
        MLU_CHECK(cnmlComputeCycleMultOpForward_V3(cyclemult_op_[i],
                                   h_reshape_blobs_[i-1]->mutable_mlu_data(),
                                   cont_blobs_[i]->mutable_mlu_data(),
                                   h_conted_blobs_[i]->mutable_mlu_data(),
                                   Caffe::forward_param(),
                                   Caffe::queue()));
      } else {
        MLU_CHECK(cnmlComputeReshapeOpForward_V3(h_reshape_op_[i-1],
                                   h_blobs_[i]->mutable_mlu_data(),
                                   h_reshape_blobs_[i-1]->mutable_mlu_data(),
                                   Caffe::forward_param(),
                                   Caffe::queue()));
        MLU_CHECK(cnmlComputeCycleMultOpForward_V3(cyclemult_op_[i],
                                   h_reshape_blobs_[i-1]->mutable_mlu_data(),
                                   cont_blobs_[i]->mutable_mlu_data(),
                                   h_conted_blobs_[i]->mutable_mlu_data(),
                                   Caffe::forward_param(),
                                   Caffe::queue()));
      }
    }
    // reshape h_cont_blobs to h_cont_reshape_blobs_
    // 1,2,18,1 to 2,18,1,1
    MLU_CHECK(cnmlComputeReshapeOpForward_V3(h_conted_reshape_op_[i],
                                  h_conted_blobs_[i]->mutable_mlu_data(),
                                  h_conted_reshape_blobs_[i]->mutable_mlu_data(),
                                  Caffe::forward_param(),
                                  Caffe::queue()));

    //   W_hh_h_{t-1} := W_hc * h_conted_{t-1}
    MLU_CHECK(cnmlComputeMlpOpForward_V3(h_conted_mlp_op_[i],
                                  h_conted_reshape_blobs_[i]->mutable_mlu_data(),
                                  w_hh_h_blobs_[i]->mutable_mlu_data(),
                                  Caffe::forward_param(),
                                  Caffe::queue()));
    // Add the outputs of the linear transformations to compute the gate input.
    //     h_neuron_input_blobs := W_hh * h_conted_{t-1} + W_xh * x_t + b_h
    //                   = W_hh_h_{t-1} + W_xh_x_t + b_h

    if (!this->static_input_) {
      MLU_CHECK(cnmlComputeAddOpForward_V3(add_op_[i],
                                           w_hh_h_blobs_[i]->mutable_mlu_data(),
                                           w_xh_x_blobs_[i]->mutable_mlu_data(),
                                           h_neuron_input_blobs_[i]->mutable_mlu_data(),
                                           Caffe::forward_param(),
                                           Caffe::queue()));
    } else {
      MLU_CHECK(cnmlComputeAddOpForward_V3(add_op_[i],
                                           w_hh_h_blobs_[i]->mutable_mlu_data(),
                                           w_xh_x_blobs_[i]->mutable_mlu_data(),
                                           temp_input_blobs_[i]->mutable_mlu_data(),
                                           Caffe::forward_param(),
                                           Caffe::queue()));

      MLU_CHECK(cnmlComputeAddOpForward_V3(add_1_op_[i],
                                           temp_input_blobs_[i]->mutable_mlu_data(),
                                           w_xh_x_static_blob_->mutable_mlu_data(),
                                           h_neuron_input_blobs_[i]->mutable_mlu_data(),
                                           Caffe::forward_param(),
                                           Caffe::queue()));
    }
    // h = tanh(h_neuron_input_t)
    MLU_CHECK(cnmlComputeActiveOpForward_V3(h_neuron_input_tanh_op_[i],
                                            h_neuron_input_blobs_[i]->mutable_mlu_data(),
                                            h_blobs_[i+1]->mutable_mlu_data(),
                                            Caffe::forward_param(),
                                            Caffe::queue()));
    if (this->static_input_) {
        MLU_CHECK(cnmlComputeMlpOpForward_V3(h_mlp_op_[i],
            h_blobs_[i+1]->mutable_mlu_data(),
            w_ho_blobs_[i]->mutable_mlu_data(),
            Caffe::forward_param(), Caffe::queue()));
    } else {
        MLU_CHECK(cnmlComputeMlpOpForward_V3(h_mlp_op_[i],
                                             h_blobs_[i+1]->mutable_mlu_data(),
                                             w_ho_blobs_[i]->mutable_mlu_data(),
                                             Caffe::forward_param(),
                                             Caffe::queue()));
    }
    // o = tanh(w_ho)
    MLU_CHECK(cnmlComputeActiveOpForward_V3(w_ho_tanh_op_[i],
                                            w_ho_blobs_[i]->mutable_mlu_data(),
                                            o_blobs_[i]->mutable_mlu_data(),
                                            Caffe::forward_param(),
                                            Caffe::queue()));
  }  // end of for loop

  // top[0]
  void* o_input_ptrs[this->T_];
  for (int i = 0; i < this->T_; i++) {
    o_input_ptrs[i] = o_blobs_[i]->mutable_mlu_data();
  }
  void* o_output_ptrs = top0_reshape_blob_->mutable_mlu_data();

  MLU_CHECK(cnmlComputeConcatOpForward_V3(concat_op_,
                                          o_input_ptrs,
                                          this->T_,
                                          &o_output_ptrs,
                                          1,
                                          Caffe::forward_param(),
                                          Caffe::queue()));

  // reshape top0_reshape_blob_ to top[0]
  // 6,18,1,1 to 3,2,18,1
  MLU_CHECK(cnmlComputeReshapeOpForward_V3(top0_reshape_op_,
                     top0_reshape_blob_->mutable_mlu_data(),
                     top[0]->mutable_mlu_data(),
                     Caffe::forward_param(),
                     Caffe::queue()));

  if (this->expose_hidden_) {
    MLU_CHECK(cnmlComputeDeviceMemcpyOpForward_V3(device_memcpy_op_,
                             h_blobs_[this->T_]->mutable_mlu_data(),
                             top1_reshape_blob_->mutable_mlu_data(),
                             Caffe::forward_param(),
                             Caffe::queue()));
    // reshape top1_reshape_blob_ to top[1]
    // 2,18,1,1 to 1,2,18
    MLU_CHECK(cnmlComputeReshapeOpForward_V3(top1_reshape_op_,
                       top1_reshape_blob_->mutable_mlu_data(),
                       top[1]->mutable_mlu_data(),
                       Caffe::forward_param(), Caffe::queue()));
  }
}

INSTANTIATE_CLASS(MLURNNLayer);

}  // namespace caffe
#endif  // USE_MLU
