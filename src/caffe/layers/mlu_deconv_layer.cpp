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
#include <string>
#include <vector>
#include "caffe/filler.hpp"
#include "caffe/layers/mlu_deconv_layer.hpp"
#include "caffe/common.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

template <typename Dtype>
void MLUDeconvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                              const vector<Blob<Dtype>*>& top) {
  BaseConvolutionLayer<Dtype>::LayerSetUp(bottom, top);
  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
  BaseDataType weight_mlu_dtype = this->layer_param_.blobs_dtype_size() > 0
                                      ? this->layer_param_.blobs_dtype(0).type()
                                      : DT_FLOAT16;
  BaseDataType mlu_dtype = DT_FLOAT16;

  this->blobs_[0].reset(new Blob<Dtype>(this->blobs_[0]->shape(), cpu_dtype,
                                        weight_mlu_dtype, CNML_FILTER));
  shared_ptr<Filler<Dtype> > weight_filler(
      GetFiller<Dtype>(this->layer_param_.convolution_param().weight_filler()));
  weight_filler->Fill(this->blobs_[0].get());

  // If necessary, initialize and fill the biases.
  if (this->bias_term_) {
    this->blobs_[1].reset(new Blob<Dtype>(this->blobs_[1]->shape(), cpu_dtype,
                                          mlu_dtype, CNML_CONST, CNML_CNHW));
    shared_ptr<Filler<Dtype> > bias_filler(
        GetFiller<Dtype>(this->layer_param_.convolution_param().bias_filler()));
    bias_filler->Fill(this->blobs_[1].get());
  }
  // Set the default optimize level 0
  this->optimization_level_ = 0;

  // Multiple groups will be broken up.
  CHECK_LE(this->group_, deconv_limit_ * deconv_limit_)
      << " Group >= " << deconv_limit_ * deconv_limit_ << "is not supported";
  if (this->group_ > deconv_limit_) {
    extra_op_ = true;
    quotient_ = this->group_ / deconv_limit_;
    remainder_ = this->group_ % deconv_limit_;
    offset_ = quotient_ + (remainder_ ? 1 : 0);
  }
  deconv_op_.resize(bottom.size() * this->group_);
  fill(deconv_op_.begin(), deconv_op_.end(), nullptr);
  if (this->group_ > 1) {
    if (extra_op_) {
      extra_concat_op_a_.resize(bottom.size() * quotient_);
      fill(extra_concat_op_a_.begin(), extra_concat_op_a_.end(), nullptr);
      if (remainder_ != 0) {
        extra_concat_op_b_.resize(bottom.size());
        fill(extra_concat_op_b_.begin(), extra_concat_op_b_.end(), nullptr);
      }
      extra_split_op_a_.resize(bottom.size() * quotient_);
      fill(extra_split_op_a_.begin(), extra_split_op_a_.end(), nullptr);
      if (remainder_) {
        extra_split_op_b_.resize(bottom.size());
        fill(extra_split_op_b_.begin(), extra_split_op_b_.end(), nullptr);
      }
    }
    split_input_op_.resize(bottom.size());
    fill(split_input_op_.begin(), split_input_op_.end(), nullptr);
    concat_output_op_.resize(bottom.size());
    fill(concat_output_op_.begin(), concat_output_op_.end(), nullptr);
  }
}

template <typename Dtype>
void MLUDeconvolutionLayer<Dtype>::Reshape_tensor(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BaseConvolutionLayer<Dtype>::Reshape(bottom, top);
  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
  BaseDataType weight_mlu_dtype = this->layer_param_.blobs_dtype_size() > 0
                                      ? this->layer_param_.blobs_dtype(0).type()
                                      : DT_FLOAT16;
  BaseDataType mlu_dtype = DT_FLOAT16;

  if (group_op_) {
    output_blobs_.resize(bottom.size());
    const int* pad_data = this->pad_.cpu_data();
    for (int i = 0; i < bottom.size(); i++) {
      vector<int> top_shape(top[0]->shape());
      top_shape[2] = top_shape[2] + pad_data[0] * 2;
      top_shape[3] = top_shape[3] + pad_data[1] * 2;
      output_blobs_[i].reset(
          new Blob<Dtype>(top_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));
    }
    return;
  }
  input_blobs_.resize(bottom.size() * this->group_);
  weight_blobs_.resize(this->group_);
  if (this->bias_term_) bias_blobs_.resize(this->group_);
  output_blobs_.resize(bottom.size() * this->group_);
  if (extra_op_) {
    extra_concat_blobs_.resize(extra_concat_op_a_.size() +
                               extra_concat_op_b_.size());
    extra_split_blobs_.resize(extra_split_op_a_.size() +
                              extra_split_op_b_.size());
  }

  vector<int> weight_shape = this->blobs_[0]->shape();
  weight_shape[0] /= this->group_;
  int stride_h = this->stride_.cpu_data()[0];
  int stride_w = this->stride_.cpu_data()[1];
  if (stride_h > weight_shape[2]) {
    weight_shape[2] = stride_h;
  }

  if (stride_w > weight_shape[3]) {
    weight_shape[3] = stride_w;
  }
  vector<int> bias_shape(4, 1);
  bias_shape[1] = this->num_output_ / this->group_;

  for (auto& blob : weight_blobs_) {
    blob.reset(new Blob<Dtype>(weight_shape, cpu_dtype, weight_mlu_dtype,
                               CNML_FILTER));
  }
  if (this->bias_term_) {
    for (auto& blob : bias_blobs_) {
      blob.reset(new Blob<Dtype>(bias_shape, cpu_dtype, mlu_dtype, CNML_CONST));
    }
  }
  if (this->group_ > 1) {
    vector<int> bottom_shape = bottom[0]->shape();
    bottom_shape[1] /= this->group_;
    vector<int> top_shape = top[0]->shape();
    top_shape[1] /= this->group_;

    for (auto& blob : input_blobs_) {
      blob.reset(
          new Blob<Dtype>(bottom_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));
    }

    for (auto& blob : output_blobs_) {
      blob.reset(new Blob<Dtype>(top_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));
    }
    bottom_shape = bottom[0]->shape();
    top_shape = top[0]->shape();
    if (extra_op_) {
      for (int i = 0; i < bottom.size(); i++) {
        vector<int> extra_concat_shape = top_shape;
        extra_concat_shape[1] = deconv_limit_ * top_shape[1] / this->group_;
        for (int j = 0; j < quotient_; j++) {
          extra_concat_blobs_[i * offset_ + j].reset(new Blob<Dtype>(
              extra_concat_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));
        }
        vector<int> extra_split_shape = bottom_shape;
        extra_split_shape[1] = deconv_limit_ * bottom_shape[1] / this->group_;
        for (int j = 0; j < quotient_; j++) {
          extra_split_blobs_[i * offset_ + j].reset(new Blob<Dtype>(
              extra_split_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));
        }
        if (remainder_) {
          extra_concat_shape[1] = remainder_ * top_shape[1] / this->group_;
          extra_concat_blobs_[i * offset_ + quotient_].reset(new Blob<Dtype>(
              extra_concat_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));
          extra_split_shape[1] = remainder_ * bottom_shape[1] / this->group_;
          extra_split_blobs_[i * offset_ + quotient_].reset(new Blob<Dtype>(
              extra_split_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));
        }
      }
    }
  }
}

template <typename Dtype>
void MLUDeconvolutionLayer<Dtype>::MLUCreateOpBindData(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int stride_h = stride_data[0];
  const int stride_w = stride_data[1];
  const int pad_h = pad_data[0];
  const int pad_w = pad_data[1];
  const int* dilation_data = this->dilation_.cpu_data();
  const int dilation_h = dilation_data[0];
  const int dilation_w = dilation_data[1];
  // create DeconvGroupOp
  if (group_op_) {
    MLU_CHECK(cnmlCreateDeconvOpParam(&group_param_,
                                      stride_h,
                                      stride_w,
                                      dilation_h,
                                      dilation_w,
                                      pad_h,
                                      pad_w));

    MLU_CHECK(cnmlCreateGrepOpParam(&crop_param_, 0, pad_h, pad_w, 0));

    for (int i = 0; i < bottom.size(); i++) {
      MLU_CHECK(cnmlCreateDeconvGroupOp(&deconv_op_[i],
          group_param_,
          bottom[i]->mlu_tensor(),
          output_blobs_[i]->mlu_tensor(),
          this->blobs_[0]->mlu_tensor(),
          this->bias_term_ ? this->blobs_[1]->mlu_tensor() : nullptr,
          this->group_));
      MLU_CHECK(cnmlCreateGrepOp(&crop_op_[i],
                                 crop_param_,
                                 output_blobs_[i]->mlu_tensor(),
                                 top[i]->mlu_tensor()));
      if (this->layer_param_.bottom_mlu_dtype_size() > 0 &&
          (this->layer_param_.bottom_mlu_dtype(0).position_size() ||
           this->layer_param_.bottom_mlu_dtype(0).scale_size())) {
        if (this->layer_param_.bottom_mlu_dtype(0).position_size()) {
            bottom[i]->set_mlu_position(
              this->layer_param_.bottom_mlu_dtype(0).position(0));
        }
        if (this->layer_param_.bottom_mlu_dtype(0).scale_size()) {
            bottom[i]->set_mlu_scale(
              this->layer_param_.bottom_mlu_dtype(0).scale(0));
        }
      }
    }
    if (this->blobs_[0]->mlu_type() == DT_INT8) {
      if (this->layer_param_.blobs_dtype_size() > 0 &&
          (this->layer_param_.blobs_dtype(0).position_size() ||
           this->layer_param_.blobs_dtype(0).scale_size())) {
        if (this->layer_param_.blobs_dtype(0).position_size()) {
          this->blobs_[0]->set_mlu_position(
              this->layer_param_.blobs_dtype(0).position(0));
        }
        if (this->layer_param_.blobs_dtype(0).scale_size()) {
          this->blobs_[0]->set_mlu_scale(
              this->layer_param_.blobs_dtype(0).scale(0));
        }
      }
    }
    MLU_CHECK(cnmlBindConstData_V2(this->blobs_[0]->mlu_tensor(),
                                   this->blobs_[0]->sync_data(),
                                   false));
    if (this->bias_term_) {
      MLU_CHECK(cnmlBindConstData_V2(this->blobs_[1]->mlu_tensor(),
                                     this->blobs_[1]->sync_data(),
                                     false));
    }
    return;
  }

  const int ori_kernel_size_h = this->blobs_[0]->height();
  const int ori_kernel_size_w = this->blobs_[0]->width();
  // copy data from blobs_ to weight_blobs_ and bias_blobs_
  for (int i = 0; i < this->group_; i++) {
    if (stride_h > ori_kernel_size_h || stride_w > ori_kernel_size_w) {
      for (int j = 0; j < weight_blobs_[i]->count(); j++) {
        Dtype* weight_data = weight_blobs_[i]->mutable_cpu_data();
        weight_data[j] = 0;
      }
      for (int j = 0;
           j < weight_blobs_[i]->num() * weight_blobs_[i]->channels(); j++) {
        for (int kh = 0; kh < ori_kernel_size_h; kh++) {
          for (int kw = 0; kw < ori_kernel_size_w; kw++) {
            int offset = this->blobs_[0]->count() / this->group_;
            Dtype* weight_data = weight_blobs_[i]->mutable_cpu_data();
            weight_data[j * stride_h * stride_w + kh * stride_h + kw] =
                this->blobs_[0]
                    ->cpu_data()[offset * i +
                                 j * ori_kernel_size_h * ori_kernel_size_w +
                                 kh * ori_kernel_size_w + kw];
          }
        }
      }
    } else {
      for (int j = 0; j < weight_blobs_[i]->count(); j++) {
        int offset = this->blobs_[0]->count() / this->group_;
        Dtype* weight_data = weight_blobs_[i]->mutable_cpu_data();
        weight_data[j] = this->blobs_[0]->cpu_data()[offset * i + j];
      }
    }
    if (this->bias_term_) {
      for (int j = 0; j < bias_blobs_[i]->count(); j++) {
        int offset = this->blobs_[1]->count() / this->group_;
        Dtype* bias_data = bias_blobs_[i]->mutable_cpu_data();
        bias_data[j] = this->blobs_[1]->cpu_data()[offset * i + j];
      }
    }
  }
  int special_sub_h =
      stride_h - ori_kernel_size_h > 0 ? stride_h - ori_kernel_size_h : 0;
  int special_sub_w =
      stride_w - ori_kernel_size_w > 0 ? stride_w - ori_kernel_size_w : 0;
  MLU_CHECK(cnmlCreateDeconvOpParam(&deconv_param_, stride_h, stride_w, pad_h,
                                    pad_h + special_sub_h, pad_w,
                                    pad_w + special_sub_w));

  for (int i = 0; i < bottom.size(); i++) {
    if (1 == this->group_) {
      MLU_CHECK(cnmlCreateDeconvOp(&deconv_op_[i],
          deconv_param_,
          bottom[i]->mlu_tensor(),
          top[i]->mlu_tensor(),
          weight_blobs_[0]->mlu_tensor(),
          this->bias_term_ ? bias_blobs_[0]->mlu_tensor() : nullptr));
      if (this->layer_param_.bottom_mlu_dtype_size() > i) {
        if (this->layer_param_.bottom_mlu_dtype(i).position_size()) {
          bottom[i]->set_mlu_position(
              this->layer_param_.bottom_mlu_dtype(i).position(0));
        }
        if (this->layer_param_.bottom_mlu_dtype(i).scale_size()) {
          bottom[i]->set_mlu_scale(
              this->layer_param_.bottom_mlu_dtype(i).scale(0));
        }
      }
    } else {
      // slice bottom
      cnmlTensor_t bottom_tensor = bottom[i]->mlu_tensor();
      cnmlTensor_t* input_tensors = new cnmlTensor_t[this->group_];
      for (int k = 0; k < this->group_; k++) {
        input_tensors[k] = input_blobs_[this->group_ * i + k]->mlu_tensor();
        int shape[4];
        cnmlGetTensorShape(input_tensors[i], shape);
      }
      if (!extra_op_) {
        MLU_CHECK(cnmlCreateNdSplitOp(&split_input_op_[i],
                                      3,
                                      &bottom_tensor,
                                      1,
                                      input_tensors,
                                      this->group_));
      } else {
        vector<cnmlTensor_t> output(offset_);
        for (int k = 0; k < offset_; k++) {
          output[k] = extra_split_blobs_[offset_ * i + k]->mlu_tensor();
        }
        MLU_CHECK(cnmlCreateNdSplitOp(&split_input_op_[i],
                                      3,
                                      &bottom_tensor,
                                      1,
                                      output.data(),
                                      offset_));
        for (int j = 0; j < quotient_; j++) {
          cnmlTensor_t input_a =
              extra_split_blobs_[i * offset_ + j]->mlu_tensor();
          vector<cnmlTensor_t> output_a(deconv_limit_);
          for (int k = 0; k < deconv_limit_; k++) {
            output_a[k] = input_tensors[j * deconv_limit_ + k];
          }
          MLU_CHECK(cnmlCreateNdSplitOp(&extra_split_op_a_[i * quotient_ + j],
                                        3,
                                        &input_a,
                                        1,
                                        output_a.data(),
                                        deconv_limit_));
          if (remainder_) {
            cnmlTensor_t input_b =
                extra_split_blobs_[i * offset_ + quotient_]->mlu_tensor();
            vector<cnmlTensor_t> output_b(remainder_);
            for (int k = 0; k < remainder_; k++) {
              output_b[k] = input_tensors[quotient_ * deconv_limit_ + k];
            }
            MLU_CHECK(cnmlCreateNdSplitOp(&extra_split_op_b_[i],
                                          3,
                                          &input_b,
                                          1,
                                          output_b.data(),
                                          remainder_));
          }
        }
      }
      // output
      cnmlTensor_t* output_tensors = new cnmlTensor_t[this->group_];
      for (int j = 0; j < this->group_; j++) {
        output_tensors[j] = output_blobs_[this->group_ * i + j]->mlu_tensor();
        int shape[4];
        cnmlGetTensorShape(output_tensors[j], shape);
      }
      // deconv
      for (int j = 0; j < this->group_; j++) {
        MLU_CHECK(cnmlCreateDeconvOp(&deconv_op_[this->group_ * i + j],
            deconv_param_,
            input_tensors[j],
            output_tensors[j],
            weight_blobs_[j]->mlu_tensor(),
            this->bias_term_ ? bias_blobs_[j]->mlu_tensor() : nullptr));

        if (this->layer_param_.bottom_mlu_dtype_size() > 0 &&
            (this->layer_param_.bottom_mlu_dtype(0).position_size() ||
             this->layer_param_.bottom_mlu_dtype(0).scale_size())) {
          if (this->layer_param_.bottom_mlu_dtype(0).position_size()) {
            input_blobs_[this->group_ * i + j]->set_mlu_position(
                this->layer_param_.bottom_mlu_dtype(0).position(0));
          }
          if (this->layer_param_.bottom_mlu_dtype(0).scale_size()) {
            input_blobs_[this->group_ * i + j]->set_mlu_scale(
                this->layer_param_.bottom_mlu_dtype(0).scale(0));
          }
        }
      }
      cnmlTensor_t output_tensor = top[i]->mlu_tensor();
      if (!extra_op_) {
        // concat output
        MLU_CHECK(cnmlCreateNdConcatOp(&concat_output_op_[i],
                                       3,
                                       output_tensors,
                                       this->group_,
                                       &output_tensor,
                                       1));
      } else {
        for (int j = 0; j < quotient_; j++) {
          vector<cnmlTensor_t> input_a(deconv_limit_);
          for (int k = 0; k < deconv_limit_; k++) {
            input_a[k] = output_tensors[j * deconv_limit_ + k];
          }
          cnmlTensor_t output_a = extra_concat_blobs_[i * offset_ + j]->mlu_tensor();
          MLU_CHECK(cnmlCreateNdConcatOp(&extra_concat_op_a_[i * quotient_ + j],
                                         3,
                                         input_a.data(),
                                         deconv_limit_,
                                         &output_a,
                                         1));
        }
        if (remainder_) {
          vector<cnmlTensor_t> input_b(remainder_);
          for (int k = 0; k < remainder_; k++) {
            input_b[k] = output_tensors[quotient_ * deconv_limit_ + k];
          }
          cnmlTensor_t output_b =
              extra_concat_blobs_[i * offset_ + quotient_]->mlu_tensor();
          MLU_CHECK(cnmlCreateNdConcatOp(&extra_concat_op_b_[i],
                                         3,
                                         input_b.data(),
                                         remainder_,
                                         &output_b,
                                         1));
        }
        vector<cnmlTensor_t> input(offset_);
        for (int k = 0; k < offset_; k++) {
          input[k] = extra_concat_blobs_[i * offset_ + k]->mlu_tensor();
        }
        MLU_CHECK(cnmlCreateNdConcatOp(&concat_output_op_[i],
                                       3,
                                       input.data(),
                                       offset_,
                                       &output_tensor,
                                       1));
      }
    }
  }
  for (int i = 0; i < this->group_; i++) {
    if (this->blobs_[0]->mlu_type() == DT_INT8) {
      if (this->layer_param_.blobs_dtype_size() > 0 &&
          (this->layer_param_.blobs_dtype(0).position_size() ||
           this->layer_param_.blobs_dtype(0).scale_size())) {
        if (this->layer_param_.blobs_dtype(0).position_size()) {
          weight_blobs_[i]->set_mlu_position(
              this->layer_param_.blobs_dtype(0).position(0));
        }
        if (this->layer_param_.blobs_dtype(0).scale_size()) {
          weight_blobs_[i]->set_mlu_scale(
              this->layer_param_.blobs_dtype(0).scale(0));
        }
      }
    }
    MLU_CHECK(cnmlBindConstData_V2(weight_blobs_[i]->mlu_tensor(),
                                   weight_blobs_[i]->sync_data(),
                                   false));
    if (this->bias_term_) {
      MLU_CHECK(cnmlBindConstData_V2(bias_blobs_[i]->mlu_tensor(),
                                     bias_blobs_[i]->sync_data(),
                                     false));
    }
  }
}

template <typename Dtype>
void MLUDeconvolutionLayer<Dtype>::MLUCompileOp() {
  if (this->group_ > 1 && !group_op_) {
    for (auto& op : concat_output_op_)
      MLU_CHECK(cnmlCompileBaseOp(op, Caffe::rt_core(), Caffe::core_number()));
    for (auto& op : split_input_op_)
      MLU_CHECK(cnmlCompileBaseOp(op, Caffe::rt_core(), Caffe::core_number()));
    if (extra_op_) {
      for (auto& op : extra_concat_op_a_)
        MLU_CHECK(cnmlCompileBaseOp(op, Caffe::rt_core(), Caffe::core_number()));
      for (auto& op : extra_split_op_a_)
        MLU_CHECK(cnmlCompileBaseOp(op, Caffe::rt_core(), Caffe::core_number()));
      if (remainder_) {
        for (auto& op : extra_concat_op_b_)
          MLU_CHECK(cnmlCompileBaseOp(op, Caffe::rt_core(), Caffe::core_number()));
        for (auto& op : extra_split_op_b_)
          MLU_CHECK(cnmlCompileBaseOp(op, Caffe::rt_core(), Caffe::core_number()));
      }
    }
  }
  for (int i = 0; i < deconv_op_.size(); i++) {
    MLU_CHECK(cnmlCompileBaseOp(deconv_op_[i], Caffe::rt_core(), Caffe::core_number()));
    if (group_op_) {
      MLU_CHECK(cnmlCompileBaseOp(crop_op_[i], Caffe::rt_core(), Caffe::core_number()));
    }
  }
}

template <typename Dtype>
void MLUDeconvolutionLayer<Dtype>::Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                                               const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < bottom.size(); i++) {
    void* bottom_ptr = bottom[i]->mutable_mlu_data();
    void* input_ptrs[this->group_];
    if (group_op_) {
      void* blobs_ptr = output_blobs_[i]->mutable_mlu_data();
      void* output_ptr = top[i]->mutable_mlu_data();
      MLU_CHECK(cnmlComputeDeconvGroupOpForward(deconv_op_[i], bottom_ptr, blobs_ptr,
                                                Caffe::forward_param(),
                                                Caffe::queue()));
      MLU_CHECK(cnmlComputeGrepOpForward_V3(crop_op_[i], blobs_ptr, output_ptr,
                                            Caffe::forward_param(),
                                            Caffe::queue()));
      break;
    }
    if (1 == this->group_) {
      // Concat operation follows DeConv, so even Deconv layer
      // is the last one in network, it is not the last
      // Operation at all, then optimization level can't be set to 1
      // when group_ is 1, conact/parallel is redundant, so they are
      // skipped. And the side affect is opt 1 can be applied.
      void* output_ptr = top[i]->mutable_mlu_data();
      MLU_CHECK(cnmlComputeDeconvOpForward_V3(deconv_op_[i], bottom_ptr, output_ptr,
                                              Caffe::forward_param(),
                                              Caffe::queue()));
    } else {
      // slice bottom
      for (int j = 0; j < this->group_; j++) {
        input_ptrs[j] = input_blobs_[this->group_ * i + j]->mutable_mlu_data();
      }
      if (!extra_op_) {
        MLU_CHECK(cnmlComputeSplitOpForward_V3(split_input_op_[i],
                                               &bottom_ptr,
                                               1,
                                               input_ptrs,
                                               this->group_,
                                               Caffe::forward_param(),
                                               Caffe::queue()));
      } else {
        void* output[offset_];
        for (int k = 0; k < offset_; k++) {
          output[k] = extra_split_blobs_[offset_ * i + k]->mutable_mlu_data();
        }
        MLU_CHECK(cnmlComputeSplitOpForward_V3(split_input_op_[i],
                                               &bottom_ptr,
                                               1,
                                               output,
                                               offset_,
                                               Caffe::forward_param(),
                                               Caffe::queue()));
        for (int j = 0; j < quotient_; j++) {
          void* input_a =
              extra_split_blobs_[i * offset_ + j]->mutable_mlu_data();
          void* output_a[deconv_limit_];
          for (int k = 0; k < deconv_limit_; k++) {
            output_a[k] = input_blobs_[i * this->group_ + j * deconv_limit_ + k]
                              ->mutable_mlu_data();
          }
          MLU_CHECK(cnmlComputeSplitOpForward_V3(extra_split_op_a_[i * quotient_ + j],
                                                 &input_a,
                                                 1,
                                                 output_a,
                                                 deconv_limit_,
                                                 Caffe::forward_param(),
                                                 Caffe::queue()));
        }
        if (remainder_) {
          void* input_b = extra_split_blobs_[i * offset_ + quotient_]->mutable_mlu_data();
          void* output_b[remainder_];
          for (int k = 0; k < remainder_; k++) {
            output_b[k] =
                input_blobs_[i * this->group_ + quotient_ * deconv_limit_ + k]
                    ->mutable_mlu_data();
          }
          MLU_CHECK(cnmlComputeSplitOpForward_V3(extra_split_op_b_[i],
                                                 &input_b,
                                                 1,
                                                 output_b,
                                                 remainder_,
                                                 Caffe::forward_param(),
                                                 Caffe::queue()));
        }
      }
      // output
      void* output_ptrs[this->group_];
      for (int j = 0; j < this->group_; j++) {
        output_ptrs[j] = output_blobs_[this->group_ * i + j]->mutable_mlu_data();
      }
      // deconv
      for (int k = 0; k < this->group_; k++) {
        MLU_CHECK(cnmlComputeDeconvOpForward_V3(deconv_op_[this->group_ * i + k],
                                                input_ptrs[k],
                                                output_ptrs[k],
                                                Caffe::forward_param(),
                                                Caffe::queue()));
      }
      // concat output
      void* output_ptr = top[i]->mutable_mlu_data();
      if (!extra_op_) {
        MLU_CHECK(cnmlComputeConcatOpForward_V3(concat_output_op_[i],
                                                output_ptrs,
                                                this->group_,
                                                &output_ptr,
                                                1,
                                                Caffe::forward_param(),
                                                Caffe::queue()));
      } else {
        for (int j = 0; j < quotient_; j++) {
          void* input_a[deconv_limit_];
          for (int k = 0; k < deconv_limit_; k++) {
            input_a[k] = output_ptrs[j * deconv_limit_ + k];
          }
          void* output_a = extra_concat_blobs_[i * offset_ + j]->mutable_mlu_data();
          MLU_CHECK(cnmlComputeConcatOpForward_V3(extra_concat_op_a_[i * quotient_ + j],
                                                  input_a,
                                                  deconv_limit_,
                                                  &output_a,
                                                  1,
                                                  Caffe::forward_param(),
                                                  Caffe::queue()));
        }
        if (remainder_) {
          void* input_b[remainder_];
          for (int k = 0; k < remainder_; k++)
            input_b[k] = output_ptrs[quotient_ * deconv_limit_ + k];
          void* output_b =
              extra_concat_blobs_[i * offset_ + quotient_]->mutable_mlu_data();
          MLU_CHECK(cnmlComputeConcatOpForward_V3(extra_concat_op_b_[i],
                                                  input_b,
                                                  remainder_,
                                                  &output_b,
                                                  1,
                                                  Caffe::forward_param(),
                                                  Caffe::queue()));
        }
        void* input[offset_];
        for (int k = 0; k < offset_; k++) {
          input[k] = extra_concat_blobs_[i * offset_ + k]->mutable_mlu_data();
        }
        MLU_CHECK(cnmlComputeConcatOpForward_V3(concat_output_op_[i],
                                                input,
                                                offset_,
                                                &output_ptr,
                                                1,
                                                Caffe::forward_param(),
                                                Caffe::queue()));
      }
    }
  }
}

template <typename Dtype>
void MLUDeconvolutionLayer<Dtype>::MLUDestroyOp() {
  for (auto& op : deconv_op_) {
    if (op != nullptr) {
      MLU_CHECK(cnmlDestroyBaseOp(&op));
      op = nullptr;
    }
  }
  for (auto& op : split_input_op_) {
    if (op != nullptr) {
      MLU_CHECK(cnmlDestroyBaseOp(&op));
      op = nullptr;
    }
  }
  for (auto& op : concat_output_op_) {
    if (op != nullptr) {
      MLU_CHECK(cnmlDestroyBaseOp(&op));
      op = nullptr;
    }
  }
  for (auto& op : extra_concat_op_a_) {
    if (op != nullptr) {
      MLU_CHECK(cnmlDestroyBaseOp(&op));
      op = nullptr;
    }
  }
  for (auto& op : extra_concat_op_b_) {
    if (op != nullptr) {
      MLU_CHECK(cnmlDestroyBaseOp(&op));
      op = nullptr;
    }
  }
  for (auto& op : extra_split_op_a_) {
    if (op != nullptr) {
      MLU_CHECK(cnmlDestroyBaseOp(&op));
      op = nullptr;
    }
  }
  for (auto& op : extra_split_op_b_) {
    if (op != nullptr) {
      MLU_CHECK(cnmlDestroyBaseOp(&op));
      op = nullptr;
    }
  }
  if (deconv_param_ != nullptr) {
    MLU_CHECK(cnmlDestroyDeconvOpParam(&deconv_param_));
    deconv_param_ = nullptr;
  }
  if (group_param_ != nullptr) {
    MLU_CHECK(cnmlDestroyDeconvOpParam(&group_param_));
    group_param_ = nullptr;
  }
  if (crop_param_ != nullptr) {
    MLU_CHECK(cnmlDestroyGrepOpParam(&crop_param_));
    crop_param_ = nullptr;
  }
}

template <typename Dtype>
void MLUDeconvolutionLayer<Dtype>::fuse(MFusion<Dtype>* fuser) {
  for (int i = 0; i < deconv_op_.size(); i++) {
    fuser->fuse(deconv_op_[i]);
    if (group_op_) fuser->fuse(crop_op_[i]);
  }
  if (this->group_ > 1 && !group_op_) {
    for (auto& op : concat_output_op_) fuser->fuse(op);
    for (auto& op : split_input_op_) fuser->fuse(op);
    if (extra_op_) {
      for (auto& op : extra_split_op_a_) fuser->fuse(op);
      for (auto& op : extra_concat_op_a_) fuser->fuse(op);
      if (remainder_) {
        for (auto& op : extra_split_op_b_) fuser->fuse(op);
        for (auto& op : extra_concat_op_b_) fuser->fuse(op);
      }
    }
  }
}

INSTANTIATE_CLASS(MLUDeconvolutionLayer);
}  // namespace caffe
#endif  //  USE_MLU
