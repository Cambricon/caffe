/*
All modification made by Cambricon Corporation: © 2018-2019 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2019, the respective contributors
All rights reserved.
For the list of contributors go to
https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md Redistribution and use
in source and binary forms, with or without modification, are permitted provided
that the following conditions are met:
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

#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/convolution3d_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/vol2col.hpp"

namespace caffe {

template <typename Dtype>
void Convolution3DLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), 1) << "Conv Layer takes a single blob as input.";
  CHECK_EQ(top.size(), 1) << "Conv Layer takes a single blob as output.";

  kernel_size_ = this->layer_param_.convolution3d_param().kernel_size();
  kernel_depth_ = this->layer_param_.convolution3d_param().kernel_depth();
  stride_ = this->layer_param_.convolution3d_param().stride();
  temporal_stride_ = this->layer_param_.convolution3d_param().temporal_stride();
  pad_ = this->layer_param_.convolution3d_param().pad();
  temporal_pad_ = this->layer_param_.convolution3d_param().temporal_pad();
  num_ = bottom[0]->shape(0);
  channels_ = bottom[0]->shape(1);
  length_ = bottom[0]->shape(2);
  height_ = bottom[0]->shape(3);
  width_ = bottom[0]->shape(4);
  num_output_ = this->layer_param_.convolution3d_param().num_output();
  filter_group_ = this->layer_param_.convolution3d_param().filter_group();
  CHECK_GT(num_output_, 0);

  // number of output filters must be divided by filter_group
  CHECK_EQ(num_output_ % filter_group_, 0);

  // The vol2col result buffer would only hold one image at a time to avoid
  // overly large memory usage.

  int height_out = (height_ + 2 * pad_ - kernel_size_) / stride_ + 1;
  int width_out = (width_ + 2 * pad_ - kernel_size_) / stride_ + 1;
  int length_out =
      (length_ + 2 * temporal_pad_ - kernel_depth_) / temporal_stride_ + 1;

  vector<int> top_shape(5);
  vector<int> weight_shape(5);
  vector<int> col_shape(5);
  vector<int> bias_shape(5);

  col_shape[0] = 1;
  col_shape[1] = channels_ * kernel_depth_ * kernel_size_ * kernel_size_;
  col_shape[2] = length_out;
  col_shape[3] = height_out;
  col_shape[4] = width_out;

  // buffer for one image
  col_buffer_.Reshape(col_shape);
  // 1, channels_ * kernel_depth_ * kernel_size_ * kernel_size_, length_out,
  // height_out, width_out);

  bias_term_ = this->layer_param_.convolution3d_param().bias_term();

  // Figure out the dimensions for individual gemms.
  M_ = num_output_ /
       filter_group_;  // doing convolution filter_group_ times per volume
  K_ = channels_ * kernel_depth_ * kernel_size_ * kernel_size_;
  N_ = length_out * height_out * width_out;

  // output size
  top_shape[0] = bottom[0]->shape(0);
  top_shape[1] = num_output_;
  top_shape[2] = length_out;
  top_shape[3] = height_out;
  top_shape[4] = width_out;

  top[0]->Reshape(top_shape);
  //(*top)[0]->Reshape(bottom[0]->num(), num_output_, length_out, height_out,
  // width_out);

  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }

    weight_shape[0] = num_output_;
    weight_shape[1] = channels_;
    weight_shape[2] = kernel_depth_;
    weight_shape[3] = kernel_size_;
    weight_shape[4] = kernel_size_;
    // Initialize the weights
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // num_output_, channels_, kernel_depth_, kernel_size_, kernel_size_));

    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution3d_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the bias term
    if (bias_term_) {
      bias_shape[0] = 1;
      bias_shape[1] = 1;
      bias_shape[2] = 1;
      bias_shape[3] = 1;
      bias_shape[4] = num_output_;
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      // 1, 1, 1, 1, num_output_));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution3d_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }

  // Set up the bias filler
  if (bias_term_) {
    bias_multiplier_.reset(new SyncedMemory(N_ * sizeof(Dtype)));
    Dtype* bias_multiplier_data =
        reinterpret_cast<Dtype*>(bias_multiplier_->mutable_cpu_data());
    for (int i = 0; i < N_; ++i) {
      bias_multiplier_data[i] = 1.;
    }
  }
#ifdef USE_MLU
  SetMean(bottom, top);
#endif
}

template <typename Dtype>
void Convolution3DLayer<Dtype>::SetMean(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
  Convolution3DParameter conv_param = this->layer_param().convolution3d_param();
  if (conv_param.has_mean_file()) {
    this->conv_first_ = true;
    string mean_file = conv_param.mean_file();
    BlobProto blob_proto;
    LOG(INFO) << "meanfile: " << mean_file;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    this->mean_.FromProto(blob_proto);
  } else if (conv_param.mean_value_size() || conv_param.has_std()) {
    this->conv_first_ = true;
    vector<int> mean_shape(bottom[0]->shape());
    this->mean_.Reshape(mean_shape);
    Dtype* data = this->mean_.mutable_cpu_data();
    if (conv_param.mean_value_size() > 0) {
      for (int i = 0; i < bottom[0]->channels(); i++) {
        caffe_set<Dtype>(this->mean_.count() / bottom[0]->channels(),
                         conv_param.mean_value(i), data);
        data += this->mean_.count() / bottom[0]->channels();
      }
    } else {
      caffe_set<Dtype>(this->mean_.count(), Dtype(0.),
                       this->mean_.mutable_cpu_data());
    }
  }
}

template <typename Dtype>
void Convolution3DLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* col_data = col_buffer_.mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();

  int weight_offset = M_ * K_;
  int top_offset = M_ * N_;
  static const int zero_array[] = {0, 0, 0, 0, 0};
  vector<int> offset_indices(zero_array, zero_array + 5);

#ifdef USE_MLU
  if (this->conv_first_) {
    bool std = this->layer_param().convolution_param().has_std();
    for (int j = 0; j < bottom[0]->count(); j++) {
      if (this->layer_param().convolution_param().has_mean_file()) {
        bottom[0]->mutable_cpu_data()[j] -=
            this->mean_.cpu_data()[j % this->mean_.count()];
      } else {
        bottom[0]->mutable_cpu_data()[j] -=
            this->mean_
                .cpu_data()[(j / (bottom[0]->count(2))) % this->mean_.count()];
      }
      if (std) {
        bottom[0]->mutable_cpu_data()[j] *=
            this->layer_param().convolution_param().std();
      }
    }
  }
#endif

  for (int n = 0; n < num_; ++n) {
    offset_indices[0] = n;

    // First, im2col
    vol2col_cpu(bottom_data + bottom[0]->offset(offset_indices), channels_,
                length_, height_, width_, kernel_size_, kernel_depth_, pad_,
                temporal_pad_, stride_, temporal_stride_, col_data);

    // Second, inner-product without filter groups
    for (int g = 0; g < filter_group_; ++g) {
      caffe_cpu_gemm<Dtype>(
          CblasNoTrans, CblasNoTrans, M_, N_, K_, (Dtype)1.,
          weight + g * weight_offset, col_data, (Dtype)0.,
          top_data + top[0]->offset(offset_indices) + g * top_offset);
    }
    // third, add bias
    if (bias_term_) {
      caffe_cpu_gemm<Dtype>(
          CblasNoTrans, CblasNoTrans, num_output_, N_, 1, (Dtype)1.,
          this->blobs_[1]->cpu_data(),
          reinterpret_cast<const Dtype*>(bias_multiplier_->cpu_data()),
          (Dtype)1., top_data + top[0]->offset(offset_indices));
    }
  }
}

template <typename Dtype>
void Convolution3DLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  Dtype* col_data = col_buffer_.mutable_cpu_data();
  Dtype* col_diff = col_buffer_.mutable_cpu_diff();
  // bias gradient if necessary
  Dtype* bias_diff = NULL;

  static const int zero_array[] = {0, 0, 0, 0, 0};
  vector<int> offset_indices(zero_array, zero_array + 5);

  if (bias_term_) {
    bias_diff = this->blobs_[1]->mutable_cpu_diff();
    memset(bias_diff, 0, sizeof(Dtype) * this->blobs_[1]->count());
    for (int n = 0; n < num_; ++n) {
      offset_indices[0] = n;
      caffe_cpu_gemv<Dtype>(
          CblasNoTrans, num_output_, N_, 1.,
          top_diff + top[0]->offset(offset_indices),
          reinterpret_cast<const Dtype*>(bias_multiplier_->cpu_data()), 1.,
          bias_diff);
    }
  }

  int weight_offset = M_ * K_;
  int top_offset = M_ * N_;

  memset(weight_diff, 0, sizeof(Dtype) * this->blobs_[0]->count());
  for (int n = 0; n < num_; ++n) {
    offset_indices[0] = n;
    // since we saved memory in the forward pass by not storing all col data,
    // we will need to recompute them.
    vol2col_cpu(bottom_data + bottom[0]->offset(offset_indices), channels_,
                length_, height_, width_, kernel_size_, kernel_depth_, pad_,
                temporal_pad_, stride_, temporal_stride_, col_data);

    // gradient w.r.t. weight. Note that we will accumulate diffs.
    for (int g = 0; g < filter_group_; ++g) {
      caffe_cpu_gemm<Dtype>(
          CblasNoTrans, CblasTrans, M_, K_, N_, (Dtype)1.,
          top_diff + top[0]->offset(offset_indices) + g * top_offset, col_data,
          (Dtype)1., weight_diff + g * weight_offset);
    }

    // gradient w.r.t. bottom data, if necessary
    if (propagate_down[0]) {
      // compute first filter group -> col_diff
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_, (Dtype)1.,
                            weight, top_diff + top[0]->offset(offset_indices),
                            (Dtype)0., col_diff);

      // accumulate the other filter groups -> col_diff
      for (int g = 1; g < filter_group_; ++g) {
        caffe_cpu_gemm<Dtype>(
            CblasTrans, CblasNoTrans, K_, N_, M_, (Dtype)1.,
            weight + g * weight_offset,
            top_diff + top[0]->offset(offset_indices) + g * top_offset,
            (Dtype)1., col_diff);
      }

      // vol2im back to the data
      col2vol_cpu(col_diff, channels_, length_, height_, width_, kernel_size_,
                  kernel_depth_, pad_, temporal_pad_, stride_, temporal_stride_,
                  bottom_diff + bottom[0]->offset(offset_indices));
    }
  }
}

INSTANTIATE_CLASS(Convolution3DLayer);

}  // namespace caffe
