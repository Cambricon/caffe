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

#include <vector>
#include "caffe/layers/reorg_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template<typename Dtype>
void ReorgLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                   const vector<Blob<Dtype> *> &top) {
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
              "allow in-place computation.";
  ReorgParameter reorg_param = this->layer_param_.reorg_param();
  CHECK_EQ(reorg_param.has_stride(), true) << this->type()
                                   << " Layer needs stride param.";
  reverse_ = reorg_param.reverse();
  stride_ = reorg_param.stride();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  batch_num_ = bottom[0]->num();
  CHECK_GT(stride_, 0);
  CHECK_EQ(channels_ % (stride_ * stride_), 0);
  CHECK_EQ(height_ % stride_, 0);
  CHECK_EQ(width_ % stride_, 0);
  diff_.Reshape(batch_num_, channels_, height_, width_);
  if (reverse_) {
      reorged_channels_ = channels_ / (stride_ * stride_);
      reorged_width_ = width_ * stride_;
      reorged_height_ = height_ * stride_;
  } else {
      reorged_channels_ = channels_ * stride_ * stride_;
      reorged_height_ = height_ / stride_;
      reorged_width_ = width_ / stride_;
  }
}

template<typename Dtype>
void ReorgLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                const vector<Blob<Dtype> *> &top) {
  top[0]->Reshape(batch_num_, reorged_channels_,
                  reorged_height_, reorged_width_);
}

template<typename Dtype>
void ReorgLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                    const vector<Blob<Dtype> *> &top) {
  const Dtype *bottom_data = bottom[0]->cpu_data();
  Dtype *top_data = top[0]->mutable_cpu_data();
  reorg_cpu(bottom_data, width_, height_,
            channels_, batch_num_, stride_, reverse_, top_data);
}

template<typename Dtype>
void ReorgLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
                                     const vector<bool> &propagate_down,
                                     const vector<Blob<Dtype> *> &bottom) {
  if (!propagate_down[0]) {
      return;
  }
  const Dtype *top_diff = diff_.mutable_cpu_diff();
  Dtype *bottom_diff = bottom[0]->mutable_cpu_diff();
  reorg_cpu(top_diff, width_, height_,
            channels_, batch_num_, stride_, !reverse_, bottom_diff);
}

STUB_GPU(ReorgLayer);
INSTANTIATE_CLASS(ReorgLayer);

}  // namespace caffe
