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

#include <algorithm>
#include <vector>

#include "caffe/layers/crelu_layer.hpp"

namespace caffe {
template <typename Dtype>
void CReLULayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_NE(bottom[0], top[0]) << "CReLU doesn't support inplace computation!"
      << " Please check prototxt!";
  negative_slope_ = this->layer_param_.crelu_param().negative_slope();
  concat_axis_ = bottom[0]->
      CanonicalAxisIndex(this->layer_param_.crelu_param().concat_axis());
  CHECK_LT(concat_axis_, bottom[0]->num_axes()) << "concat axis out of range.";
}

template <typename Dtype>
void CReLULayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape = bottom[0]->shape();
  top_shape[concat_axis_] = 2 * bottom[0]->shape(concat_axis_);
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
inline Dtype ReLU(Dtype x, Dtype negative_slope) {
  return std::max(x, Dtype(0)) + negative_slope * std::min(x, Dtype(0));
}

template <typename Dtype>
void CReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count(concat_axis_);
  for (int i = 0; i < bottom[0]->count(0, concat_axis_); ++i) {
    for (int j = 0; j < count; ++j) {
      top_data[2*i*count + j] = ReLU(bottom_data[i * count + j], negative_slope_);
      top_data[(2*i+1)*count + j] = ReLU(-bottom_data[i * count + j], negative_slope_);
    }
  }
}

INSTANTIATE_CLASS(CReLULayer);
// REGISTER_LAYER_CLASS(CReLU);
}  // namespace caffe
