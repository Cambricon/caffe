/*
All modification made by Cambricon Corporation: © 2018--2019 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2018, the respective contributors
All rights reserved.

Copyright Cambricon Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "caffe/layers/axpy_layer.hpp"
namespace caffe {
template <typename Dtype>
void AxpyLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0));
  CHECK_EQ(bottom[0]->shape(1), bottom[1]->shape(1));
  if (bottom[0]->num_axes() == 4) {
    CHECK_EQ(bottom[0]->shape(2), 1);
    CHECK_EQ(bottom[0]->shape(3), 1);
  }
  CHECK(bottom[1]->shape() == bottom[2]->shape());
  top[0]->ReshapeLike(*bottom[1]);
  int spatial_dim = bottom[1]->count(2);
  if (spatial_sum_multiplier_.count() < spatial_dim) {
    spatial_sum_multiplier_.Reshape(vector<int>(1, spatial_dim));
    caffe_set(spatial_dim, Dtype(1),
        spatial_sum_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void AxpyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int channel_dim = bottom[1]->channels();
  int spatial_dim = bottom[1]->count(2);
  const Dtype* scale_data = bottom[0]->cpu_data();
  const Dtype* x_data = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_copy(bottom[2]->count(), bottom[2]->cpu_data(), top_data);
  for (int n = 0; n < bottom[1]->num(); ++n) {
    for (int c = 0; c < channel_dim; ++c) {
      int scale_offset = n * channel_dim + c;
      caffe_axpy(spatial_dim, scale_data[scale_offset],
          x_data + scale_offset * spatial_dim,
          top_data + scale_offset * spatial_dim);
    }
  }
}

template <typename Dtype>
void AxpyLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int count = top[0]->count();
  const Dtype* top_diff = top[0]->cpu_diff();
  if (propagate_down[0]) {
    int spatial_dim = bottom[1]->count(2);
    const Dtype* x_data = bottom[1]->cpu_data();
    Dtype* x_diff = bottom[1]->mutable_cpu_diff();
    Dtype* scale_diff = bottom[0]->mutable_cpu_diff();
    caffe_mul(count, top_diff, x_data, x_diff);
    caffe_set(bottom[0]->count(), Dtype(0), scale_diff);
    caffe_cpu_gemv(CblasNoTrans, bottom[0]->count(), spatial_dim, Dtype(1),
        x_diff, spatial_sum_multiplier_.cpu_data(), Dtype(1), scale_diff);
    if (!propagate_down[1]) {
      caffe_set(bottom[1]->count(), Dtype(0), x_diff);
    }
  }
  if (propagate_down[0]) {
    int channel_dim = bottom[1]->channels();
    int spatial_dim = bottom[1]->count(2);
    const Dtype* scale_data = bottom[0]->cpu_data();
    Dtype* x_diff = bottom[1]->mutable_cpu_diff();
    for (int n = 0; n < bottom[1]->num(); ++n) {
      for (int c = 0; c < channel_dim; ++c) {
        int scale_offset = n * channel_dim + c;
        caffe_cpu_scale(spatial_dim, scale_data[scale_offset],
            top_diff + scale_offset * spatial_dim,
            x_diff + scale_offset * spatial_dim);
      }
    }
  }
  if (propagate_down[2]) {
    caffe_copy(count, top_diff, bottom[2]->mutable_cpu_diff());
  }
}

STUB_GPU(AxpyLayer);
INSTANTIATE_CLASS(AxpyLayer);

}  // namespace caffe
