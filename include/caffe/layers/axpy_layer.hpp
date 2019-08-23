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

#ifndef INCLUDE_CAFFE_LAYERS_AXPY_LAYER_HPP_
#define INCLUDE_CAFFE_LAYERS_AXPY_LAYER_HPP_
#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
namespace caffe {
/**
 * @brief For reduce memory and time both on training and testing, we combine
 *        channel-wise scale operation and element-wise addition operation
 *        into a single layer called "axpy".
 */
template <typename Dtype>
class AxpyLayer: public Layer<Dtype> {
  public:
  explicit AxpyLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Axpy"; }
  virtual inline int ExactNumBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

  protected:
/**
 * @param Formulation:
 *            F = a * X + Y
 *	  Shape info:
 *            a:  N x C          --> bottom[0]
 *            X:  N x C x H x W  --> bottom[1]
 *            Y:  N x C x H x W  --> bottom[2]
 *            F:  N x C x H x W  --> top[0]
 */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  Blob<Dtype> spatial_sum_multiplier_;
};

}  // namespace caffe
#endif  // INCLUDE_CAFFE_LAYERS_AXPY_LAYER_HPP_
