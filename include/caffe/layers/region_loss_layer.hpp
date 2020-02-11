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

#ifndef INCLUDE_CAFFE_LAYERS_REGION_LOSS_LAYER_HPP_
#define INCLUDE_CAFFE_LAYERS_REGION_LOSS_LAYER_HPP_

#include <map>
#include <string>
#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/loss_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/tree.hpp"

namespace caffe {
template <typename Dtype>
Dtype Overlap(Dtype x1, Dtype w1, Dtype x2, Dtype w2);

template <typename Dtype>
Dtype Calc_iou(const vector<Dtype>& box, const vector<Dtype>& truth);

template <typename Dtype>
void disp(Blob<Dtype>* swap);

template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 1. / (1. + exp(-x));
}

template <typename Dtype>
Dtype softmax_region(Dtype* input, int classes);
// template <typename Dtype>
// Dtype softmax_region(Dtype* input, int n, float temp, Dtype* output);

// template <typename Dtype>
// Dtype* flatten(Dtype* input_data, int size, int channels, int batch, int
// forward);
template <typename Dtype>
void softmax_tree(Dtype* input, tree* t);

template <typename Dtype>
Dtype get_hierarchy_prob(Dtype* input_data, tree* t, int c);

template <typename Dtype>
vector<Dtype> get_region_box(Dtype* x, vector<Dtype> biases, int n, int index,
                             int i, int j, int w, int h);

template <typename Dtype>
Dtype delta_region_box(vector<Dtype> truth, Dtype* x, vector<Dtype> biases,
                       int n, int index, int i, int j, int w, int h,
                       Dtype* delta, float scale);

template <typename Dtype>
void delta_region_class(Dtype* input_data, Dtype* diff, int index,
                        int class_label, int classes, string softmax_tree,
                        tree* t, float scale, Dtype* avg_cat);

template <typename Dtype>
class RegionLossLayer : public LossLayer<Dtype> {
  public:
  explicit RegionLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "RegionLoss"; }

  protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);

  int side_;
  int bias_match_;
  int num_class_;
  int coords_;
  int num_;
  int softmax_;
  string softmax_tree_;
  float jitter_;
  int rescore_;

  float object_scale_;
  float class_scale_;
  float noobject_scale_;
  float coord_scale_;

  int absolute_;
  float thresh_;
  int random_;
  vector<Dtype> biases_;

  Blob<Dtype> diff_;
  Blob<Dtype> real_diff_;
  tree t_;

  string class_map_;
  map<int, int> cls_map_;
};

}  // namespace caffe

#endif  // INCLUDE_CAFFE_LAYERS_REGION_LOSS_LAYER_HPP_
