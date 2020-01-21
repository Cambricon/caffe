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

#include <cfloat>
#include <algorithm>

#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/psroi_pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {
template <typename Dtype>
void PSROIPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  PSROIPoolingParameter psroi_pooling_param =
    this->layer_param_.psroi_pooling_param();
  spatial_scale_ = psroi_pooling_param.spatial_scale();
  // LOG(INFO) << "Spatial scale: " << spatial_scale_;
  CHECK_GT(psroi_pooling_param.output_dim(), 0)
    << "output_dim must be > 0";
  CHECK_GT(psroi_pooling_param.group_size(), 0)
    << "group_size must be > 0";
  output_dim_ = psroi_pooling_param.output_dim();
  group_size_ = psroi_pooling_param.group_size();
}

template <typename Dtype>
void PSROIPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  LOG(INFO) << "psroipooling reshape";
  int channels = bottom[0]->channels();
  CHECK_EQ(channels, output_dim_*group_size_*group_size_)
    << "input channel number does not match layer parameters";
  top[0]->Reshape(
    bottom[1]->num(), output_dim_, group_size_, group_size_);
  mapping_channel_.Reshape(
    bottom[1]->num(), output_dim_, group_size_, group_size_);
}

template <typename Dtype>
void PSROIPoolingLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_rois = bottom[1]->cpu_data();
  int rois_num = bottom[1]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  int pooled_height = group_size_;
  int pooled_width = group_size_;
  int* mapping_channel = mapping_channel_.mutable_cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int count = top[0]->count();
  caffe_set(count, Dtype(0), top_data);
  caffe_set(count, -1, mapping_channel);
  for (int n = 0; n < rois_num; ++n) {
    int roi_add = n*5;
    // [start, end) interval for spatial sampling
    int roi_batch_ind = bottom_rois[roi_add];
    Dtype roi_start_w =
        static_cast<Dtype>(round(bottom_rois[roi_add + 1])) * spatial_scale_;
    Dtype roi_start_h =
        static_cast<Dtype>(round(bottom_rois[roi_add + 2])) * spatial_scale_;
    Dtype roi_end_w =
        static_cast<Dtype>(round(bottom_rois[roi_add + 3])
                           + 1.) * spatial_scale_;
    Dtype roi_end_h =
        static_cast<Dtype>(round(bottom_rois[roi_add + 4])
                           + 1.) * spatial_scale_;

    // Force too small ROIs to be 1x1
    Dtype roi_width = max<Dtype>(roi_end_w - roi_start_w, 0.1);  // avoid 0
    Dtype roi_height = max<Dtype>(roi_end_h - roi_start_h, 0.1);

    // Compute w and h at bottom, prepare pooling in rois
    Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);
    Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);
    // Suppose rois anchor is feathure map, pooling in the rois feathure map
    for (int ctop = 0; ctop < output_dim_; ++ctop) {
      for (int ph = 0; ph < pooled_height; ++ph) {
        for (int pw = 0; pw < pooled_width; ++pw) {
          // The output is in order (n, ctop, ph, pw)
          int index = n*output_dim_*pooled_height*pooled_width
                      + ctop*pooled_height*pooled_width + ph*pooled_width + pw;
          int hstart = floor(static_cast<Dtype>(ph) * bin_size_h
                          + roi_start_h);
          int wstart = floor(static_cast<Dtype>(pw)* bin_size_w
                          + roi_start_w);
          int hend = ceil(static_cast<Dtype>(ph + 1) * bin_size_h
                          + roi_start_h);
          int wend = ceil(static_cast<Dtype>(pw + 1) * bin_size_w
                      + roi_start_w);
          // Add roi offsets and clip to input boundaries
          hstart = min(max(hstart, 0), height);
          hend = min(max(hend, 0), height);
          wstart = min(max(wstart, 0), width);
          wend = min(max(wend, 0), width);
          // bottom_rois may give locs that is not a anchor, then it's empty
          bool is_empty = (hend <= hstart) || (wend <= wstart);
          int gw = pw;
          int gh = ph;
          int c = (ctop*group_size_ + gh)*group_size_ + gw;
          // sum the data in the pooling group and get average pooling
          Dtype out_sum = 0;
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              int bottom_index = h*width + w;
              out_sum += bottom_data[(roi_batch_ind * channels + c)
                                      * height * width + bottom_index];
            }
          }
          Dtype bin_area = (hend - hstart)*(wend - wstart);
          if (is_empty) {
            top_data[index] = 0;
          } else {
            top_data[index] = out_sum/bin_area;
          }
          mapping_channel[index] = c;
        }
      }
    }
  }
}

template <typename Dtype>
void PSROIPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
NOT_IMPLEMENTED;
}


STUB_GPU(PSROIPoolingLayer);

INSTANTIATE_CLASS(PSROIPoolingLayer);
}  // namespace caffe
