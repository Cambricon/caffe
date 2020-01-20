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

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/roi_align_layer.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {
template <typename Dtype>
void ROIAlignLayer<Dtype>::pre_calc_for_bilinear_interpolate(
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int iy_upper,
    const int ix_upper,
    Dtype roi_start_h,
    Dtype roi_start_w,
    Dtype bin_size_h,
    Dtype bin_size_w,
    int roi_bin_grid_h,
    int roi_bin_grid_w,
    std::vector<PreCalc> *pre_calc ) {
  int pre_calc_index = 0;
  Dtype half = 0.5;
  Dtype one = 1.0;
  Dtype zero = 0.0;
  for (int ph = 0; ph < pooled_height; ph++) {
    for (int pw = 0; pw < pooled_width; pw++) {
      for (int iy = 0; iy < iy_upper; iy++) {
        const Dtype yy = roi_start_h + ph * bin_size_h +
           (static_cast<Dtype>(iy) + half) * bin_size_h /
            static_cast<Dtype>(roi_bin_grid_h);  // e.g., 0.5, 1.5
        for (int ix = 0; ix < ix_upper; ix++) {
          const Dtype xx = roi_start_w + pw * bin_size_w +
             (static_cast<Dtype>(ix) + half) * bin_size_w /
              static_cast<Dtype>(roi_bin_grid_w);
          Dtype x = xx;
          Dtype y = yy;
          // deal with: inverse elements are out of feature map boundary
          if (y < -1.0 || y > height || x < -1.0 || x > width) {
            // empty
            PreCalc pc;
            pc.pos1 = zero;
            pc.pos2 = zero;
            pc.pos3 = zero;
            pc.pos4 = zero;
            pc.w1 = zero;
            pc.w2 = zero;
            pc.w3 = zero;
            pc.w4 = zero;
            pre_calc->at(pre_calc_index) = pc;
            pre_calc_index += 1;
            continue;
          }

          if (y <= zero) {
            y = zero;
          }
          if (x <= zero) {
            x = zero;
          }

          int y_low = floor(y);
          int x_low = floor(x);
          int y_high;
          int x_high;

          if (y_low >= height - 1) {
            y_high = y_low = height - 1;
            y = (Dtype)y_low;
          } else {
            y_high = ceil(y);
          }

          if (x_low >= width - 1) {
            x_high = x_low = width - 1;
            x = (Dtype)x_low;
          } else {
            x_high = ceil(x);
          }

          Dtype ly = y - static_cast<Dtype>(y_low);
          Dtype lx = x - static_cast<Dtype>(x_low);
          Dtype hy = one - ly, hx = one - lx;
          Dtype w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
          // save weights and indeces
          PreCalc pc;
          pc.pos1 = y_low * width + x_low;
          pc.pos2 = y_low * width + x_high;
          pc.pos3 = y_high * width + x_low;
          pc.pos4 = y_high * width + x_high;
          pc.w1 = w1;
          pc.w2 = w2;
          pc.w3 = w3;
          pc.w4 = w4;
          pre_calc->at(pre_calc_index) = pc;

          pre_calc_index += 1;
        }
      }
    }
  }
}

template <typename Dtype>
void ROIAlignLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
  ROIAlignParameter roi_align_param = this->layer_param_.roi_align_param();
  CHECK_GT(roi_align_param.pooled_h(), 0) << "pooled_h must be > 0";
  CHECK_GT(roi_align_param.pooled_w(), 0) << "pooled_w must be > 0";
  pooled_height_ = roi_align_param.pooled_h();
  pooled_width_ = roi_align_param.pooled_w();
  spatial_scale_ = roi_align_param.spatial_scale();
  sampling_ratio_ = roi_align_param.sampling_ratio();
  CHECK(bottom[1]->num_axes() == 2 || bottom[1]->num_axes() == 4)
      << "Invalid shape: bottom[1]->shape() must be {m, 5} or {1, m, 1, 5}";
  if (bottom[1]->num_axes() == 2) {  // {m, 5}
    rois_num_ = bottom[1]->num();
    CHECK_EQ(bottom[1]->channels(), 5)
        << "Invalid shape: bottom[1]->shape() should be {"
        << rois_num_
        << ", 5}";
  } else {  // {1, m, 1, 5}
    rois_num_ = bottom[1]->channels();
    CHECK_EQ(bottom[1]->width(), 5)
        << "Invalid shape: bottom[1]->shape() should be {1, "
        << rois_num_
        << ", 1, 5}";
  }
  roi_cols_ = 5;
}

template <typename Dtype>
void ROIAlignLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  top[0]->Reshape(rois_num_ * bottom[0]->num(), channels_,
                  pooled_height_, pooled_width_);
}

template <typename Dtype>
void ROIAlignLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_rois = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int top_count = top[0]->count();
  caffe_set(top_count, Dtype(-FLT_MAX), top_data);
  for (int bi = 0; bi < bottom[0]->num(); bi++) {
    for (int n = 0; n < rois_num_; n++) {
      int index_n = n * channels_ * pooled_width_ * pooled_height_
      + bi * rois_num_ * channels_ * pooled_width_ * pooled_height_;
      const Dtype* offset_bottom_rois = bottom_rois + n * roi_cols_
                                      + bi * rois_num_ * roi_cols_;
      int roi_batch_ind = 0;
      roi_batch_ind = offset_bottom_rois[0];
      offset_bottom_rois++;

      // Do not using rounding; this implementation detail is critical
      Dtype roi_start_w = offset_bottom_rois[0] * spatial_scale_;
      Dtype roi_start_h = offset_bottom_rois[1] * spatial_scale_;
      Dtype roi_end_w = offset_bottom_rois[2] * spatial_scale_;
      Dtype roi_end_h = offset_bottom_rois[3] * spatial_scale_;

      // Force malformed ROIs to be 1x1
      Dtype roi_width = max(roi_end_w - roi_start_w, Dtype(1.0));
      Dtype roi_height = max(roi_end_h - roi_start_h, Dtype(1.0));
      Dtype bin_size_h = static_cast<Dtype>(roi_height) /
        static_cast<Dtype>(pooled_height_);
      Dtype bin_size_w = static_cast<Dtype>(roi_width) /
        static_cast<Dtype>(pooled_width_);

      // We use roi_bin_grid to sample the grid and mimic integral
      int roi_bin_grid_h = (sampling_ratio_ > 0) ? sampling_ratio_:
        ceil(roi_height / pooled_height_);  // e.g., = 2
      int roi_bin_grid_w = (sampling_ratio_ > 0) ? sampling_ratio_:
        ceil(roi_width / pooled_width_);  // e.g., = 2

      // We do average (integral) pooling inside a bin
      const Dtype count = roi_bin_grid_h * roi_bin_grid_w;  // e.g. = 4

      // we want to precalculate indeces and weights shared by all chanels,
      // this is the key point of optimiation
      vector<PreCalc> pre_calc(
          roi_bin_grid_h * roi_bin_grid_w * pooled_width_ * pooled_height_);
      pre_calc_for_bilinear_interpolate(
          height_,
          width_,
          pooled_height_,
          pooled_width_,
          roi_bin_grid_h,
          roi_bin_grid_w,
          roi_start_h,
          roi_start_w,
          bin_size_h,
          bin_size_w,
          roi_bin_grid_h,
          roi_bin_grid_w,
          &pre_calc);
      for (int c = 0; c < channels_; c++) {
        int index_n_c = index_n + c * pooled_width_ * pooled_height_;
        const Dtype* offset_bottom_data =
              bottom_data + (roi_batch_ind * channels_ + c) * height_ * width_;
        int pre_calc_index = 0;
        for (int ph = 0; ph < pooled_height_; ph++) {
          for (int pw = 0; pw < pooled_width_; pw++) {
            int index = index_n_c + ph * pooled_width_ + pw;
            Dtype output_val = 0.0;
            for (int iy = 0; iy < roi_bin_grid_h; iy++) {
              for (int ix = 0; ix < roi_bin_grid_w; ix++) {
                PreCalc pc = pre_calc[pre_calc_index];
                output_val += pc.w1 * offset_bottom_data[pc.pos1] +
                              pc.w2 * offset_bottom_data[pc.pos2] +
                              pc.w3 * offset_bottom_data[pc.pos3] +
                              pc.w4 * offset_bottom_data[pc.pos4];
                pre_calc_index += 1;
              }
            }
            output_val /= count;
            top_data[index] = output_val;
          }  // for pw
        }  // for ph
      }   // for c
    }    // for n
  }     // for bi
}

template <typename Dtype>
void ROIAlignLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

STUB_GPU(ROIAlignLayer);

INSTANTIATE_CLASS(ROIAlignLayer);

}  // namespace caffe
