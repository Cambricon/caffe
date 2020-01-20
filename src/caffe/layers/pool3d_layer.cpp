/*
All modification made by Cambricon Corporation: © 2018-2019 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2019, the respective contributors
All rights reserved.
For the list of contributors go to
https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md
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

#include "caffe/layer.hpp"
#include "caffe/layers/pool3d_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::max;
using std::min;

template <typename Dtype>
void Pooling3DLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), 1) << "Pooling3DLayer takes a single blob as input.";
  CHECK_EQ(top.size(), 1) << "Pooling3DLayer takes a single blob as output.";
  kernel_size_ = this->layer_param_.pooling3d_param().kernel_size();
  kernel_depth_ = this->layer_param_.pooling3d_param().kernel_depth();
  stride_ = this->layer_param_.pooling3d_param().stride();
  temporal_stride_ = this->layer_param_.pooling3d_param().temporal_stride();
  pad_ = this->layer_param_.pooling3d_param().pad();

  if (pad_ != 0) {
    CHECK_EQ(this->layer_param_.pooling3d_param().pool(),
             Pooling3DParameter_PoolMethod_AVE)
        << "Padding implemented only for average pooling.";
  }

  channels_ = bottom[0]->shape(1);
  length_ = bottom[0]->shape(2);
  height_ = bottom[0]->shape(3);
  width_ = bottom[0]->shape(4);

  pooled_height_ =
      static_cast<int>(ceil(
          static_cast<float>(height_ + 2 * pad_ - kernel_size_) / stride_)) +
      1;
  pooled_width_ =
      static_cast<int>(ceil(
          static_cast<float>(width_ + 2 * pad_ - kernel_size_) / stride_)) +
      1;
  pooled_length_ =
      static_cast<int>(ceil(static_cast<float>(length_ - kernel_depth_) /
                            temporal_stride_)) +
      1;

  vector<int> top_shape(5);
  top_shape[0] = bottom[0]->shape(0);
  top_shape[1] = channels_;
  top_shape[2] = pooled_length_;
  top_shape[3] = pooled_height_;
  top_shape[4] = pooled_width_;

  top[0]->Reshape(top_shape);
  // top[0]->Reshape(bottom[0]->num(), channels_, pooled_length_,
  // pooled_height_, pooled_width_);

  // If stochastic pooling, we will initialize the random index part.
  if (this->layer_param_.pooling3d_param().pool() ==
      Pooling3DParameter_PoolMethod_STOCHASTIC) {
    rand_idx_.Reshape(top_shape);
    // rand_idx_.Reshape(bottom[0]->num(), channels_, pooled_length_,
    // pooled_height_, pooled_width_);
  }
}

template <typename Dtype>
void Pooling3DLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  static const int index_array[] = {0, 1, 0, 0, 0};
  vector<int> offset_indices(index_array, index_array + 5);

  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  int top_count = top[0]->count();
  switch (this->layer_param_.pooling3d_param().pool()) {
    case Pooling3DParameter_PoolMethod_MAX:
      // Initialize
      for (int i = 0; i < top_count; ++i) {
        top_data[i] = Dtype(-FLT_MAX);
      }
      // The main loop
      for (int n = 0; n < bottom[0]->shape(0); ++n) {
        for (int c = 0; c < channels_; ++c) {
          for (int pl = 0; pl < pooled_length_; ++pl) {
            for (int ph = 0; ph < pooled_height_; ++ph) {
              for (int pw = 0; pw < pooled_width_; ++pw) {
                int hstart = ph * stride_;
                int wstart = pw * stride_;
                int lstart = pl * temporal_stride_;
                int hend = min(hstart + kernel_size_, height_);
                int wend = min(wstart + kernel_size_, width_);
                int lend = min(lstart + kernel_depth_, length_);
                for (int l = lstart; l < lend; ++l) {
                  for (int h = hstart; h < hend; ++h) {
                    for (int w = wstart; w < wend; ++w) {
                      top_data[(pl * pooled_height_ + ph) * pooled_width_ +
                               pw] =
                          max(top_data[(pl * pooled_height_ + ph) *
                                           pooled_width_ +
                                       pw],
                              bottom_data[(l * height_ + h) * width_ + w]);
                    }
                  }
                }
              }
            }
          }
          // compute offset
          bottom_data += bottom[0]->offset(offset_indices);
          top_data += top[0]->offset(offset_indices);
        }
      }
      break;
    case Pooling3DParameter_PoolMethod_AVE:
      for (int i = 0; i < top_count; ++i) {
        top_data[i] = 0;
      }
      // The main loop
      for (int n = 0; n < bottom[0]->shape(0); ++n) {
        for (int c = 0; c < channels_; ++c) {
          for (int pl = 0; pl < pooled_length_; ++pl) {
            for (int ph = 0; ph < pooled_height_; ++ph) {
              for (int pw = 0; pw < pooled_width_; ++pw) {
                int hstart = ph * stride_ - pad_;
                int wstart = pw * stride_ - pad_;
                int lstart = pl * temporal_stride_;
                int hend = min(hstart + kernel_size_, height_ + pad_);
                int wend = min(wstart + kernel_size_, width_ + pad_);
                int lend = min(lstart + kernel_depth_, length_);
                int pool_size =
                    (hend - hstart) * (wend - wstart) * (lend - lstart);
                hstart = max(hstart, 0);
                wstart = max(wstart, 0);

                hend = min(hend, height_);
                wend = min(wend, width_);
                lend = min(lend, length_);
                for (int l = lstart; l < lend; ++l) {
                  for (int h = hstart; h < hend; ++h) {
                    for (int w = wstart; w < wend; ++w) {
                      top_data[(pl * pooled_height_ + ph) * pooled_width_ +
                               pw] +=
                          bottom_data[(l * height_ + h) * width_ + w];
                    }
                  }
                }
                top_data[(pl * pooled_height_ + ph) * pooled_width_ + pw] /=
                    pool_size;
              }
            }
          }
          // compute offset
          bottom_data += bottom[0]->offset(offset_indices);
          top_data += top[0]->offset(offset_indices);
        }
      }
      break;
    case Pooling3DParameter_PoolMethod_STOCHASTIC:
      NOT_IMPLEMENTED;
      break;
    default:
      LOG(FATAL) << "Unknown pooling method.";
  }
}

template <typename Dtype>
void Pooling3DLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                         const vector<bool>& propagate_down,
                                         const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  static const int index_array[] = {0, 1, 0, 0, 0};
  vector<int> offset_indices(index_array, index_array + 5);

  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  memset(bottom_diff, 0, bottom[0]->count() * sizeof(Dtype));
  switch (this->layer_param_.pooling3d_param().pool()) {
    case Pooling3DParameter_PoolMethod_MAX:
      // The main loop
      for (int n = 0; n < top[0]->shape(0); ++n) {
        for (int c = 0; c < channels_; ++c) {
          for (int pl = 0; pl < pooled_length_; ++pl) {
            for (int ph = 0; ph < pooled_height_; ++ph) {
              for (int pw = 0; pw < pooled_width_; ++pw) {
                int hstart = ph * stride_;
                int wstart = pw * stride_;
                int lstart = pl * temporal_stride_;
                int hend = min(hstart + kernel_size_, height_);
                int wend = min(wstart + kernel_size_, width_);
                int lend = min(lstart + kernel_depth_, length_);
                for (int l = lstart; l < lend; ++l) {
                  for (int h = hstart; h < hend; ++h) {
                    for (int w = wstart; w < wend; ++w) {
                      bottom_diff[(l * height_ + h) * width_ + w] +=
                          top_diff[(pl * pooled_height_ + ph) * pooled_width_ +
                                   pw] *
                          (bottom_data[(l * height_ + h) * width_ + w] ==
                           top_data[(pl * pooled_height_ + ph) * pooled_width_ +
                                    pw]);
                    }
                  }
                }
              }
            }
          }
          // offset
          bottom_data += bottom[0]->offset(offset_indices);
          top_data += top[0]->offset(offset_indices);
          bottom_diff += bottom[0]->offset(offset_indices);
          top_diff += top[0]->offset(offset_indices);
        }
      }
      break;
    case Pooling3DParameter_PoolMethod_AVE:
      // The main loop0, 1
      for (int n = 0; n < top[0]->shape(0); ++n) {
        for (int c = 0; c < channels_; ++c) {
          for (int pl = 0; pl < pooled_length_; ++pl) {
            for (int ph = 0; ph < pooled_height_; ++ph) {
              for (int pw = 0; pw < pooled_width_; ++pw) {
                int hstart = ph * stride_ - pad_;
                int wstart = pw * stride_ - pad_;
                int lstart = pl * temporal_stride_;
                int hend = min(hstart + kernel_size_, height_ + pad_);
                int wend = min(wstart + kernel_size_, width_ + pad_);
                int lend = min(lstart + kernel_depth_, length_);
                int pool_size =
                    (hend - hstart) * (wend - wstart) * (lend - lstart);
                hstart = max(hstart, 0);
                wstart = max(wstart, 0);
                hend = min(hend, height_);
                wend = min(wend, width_);
                lend = min(lend, length_);
                for (int l = lstart; l < lend; ++l) {
                  for (int h = hstart; h < hend; ++h) {
                    for (int w = wstart; w < wend; ++w) {
                      bottom_diff[(l * height_ + h) * width_ + w] +=
                          top_diff[(pl * pooled_height_ + ph) * pooled_width_ +
                                   pw] /
                          pool_size;
                    }
                  }
                }
              }
            }
          }
          // offset
          bottom_data += bottom[0]->offset(offset_indices);
          top_data += top[0]->offset(offset_indices);
          bottom_diff += bottom[0]->offset(offset_indices);
          top_diff += top[0]->offset(offset_indices);
        }
      }
      break;
    case Pooling3DParameter_PoolMethod_STOCHASTIC:
      NOT_IMPLEMENTED;
      break;
    default:
      LOG(FATAL) << "Unknown pooling method.";
  }
}

INSTANTIATE_CLASS(Pooling3DLayer);

}  // namespace caffe
