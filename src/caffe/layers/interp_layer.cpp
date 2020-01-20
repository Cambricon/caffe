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
#include "caffe/layer.hpp"
#include "caffe/layers/interp_layer.hpp"
#include "caffe/util/interp.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void InterpLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top) {
  InterpParameter interp_param = this->layer_param_.interp_param();
  pad_beg_ = interp_param.pad_beg();
  pad_end_ = interp_param.pad_end();
  CHECK_LE(pad_beg_, 0) << "Only supports non-pos padding (cropping) for now";
  CHECK_LE(pad_end_, 0) << "Only supports non-pos padding (cropping) for now";
}

template <typename Dtype>
void InterpLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top) {
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_in_ = bottom[0]->height();
  width_in_ = bottom[0]->width();
  height_in_eff_ = height_in_ + pad_beg_ + pad_end_;
  width_in_eff_ = width_in_ + pad_beg_ + pad_end_;

  InterpParameter interp_param = this->layer_param_.interp_param();
  if (interp_param.has_shrink_factor() && !interp_param.has_zoom_factor()) {
    const int shrink_factor = interp_param.shrink_factor();
    CHECK_GE(shrink_factor, 1) << "Shrink factor must be positive";
    height_out_ = (height_in_eff_ - 1) / shrink_factor + 1;
    width_out_ = (width_in_eff_ - 1) / shrink_factor + 1;
  } else if (interp_param.has_zoom_factor() && !interp_param.has_shrink_factor()) {
    const int zoom_factor = interp_param.zoom_factor();
    CHECK_GE(zoom_factor, 1) << "Zoom factor must be positive";
    height_out_ = height_in_eff_ + (height_in_eff_ - 1) * (zoom_factor - 1);
    width_out_ = width_in_eff_ + (width_in_eff_ - 1) * (zoom_factor - 1);
  } else if (interp_param.has_height() && interp_param.has_width()) {
    height_out_  = interp_param.height();
    width_out_  = interp_param.width();
  } else if (interp_param.has_shrink_factor() && interp_param.has_zoom_factor()) {
    const int shrink_factor = interp_param.shrink_factor();
    const int zoom_factor = interp_param.zoom_factor();
    CHECK_GE(shrink_factor, 1) << "Shrink factor must be positive";
    CHECK_GE(zoom_factor, 1) << "Zoom factor must be positive";
    height_out_ = (height_in_eff_ - 1) / shrink_factor + 1;
    width_out_ = (width_in_eff_ - 1) / shrink_factor + 1;
    height_out_ = height_out_ + (height_out_ - 1) * (zoom_factor - 1);
    width_out_ = width_out_ + (width_out_ - 1) * (zoom_factor - 1);
  } else {
    LOG(FATAL);
  }
  CHECK_GT(height_in_eff_, 0) << "height should be positive";
  CHECK_GT(width_in_eff_, 0) << "width should be positive";
  CHECK_GT(height_out_, 0) << "height should be positive";
  CHECK_GT(width_out_, 0) << "width should be positive";

  top[0]->Reshape(num_, channels_, height_out_, width_out_);
}

template <typename Dtype>
void InterpLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top) {
  caffe_cpu_interp2<Dtype>(num_ * channels_,
                           bottom[0]->cpu_data(),
                           - pad_beg_,
                           - pad_beg_,
                           height_in_eff_,
                           width_in_eff_,
                           height_in_,
                           width_in_,
                           top[0]->mutable_cpu_data(),
                           0,
                           0,
                           height_out_,
                           width_out_,
                           height_out_,
                           width_out_,
                           false);
}

template <typename Dtype>
void InterpLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                      const vector<bool>& propagate_down,
                                      const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  caffe_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_cpu_diff());
  caffe_cpu_interp2_backward<Dtype>(num_ * channels_,
                                    bottom[0]->mutable_cpu_diff(),
                                    - pad_beg_,
                                    - pad_beg_,
                                    height_in_eff_,
                                    width_in_eff_,
                                    height_in_,
                                    width_in_,
                                    top[0]->cpu_diff(),
                                    0,
                                    0,
                                    height_out_,
                                    width_out_,
                                    height_out_,
                                    width_out_,
                                    false);
}

INSTANTIATE_CLASS(InterpLayer);

}  // namespace caffe
