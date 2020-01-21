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

#ifdef USE_MLU
#include <algorithm>
#include <vector>
#include "caffe/layers/mlu_prior_box_layer.hpp"

namespace caffe {

template <typename Dtype>
void MLUPriorBoxLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
  PriorBoxLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void MLUPriorBoxLayer<Dtype>::Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
  const int layer_width = bottom[0]->width();
  const int layer_height = bottom[0]->height();
  vector<int> top_shape(3, 1);
  // Since all images in a batch has same height and width, we only need to
  // generate one set of priors which can be shared across all images.
  top_shape[0] = 1;
  // 2 channels. First channel stores the mean of each prior coordinate.
  // Second channel stores the variance of each prior coordinate.
  top_shape[1] = 2;
  top_shape[2] = layer_width * layer_height * this->num_priors_ * 4;
  CHECK_GT(top_shape[2], 0);
  // top[0]->Reshape(top_shape);
  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
  BaseDataType mlu_dtype = bottom[0]->mlu_type();
  top[0]->Reshape(top_shape,
                  cpu_dtype,
                  mlu_dtype,
                  CNML_TENSOR);
  priorbox_blob_.Reshape(top_shape,
                  cpu_dtype,
                  mlu_dtype,
                  CNML_CONST);
}

template <typename Dtype>
void MLUPriorBoxLayer<Dtype>::MLUCreateOpBindData(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // copied from forward_cpu
  const int layer_width = bottom[0]->width();
  const int layer_height = bottom[0]->height();
  int img_width, img_height;
  if (this->img_h_ == 0 || this->img_w_ == 0) {
    img_width = bottom[1]->width();
    img_height = bottom[1]->height();
  } else {
    img_width = this->img_w_;
    img_height = this->img_h_;
  }
  float step_w, step_h;
  if (this->step_w_ == 0 || this->step_h_ == 0) {
    step_w = static_cast<float>(img_width) / layer_width;
    step_h = static_cast<float>(img_height) / layer_height;
  } else {
    step_w = this->step_w_;
    step_h = this->step_h_;
  }
  Dtype* top_data = priorbox_blob_.mutable_cpu_data();
  int dim = layer_height * layer_width * this->num_priors_ * 4;
  int idx = 0;
  for (int h = 0; h < layer_height; ++h) {
    for (int w = 0; w < layer_width; ++w) {
      float center_x = (w + this->offset_) * step_w;
      float center_y = (h + this->offset_) * step_h;
      float box_width, box_height;
      for (int s = 0; s < this->min_sizes_.size(); ++s) {
        int min_size_ = this->min_sizes_[s];
        // first prior: aspect_ratio = 1, size = min_size
        box_width = box_height = min_size_;
        // xmin
        top_data[idx++] = (center_x - box_width / 2.) / img_width;
        // ymin
        top_data[idx++] = (center_y - box_height / 2.) / img_height;
        // xmax
        top_data[idx++] = (center_x + box_width / 2.) / img_width;
        // ymax
        top_data[idx++] = (center_y + box_height / 2.) / img_height;

        if (this->max_sizes_.size() > 0) {
          CHECK_EQ(this->min_sizes_.size(), this->max_sizes_.size());
          int max_size_ = this->max_sizes_[s];
          if (this->p_type ==
            PriorBoxLayer<Dtype>::PriorType::CLASSICAL_PRIOR) {
            // second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
            box_width = box_height = sqrt(min_size_ * max_size_);
            // xmin
            top_data[idx++] = (center_x - box_width / 2.) / img_width;
            // ymin
            top_data[idx++] = (center_y - box_height / 2.) / img_height;
            // xmax
            top_data[idx++] = (center_x + box_width / 2.) / img_width;
            // ymax
            top_data[idx++] = (center_y + box_height / 2.) / img_height;
          } else if (
            this->p_type ==  PriorBoxLayer<Dtype>::PriorType::DENSE_PRIOR) {
            // intermediate size
            for (int scale_i = 1; scale_i < this->inner_scale; scale_i++) {
              float temp_size = min_size_ * \
                 pow((max_size_ * max_size_) / (min_size_ * min_size_),
                 static_cast<float>(scale_i)/(this->inner_scale * 2));
              // add together normal ratio(=1) to what has been set by users
              bool has_normal_ratio_ = false;
              for (int r = -1; r < this->aspect_ratios_.size(); ++r) {
                float ar = (r == -1 ? 1 : this->aspect_ratios_[r]);
                if (fabs(ar - 1.) < 1e-6 && has_normal_ratio_) {
                  continue;
                } else {
                  has_normal_ratio_ = true;
                }
                box_width = temp_size * sqrt(ar);
                box_height = temp_size / sqrt(ar);
                // xmin
                top_data[idx++] = (center_x - box_width / 2.) / img_width;
                // ymin
                top_data[idx++] = (center_y - box_height / 2.) / img_height;
                // xmax
                top_data[idx++] = (center_x + box_width / 2.) / img_width;
                // ymax
                top_data[idx++] = (center_y + box_height / 2.) / img_height;
              }
            }
          }
        }
        // rest of priors
        for (int r = 0; r < this->aspect_ratios_.size(); ++r) {
          float ar = this->aspect_ratios_[r];
          if (fabs(ar - 1.) < 1e-6) {
            continue;
          }
          box_width = min_size_ * sqrt(ar);
          box_height = min_size_ / sqrt(ar);
          // xmin
          top_data[idx++] = (center_x - box_width / 2.) / img_width;
          // ymin
          top_data[idx++] = (center_y - box_height / 2.) / img_height;
          // xmax
          top_data[idx++] = (center_x + box_width / 2.) / img_width;
          // ymax
          top_data[idx++] = (center_y + box_height / 2.) / img_height;
        }
      }
    }
  }
  // clip the prior's coordidate such that it is within [0, 1]
  if (this->clip_) {
    for (int d = 0; d < dim; ++d) {
      top_data[d] = std::min<Dtype>(std::max<Dtype>(top_data[d], 0.), 1.);
    }
  }
  // set the variance.
  top_data += top[0]->offset(0, 1);
  if (this->variance_.size() == 1) {
    caffe_set<Dtype>(dim, Dtype(this->variance_[0]), top_data);
  } else {
    int count = 0;
    for (int h = 0; h < layer_height; ++h) {
      for (int w = 0; w < layer_width; ++w) {
        for (int i = 0; i < this->num_priors_; ++i) {
          for (int j = 0; j < 4; ++j) {
            top_data[count] = this->variance_[j];
            ++count;
          }
        }
      }
    }
  }
  // priorbox const tensor
  MLU_CHECK(cnmlBindConstData_V2(priorbox_blob_.mlu_tensor(),
                                 reinterpret_cast<void*>(priorbox_blob_.sync_data()),
                                 false));

  MLU_CHECK(cnmlCreateDeviceMemcpyOp(&device_copy_op_ptr_,
                                      priorbox_blob_.mlu_tensor(),
                                      top[0]->mlu_tensor()));
}

template <typename Dtype>
void MLUPriorBoxLayer<Dtype>::MLUCompileOp() {
  MLU_CHECK(cnmlCompileBaseOp(device_copy_op_ptr_,
                              Caffe::rt_core(),
                              Caffe::core_number()));
}

// Attention: Prior should not be fused alone
template <typename Dtype>
void MLUPriorBoxLayer<Dtype>::fuse(MFusion<Dtype>* fuser) {
  fuser->fuse(device_copy_op_ptr_);
}
template <typename Dtype>
void MLUPriorBoxLayer<Dtype>::MLUDestroyOp() {
}

template <typename Dtype>
MLUPriorBoxLayer<Dtype>::~MLUPriorBoxLayer() {
  MLUDestroyOp();
}

template <typename Dtype>
void MLUPriorBoxLayer<Dtype>::Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  MLU_CHECK(cnmlComputeDeviceMemcpyOpForward_V3(device_copy_op_ptr_,
                                                nullptr,
                                                top[0]->mutable_mlu_data(),
                                                Caffe::forward_param(),
                                                Caffe::queue()));
}

INSTANTIATE_CLASS(MLUPriorBoxLayer);

}  // namespace caffe
#endif  // USE_MLU
