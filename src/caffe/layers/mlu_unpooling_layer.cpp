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

#ifdef USE_MLU
#include "caffe/layers/mlu_unpooling_layer.hpp"
#include <algorithm>
#include <sstream>
#include <vector>

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void MLUUnPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  UnPoolingLayer<Dtype>::LayerSetUp(bottom, top);
  mlu_mask_ = new Blob<Dtype>();
}

template <typename Dtype>
void MLUUnPoolingLayer<Dtype>::FillMask() {
  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
  mlu_mask_->Reshape(1, 1, this->unpooled_height_, this->unpooled_width_,
                     cpu_dtype, DT_FLOAT16, CNML_CONST);
  Dtype* mask = mlu_mask_->mutable_cpu_data();
  switch (this->layer_param_.unpooling_param().unpool()) {
    case UnPoolingParameter_UnPoolMethod_FIXED:
      // mask_ records map of positions from bottom to top
      caffe_set(mlu_mask_->count(), Dtype(-1), mask);
      for (int h = 0; h < height_; ++h) {
        for (int w = 0; w < width_; ++w) {
          int uhstart = h * this->out_stride_h_ - this->out_pad_h_;
          int uwstart = w * this->out_stride_w_ - this->out_pad_w_;
          int uhend = uhstart + this->out_kernel_h_;
          int uwend = uwstart + this->out_kernel_w_;
          int uhmid = floor((uhstart + uhend - 1) / 2);
          int uwmid = floor((uwstart + uwend - 1) / 2);
          uhmid = min(max(uhmid, 0), this->unpooled_height_ - 1);
          uwmid = min(max(uwmid, 0), this->unpooled_width_ - 1);
          const int unpool_index = uhmid * this->unpooled_width_ + uwmid;
          const int index = h * width_ + w;
          mask[unpool_index] = index;
        }
      }
      break;
    case UnPoolingParameter_UnPoolMethod_DIV:
    case UnPoolingParameter_UnPoolMethod_REP:
      caffe_set(mlu_mask_->count(), Dtype(0.0), mask);
      for (int h = 0; h < height_; ++h) {
        for (int w = 0; w < width_; ++w) {
          int uhstart = h * this->out_stride_h_ - this->out_pad_h_;
          int uwstart = w * this->out_stride_w_ - this->out_pad_w_;
          int uhend =
              min(uhstart + this->out_kernel_h_, this->unpooled_height_);
          int uwend = min(uwstart + this->out_kernel_w_, this->unpooled_width_);
          uhstart = max(uhstart, 0);
          uwstart = max(uwstart, 0);
          for (int uh = uhstart; uh < uhend; ++uh) {
            for (int uw = uwstart; uw < uwend; ++uw) {
              const int unpool_index = uh * this->unpooled_width_ + uw;
              mask[unpool_index] += 1;
            }
          }
        }
      }
      break;
    default:
      LOG(FATAL) << "Unknown unpooling method.";
  }
}
template <typename Dtype>
void MLUUnPoolingLayer<Dtype>::Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                                              const vector<Blob<Dtype>*>& top) {
  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
  BaseDataType mlu_dtype = bottom[0]->mlu_type();

  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  this->unpooled_height_ = (height_ - 1) * this->out_stride_h_ -
                           2 * this->out_pad_h_ + this->out_kernel_h_;
  this->unpooled_width_ = (width_ - 1) * this->out_stride_w_ -
                          2 * this->out_pad_w_ + this->out_kernel_w_;
  top[0]->Reshape(num_, channels_, this->unpooled_height_,
                  this->unpooled_width_, cpu_dtype, mlu_dtype, CNML_TENSOR);

  this->FillMask();
}

template <typename Dtype>
void MLUUnPoolingLayer<Dtype>::fuse(MFusion<Dtype>* fuser) {
  fuser->fuse(unpooling_op_ptr_);
}

template <typename Dtype>
void MLUUnPoolingLayer<Dtype>::MLUCompileOp() {
  MLU_CHECK(cnmlCompileBaseOp(unpooling_op_ptr_,
                              Caffe::rt_core(),
                              Caffe::core_number()));
}

template <typename Dtype>
void MLUUnPoolingLayer<Dtype>::MLUDestroyOp() {
  if (unpooling_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&unpooling_op_ptr_));
    unpooling_op_ptr_ = nullptr;
  }
  if (unpooling_param_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyUnpoolOpParam(&unpooling_param_ptr_));
    unpooling_param_ptr_ = nullptr;
  }
}

template <typename Dtype>
void MLUUnPoolingLayer<Dtype>::MLUCreateOpBindData(
                               const vector<Blob<Dtype>*>& bottom,
                               const vector<Blob<Dtype>*>& top) {
  cnmlUnpoolMode_t unpool_mode_;
  switch (this->layer_param_.unpooling_param().unpool()) {
    case UnPoolingParameter_UnPoolMethod_FIXED:
      unpool_mode_ = CNML_MIDUNPOOL;
      MLU_CHECK(cnmlCreateUnpoolOpParam(&unpooling_param_ptr_,
                                        this->out_kernel_h_,
                                        this->out_kernel_w_,
                                        this->out_stride_h_,
                                        this->out_stride_w_,
                                        unpool_mode_));
      MLU_CHECK(cnmlCreateUnpoolOp(&unpooling_op_ptr_,
                                   bottom[0]->mlu_tensor(),
                                   bottom[0]->mlu_tensor(),
                                   top[0]->mlu_tensor(),
                                   unpooling_param_ptr_));
      break;
    case UnPoolingParameter_UnPoolMethod_DIV:
      unpool_mode_ = CNML_DIV;
      MLU_CHECK(cnmlCreateUnpoolOpParam(&unpooling_param_ptr_,
                                        this->out_kernel_h_,
                                        this->out_kernel_w_,
                                        this->out_stride_h_,
                                        this->out_stride_w_,
                                        unpool_mode_));
      MLU_CHECK(cnmlCreateUnpoolOp(&unpooling_op_ptr_,
                                   bottom[0]->mlu_tensor(),
                                   mlu_mask_->mlu_tensor(),
                                   top[0]->mlu_tensor(),
                                   unpooling_param_ptr_));
      MLU_CHECK(cnmlBindConstData_V2(mlu_mask_->mlu_tensor(),
                                  mlu_mask_->sync_data(),
                                  false));
      break;
    case UnPoolingParameter_UnPoolMethod_REP:
      unpool_mode_ = CNML_REP;
      MLU_CHECK(cnmlCreateUnpoolOpParam(&unpooling_param_ptr_,
                                        this->out_kernel_h_,
                                        this->out_kernel_w_,
                                        this->out_stride_h_,
                                        this->out_stride_w_,
                                        unpool_mode_));
      MLU_CHECK(cnmlCreateUnpoolOp(&unpooling_op_ptr_,
                                   bottom[0]->mlu_tensor(),
                                   mlu_mask_->mlu_tensor(),
                                   top[0]->mlu_tensor(),
                                   unpooling_param_ptr_));
      MLU_CHECK(cnmlBindConstData_V2(mlu_mask_->mlu_tensor(),
                                  mlu_mask_->sync_data(),
                                  false));
      break;
    default:
      LOG(FATAL) << "Unknown unpooling method.";
  }
}

template <typename Dtype>
void MLUUnPoolingLayer<Dtype>::Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {
  switch (this->layer_param_.unpooling_param().unpool()) {
    case UnPoolingParameter_UnPoolMethod_FIXED:
      MLU_CHECK(cnmlComputeUnpoolOpForward_V3(unpooling_op_ptr_,
                                              bottom[0]->mutable_mlu_data(),
                                              bottom[0]->mutable_mlu_data(),
                                              top[0]->mutable_mlu_data(),
                                              Caffe::forward_param(),
                                              Caffe::queue()));
      break;
    case UnPoolingParameter_UnPoolMethod_DIV:
    case UnPoolingParameter_UnPoolMethod_REP:
      MLU_CHECK(cnmlComputeUnpoolOpForward_V3(unpooling_op_ptr_,
                                              bottom[0]->mutable_mlu_data(),
                                              nullptr,
                                              top[0]->mutable_mlu_data(),
                                              Caffe::forward_param(),
                                              Caffe::queue()));
      break;
    default:
      LOG(FATAL) << "Unknown unpooling method.";
  }
}

INSTANTIATE_CLASS(MLUUnPoolingLayer);

}  // namespace caffe
#endif  // USE_MLU
