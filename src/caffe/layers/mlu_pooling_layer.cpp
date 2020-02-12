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
#include <algorithm>
#include <cfloat>
#include <sstream>
#include <vector>
#include "caffe/layers/mlu_pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MLUPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
  PoolingLayer<Dtype>::LayerSetUp(bottom, top);
  top_size_ = top.size();
}

template <typename Dtype>
void MLUPoolingLayer<Dtype>::Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
    << "corresponding to (num, channels, height, width)";
  this->channels_ = bottom[0]->channels();
  this->height_ = bottom[0]->height();
  this->width_ = bottom[0]->width();
  if (this->global_pooling_) {
    this->kernel_h_ = bottom[0]->height();
    this->kernel_w_ = bottom[0]->width();
  }
  // Modified to support ceil_mode in densenet.
  if (this->ceil_mode_) {
    this->pooled_height_ = static_cast<int>(ceil(static_cast<float>(
            this->height_ + 2 * this->pad_h_ - this->kernel_h_) /
          this->stride_h_)) + 1;
    this->pooled_width_ = static_cast<int>(ceil(static_cast<float>(
            this->width_ + 2 * this->pad_w_ - this->kernel_w_) /
          this->stride_w_)) + 1;
  } else {
    this->pooled_height_ = static_cast<int>(floor(static_cast<float>(
            this->height_ + 2 * this->pad_h_ - this->kernel_h_) /
          this->stride_h_)) + 1;
    this->pooled_width_ = static_cast<int>(floor(static_cast<float>(
            this->width_ + 2 * this->pad_w_ - this->kernel_w_) /
          this->stride_w_)) + 1;
  }
  if (this->pad_h_ || this->pad_w_) {
    // If we have padding, ensure that the last pooling starts strictly
    // inside the image (instead of at the padding); otherwise clip the last.
    pad_down_ = this->pad_h_;
    pad_right_ = this->pad_w_;
    if ((this->pooled_height_ - 1) * this->stride_h_ >=
        this->height_ + this->pad_h_) {
      --this->pooled_height_;
      pad_down_ = 0;
    }
    if ((this->pooled_width_ - 1) * this->stride_w_ >=
        this->width_ + this->pad_w_) {
      --this->pooled_width_;
      pad_right_ = 0;
    }
    CHECK_LT((this->pooled_height_ - 1) * this->stride_h_,
        this->height_ + this->pad_h_);
    CHECK_LT((this->pooled_width_ - 1) * this->stride_w_,
        this->width_ + this->pad_w_);
  }

  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
  BaseDataType mlu_dtype = bottom[0]->mlu_type();
  top[0]->Reshape(bottom[0]->num(), this->channels_, this->pooled_height_,
                  this->pooled_width_, cpu_dtype, mlu_dtype, CNML_TENSOR);
  if (top.size() > 1) {
      top[1]->Reshape(top[0]->shape(), cpu_dtype, DT_INT16, CNML_TENSOR);
  }
  if (this->pad_h_ || this->pad_w_) {
    vector<int> addpad_shape = bottom[0]->shape();
    addpad_shape[2] = bottom[0]->height() + this->pad_h_ + pad_down_;
    addpad_shape[3] = bottom[0]->width() + this->pad_w_ + pad_right_;
    addpad_.Reshape(addpad_shape, cpu_dtype, mlu_dtype, CNML_TENSOR);
  }
}

template <typename Dtype>
void MLUPoolingLayer<Dtype>::MLUCreateOpBindData(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // add pad
  if (this->pad_h_ || this->pad_w_) {
    MLU_CHECK(cnmlCreateAddPadOpParam_V2(&pool_addpad_op_param_,
                                          this->pad_h_,
                                          pad_down_,
                                          this->pad_w_,
                                          pad_right_,
                                          0));

    MLU_CHECK(cnmlCreateAddPadOp(&addpad_op_ptr_,
                                  pool_addpad_op_param_,
                                  bottom[0]->mlu_tensor(),
                                  addpad_.mlu_tensor()));
  }

  // pool_mode
  cnmlPoolMode_t pool_mode = CNML_POOL_MAX;
  if (this->layer_param_.pooling_param().pool()==
           PoolingParameter_PoolMethod_AVE) {
    pool_mode = CNML_POOL_AVG;
  }

  /* PoolOpParam */
  if (this->ceil_mode_) {
    MLU_CHECK(cnmlCreatePoolOpParam(&pool_param_ptr_,
                               this->kernel_h_,
                               this->kernel_w_,
                               this->stride_h_,
                               this->stride_w_,
                               0, /* origin pad_h ignored */
                               0, /* origin pad_w ignored */
                               0, /* dilation_h not set */
                               0, /* dilation_w not set */
                               pool_mode,
                               CNML_POOL_KFULL,
                               false));
  } else {   //  Use KVALID for ceil_mode_ = false
    MLU_CHECK(cnmlCreatePoolOpParam(&pool_param_ptr_,
                               this->kernel_h_,
                               this->kernel_w_,
                               this->stride_h_,
                               this->stride_w_,
                               0, /* origin pad_h ignored */
                               0, /* origin pad_w ignored */
                               0, /* dilation_h not set */
                               0, /* dilation_w not set */
                               pool_mode,
                               CNML_POOL_KVALID,
                               false));
  }

  /* PoolOp */
  MLU_CHECK(cnmlCreatePoolOp(&pool_op_ptr_,
                     pool_param_ptr_,
                     addpad_op_ptr_? addpad_.mlu_tensor() : bottom[0]->mlu_tensor(),
                     top[0]->mlu_tensor()));

  // MAX POOL layers can output an extra top blob for the index
  // The output of index can be used as the second input of upsample layer.
  if (top_size_ > 1) {
    pool_mode = CNML_POOL_MAXINDEX;
    MLU_CHECK(cnmlCreatePoolOpParam(&pool_index_param_ptr_,
                                    this->kernel_h_,
                                    this->kernel_w_,
                                    this->stride_h_,
                                    this->stride_w_,
                                    0, /* origin pad_h ignored */
                                    0, /* origin pad_w ignored */
                                    0, /* dilation_h not set */
                                    0, /* dilation_w not set */
                                    pool_mode,
                                    CNML_POOL_KVALID,
                                    false));

    MLU_CHECK(cnmlCreatePoolOp(&pool_index_op_ptr_,
                                pool_index_param_ptr_,
                                addpad_op_ptr_? addpad_.mlu_tensor() :
                                bottom[0]->mlu_tensor(),
                                top[1]->mlu_tensor()));
  }
}

template <typename Dtype>
void MLUPoolingLayer<Dtype>::MLUCompileOp() {
  if (this->pad_h_ || this->pad_w_) {
    MLU_CHECK(cnmlCompileBaseOp(addpad_op_ptr_,
                                Caffe::rt_core(),
                                Caffe::core_number()));
  }
  MLU_CHECK(cnmlCompileBaseOp(pool_op_ptr_,
                              Caffe::rt_core(),
                              Caffe::core_number()));
  if (top_size_ > 1) {
    MLU_CHECK(cnmlCompileBaseOp(pool_index_op_ptr_,
                                Caffe::rt_core(),
                                Caffe::core_number()));
  }
}

template <typename Dtype>
void MLUPoolingLayer<Dtype>::Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  if ((this->pad_h_ || this->pad_w_) && addpad_op_ptr_) {
    MLU_CHECK(cnmlComputeAddPadOpForward_V3(addpad_op_ptr_,
                                            bottom[0]->mutable_mlu_data(),
                                            addpad_.mutable_mlu_data(),
                                            Caffe::forward_param(),
                                            Caffe::queue()));
  }
  auto pool_input = addpad_op_ptr_? addpad_.mutable_mlu_data() :
                                    bottom[0]->mutable_mlu_data();
  MLU_CHECK(cnmlComputePoolOpForward_V3(pool_op_ptr_,
                                        pool_input,
                                        top[0]->mutable_mlu_data(),
                                        Caffe::forward_param(),
                                        Caffe::queue()));
  if (top_size_ > 1) {
    MLU_CHECK(cnmlComputePoolOpForward_V3(pool_index_op_ptr_,
                                          pool_input,
                                          top[1]->mutable_mlu_data(),
                                          Caffe::forward_param(),
                                          Caffe::queue()));
  }
}

template <typename Dtype>
void MLUPoolingLayer<Dtype>::fuse(MFusion<Dtype>* fuser) {
  if (this->pad_h_ || this->pad_w_) {
    fuser->fuse(addpad_op_ptr_);
  }
  fuser->fuse(pool_op_ptr_);
  if (top_size_ > 1) {
    fuser->fuse(pool_index_op_ptr_);
  }
}

template <typename Dtype>
void MLUPoolingLayer<Dtype>::MLUDestroyOp() {
  if ((this->pad_h_ || this->pad_w_) && pool_addpad_op_param_ != nullptr) {
    MLU_CHECK(cnmlDestroyAddPadOpParam(&pool_addpad_op_param_));
    pool_addpad_op_param_ = nullptr;
  }
  if ((this->pad_h_ || this->pad_w_) && addpad_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&addpad_op_ptr_));
    addpad_op_ptr_ = nullptr;
  }
  if (pool_param_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyPoolOpParam(&pool_param_ptr_));
    this->pool_param_ptr_ = nullptr;
  }
  if (pool_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&pool_op_ptr_));
    pool_op_ptr_ = nullptr;
  }
  if (pool_index_param_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyPoolOpParam(&pool_index_param_ptr_));
    this->pool_index_param_ptr_ = nullptr;
  }
  if (pool_index_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&pool_index_op_ptr_));
    pool_index_op_ptr_ = nullptr;
  }
}

INSTANTIATE_CLASS(MLUPoolingLayer);

}  // namespace caffe
#endif  // USE_MLU
