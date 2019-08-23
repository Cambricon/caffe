/*
All modification made by Cambricon Corporation: © 2018--2019 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2018, the respective contributors
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
#include <utility>
#include <vector>
#include "caffe/layers/mlu_argmax_layer.hpp"

namespace caffe {

template <typename Dtype>
void MLUArgMaxLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
  ArgMaxLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void MLUArgMaxLayer<Dtype>::Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {
  ArgMaxLayer<Dtype>::Reshape(bottom, top);
  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
  BaseDataType mlu_dtype = DT_FLOAT16;
  if (this->out_max_val_) {
    top[0]->Reshape(top[0]->shape(),
                    cpu_dtype,
                    mlu_dtype,
                    CNML_TENSOR);
  } else {
    top[0]->Reshape(top[0]->shape(),
                    cpu_dtype,
                    DT_INT16,
                    CNML_TENSOR);
  }
  if (this->has_axis_) {
  /**
   *  out_max_val == 1 -> new index, result is fp16 in top[0]
   *  out_max_val == 0 -> new value, result is int16 in top[0]
   */
      if (this->out_max_val_) {
        index_blob_ = new Blob<Dtype>(top[0]->shape(),
                                      cpu_dtype,
                                      DT_INT16,
                                      CNML_TENSOR);
      } else {
        value_blob_ = new Blob<Dtype>(top[0]->shape(),
                                      cpu_dtype,
                                      mlu_dtype,
                                      CNML_TENSOR);
      }
  } else {
    vector<int> shape(4, 1);
    shape[0] = bottom[0]->shape(0);
    shape[2] = bottom[0]->count(1);
    bottom_reshape_blob_ =
        new Blob<Dtype>(shape, cpu_dtype, mlu_dtype, CNML_TENSOR);
    shape[2] = this->top_k_;
    value_blob_ = new Blob<Dtype>(shape, cpu_dtype, mlu_dtype, CNML_TENSOR);
    index_blob_ = new Blob<Dtype>(shape, cpu_dtype, DT_INT16, CNML_TENSOR);
    shape = top[0]->shape();
    if (this->out_max_val_) {
      shape[1] /= 2;
    }
    cast_blob_ = new Blob<Dtype>(shape, cpu_dtype, mlu_dtype, CNML_TENSOR);
  }
}

template <typename Dtype>
MLUArgMaxLayer<Dtype>::~MLUArgMaxLayer() {
  MLUDestroyOp();
// Delete Blobs
  delete value_blob_;
  value_blob_ = nullptr;
  delete index_blob_;
  index_blob_ = nullptr;
  delete bottom_reshape_blob_;
  bottom_reshape_blob_ = nullptr;
  delete cast_blob_;
  cast_blob_ = nullptr;
}

template <typename Dtype>
void MLUArgMaxLayer<Dtype>::MLUDestroyOp() {
// Delete param
  if (bottom_reshape_param_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyReshapeOpParam(&bottom_reshape_param_ptr_));
    bottom_reshape_param_ptr_ = nullptr;
  }
  if (concat_param_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyConcatOpParam(&concat_param_ptr_));
    concat_param_ptr_ = nullptr;
  }

// Delete Op
  if (topk_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&topk_op_ptr_));
    topk_op_ptr_ = nullptr;
  }
  if (bottom_reshape_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&bottom_reshape_op_ptr_));
    bottom_reshape_op_ptr_ = nullptr;
  }
  if (concat_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&concat_op_ptr_));
    concat_op_ptr_ = nullptr;
  }
  if (cast_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&cast_op_ptr_));
    cast_op_ptr_ = nullptr;
  }
}

template <typename Dtype>
void MLUArgMaxLayer<Dtype>::MLUCreateOpBindData(
                            const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top) {
  cnmlDimension_t ch;
  if (this->has_axis_) {
    if (this->axis_ == 0) {
      ch = cnmlDimension_t::CNML_DIM_N;
    } else if (this->axis_ == 2) {
      ch = cnmlDimension_t::CNML_DIM_H;
    } else if (this->axis_ == 3) {
      ch = cnmlDimension_t::CNML_DIM_W;
    } else {
      ch = cnmlDimension_t::CNML_DIM_C;
    }
    if (this->out_max_val_) {
      MLU_CHECK(cnmlCreateTopkOp(&topk_op_ptr_,
                                 this->top_k_,
                                 bottom[0]->mlu_tensor(),
                                 top[0]->mlu_tensor(),
                                 index_blob_->mlu_tensor(),
                                 ch));
    } else {
      MLU_CHECK(cnmlCreateTopkOp(&topk_op_ptr_,
                                 this->top_k_,
                                 bottom[0]->mlu_tensor(),
                                 value_blob_->mlu_tensor(),
                                 top[0]->mlu_tensor(),
                                 ch));
    }
  } else {
    MLU_CHECK(cnmlCreateReshapeOpParam(&bottom_reshape_param_ptr_,
                                       bottom_reshape_blob_->num(),
                                       bottom_reshape_blob_->channels(),
                                       bottom_reshape_blob_->height(),
                                       bottom_reshape_blob_->width(),
                                       CNML_NCHW));
    MLU_CHECK(cnmlCreateReshapeOp(&bottom_reshape_op_ptr_,
                                  bottom_reshape_param_ptr_,
                                  bottom[0]->mlu_tensor(),
                                  bottom_reshape_blob_->mlu_tensor()));
    if (this->out_max_val_) {
      MLU_CHECK(cnmlCreateTopkOp(&topk_op_ptr_,
                                 this->top_k_,
                                 bottom_reshape_blob_->mlu_tensor(),
                                 value_blob_->mlu_tensor(),
                                 index_blob_->mlu_tensor(),
                                 cnmlDimension_t::CNML_DIM_H));
      MLU_CHECK(cnmlCreateCastOp(&cast_op_ptr_,
                                 CNML_CAST_INT16_TO_FLOAT16,
                                 index_blob_->mlu_tensor(),
                                 cast_blob_->mlu_tensor()));

      cnmlTensor_t input_concat[2];
      input_concat[0] = cast_blob_->mlu_tensor();
      input_concat[1] = value_blob_->mlu_tensor();
      cnmlTensor_t output[1];
      output[0] = top[0]->mlu_tensor();
      MLU_CHECK(cnmlCreateConcatOpParam(&concat_param_ptr_,
                                        2, 1,
                                        cnmlConcatMode_t::CNML_CONCAT_FEAT));
      MLU_CHECK(cnmlCreateConcatOp(&concat_op_ptr_,
                                   concat_param_ptr_,
                                   input_concat,
                                   2,
                                   output,
                                   1));
    } else {
      MLU_CHECK(cnmlCreateTopkOp(&topk_op_ptr_,
                                 this->top_k_,
                                 bottom_reshape_blob_->mlu_tensor(),
                                 value_blob_->mlu_tensor(),
                                 top[0]->mlu_tensor(),
                                 cnmlDimension_t::CNML_DIM_H));
    }
  }
}


template <typename Dtype>
void MLUArgMaxLayer<Dtype>::Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
  if (this->has_axis_) {
    if (this->out_max_val_) {
      MLU_CHECK(cnmlComputeTopkOpForward_V3(topk_op_ptr_,
                                bottom[0]->mutable_mlu_data(),
                                top[0]->mutable_mlu_data(),
                                index_blob_->mutable_mlu_data(),
                                Caffe::forward_param(), Caffe::queue()));
    } else {
      MLU_CHECK(cnmlComputeTopkOpForward_V3(topk_op_ptr_,
                                bottom[0]->mutable_mlu_data(),
                                value_blob_->mutable_mlu_data(),
                                top[0]->mutable_mlu_data(),
                                Caffe::forward_param(), Caffe::queue()));
    }

  } else {
    // reshape to (n, c), then topk
    // concat if out_max_val_
    MLU_CHECK(cnmlComputeReshapeOpForward_V3(bottom_reshape_op_ptr_,
                                      bottom[0]->mutable_mlu_data(),
                                      bottom_reshape_blob_->mutable_mlu_data(),
                                      Caffe::forward_param(), Caffe::queue()));
    if (this->out_max_val_) {
      MLU_CHECK(cnmlComputeTopkOpForward_V3(topk_op_ptr_,
          bottom_reshape_blob_->mutable_mlu_data(),
          value_blob_->mutable_mlu_data(),
          index_blob_->mutable_mlu_data(),
          Caffe::forward_param(), Caffe::queue()));
      MLU_CHECK(cnmlComputeCastOpForward_V3(cast_op_ptr_,
                                        index_blob_->mutable_mlu_data(),
                                        cast_blob_->mutable_mlu_data(),
                                        Caffe::forward_param(), Caffe::queue()));
      void* input_concat[2];
      input_concat[0] = cast_blob_->mutable_mlu_data();
      input_concat[1] = value_blob_->mutable_mlu_data();
      void* output[1];
      output[0] = top[0]->mutable_mlu_data();
      MLU_CHECK(cnmlComputeConcatOpForward_V3(concat_op_ptr_,
                                          input_concat,
                                          2,
                                          output,
                                          1,
                                          Caffe::forward_param(), Caffe::queue()));
    } else {
      MLU_CHECK(cnmlComputeTopkOpForward_V3(topk_op_ptr_,
          bottom_reshape_blob_->mutable_mlu_data(),
          value_blob_->mutable_mlu_data(),
          top[0]->mutable_mlu_data(),
          Caffe::forward_param(), Caffe::queue()));
    }
  }
}

template<typename Dtype>
void MLUArgMaxLayer<Dtype>::fuse(MFusion<Dtype>* fuser) {
  if (this->has_axis_) {
    fuser->fuse(topk_op_ptr_);
  } else {
    fuser->fuse(bottom_reshape_op_ptr_);
    if (this->out_max_val_) {
      fuser->fuse(topk_op_ptr_);
      fuser->fuse(cast_op_ptr_);
      fuser->fuse(concat_op_ptr_);
    } else {
      fuser->fuse(topk_op_ptr_);
    }
  }
}

template<typename Dtype>
void MLUArgMaxLayer<Dtype>::MLUCompileOp() {
  if (this->has_axis_) {
    MLU_CHECK(cnmlCompileBaseOp(topk_op_ptr_,
                                Caffe::rt_core(),
                                Caffe::model_parallel()));
  } else {
    MLU_CHECK(cnmlCompileBaseOp(bottom_reshape_op_ptr_,
                                Caffe::rt_core(),
                                Caffe::model_parallel()));
    if (this->out_max_val_) {
      MLU_CHECK(cnmlCompileBaseOp(topk_op_ptr_,
                                  Caffe::rt_core(),
                                  Caffe::model_parallel()));
      MLU_CHECK(cnmlCompileBaseOp(cast_op_ptr_,
                                  Caffe::rt_core(),
                                  Caffe::model_parallel()));
      MLU_CHECK(cnmlCompileBaseOp(concat_op_ptr_,
                                  Caffe::rt_core(),
                                  Caffe::model_parallel()));
    } else {
      MLU_CHECK(cnmlCompileBaseOp(topk_op_ptr_,
                                  Caffe::rt_core(),
                                  Caffe::model_parallel()));
    }
  }
}

INSTANTIATE_CLASS(MLUArgMaxLayer);
}   //  namespace caffe
#endif
