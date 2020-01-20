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

#ifndef INCLUDE_CAFFE_LAYERS_MLU_SCALE_LAYER_HPP_
#define INCLUDE_CAFFE_LAYERS_MLU_SCALE_LAYER_HPP_
#ifdef USE_MLU

#include <vector>
#include "caffe/layers/scale_layer.hpp"

namespace caffe {

/**
 * @brief MLU acceleration of ScaleLayer
 *        Computes the elementwise product of two input Blobs, with the shape of
 *  the latter Blob "broadcast" to match the shape of the former.Equivalent to
 *  tiling the latter Blob, then computing the elementwise product.
 */
template <typename Dtype>
class MLUScaleLayer : public ScaleLayer<Dtype> {
  public:
  explicit MLUScaleLayer(const LayerParameter& param)
      : ScaleLayer<Dtype>(param), mlu_scale_op_ptr_(NULL),
        mlu_cmul_op_ptr_(NULL), mlu_cadd_op_ptr_(NULL),
        inter_scale_shape(4, 1), real_bottom_shape(4, 1),
        inter_bottom_shape(4, 1), alpha_shape(), reshape_param_(nullptr),
        reshape1_param_(nullptr), reshape2_param_(nullptr),
        transpose_bottom_d2h_param_(nullptr), transpose_bottom_h2d_param_(nullptr),
        transpose_alpha_d2h_param_(nullptr), transpose_alpha_h2d_param_(nullptr),
        transpose_top_d2h_param_(nullptr), transpose_top_h2d_param_(nullptr),
        reshape_op0_ptr_(nullptr), reshape_op1_ptr_(nullptr),
        reshape_op2_ptr_(nullptr),
        transpose_bottom_d2h_op_ptr_(nullptr), transpose_bottom_h2d_op_ptr_(nullptr),
        transpose_alpha_d2h_op_ptr_(nullptr), transpose_alpha_h2d_op_ptr_(nullptr),
        transpose_top_d2h_op_ptr_(nullptr), transpose_top_h2d_op_ptr_(nullptr),
        bias_param_id_(-1), need_reshape_(true) {
        op_ptrs_.push_back(&mlu_scale_op_ptr_);
        op_ptrs_.push_back(&mlu_cmul_op_ptr_);
        op_ptrs_.push_back(&mlu_cadd_op_ptr_);
        op_ptrs_.push_back(&reshape_op0_ptr_);
        op_ptrs_.push_back(&reshape_op1_ptr_);
        op_ptrs_.push_back(&reshape_op2_ptr_);
        op_ptrs_.push_back(&transpose_bottom_d2h_op_ptr_);
        op_ptrs_.push_back(&transpose_bottom_h2d_op_ptr_);
        op_ptrs_.push_back(&transpose_alpha_d2h_op_ptr_);
        op_ptrs_.push_back(&transpose_alpha_h2d_op_ptr_);
        op_ptrs_.push_back(&transpose_top_d2h_op_ptr_);
        op_ptrs_.push_back(&transpose_top_h2d_op_ptr_);
        }
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                              const vector<Blob<Dtype>*>& top);
  virtual ~MLUScaleLayer();
  virtual inline bool mfus_supported() { return true; }
  virtual void fuse(MFusion<Dtype>* fuser);
  const int need_reshape() { return need_reshape_ ; }

  protected:
  virtual void MLUDestroyOp();
  virtual void MLUCreateOpBindData(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top);
  virtual void MLUCreateOpBindCycleMult(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top);
  virtual void MLUCreateOpBindScale(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top);
  virtual void MLUCreateOpBindCycleMult_(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top);
  virtual void MLUCreateOpBindScale_(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top);
  virtual void MLUCompileOp();
  virtual void Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void ForwardMLU(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void ForwardMLU_(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);

  cnmlBaseOp_t mlu_scale_op_ptr_;
  cnmlBaseOp_t mlu_cmul_op_ptr_;
  cnmlBaseOp_t mlu_cadd_op_ptr_;
  Blob<Dtype> true_bias_data_;  //  ScaleOp needs beta anyway
  vector<int> inter_scale_shape;
  vector<int> real_bottom_shape;
  vector<int> inter_bottom_shape;
  vector<int> alpha_shape;
  cnmlReshapeOpParam_t reshape_param_;
  cnmlReshapeOpParam_t reshape1_param_;
  cnmlReshapeOpParam_t reshape2_param_;
  cnmlNdTransposeOpParam_t transpose_bottom_d2h_param_;
  cnmlNdTransposeOpParam_t transpose_bottom_h2d_param_;
  cnmlNdTransposeOpParam_t transpose_alpha_d2h_param_;
  cnmlNdTransposeOpParam_t transpose_alpha_h2d_param_;
  cnmlNdTransposeOpParam_t transpose_top_d2h_param_;
  cnmlNdTransposeOpParam_t transpose_top_h2d_param_;
  cnmlBaseOp_t reshape_op0_ptr_;
  cnmlBaseOp_t reshape_op1_ptr_;
  cnmlBaseOp_t reshape_op2_ptr_;
  cnmlBaseOp_t transpose_bottom_d2h_op_ptr_;
  cnmlBaseOp_t transpose_bottom_h2d_op_ptr_;
  cnmlBaseOp_t transpose_alpha_d2h_op_ptr_;
  cnmlBaseOp_t transpose_alpha_h2d_op_ptr_;
  cnmlBaseOp_t transpose_top_d2h_op_ptr_;
  cnmlBaseOp_t transpose_top_h2d_op_ptr_;
  Blob<Dtype> op_bottom0_blob_;
  Blob<Dtype> op_bottom1_blob_;
  Blob<Dtype> op_top0_blob_;
  Blob<Dtype> op_top1_blob_;
  Blob<Dtype> transpose_bottom_d2h_blob_;
  Blob<Dtype> transpose_bottom_h2d_blob_;
  Blob<Dtype> transpose_alpha_d2h_blob_;
  Blob<Dtype> transpose_alpha_h2d_blob_;
  Blob<Dtype> transpose_top_d2h_blob_;
  Blob<Dtype> transpose_top_h2d_blob_;
  int bias_param_id_;
  bool need_reshape_;
  vector<cnmlBaseOp_t*> op_ptrs_;
};
}  // namespace caffe
#endif  // USE_MLU
#endif  // INCLUDE_CAFFE_LAYERS_MLU_SCALE_LAYER_HPP_
