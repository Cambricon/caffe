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

#ifndef INCLUDE_CAFFE_LAYERS_MLU_NORMALIZE_LAYER_HPP_
#define INCLUDE_CAFFE_LAYERS_MLU_NORMALIZE_LAYER_HPP_
#ifdef USE_MLU

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/normalize_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief This layer realize L2-Normalization, when across_spatial is true, normalization
 * will take in one whole batch which square-sum number is channel*height*width.when
 * false, normalization will take in one channel, then suqare-sum number is height*width
 * with channel_shared true you can scale the results with one same value, else each
 * channel with different values.
 *
 */
template <typename Dtype>
class MLUNormalizeLayer : public NormalizeLayer<Dtype> {
  public:
  explicit MLUNormalizeLayer(const LayerParameter &param)
    : NormalizeLayer<Dtype>(param), mlp_weight_blob_(nullptr),
      mlp_bias_blob_(nullptr), eps_blob_(nullptr),
      div0_blob_(nullptr), mult_blob_(nullptr),
      gemv_weight_(nullptr), gemm_weight_(nullptr),
      gemm_matrix_(nullptr), max_op_ptr_(nullptr),
      power_op_ptr_(nullptr), bMult_op_ptr_(nullptr),
      sqr_op_ptr_(nullptr), gemv_op_ptr_(nullptr),
      sqrt_op_ptr_(nullptr), gemm1_op_ptr_(nullptr),
      div_op_ptr_(nullptr), mult_op_ptr_(nullptr),
      absval_op_ptr_(nullptr), mlp_op_ptr_(nullptr),
      add_op_ptr_(nullptr), pow_op_ptr_(nullptr),
      div1_op_ptr_(nullptr),
      cyclemult_op_ptr_(nullptr), mult1_op_ptr_(nullptr),
      reshape_op_ptr_(nullptr), reshape_param_(nullptr),
      reshape1_op_ptr_(nullptr), reshape1_param_(nullptr),
      reshape2_op_ptr_(nullptr), reshape2_param_(nullptr),
      gemv_param_ptr_(nullptr), int8_mode(false),
      trans_div1_d2h_layout_(nullptr), trans_div1_d2h_param_(nullptr),
      trans_div1_h2d_layout_(nullptr), trans_div1_h2d_param_(nullptr),
      trans_bMult_d2h_layout_(nullptr), trans_bMult_d2h_param_(nullptr),
      trans_bMult_h2d_layout_(nullptr), trans_bMult_h2d_param_(nullptr),
      trans_scale_d2h_layout_(nullptr), trans_scale_d2h_param_(nullptr),
      trans_scale_h2d_layout_(nullptr), trans_scale_h2d_param_(nullptr) {}
  virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                          const vector<Blob<Dtype> *> &top);
  virtual void Reshape_tensor(const vector<Blob<Dtype> *> &bottom,
                              const vector<Blob<Dtype> *> &top);
  virtual inline bool mfus_supported() { return true; }
  virtual void fuse(MFusion<Dtype> *fuser);
  virtual ~MLUNormalizeLayer();

  protected:
  virtual void Forward_mlu(const vector<Blob<Dtype> *> &bottom,
                           const vector<Blob<Dtype> *> &top);
  virtual void MLUDestroyOp();
  virtual void MLUCreateOpBindData(const vector<Blob<Dtype> *> &bottom,
                                   const vector<Blob<Dtype> *> &top);
  virtual void MLUCompileOp();
  Blob<Dtype> max_blob_;
  Blob<Dtype> index_blob_;
  Blob<Dtype> power_blob_;
  Blob<Dtype> bMult_blob_;
  Blob<Dtype> sqr_blob_;
  Blob<Dtype> gemv_blob_;
  Blob<Dtype> powx_blob_;
  Blob<Dtype> gemm1_blob_;
  Blob<Dtype> div_blob_;
  Blob<Dtype> gemm2_blob_;

  Blob<Dtype> abs_blob_;
  Blob<Dtype> mlp_blob_;
  Blob<Dtype> add_blob_;
  Blob<Dtype> pow_blob_;
  Blob<Dtype> div1_blob_;
  Blob<Dtype> scale_blob_;
  Blob<Dtype> reshape_blob_;  // reshape div1_blob_
  Blob<Dtype> reshape1_blob_;  // reshape bottom[0]
  Blob<Dtype> reshape2_blob_;  // reshape scale_blob_

  Blob<Dtype> *mlp_weight_blob_;
  Blob<Dtype> *mlp_bias_blob_;
  Blob<Dtype> *eps_blob_;
  Blob<Dtype> *div0_blob_;  // the value is 1.0
  Blob<Dtype> *mult_blob_;  // the value is blobs_[0]

  // inner weights
  Blob<Dtype> *gemv_weight_;
  Blob<Dtype> *gemm_weight_;
  Blob<Dtype> *gemm_matrix_;

  cnmlBaseOp_t max_op_ptr_;
  cnmlBaseOp_t power_op_ptr_;
  cnmlBaseOp_t bMult_op_ptr_;
  cnmlBaseOp_t sqr_op_ptr_;
  cnmlBaseOp_t gemv_op_ptr_;
  cnmlBaseOp_t sqrt_op_ptr_;
  cnmlBaseOp_t gemm1_op_ptr_;
  cnmlBaseOp_t div_op_ptr_;
  cnmlBaseOp_t mult_op_ptr_;

  cnmlBaseOp_t absval_op_ptr_;
  cnmlBaseOp_t mlp_op_ptr_;
  cnmlBaseOp_t add_op_ptr_;
  cnmlBaseOp_t pow_op_ptr_;
  cnmlBaseOp_t div1_op_ptr_;
  cnmlBaseOp_t cyclemult_op_ptr_;
  cnmlBaseOp_t mult1_op_ptr_;

  cnmlBaseOp_t reshape_op_ptr_;  // div1_blob_   n,1,1,1 to 1,n,1,1
  cnmlReshapeOpParam_t reshape_param_;

  cnmlBaseOp_t reshape1_op_ptr_;  // bottom[0]    n,c,h, w to 1, n, c*h, w
  cnmlReshapeOpParam_t reshape1_param_;

  cnmlBaseOp_t reshape2_op_ptr_;  // scale_blob_
  cnmlReshapeOpParam_t reshape2_param_;

  cnmlConvOpParam_t gemv_param_ptr_;
  bool int8_mode;
  vector<cnmlQuantizedParam_t> quant_params;
  Blob<Dtype> div1_d2h_blob_;
  Blob<Dtype> div1_h2d_blob_;
  Blob<Dtype> bMult_d2h_blob_;
  Blob<Dtype> bMult_h2d_blob_;
  Blob<Dtype> scale_d2h_blob_;
  Blob<Dtype> scale_h2d_blob_;
  cnmlBaseOp_t trans_div1_d2h_layout_;  // NHWC --> NCHW
  cnmlNdTransposeOpParam_t trans_div1_d2h_param_;
  cnmlBaseOp_t trans_div1_h2d_layout_;  // NCHW --> NHWC
  cnmlNdTransposeOpParam_t trans_div1_h2d_param_;
  cnmlBaseOp_t trans_bMult_d2h_layout_;  // NHWC --> NCHW
  cnmlNdTransposeOpParam_t trans_bMult_d2h_param_;
  cnmlBaseOp_t trans_bMult_h2d_layout_;  // NCHW --> NHWC
  cnmlNdTransposeOpParam_t trans_bMult_h2d_param_;
  cnmlBaseOp_t trans_scale_d2h_layout_;  // NHWC --> NCHW
  cnmlNdTransposeOpParam_t trans_scale_d2h_param_;
  cnmlBaseOp_t trans_scale_h2d_layout_;  // NCHW --> NHWC
  cnmlNdTransposeOpParam_t trans_scale_h2d_param_;
};
}  // namespace caffe
#endif  // USE_MLU
#endif  // INCLUDE_CAFFE_LAYERS_MLU_NORMALIZE_LAYER_HPP_
