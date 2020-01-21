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

#ifndef INCLUDE_CAFFE_LAYERS_MLU_LRN_LAYER_HPP_
#define INCLUDE_CAFFE_LAYERS_MLU_LRN_LAYER_HPP_

#ifdef USE_MLU
#include <vector>

#include "caffe/layers/lrn_layer.hpp"

namespace caffe {
/**
 * @brief Normalize the input in a local region across or within feature maps.
 *
 */
template <typename Dtype>
class MLULRNLayer : public LRNLayer<Dtype> {
  public:
  explicit MLULRNLayer(const LayerParameter& param)
      : LRNLayer<Dtype>(param),
        mlu_lrn_param_ptr_(nullptr),
        mlu_lrn_op_ptr_(nullptr),
        mlu_square_op_ptr_(nullptr),
        mlu_pool_op_ptr_(nullptr),
        mlu_addpad_op_ptr_(nullptr),
        mlu_scale_op_ptr_(nullptr),
        mlu_power_op_ptr_(nullptr),
        mlu_product_op_ptr_(nullptr),
        mlu_scale1_op_ptr_(nullptr),
        input_quant_params(nullptr),
        output_quant_params(nullptr) {}
  virtual ~MLULRNLayer();
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                              const vector<Blob<Dtype>*>& top);
  virtual inline bool mfus_supported() { return true; }
  virtual void fuse(MFusion<Dtype>* fuser);

  protected:
  virtual void Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void MLUDestroyOp();
  virtual void MLUCreateOpBindData(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top);
  virtual void MLUCompileOp();

  cnmlLrnOpParam_t mlu_lrn_param_ptr_;
  cnmlBaseOp_t mlu_lrn_op_ptr_;
  cnmlBaseOp_t mlu_square_op_ptr_;
  cnmlBaseOp_t mlu_pool_op_ptr_;
  cnmlBaseOp_t mlu_addpad_op_ptr_;
  cnmlBaseOp_t mlu_scale_op_ptr_;
  cnmlBaseOp_t mlu_power_op_ptr_;
  cnmlBaseOp_t mlu_product_op_ptr_;

  cnmlBaseOp_t mlu_scale1_op_ptr_;

  Blob<Dtype> square_blob_;
  Blob<Dtype> pool_blob_;
  Blob<Dtype> addpad_blob_;
  Blob<Dtype> alpha_blob_;
  Blob<Dtype> beta_blob_;
  Blob<Dtype> temp_blob_;
  Blob<Dtype> power_blob_;

  Blob<Dtype> scale_blob_;
  Blob<Dtype> alpha1_blob_;
  Blob<Dtype> beta1_blob_;

  cnmlPoolOpParam_t pool_param_ptr_;
  cnmlAddPadOpParam_t addpad_param_ptr_;
  cnmlQuantizedParam_t input_quant_params;
  cnmlQuantizedParam_t output_quant_params;
};

#endif  // USE_MLU
}  // namespace caffe

#endif  // INCLUDE_CAFFE_LAYERS_MLU_LRN_LAYER_HPP_
