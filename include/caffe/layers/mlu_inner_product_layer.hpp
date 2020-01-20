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

#ifndef INCLUDE_CAFFE_LAYERS_MLU_INNER_PRODUCT_LAYER_HPP_
#define INCLUDE_CAFFE_LAYERS_MLU_INNER_PRODUCT_LAYER_HPP_
#ifdef USE_MLU

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/inner_product_layer.hpp"
/**
 * @brief Also known as a "fully-connected" layer, computes an inner product
 *        with a set of learned weights, and (optionally) adds biases.
 *
 * @param
 * num_output:
 * The number of outputs of the layer
 *
 * transpose:
 * Specify whether to transpose the weight matrix or not.
 * If transpose == true, any operations will be performed on the transpose
 * of the weight matrix. The weight matrix itself is not going to be transposed
 * but rather the transfer flag of operations will be toggled accordingly.
 *
 * MLU does not support param axis, that means you should always use axis = 1
 * for MLU.
 */

namespace caffe {

template <typename Dtype>
class MLUInnerProductLayer : public InnerProductLayer<Dtype> {
  public:
  explicit MLUInnerProductLayer(const LayerParameter& param)
      : InnerProductLayer<Dtype>(param), mlp_op_ptr_(nullptr),
      input_quant_param_(nullptr),
      reshape_op_ptr_(nullptr),
      reshape_param_(nullptr),
      reshape_(false),
      transpose_op_param_ptr_(nullptr),
      transpose_pro_op_ptr_(nullptr) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual ~MLUInnerProductLayer();
  virtual inline bool mfus_supported() { return true; }
  virtual void fuse(MFusion<Dtype>* fuser) {
    if (reshape_) {
      fuser->fuse(reshape_op_ptr_);
      fuser->fuse(transpose_pro_op_ptr_);
    }
    fuser->fuse(mlp_op_ptr_);
  }
  virtual void Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  protected:
  virtual void Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void MLUDestroyOp();
  virtual void MLUCreateOpBindData(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top);
  virtual void MLUCompileOp();
  void bindDataAndSetComputingDataType(shared_ptr<Blob<Dtype>> blob,
                                 cnmlBaseOp_t op, BaseDataType type);
  cnmlBaseOp_t mlp_op_ptr_;
  cnmlQuantizedParam_t input_quant_param_;
  cnmlBaseOp_t reshape_op_ptr_;
  cnmlReshapeOpParam_t reshape_param_;
  Blob<Dtype> reshape;
  bool reshape_;
  cnmlNdTransposeOpParam_t transpose_op_param_ptr_;
  cnmlBaseOp_t transpose_pro_op_ptr_;
  Blob<Dtype> transpose;
};
}  // namespace caffe

#endif  // USE_MLU
#endif  // INCLUDE_CAFFE_LAYERS_MLU_INNER_PRODUCT_LAYER_HPP_
