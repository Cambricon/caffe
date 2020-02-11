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

#ifndef INCLUDE_CAFFE_LAYERS_MLU_ARGMAX_LAYER_HPP_
#define INCLUDE_CAFFE_LAYERS_MLU_ARGMAX_LAYER_HPP_
#ifdef USE_MLU

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/argmax_layer.hpp"

namespace caffe {
/**
 * @brief MLU acceleration of ArgMaxLayer
 *        ArgMaxLayer Compute the index of the @f$ K @f$ max values for each datum across
 *        specified axis or full image.
 */
template <typename Dtype>
class MLUArgMaxLayer : public ArgMaxLayer<Dtype> {
/**
 * @param param provides ArgMaxParameter argmax_param,
 *     with ArgMaxLayer options:
 *   - top_k (\b optional uint, default 1)
 *     the number @f$ K @f$ of maximal items to output.
 *   - out_max_val (\b optional bool, default false)
 *     if set, output a vector of pars (max_ind, max_val)
 *     unless axis is set, then output max_val along the specified axis.
 *   - axis (\b optional int).
 *     if set, maximise along the specified axis, else maximise the flattened
 *     trailing dimensions for each index of the first / num dimension
 */
  public:
  explicit MLUArgMaxLayer(const LayerParameter& param)
      : ArgMaxLayer<Dtype>(param),
        bottom_reshape_param_ptr_(nullptr),
        bottom_reshape_op_ptr_(nullptr),
        concat_op_ptr_(nullptr),
        topk_op_ptr_(nullptr),
        cast_op_ptr_(nullptr),
        value_blob_(nullptr),
        index_blob_(nullptr),
        bottom_reshape_blob_(nullptr),
        cast_blob_(nullptr),
        trans_d2h_layout_(nullptr),
        trans_d2h_param_(nullptr),
        trans_h2d_layout_(nullptr),
        trans_h2d_param_(nullptr) {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                              const vector<Blob<Dtype>*>& top);
  virtual inline bool mfus_supported() { return true; }
  virtual void fuse(MFusion<Dtype>* fuser);
  virtual ~MLUArgMaxLayer();

  protected:
  /**
   * @param bottom input Blob vector (length 1)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the inputs @f$ x @f$
   * @param top output Blob vector (length 1)
   *   -# @f$ (N \times 1 \times K) @f$ or, if out_max_val
   *      @f$ (N \times 2 \times K) @f$ unless axis set than e.g.
   *      @f$ (N \times K \times H \times W) @f$ if axis == 1
   *      the computed outputs @f$
   *       y_n = \arg\max\limits_i x_{ni}
   *      @f$ (for @f$ K = 1 @f$).
   */
  virtual void MLUDestroyOp();
  virtual void MLUCreateOpBindData(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top);
  virtual void MLUCompileOp();
  virtual void Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);

  cnmlReshapeOpParam_t bottom_reshape_param_ptr_;
  cnmlBaseOp_t bottom_reshape_op_ptr_;
  cnmlBaseOp_t concat_op_ptr_;
  cnmlBaseOp_t topk_op_ptr_;
  cnmlBaseOp_t cast_op_ptr_;

  Blob<Dtype>* value_blob_;
  Blob<Dtype>* index_blob_;
  Blob<Dtype>* bottom_reshape_blob_;
  Blob<Dtype>* cast_blob_;
  Blob<Dtype>  d2h_blob_;
  Blob<Dtype>  h2d_blob_;
  cnmlBaseOp_t trans_d2h_layout_;  // NHWC --> NCHW
  cnmlNdTransposeOpParam_t trans_d2h_param_;
  cnmlBaseOp_t trans_h2d_layout_;  // NCHW --> NHWC
  cnmlNdTransposeOpParam_t trans_h2d_param_;
};

}  // namespace caffe
#endif  // USE_MLU
#endif  // INCLUDE_CAFFE_LAYERS_MLU_ARGMAX_LAYER_HPP_
