/*
All modification made by Cambricon Corporation: © 2019 Cambricon Corporation
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

#ifndef INCLUDE_CAFFE_LAYERS_MLU_PSROI_POOLING_LAYER_HPP_
#define INCLUDE_CAFFE_LAYERS_MLU_PSROI_POOLING_LAYER_HPP_
#ifdef USE_MLU

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layers/psroi_pooling_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Position-Sensitive ROI pooling thransform the input ROI features to
 *        C+1 dimentional feature vector.
 */
template <typename Dtype>
class MLUPSROIPoolingLayer : public PSROIPoolingLayer<Dtype> {
  public:
  explicit MLUPSROIPoolingLayer(const LayerParameter& param)
      : PSROIPoolingLayer<Dtype>(param),
      psroi_pool_op_ptr_(nullptr),
      bottom_reshape_op_ptr1_(nullptr),
      bottom_reshape_param_ptr1_(nullptr),
      bottom_transpose_op_ptr_(nullptr),
      bottom_transpose_param_ptr_(nullptr),
      psroi_pooling_ptr_param(nullptr) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                              const vector<Blob<Dtype>*>& top);
  virtual ~MLUPSROIPoolingLayer();
  virtual inline bool mfus_supported() { return true; }
  virtual void fuse(MFusion<Dtype>* fuser) {
    fuser->fuse(bottom_reshape_op_ptr1_);
    fuser->fuse(bottom_transpose_op_ptr_);
    fuser->fuse(psroi_pool_op_ptr_);
  }

  protected:
  virtual void MLUDestroyOp();
  virtual void MLUCreateOpBindData(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top);
  virtual void MLUCompileOp();
  virtual void Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  cnmlBaseOp_t psroi_pool_op_ptr_;
  cnmlBaseOp_t bottom_reshape_op_ptr1_;
  cnmlReshapeOpParam_t bottom_reshape_param_ptr1_;

  cnmlBaseOp_t bottom_transpose_op_ptr_;
  cnmlNdTransposeOpParam_t bottom_transpose_param_ptr_;
  cnmlPluginPsRoiPoolOpParam_t psroi_pooling_ptr_param;

  Blob<Dtype>* bottom_reshape_blob1_;
  Blob<Dtype>* bottom_transpose_blob_;
};
}  // namespace caffe
#endif  // USE_MLU
#endif  // INCLUDE_CAFFE_LAYERS_MLU_PSROI_POOLING_LAYER_HPP_
