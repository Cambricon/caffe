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

#ifndef INCLUDE_CAFFE_LAYERS_MLU_CONV_DEPTHWISE_LAYER_HPP_
#define INCLUDE_CAFFE_LAYERS_MLU_CONV_DEPTHWISE_LAYER_HPP_
#ifdef USE_MLU
#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/conv_depthwise_layer.hpp"

namespace caffe {

/**
 * @brief MLU acceleration of ConvolutionDepthwiseLayer
 * Convdepthwise is a specifial type of convolution. When a convolution layer
 * is a depthwise one if it matches below criteira:
 * group is greater than 1
 * Channels of bottom 0 equals group_
 * n_output divides group
 *
 * This layer is a special case of ConvolutionLayer and MLUConvolutionLayer.
 * Refer to those two layers for more details.
*/

template <typename Dtype>
class MLUConvolutionDepthwiseLayer : public ConvolutionDepthwiseLayer<Dtype> {
  public:
  explicit MLUConvolutionDepthwiseLayer(const LayerParameter& param)
      : ConvolutionDepthwiseLayer<Dtype>(param),
        conv_depthwise_op_ptrs_(nullptr),
        mlu_addpad_op_ptrs_(nullptr),
        mlu_addpad_param_ptr_(nullptr),
        depthwise_param_ptr_(nullptr) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                              const vector<Blob<Dtype>*>& top);
  virtual inline bool mfus_supported() { return true; }
  virtual void fuse(MFusion<Dtype>* fuser) {
    if (add_pad_) {
      for (int i = 0; i < bottom_size_; i++) {
        fuser->fuse(mlu_addpad_op_ptrs_[i]);
      }
    }
    for (int i = 0; i < bottom_size_; i++) {
      fuser->fuse(conv_depthwise_op_ptrs_[i]);
    }
  }
  virtual ~MLUConvolutionDepthwiseLayer();

  protected:
  virtual void MLUCreateOpBindData(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top);
  virtual void Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void MLUDestroyOp();
  virtual void MLUCompileOp();

  size_t bottom_size_;

  cnmlBaseOp_t* conv_depthwise_op_ptrs_;
  cnmlBaseOp_t* mlu_addpad_op_ptrs_;
  cnmlAddPadOpParam_t mlu_addpad_param_ptr_;
  vector<Blob<Dtype>*> addpad_;
  bool add_pad_;

  private:
  int num_kernels_im2col_;
  int num_kernels_col2im_;
  int conv_out_channels_;
  int conv_in_channels_;
  int conv_out_spatial_dim_;
  int col_offset_;
  int output_offset_;

  cnmlConvDepthwiseOpParam_t depthwise_param_ptr_;
  Blob<Dtype> bias_multiplier_;
};
}  // namespace caffe
#endif  // USE_MLU
#endif  // INCLUDE_CAFFE_LAYERS_MLU_CONV_DEPTHWISE_LAYER_HPP_
