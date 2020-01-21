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

#ifndef INCLUDE_CAFFE_LAYERS_MLU_ELTWISE_LAYER_HPP_
#define INCLUDE_CAFFE_LAYERS_MLU_ELTWISE_LAYER_HPP_
#ifdef USE_MLU
#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/eltwise_layer.hpp"

namespace caffe {

/**
 * @brief MLU acceleration of EltwiseLayer
 *        EltwiseLayer Compute elementwise operations, such as product and sum,
 *        along multiple input Blobs.
 *
 */
template <typename Dtype>
class MLUEltwiseLayer : public EltwiseLayer<Dtype> {
  public:
  explicit MLUEltwiseLayer(const LayerParameter& param)
      : EltwiseLayer<Dtype>(param), beta_(nullptr){}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual ~MLUEltwiseLayer();
  virtual inline bool mfus_supported() { return true; }
  virtual void fuse(MFusion<Dtype>* fuser);
  virtual void Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                              const vector<Blob<Dtype>*>& top);

  protected:
  /**
   *@brief destroy the Ops have been setup
   *
   */
  virtual void MLUDestroyOp();
  /**
   *@brief Create Eltwise Op
   *
   */
  virtual void MLUCreateOpBindData(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top);
  /**
   *@brief Complile the Op to instructions
   *
   */
  virtual void MLUCompileOp();
  /**
   *@param bottom input blob vector(length >= 2)
   *@param top output blob vector(length 1)
   *
   */
  virtual void Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  vector<Blob<Dtype>*> alpha_;
  Blob<Dtype>* beta_;          // all is 0, for ScaleOp;
  vector<Blob<Dtype>*> temp_;
  vector<Blob<Dtype>*> temp2_;
  vector<cnmlBaseOp_t> vec_scale_op_ptr_;
  vector<cnmlBaseOp_t> vec_add_op_ptr_;
  vector<cnmlBaseOp_t> vec_mult_op_ptr_;
  vector<cnmlBaseOp_t> vec_max_op_ptr_;
};

}  // namespace caffe
#endif  // USE_MLU
#endif  // INCLUDE_CAFFE_LAYERS_MLU_ELTWISE_LAYER_HPP_
