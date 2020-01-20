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
#ifndef INCLUDE_CAFFE_LAYERS_MLU_CRELU_LAYER_HPP_
#define INCLUDE_CAFFE_LAYERS_MLU_CRELU_LAYER_HPP_
#ifdef USE_MLU
#include <vector>
#include "caffe/layers/crelu_layer.hpp"

namespace caffe {
/**
 *  @brief CNML implementation of CReLULayer.
 *
 * CReLU(x) = [ ReLU(x), ReLU(-x)]
 */
template <typename Dtype>
class MLUCReLULayer : public CReLULayer<Dtype> {
  public:
  explicit MLUCReLULayer(const LayerParameter& param)
    : CReLULayer<Dtype>(param), minus_op_ptr_(nullptr),
      concat_op_ptr_(nullptr), prelu_op_ptr_(nullptr) {}
  virtual inline bool mfus_supported() { return true; }
  virtual void fuse(MFusion<Dtype>* fuser) {
    fuser->fuse(minus_op_ptr_);
    fuser->fuse(concat_op_ptr_);
    fuser->fuse(prelu_op_ptr_);
  }
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                              const vector<Blob<Dtype>*>& top);
  virtual ~MLUCReLULayer();

  protected:
  virtual void MLUDestroyOp();
  virtual void MLUCreateOpBindData(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top);
  virtual void MLUCompileOp() {
    const unsigned int p = Caffe::core_number();  //  avoid the line wrap
    MLU_CHECK(cnmlCompileBaseOp(minus_op_ptr_, Caffe::rt_core(), p));
    MLU_CHECK(cnmlCompileBaseOp(concat_op_ptr_, Caffe::rt_core(), p));
    MLU_CHECK(cnmlCompileBaseOp(prelu_op_ptr_, Caffe::rt_core(), p));
  }
  virtual void Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);

  cnmlBaseOp_t minus_op_ptr_;
  cnmlBaseOp_t concat_op_ptr_;
  cnmlBaseOp_t prelu_op_ptr_;

  Blob<Dtype> negative_input_;
  Blob<Dtype> concated_data_;
  // negative slope for relu
  Blob<Dtype> negative_slope_b_;
};

}  // namespace caffe
#endif  // USE_MLU
#endif  // INCLUDE_CAFFE_LAYERS_MLU_CRELU_LAYER_HPP_
