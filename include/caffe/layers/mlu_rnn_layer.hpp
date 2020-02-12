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

#ifndef INCLUDE_CAFFE_LAYERS_MLU_RNN_LAYER_HPP_
#define INCLUDE_CAFFE_LAYERS_MLU_RNN_LAYER_HPP_
#ifdef USE_MLU

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layers/rnn_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Processes time-varying inputs using a simple recurrent neural network
 *        (RNN). Implemented as a network unrolling the RNN computation in time.
 *
 * Given time-varying inputs @f$ x_t @f$, coputes hidden state @f$
 *     h_t := \tanh[ W_{hh} h_{t_1} +W_{xh} x_t + b_h ]
 * @f$, and outputs @f$
 *     o_t := \tanh[ W{ho} h_t + b_o ]
 * @f$
 */

template <typename Dtype>
class MLURNNLayer : public RNNLayer<Dtype> {
  public:
  explicit MLURNNLayer(const LayerParameter& param)
      : RNNLayer<Dtype>(param), bottom0_reshape_param_ptr_(nullptr),
      bottom1_reshape_param_ptr_(nullptr), bottom2_reshape_param_ptr_(nullptr),
      bottom3_reshape_param_ptr_(nullptr), top0_reshape_param_ptr_(nullptr),
      top1_reshape_param_ptr_(nullptr), cont_slice_op_(nullptr),
      w_xh_x_slice_op_(nullptr), x_mlp_op_(nullptr),
      x_static_mlp_op_(nullptr), device_memcpy_op_(nullptr),
      bottom0_reshape_op_(nullptr), bottom1_reshape_op_(nullptr),
      bottom2_reshape_op_(nullptr), bottom3_reshape_op_(nullptr),
      top0_reshape_op_(nullptr), top1_reshape_op_(nullptr),
      concat_op_(nullptr), num_output_(0),
      channel(1), height(1), width(1) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                              const vector<Blob<Dtype>*>& top);
  virtual inline bool mfus_supported() { return true; }
  virtual void fuse(MFusion<Dtype>* fuser);
  virtual ~MLURNNLayer();

  protected:
  virtual void MLUDestroyOp();
  virtual void MLUCreateOpBindData(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top);
  virtual void MLUCompileOp();
  virtual void Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  shared_ptr<Blob<Dtype> > w_xh_x_blob_;  // w_xh_blob_  MLP
  vector<shared_ptr<Blob<Dtype> > > w_xh_x_blobs_;
  vector<shared_ptr<Blob<Dtype> > > h_neuron_input_blobs_;  // Eltwise
  vector<shared_ptr<Blob<Dtype> > > temp_input_blobs_;  // temp in Eltwise
  shared_ptr<Blob<Dtype> >  w_xh_x_static_blob_;  // static_input_ is true
  vector<shared_ptr<Blob<Dtype> > > cont_blobs_;
  vector<shared_ptr<Blob<Dtype> > > h_conted_blobs_;
  vector<shared_ptr<Blob<Dtype> > > w_hh_h_blobs_;  // w_hh_h_blobs MLP
  vector<shared_ptr<Blob<Dtype> > > h_blobs_;  // h_blobs_ tanh
  vector<shared_ptr<Blob<Dtype> > > o_blobs_;  // o_blobs_ tanh
  vector<shared_ptr<Blob<Dtype> > > w_ho_blobs_;  // w_ho_blobs_ MLP
  vector<shared_ptr<Blob<Dtype> > > h_reshape_blobs_;  // reshaped h container
  vector<shared_ptr<Blob<Dtype> > > h_conted_reshape_blobs_;

  vector<shared_ptr<Blob<Dtype> > > bak_weight1_blobs_;
  vector<shared_ptr<Blob<Dtype> > > bak_weight2_blobs_;
  vector<shared_ptr<Blob<Dtype> > > bak_beta_blobs_;

  shared_ptr<Blob<Dtype> > bottom0_reshape_blob_;
  shared_ptr<Blob<Dtype> > bottom1_reshape_blob_;
  shared_ptr<Blob<Dtype> > bottom2_reshape_blob_;
  shared_ptr<Blob<Dtype> > bottom3_reshape_blob_;

  shared_ptr<Blob<Dtype> > top0_reshape_blob_;
  shared_ptr<Blob<Dtype> > top1_reshape_blob_;

  cnmlReshapeOpParam_t bottom0_reshape_param_ptr_;
  cnmlReshapeOpParam_t bottom1_reshape_param_ptr_;
  cnmlReshapeOpParam_t bottom2_reshape_param_ptr_;
  cnmlReshapeOpParam_t bottom3_reshape_param_ptr_;

  cnmlReshapeOpParam_t top0_reshape_param_ptr_;
  cnmlReshapeOpParam_t top1_reshape_param_ptr_;
  vector<cnmlReshapeOpParam_t> h_reshape_param_ptr_;
  vector<cnmlReshapeOpParam_t> h_conted_reshape_param_ptr_;

  cnmlBaseOp_t cont_slice_op_;  // slice input cont
  cnmlBaseOp_t w_xh_x_slice_op_;  // slice  w_xh_x_blob_
  cnmlBaseOp_t x_mlp_op_;  // InnerProduct op for input x
  cnmlBaseOp_t x_static_mlp_op_;  // InnerProduct op for input x_statici
  cnmlBaseOp_t device_memcpy_op_;  // copy last h to top[1]

  cnmlBaseOp_t bottom0_reshape_op_;
  cnmlBaseOp_t bottom1_reshape_op_;
  cnmlBaseOp_t bottom2_reshape_op_;
  cnmlBaseOp_t bottom3_reshape_op_;

  cnmlBaseOp_t top0_reshape_op_;
  cnmlBaseOp_t top1_reshape_op_;

  vector<cnmlBaseOp_t> cyclemult_op_;  // scale op for h and cont
  vector<cnmlBaseOp_t> h_conted_mlp_op_;  // InnerProduct op for h_conted
  vector<cnmlBaseOp_t> h_mlp_op_;  // InnerProduct op for h
  vector<cnmlBaseOp_t> w_ho_tanh_op_;  // tanh op for w_ho
  vector<cnmlBaseOp_t> h_neuron_input_tanh_op_;  // tanh op for h_neuron_input
  vector<cnmlBaseOp_t> add_op_;  // eltwise sum op for w_hh_h,w_xh_x
  vector<cnmlBaseOp_t> add_1_op_;  // eltwise sum op 2 if there is x_static
  cnmlBaseOp_t concat_op_;  // concat op for o
  vector<cnmlBaseOp_t> h_reshape_op_;  // reshape h to use cyclemult
  vector<cnmlBaseOp_t> h_conted_reshape_op_;  // reshape back
  int num_output_;
  int channel, height, width;
};
}  // namespace caffe
#endif  // USE_MLU
#endif  // INCLUDE_CAFFE_LAYERS_MLU_RNN_LAYER_HPP_
