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

#ifdef USE_MLU
#include <vector>
#include "caffe/layers/mlu_shufflechannel_layer.hpp"

namespace caffe {

template <typename Dtype>
void MLUShuffleChannelLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                               const vector<Blob<Dtype>*>& top) {
  group_ = this->layer_param_.shuffle_channel_param().group();
  ShuffleChannelLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void MLUShuffleChannelLayer<Dtype>::Reshape_tensor(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
  BaseDataType mlu_dtype = bottom[0]->mlu_type();
  std::vector<int> top_shape = bottom[0]->shape();
  top[0]->Reshape(top_shape, cpu_dtype, mlu_dtype, CNML_TENSOR);
}

template <typename Dtype>
void MLUShuffleChannelLayer<Dtype>::MLUCreateOpBindData(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int bottomSize = bottom.size();
  int topSize = top.size();
  cnmlTensor_t* mluTensor_input = new cnmlTensor_t[bottomSize];
  cnmlTensor_t* mluTensor_output = new cnmlTensor_t[topSize];

  for (int i = 0; i < bottom.size(); i++) {
    mluTensor_input[i] = bottom[i]->mlu_tensor();
  }
  for (int i = 0; i < top.size(); i++) {
    mluTensor_output[i] = top[i]->mlu_tensor();
  }

  MLU_CHECK(cnmlCreateShuffleChannelOp(&shufflechannel_op_ptr_,
                                       mluTensor_input,
                                       mluTensor_output,
                                       this->group_));
  delete [] mluTensor_input;
  delete [] mluTensor_output;
}

template <typename Dtype>
void MLUShuffleChannelLayer<Dtype>::fuse(MFusion<Dtype>* fuser) {
    fuser->fuse(shufflechannel_op_ptr_);
}

template <typename Dtype>
MLUShuffleChannelLayer<Dtype>::~MLUShuffleChannelLayer() {
  MLUDestroyOp();
}

template <typename Dtype>
void MLUShuffleChannelLayer<Dtype>::MLUDestroyOp() {
  if (shufflechannel_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&shufflechannel_op_ptr_));
    shufflechannel_op_ptr_ = nullptr;
  }
}

template <typename Dtype>
void MLUShuffleChannelLayer<Dtype>::MLUCompileOp() {
  MLU_CHECK(cnmlCompileBaseOp(shufflechannel_op_ptr_,
                              Caffe::rt_core(),
                              Caffe::core_number()));
}

template <typename Dtype>
void MLUShuffleChannelLayer<Dtype>::Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                                                const vector<Blob<Dtype>*>& top) {
  void* inputs[bottom.size()];
  void* outputs[top.size()];
  for ( int i = 0; i < bottom.size(); i++ ) {
    inputs[i] = bottom[i]->mutable_mlu_data();
  }
  for ( int i = 0; i < top.size(); i++ ) {
    outputs[i] = top[i]->mutable_mlu_data();
  }
  MLU_CHECK(cnmlComputeShuffleChannelOpForward_V3(shufflechannel_op_ptr_,
                                               inputs,
                                               outputs,
                                               Caffe::forward_param(),
                                               Caffe::queue()));
}

INSTANTIATE_CLASS(MLUShuffleChannelLayer);

}  // namespace caffe
#endif  // USE_MLU
