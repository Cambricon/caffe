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
#include "caffe/layers/mlu_absval_layer.hpp"

namespace caffe {


template <typename Dtype>
void MLUAbsValLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
  AbsValLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void MLUAbsValLayer<Dtype>::Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {
  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
  BaseDataType mlu_dtype = bottom[0]->mlu_type();
  AbsValLayer<Dtype>::Reshape(bottom, top);
  top[0]->Reshape(top[0]->shape(), cpu_dtype, mlu_dtype, CNML_TENSOR);
}

template <typename Dtype>
void MLUAbsValLayer<Dtype>::MLUDestroyOp() {
  if (absval_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&absval_op_ptr_));
    absval_op_ptr_ = nullptr;
  }
}

template <typename Dtype>
void MLUAbsValLayer<Dtype>::MLUCreateOpBindData(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  MLU_CHECK(cnmlCreateAbsOp(&absval_op_ptr_,
                              bottom[0]->mlu_tensor(),
                              top[0]->mlu_tensor()));
}

template <typename Dtype>
void MLUAbsValLayer<Dtype>::MLUCompileOp() {
  MLU_CHECK(cnmlCompileBaseOp(absval_op_ptr_,
                              Caffe::rt_core(),
                              Caffe::core_number()));
}

template <typename Dtype>
MLUAbsValLayer<Dtype>::~MLUAbsValLayer() {
  MLUDestroyOp();
}

template <typename Dtype>
void MLUAbsValLayer<Dtype>::Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
  MLU_CHECK(cnmlComputeActiveOpForward_V3(absval_op_ptr_,
                                        bottom[0]->mutable_mlu_data(),
                                        top[0]->mutable_mlu_data(),
                                        Caffe::forward_param(),
                                        Caffe::queue()));
}

INSTANTIATE_CLASS(MLUAbsValLayer);

}  // namespace caffe
#endif



