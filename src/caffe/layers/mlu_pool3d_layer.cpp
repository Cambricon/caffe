/*
All modification made by Cambricon Corporation: © 2018-2019 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2019, the respective contributors
All rights reserved.
For the list of contributors go to
https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md
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
#include <algorithm>
#include <cfloat>
#include <sstream>
#include <vector>
#include "caffe/layers/mlu_pool3d_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MLUPooling3DLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  Pooling3DLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void MLUPooling3DLayer<Dtype>::Reshape_tensor(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
  BaseDataType mlu_dtype = DT_FLOAT16;
  top[0]->Reshape(top[0]->shape(), cpu_dtype, mlu_dtype, CNML_TENSOR);
}

template <typename Dtype>
void MLUPooling3DLayer<Dtype>::MLUCreateOpBindData(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // pool_mode
  cnmlPoolMode_t pool_mode = CNML_POOL_MAX;
  if (this->layer_param_.pooling3d_param().pool() ==
      Pooling3DParameter_PoolMethod_AVE) {
    pool_mode = CNML_POOL_AVG;
  }
  // dilations
  int dim_size = bottom[0]->shape().size();
  int array_length = dim_size - 2;
  int dilations[array_length];
  for (int i = 0; i < array_length; i++) {
    dilations[i] = 1;
  }
  // kernel_size
  int kernel_size[array_length];
  kernel_size[0] = this->kernel_depth_;
  kernel_size[1] = kernel_size[2] = this->kernel_size_;

  // strides
  int strides[array_length];
  strides[0] = this->temporal_stride_;
  strides[1] = this->stride_;
  strides[2] = this->stride_;

  int paddings[array_length][2];
  for (int i = 0; i < array_length; i++) {
    for (int j = 0; j < 2; j++) {
      paddings[i][j] = this->pad_;
    }
  }

  // createop
  MLU_CHECK(cnmlCreateNdPoolOpParam(&param_ptr_, pool_mode, CNML_POOL_KFULL, false,
                                    array_length, kernel_size, dilations,
                                    strides, paddings));

  MLU_CHECK(cnmlCreateNdPoolOp(&pool_op_, param_ptr_, bottom[0]->mlu_tensor(),
                               top[0]->mlu_tensor()));

  MLU_CHECK(cnmlSetOperationComputingLayout(pool_op_, CNML_NDHWC));
}

template <typename Dtype>
void MLUPooling3DLayer<Dtype>::MLUCompileOp() {
  MLU_CHECK(
      cnmlCompileBaseOp(pool_op_, Caffe::rt_core(), Caffe::core_number()));
}

template <typename Dtype>
void MLUPooling3DLayer<Dtype>::Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {
  MLU_CHECK(cnmlComputeNdPoolOpForward(pool_op_, bottom[0]->mutable_mlu_data(),
                                       top[0]->mutable_mlu_data(),
                                       Caffe::forward_param(), Caffe::queue()));
}

template <typename Dtype>
void MLUPooling3DLayer<Dtype>::fuse(MFusion<Dtype>* fuser) {
  fuser->fuse(pool_op_);
}

template <typename Dtype>
void MLUPooling3DLayer<Dtype>::MLUDestroyOp() {
  if (pool_op_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&pool_op_));
    pool_op_ = nullptr;
  }
  if (param_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyNdPoolOpParam(&param_ptr_));
    param_ptr_ = nullptr;
  }
}

INSTANTIATE_CLASS(MLUPooling3DLayer);

}  // namespace caffe
#endif  // USE_MLU
