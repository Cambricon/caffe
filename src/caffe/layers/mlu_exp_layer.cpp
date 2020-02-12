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
#include <cmath>
#include "caffe/layers/mlu_exp_layer.hpp"

namespace caffe {

template <typename Dtype>
void MLUEXPLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top) {
  ExpLayer<Dtype>::LayerSetUp(bottom, top);
  base = this->layer_param_.exp_param().base();
  shift = this->layer_param_.exp_param().shift();
  scale = this->layer_param_.exp_param().scale();
}

template <typename Dtype>
void MLUEXPLayer<Dtype>::Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
  BaseDataType mlu_dtype = bottom[0]->mlu_type();
  top[0]->Reshape(bottom[0]->shape(), cpu_dtype, mlu_dtype, CNML_TENSOR);
  scale_blob_.Reshape(bottom[0]->shape(), cpu_dtype, mlu_dtype, CNML_TENSOR);

  // alpha: 1*c*1*1, beta: 1*1*1*1
  vector<int> param_shape(4, 1);
  beta_blob_.Reshape(param_shape, cpu_dtype, mlu_dtype, CNML_CONST);
  param_shape[1] = bottom[0]->channels();
  alpha_blob_.Reshape(param_shape, cpu_dtype, mlu_dtype, CNML_CONST);
}

template <typename Dtype>
void MLUEXPLayer<Dtype>::MLUCreateOpBindData(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
  // MLUEXPLayer is implemented through ScaleOp and ExpOp.
  // f(x) = e ^ (scale * log(base) * x + shift * log(base));

  /* alpha, beta */
  if (base == Dtype(-1)) {
    // log(base) = 1
    caffe_set(alpha_blob_.count(), Dtype(scale), alpha_blob_.mutable_cpu_data());
    caffe_set(beta_blob_.count(), Dtype(shift), beta_blob_.mutable_cpu_data());
  } else {
    caffe_set(alpha_blob_.count(), Dtype(scale*log(base)),
        alpha_blob_.mutable_cpu_data());
    caffe_set(beta_blob_.count(), Dtype(shift*log(base)),
        beta_blob_.mutable_cpu_data());
  }
  /* ScaleOp: */
  MLU_CHECK(cnmlCreateScaleOp(&scale_op_ptr_,
                              bottom[0]->mlu_tensor(),
                              scale_blob_.mlu_tensor(),
                              alpha_blob_.mlu_tensor(),
                              beta_blob_.mlu_tensor()));

  MLU_CHECK(cnmlBindConstData_V2(alpha_blob_.mlu_tensor(),
                              alpha_blob_.sync_data(),
                              false));
  MLU_CHECK(cnmlBindConstData_V2(beta_blob_.mlu_tensor(),
                              beta_blob_.sync_data(),
                              false));

  /* ExpOp: */
  MLU_CHECK(cnmlCreateExpOp(&exp_op_ptr_,
                              scale_blob_.mlu_tensor(),
                              top[0]->mlu_tensor()));
}

template <typename Dtype>
void MLUEXPLayer<Dtype>::MLUCompileOp() {
  MLU_CHECK(cnmlCompileBaseOp(scale_op_ptr_,
                              Caffe::rt_core(),
                              Caffe::core_number()));
  MLU_CHECK(cnmlCompileBaseOp(exp_op_ptr_,
                              Caffe::rt_core(),
                              Caffe::core_number()));
}


template <typename Dtype>
void MLUEXPLayer<Dtype>::Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top) {
  MLU_CHECK(cnmlComputeScaleOpForward_V3(scale_op_ptr_,
                              bottom[0]->mutable_mlu_data(),
                              scale_blob_.mutable_mlu_data(),
                              Caffe::forward_param(),
                              Caffe::queue()));
  MLU_CHECK(cnmlComputeExpOpForward_V3(exp_op_ptr_,
                              scale_blob_.mutable_mlu_data(),
                              top[0]->mutable_mlu_data(),
                              Caffe::forward_param(),
                              Caffe::queue()));
}

template <typename Dtype>
void MLUEXPLayer<Dtype>::fuse(MFusion<Dtype>* fuser) {
  fuser->fuse(scale_op_ptr_);
  fuser->fuse(exp_op_ptr_);
}


template <typename Dtype>
void MLUEXPLayer<Dtype>::MLUDestroyOp() {
  if (scale_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&scale_op_ptr_));
    scale_op_ptr_ = nullptr;
  }
  if (exp_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&exp_op_ptr_));
    exp_op_ptr_ = nullptr;
  }
}

INSTANTIATE_CLASS(MLUEXPLayer);

}  // namespace caffe
#endif
