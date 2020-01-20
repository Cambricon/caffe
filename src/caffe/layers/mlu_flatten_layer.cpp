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

#ifdef USE_MLU
#include <vector>
#include "caffe/layers/mlu_flatten_layer.hpp"

namespace caffe {

template <typename Dtype>
void MLUFlattenLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
  FlattenLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void MLUFlattenLayer<Dtype>::Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
  FlattenLayer<Dtype>::Reshape(bottom, top);
  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
  BaseDataType mlu_dtype = bottom[0]->mlu_type();
  top[0]->Reshape(top[0]->shape(), cpu_dtype, mlu_dtype, CNML_TENSOR);
  transpose_d2h_blob_.Reshape(bottom[0]->shape(), cpu_dtype,
                              mlu_dtype, CNML_TENSOR, CNML_NHWC);
  transpose_h2d_blob_.Reshape(top[0]->shape(), cpu_dtype,
                              mlu_dtype, CNML_TENSOR, CNML_NHWC);
}

template <typename Dtype>
void MLUFlattenLayer<Dtype>::MLUCreateOpBindData(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // trans1
  int bottom_axes = bottom[0]->mlu_shape().size();
  vector<int> trans_dim_order_d2h(bottom_axes, 0);
  trans_dim_order_d2h[1] = bottom_axes - 1;
  for (int i = 2; i < bottom_axes; i++) {
    trans_dim_order_d2h[i] = i-1;
  }
  MLU_CHECK(cnmlCreateNdTransposeOpParam(&transpose_d2h_param_,
        trans_dim_order_d2h.data(),
        bottom_axes));

  MLU_CHECK(cnmlCreateNdTransposeProOp(&transpose_d2h_op_,
                                     bottom[0]->mlu_tensor(),
                                     transpose_d2h_blob_.mlu_tensor(),
                                     transpose_d2h_param_));
  // reshape
  int output_axes = top[0]->mlu_shape().size();
  vector<int> transpose_h2d_shape = transpose_h2d_blob_.mlu_shape();

  MLU_CHECK(cnmlCreateNdReshapeOpParam(&reshape_param_,
                                     transpose_h2d_shape.data(),
                                     output_axes));
  MLU_CHECK(cnmlCreateReshapeOp(&reshape_op_ptr_,
                              reshape_param_,
                              transpose_d2h_blob_.mlu_tensor(),
                              transpose_h2d_blob_.mlu_tensor()));
  // NCHW => NHWC
  vector<int> trans_dim_order_h2d(output_axes, 0);
  trans_dim_order_h2d[output_axes - 1] = 1;
  for (int i = 1; i < output_axes - 1; i++) {
    trans_dim_order_h2d[i] = i + 1;
  }
  MLU_CHECK(cnmlCreateNdTransposeOpParam(&transpose_h2d_param_,
        trans_dim_order_h2d.data(),
        output_axes));
  MLU_CHECK(cnmlCreateNdTransposeProOp(&transpose_h2d_op_,
                                     transpose_h2d_blob_.mlu_tensor(),
                                     top[0]->mlu_tensor(),
                                     transpose_h2d_param_));
}

template <typename Dtype>
void MLUFlattenLayer<Dtype>::MLUCompileOp() {
  MLU_CHECK(cnmlCompileBaseOp(transpose_d2h_op_,
                              Caffe::rt_core(),
                              Caffe::core_number()));
  MLU_CHECK(cnmlCompileBaseOp(reshape_op_ptr_,
                              Caffe::rt_core(),
                              Caffe::core_number()));
  MLU_CHECK(cnmlCompileBaseOp(transpose_h2d_op_,
                              Caffe::rt_core(),
                              Caffe::core_number()));
}

template <typename Dtype>
void MLUFlattenLayer<Dtype>::Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  MLU_CHECK(cnmlComputeTransposeProOpForward_V3(transpose_d2h_op_,
                              bottom[0]->mutable_mlu_data(),
                              transpose_d2h_blob_.mutable_mlu_data(),
                              Caffe::forward_param(), Caffe::queue()));
  MLU_CHECK(cnmlComputeReshapeOpForward_V3(reshape_op_ptr_,
                              transpose_d2h_blob_.mutable_mlu_data(),
                              transpose_h2d_blob_.mutable_mlu_data(),
                              Caffe::forward_param(), Caffe::queue()));
  MLU_CHECK(cnmlComputeTransposeProOpForward_V3(transpose_h2d_op_,
                              transpose_h2d_blob_.mutable_mlu_data(),
                              top[0]->mutable_mlu_data(),
                              Caffe::forward_param(), Caffe::queue()));
}

template <typename Dtype>
void MLUFlattenLayer<Dtype>::fuse(MFusion<Dtype>* fuser) {
  fuser->fuse(transpose_d2h_op_);
  fuser->fuse(reshape_op_ptr_);
  fuser->fuse(transpose_h2d_op_);
}

template <typename Dtype>
void MLUFlattenLayer<Dtype>::MLUDestroyOp() {
  if (transpose_d2h_op_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&transpose_d2h_op_));
    transpose_d2h_op_ = nullptr;
  }

  if (transpose_d2h_param_ != nullptr) {
    MLU_CHECK(cnmlDestroyNdTransposeOpParam(&transpose_d2h_param_));
    transpose_d2h_param_ = nullptr;
  }
  if (reshape_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&reshape_op_ptr_));
    reshape_op_ptr_ = nullptr;
  }

  if (reshape_param_ != nullptr) {
    MLU_CHECK(cnmlDestroyReshapeOpParam(&reshape_param_));
    reshape_param_ = nullptr;
  }
  if (transpose_h2d_op_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&transpose_h2d_op_));
    transpose_h2d_op_ = nullptr;
  }

  if (transpose_h2d_param_ != nullptr) {
    MLU_CHECK(cnmlDestroyNdTransposeOpParam(&transpose_h2d_param_));
    transpose_h2d_param_ = nullptr;
  }
  if (transpose_h2d_op_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&transpose_h2d_op_));
    transpose_h2d_op_ = nullptr;
  }

  if (transpose_h2d_param_ != nullptr) {
    MLU_CHECK(cnmlDestroyNdTransposeOpParam(&transpose_h2d_param_));
    transpose_h2d_param_ = nullptr;
  }
}

INSTANTIATE_CLASS(MLUFlattenLayer);

}  // namespace caffe

#endif  // USE_MLU
