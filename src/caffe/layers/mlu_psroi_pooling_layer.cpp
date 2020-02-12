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

#ifdef USE_MLU
#include <vector>
#include "caffe/layers/mlu_psroi_pooling_layer.hpp"

namespace caffe {
template <typename Dtype>
void MLUPSROIPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
  PSROIPoolingLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void MLUPSROIPoolingLayer<Dtype>::Reshape_tensor(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->channels(),
      this->output_dim_*this->group_size_*this->group_size_)
      << "input channel number does not match layer parameters";

  vector<int> top_shape(4, 1);
  top_shape[0] = bottom[1]->channels() * bottom[0]->num();
  top_shape[1] = this->output_dim_;
  top_shape[2] = this->group_size_;
  top_shape[3] = this->group_size_;

  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
  BaseDataType mlu_dtype = bottom[0]->mlu_type();
  top[0]->Reshape(top_shape, cpu_dtype, mlu_dtype, CNML_TENSOR);

  vector<int> reshape_shape1(4, 1);
  reshape_shape1[0] = bottom[0]->num();
  reshape_shape1[1] = this->group_size_ * this->group_size_;
  reshape_shape1[2] = bottom[0]->height() * bottom[0]->width();
  reshape_shape1[3] = this->output_dim_;

  vector<int> transpose_shape(4, 1);
  transpose_shape[0] = bottom[0]->num();
  transpose_shape[1] = this->output_dim_;
  transpose_shape[2] = bottom[0]->height() * bottom[0]->width();
  transpose_shape[3] = this->group_size_ * this->group_size_;

  bottom_reshape_blob1_ = new Blob<Dtype>(reshape_shape1, cpu_dtype,
                                          mlu_dtype, CNML_TENSOR);
  bottom_transpose_blob_ = new Blob<Dtype>(transpose_shape, cpu_dtype,
                                           mlu_dtype, CNML_TENSOR);
}

template <typename Dtype>
void MLUPSROIPoolingLayer<Dtype>::MLUCreateOpBindData(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // reshape bottom shape(n, output_dim * group_size_ * group_size_, h, w) to
  // shape(n, output_dim, group_size_ * group_size_, h * w)
  int bottom_dim[4];
  bottom_dim[0] = bottom_reshape_blob1_->mlu_shape()[0];
  bottom_dim[1] = bottom_reshape_blob1_->mlu_shape()[1];
  bottom_dim[2] = bottom_reshape_blob1_->mlu_shape()[2];
  bottom_dim[3] = bottom_reshape_blob1_->mlu_shape()[3];
  MLU_CHECK(cnmlCreateNdReshapeOpParam(&bottom_reshape_param_ptr1_,
                                     bottom_dim,
                                     4));
  MLU_CHECK(cnmlCreateReshapeOp(&bottom_reshape_op_ptr1_,
                                bottom_reshape_param_ptr1_,
                                bottom[0]->mlu_tensor(),
                                bottom_reshape_blob1_->mlu_tensor()));
  // transpose (n aligned(output_dim) , group_size_ * group_size_, h * w) to
  // (n, h * w, group_size_ * group_size_, aligned(output_dim))
  int dim[4] = {0, 1, 3, 2};
  MLU_CHECK(cnmlCreateNdTransposeOpParam(&bottom_transpose_param_ptr_,
                                         dim,
                                         4));
  MLU_CHECK(cnmlCreateNdTransposeProOp(&bottom_transpose_op_ptr_,
                                       bottom_reshape_blob1_->mlu_tensor(),
                                       bottom_transpose_blob_->mlu_tensor(),
                                       bottom_transpose_param_ptr_));

  int num_rois;
  if (bottom[1]->num_axes() == 2) {
    num_rois = bottom[1]->num();
  } else {
    num_rois = bottom[1]->channels();
  }
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  int batchsize = bottom[0]->num();
  int rois_offset = 5;
  int int8_mode = false;

  cnmlTensor_t mluTensor_inputs[2];
  mluTensor_inputs[0] = bottom_transpose_blob_->mlu_tensor();
  mluTensor_inputs[1] = bottom[1]->mlu_tensor();

  cnmlTensor_t mluTensor_outputs[1];
  mluTensor_outputs[0] = top[0]->mlu_tensor();

  MLU_CHECK(cnmlCreatePluginPsRoiPoolOpParam(&psroi_pooling_ptr_param,
        batchsize,
        int8_mode,
        this->output_dim_,
        this->group_size_,
        height,
        width,
        this->group_size_,
        this->group_size_,
        num_rois,
        rois_offset,
        this->spatial_scale_,
        Caffe::rt_core()));

  MLU_CHECK(cnmlCreatePluginPsRoiPoolOp(&psroi_pool_op_ptr_,
                     psroi_pooling_ptr_param,
                     mluTensor_inputs,
                     mluTensor_outputs));
}

template <typename Dtype>
void MLUPSROIPoolingLayer<Dtype>::MLUCompileOp() {
  MLU_CHECK(cnmlCompileBaseOp(bottom_reshape_op_ptr1_,
                              Caffe::rt_core(),
                              Caffe::core_number()));
  MLU_CHECK(cnmlCompileBaseOp(bottom_transpose_op_ptr_,
                              Caffe::rt_core(),
                              Caffe::core_number()));
  MLU_CHECK(cnmlCompileBaseOp(psroi_pool_op_ptr_,
                              Caffe::rt_core(),
                              Caffe::core_number()));
}
template <typename Dtype>
void MLUPSROIPoolingLayer<Dtype>::MLUDestroyOp() {
  if (bottom_reshape_param_ptr1_ != nullptr) {
    MLU_CHECK(cnmlDestroyReshapeOpParam(&bottom_reshape_param_ptr1_));
    bottom_reshape_param_ptr1_ = nullptr;
  }
  if (bottom_reshape_op_ptr1_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&bottom_reshape_op_ptr1_));
    bottom_reshape_op_ptr1_ = nullptr;
  }
  if (bottom_transpose_param_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyNdTransposeOpParam(&bottom_transpose_param_ptr_));
    bottom_transpose_param_ptr_ = nullptr;
  }
  if (bottom_transpose_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&bottom_transpose_op_ptr_));
    bottom_transpose_op_ptr_ = nullptr;
  }
  if (psroi_pool_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&psroi_pool_op_ptr_));
    psroi_pool_op_ptr_ = nullptr;
  }
  if (psroi_pooling_ptr_param != nullptr) {
    MLU_CHECK(cnmlDestroyPluginPsRoiPoolOpParam(&psroi_pooling_ptr_param));
    psroi_pooling_ptr_param = nullptr;
  }
}

template <typename Dtype>
MLUPSROIPoolingLayer<Dtype>::~MLUPSROIPoolingLayer() {
  MLUDestroyOp();
}

template <typename Dtype>
void MLUPSROIPoolingLayer<Dtype>::Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                                              const vector<Blob<Dtype>*>& top) {
  // reshape bottom shape(n, output_dim * group_size_ * group_size_, h, w) to
  // shape(n, output_dim, group_size_ * group_size_, h * w)
  MLU_CHECK(cnmlComputeReshapeOpForward_V3(bottom_reshape_op_ptr1_,
                                           bottom[0]->mutable_mlu_data(),
                                           bottom_reshape_blob1_->mutable_mlu_data(),
                                           Caffe::forward_param(), Caffe::queue()));
  // transpose (n, output_dim, group_size_ * group_size_, h * w) to
  // (n, h * w, group_size_ * group_size_, aligned(output_dim))
  MLU_CHECK(cnmlComputeNdTransposeProOpForward(bottom_transpose_op_ptr_,
                                         bottom_reshape_blob1_->mutable_mlu_data(),
                                         bottom_transpose_blob_->mutable_mlu_data(),
                                         Caffe::forward_param(), Caffe::queue()));
  void* mluTensor_input_ptrs[2];
  mluTensor_input_ptrs[0] = bottom_transpose_blob_->mutable_mlu_data();
  mluTensor_input_ptrs[1] = bottom[1]->mutable_mlu_data();
  void* mluTensor_output_ptrs[1];
  mluTensor_output_ptrs[0] = top[0]->mutable_mlu_data();

  MLU_CHECK(cnmlComputePluginPsroipoolOpForward(psroi_pool_op_ptr_,
                                          mluTensor_input_ptrs,
                                          bottom.size(),
                                          mluTensor_output_ptrs,
                                          top.size(),
                                          Caffe::forward_param(),
                                          Caffe::queue()));
}

INSTANTIATE_CLASS(MLUPSROIPoolingLayer);

}  // namespace caffe
#endif  // USE_MLU
