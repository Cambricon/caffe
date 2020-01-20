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
#include "caffe/layers/mlu_roi_pooling_layer.hpp"
namespace caffe {
template <typename Dtype>
void MLUROIPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {
  ROIPoolingLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void MLUROIPoolingLayer<Dtype>::Reshape_tensor(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  this->channels_ = bottom[0]->channels();
  this->height_ = bottom[0]->height();
  this->width_ = bottom[0]->width();
  vector<int> top_shape(4, 1);
  top_shape[0] = bottom[1]->channels() * bottom[0]->num();
  top_shape[1] = this->channels_;
  top_shape[2] = this->pooled_height_;
  top_shape[3] = this->pooled_width_;

  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
  BaseDataType mlu_dtype = bottom[0]->mlu_type();
  top[0]->Reshape(top_shape, cpu_dtype, mlu_dtype, CNML_TENSOR);
}

template <typename Dtype>
void MLUROIPoolingLayer<Dtype>::fuse(MFusion<Dtype>* fuser) {
  fuser->fuse(roi_pool_op_ptr_);
}

template <typename Dtype>
MLUROIPoolingLayer<Dtype>::~MLUROIPoolingLayer() {
  MLUDestroyOp();
}

template <typename Dtype>
void MLUROIPoolingLayer<Dtype>::MLUDestroyOp() {
  if (roi_pool_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&roi_pool_op_ptr_));
    roi_pool_op_ptr_ = nullptr;
  }
  if (roi_pool_ptr_param != nullptr) {
    MLU_CHECK(cnmlDestroyPluginRoiPoolOpParam(&roi_pool_ptr_param));
    roi_pool_ptr_param = nullptr;
  }
}

template <typename Dtype>
void MLUROIPoolingLayer<Dtype>::MLUCreateOpBindData(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int int8_mode = this->int8_context;
  int batch_size = bottom[0]->num();
  cnmlTensor_t mluTensor_inputs[2];
  mluTensor_inputs[0] = bottom[0]->mlu_tensor();
  mluTensor_inputs[1] = bottom[1]->mlu_tensor();

  cnmlTensor_t mluTensor_outputs[1];
  mluTensor_outputs[0] = top[0]->mlu_tensor();

  MLU_CHECK(cnmlCreatePluginRoiPoolOpParam(&roi_pool_ptr_param,
        this->channels_,
        this->height_,
        this->width_,
        this->pooled_height_,
        this->pooled_width_,
        this->rois_num_,
        this->roi_cols_,
        batch_size,
        this->spatial_scale_,
        int8_mode,
        Caffe::rt_core()));

  MLU_CHECK(cnmlCreatePluginRoiPoolOp(&roi_pool_op_ptr_,
        roi_pool_ptr_param,
        mluTensor_inputs,
        mluTensor_outputs
        ));

}

template <typename Dtype>
void MLUROIPoolingLayer<Dtype>::MLUCompileOp() {
  MLU_CHECK(cnmlCompileBaseOp(
      roi_pool_op_ptr_,
      Caffe::rt_core(),
      Caffe::core_number()));
}

template <typename Dtype>
void MLUROIPoolingLayer<Dtype>::Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
  void* mlutensor_input_ptrs[2];
  mlutensor_input_ptrs[0] = bottom[0]->mutable_mlu_data();
  mlutensor_input_ptrs[1] = bottom[1]->mutable_mlu_data();
  void* mlutensor_output_ptrs[1];
  mlutensor_output_ptrs[0] = top[0]->mutable_mlu_data();

  MLU_CHECK(cnmlComputePluginRoiPoolOpForward(roi_pool_op_ptr_,
                                       mlutensor_input_ptrs,
                                       bottom.size(),
                                       mlutensor_output_ptrs,
                                       top.size(),
                                       Caffe::queue()));
}

INSTANTIATE_CLASS(MLUROIPoolingLayer);

}  // namespace caffe
#endif  // USE_MLU
