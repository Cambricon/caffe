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

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <vector>
#include "caffe/layers/mlu_slice_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MLUSliceLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
  SliceLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
MLUSliceLayer<Dtype>::~MLUSliceLayer() {
  MLUDestroyOp();
}

template <typename Dtype>
void MLUSliceLayer<Dtype>::Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
  if (top.size() == 1) {
    MLU_CHECK(cnmlComputeDeviceMemcpyOpForward_V3(slice_op_ptr_,
                                                  bottom[0]->mutable_mlu_data(),
                                                  top[0]->mutable_mlu_data(),
                                                  Caffe::forward_param(),
                                                  Caffe::queue()));
    return;
  }

  void* mlutensor_input_ptrs[bottom.size()];
  void* mlutensor_output_ptrs[top.size()];
  cnmlTensor_t mlu_tensor_input[bottom.size()];
  cnmlTensor_t mlu_tensor_output[top.size()];
  for (int i = 0; i < bottom.size(); i++) {
    mlutensor_input_ptrs[i] = bottom[i]->mutable_mlu_data();
    mlu_tensor_input[i] = bottom[i]->mlu_tensor();
  }
  for (int i = 0; i < top.size(); i++) {
    mlutensor_output_ptrs[i] =  top[i]->mutable_mlu_data();
    mlu_tensor_output[i] = top[i]->mlu_tensor();
  }
  MLU_CHECK(cnmlComputeNdSplitOpForward_V2(slice_op_ptr_,
                                           mlu_tensor_input,
                                           mlutensor_input_ptrs,
                                           bottom.size(),
                                           mlu_tensor_output,
                                           mlutensor_output_ptrs,
                                           top.size(),
                                           Caffe::queue(),
                                           nullptr));
}

template <typename Dtype>
void MLUSliceLayer<Dtype>::MLUDestroyOp() {
  if (slice_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&slice_op_ptr_));
    slice_op_ptr_ = nullptr;
  }
}

template <typename Dtype>
void MLUSliceLayer<Dtype>::MLUCreateOpBindData(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  if (top.size() == 1) {
    MLU_CHECK(cnmlCreateDeviceMemcpyOp(&slice_op_ptr_,
                                      bottom[0]->mlu_tensor(),
                                      top[0]->mlu_tensor()));
    return;
  }
  int kBottomSize = bottom.size();
  cnmlTensor_t mlutensor_inputs[kBottomSize];
  for (int i = 0; i < bottom.size(); i++)
    mlutensor_inputs[i] = bottom[0]->mlu_tensor();

  int kTopSize = top.size();
  cnmlTensor_t mlutensor_outputs[kTopSize];
  for (int i = 0; i < top.size(); i++)
    mlutensor_outputs[i] = top[i]->mlu_tensor();
  int length = bottom[0]->shape().size();
  vector<int> dim_order(length, 1);  // = {0, 3, 2, 1};
  dim_order[0] = 0;
  dim_order[1] = length - 1;
  for (int i = 2; i < length; i++) {
      dim_order[i] = i-1;
  }
  int slice_axis = dim_order[this->slice_axis_];

  MLU_CHECK(cnmlCreateNdSplitOp(&slice_op_ptr_,
                                 slice_axis,
                                 mlutensor_inputs,
                                 bottom.size(),
                                 mlutensor_outputs,
                                 top.size()));
}

template <typename Dtype>
void MLUSliceLayer<Dtype>::Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  SliceLayer<Dtype>::Reshape(bottom, top);
  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
  for (int i = 0; i < top.size(); ++i) {
    BaseDataType mlu_dtype = bottom[0]->mlu_type();
    top[i]->Reshape(top[i]->shape(), cpu_dtype, mlu_dtype, CNML_TENSOR);
  }
}

INSTANTIATE_CLASS(MLUSliceLayer);

}  // namespace caffe
#endif  // USE_MLU
