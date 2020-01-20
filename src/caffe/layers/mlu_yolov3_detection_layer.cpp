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
#include "caffe/layers/mlu_yolov3_detection_layer.hpp"
#include <sstream>
#include <vector>

namespace caffe {
template <typename Dtype>
void MLUYolov3DetectionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                                const vector<Blob<Dtype>*>& top) {
  Yolov3DetectionLayer<Dtype>::LayerSetUp(bottom, top);
  vector<int> const_shape(4, 1);
  const_shape[1] = 64;
  this->c_arr_blob_.Reshape(const_shape, DT_INT32, DT_INT32, CNML_CONST);
  this->h_arr_blob_.Reshape(const_shape, DT_INT32, DT_INT32, CNML_CONST);
  this->w_arr_blob_.Reshape(const_shape, DT_INT32, DT_INT32, CNML_CONST);
  const_shape[1] = 64;
  this->biases_blob_.Reshape(const_shape, DT_FLOAT32, DT_FLOAT32, CNML_CONST);
}

template <typename Dtype>
void MLUYolov3DetectionLayer<Dtype>::Reshape_tensor(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
  BaseDataType mlu_dtype = bottom[0]->mlu_type();
  vector<int> top_shape(4, 1);
  top_shape[0] = bottom[0]->num();
  top_shape[1] = 7 * this->num_box_ + 64;
  top[0]->Reshape(top_shape, cpu_dtype, mlu_dtype, CNML_TENSOR);

  int batch_size = bottom[0]->shape(0);
  int* h_arr = reinterpret_cast<int*>(this->h_arr_blob_.mutable_cpu_data());
  int* w_arr = reinterpret_cast<int*>(this->w_arr_blob_.mutable_cpu_data());

  vector<int> fake_input_shape = {1, 64, 1, 1};
  fake_input_blob0_.reset(
      new Blob<Dtype>(fake_input_shape, cpu_dtype, mlu_dtype, CNML_CONST));
  Dtype* fake_input_ptr0 = fake_input_blob0_->mutable_cpu_data();
  for (int i = 0; i < fake_input_blob0_->count(); i++) {
    fake_input_ptr0[i] = 0.1;
  }
  this->fake_input_blob0_->Reshape(fake_input_shape, cpu_dtype, mlu_dtype,
                                   CNML_CONST);
  // buffer_blob
  int buffer_size =
      255 * (h_arr[0] * w_arr[0] + h_arr[1] * w_arr[1] + h_arr[2] * w_arr[2]);
  vector<int> buffer_shape = {batch_size, buffer_size, 1, 1};
  buffer_blob_.reset(
      new Blob<Dtype>(buffer_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));
}

template <typename Dtype>
void MLUYolov3DetectionLayer<Dtype>::MLUCreateOpBindData(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // create detection op
  int batch_size = bottom[0]->num();
  int kBottomSize = bottom.size();
  int max_box_num = this->num_box_;
  int im_w = this->im_w_;
  int im_h = this->im_h_;

  // inputs
  cnmlTensor_t mlutensor_inputs[kBottomSize];
  mlutensor_inputs[0] = bottom[0]->mlu_tensor();
  mlutensor_inputs[1] = bottom[1]->mlu_tensor();
  mlutensor_inputs[2] = bottom[2]->mlu_tensor();

  // outputs
  int kTopSize = 2;
  cnmlTensor_t mlutensor_outputs[kTopSize];
  mlutensor_outputs[0] = top[0]->mlu_tensor();
  mlutensor_outputs[1] = buffer_blob_->mlu_tensor();

  // static
  MLU_CHECK(cnmlCreatePluginYolov3DetectionOutputOpParam(&yolov3_ptr_param_,
        batch_size,
        kBottomSize,
        this->num_classes_,
        this->anchor_num_,
        max_box_num,
        im_w,
        im_h,
        this->confidence_threshold_,
        this->nms_threshold_,
        Caffe::rt_core(),
        reinterpret_cast<int*>(this->w_arr_blob_.sync_data()),
        reinterpret_cast<int*>(this->h_arr_blob_.sync_data()),
        reinterpret_cast<float*>(this->biases_blob_.sync_data())));

  MLU_CHECK(cnmlCreatePluginYolov3DetectionOutputOp(&yolo_op_ptr_,
                     yolov3_ptr_param_,
                     mlutensor_inputs,
                     mlutensor_outputs));
}

template <typename Dtype>
void MLUYolov3DetectionLayer<Dtype>::MLUDestroyOp() {
  if (this->yolo_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&this->yolo_op_ptr_));
    this->yolo_op_ptr_ = nullptr;
  }
  if (yolov3_ptr_param_ != nullptr) {
    MLU_CHECK(cnmlDestroyPluginYolov3DetectionOutputOpParam(&yolov3_ptr_param_));
    yolov3_ptr_param_ = nullptr;
  }
}

template <typename Dtype>
void MLUYolov3DetectionLayer<Dtype>::MLUCompileOp() {
  MLU_CHECK(cnmlCompileBaseOp(yolo_op_ptr_, Caffe::rt_core(),
                              Caffe::core_number()));
}

template <typename Dtype>
void MLUYolov3DetectionLayer<Dtype>::fuse(MFusion<Dtype>* fuser) {
  fuser->fuse(this->yolo_op_ptr_);
}

template <typename Dtype>
MLUYolov3DetectionLayer<Dtype>::~MLUYolov3DetectionLayer() {
  MLUDestroyOp();
}

template <typename Dtype>
void MLUYolov3DetectionLayer<Dtype>::Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                                                 const vector<Blob<Dtype>*>& top) {
  int bottom_size = bottom.size();
  void* mlutensor_input_ptrs[bottom_size];
  mlutensor_input_ptrs[0] = bottom[0]->mutable_mlu_data();
  mlutensor_input_ptrs[1] = bottom[1]->mutable_mlu_data();
  mlutensor_input_ptrs[2] = bottom[2]->mutable_mlu_data();

  int top_size = 2;
  void* mlutensor_output_ptrs[top_size];
  mlutensor_output_ptrs[0] = top[0]->mutable_mlu_data();
  mlutensor_output_ptrs[1] = buffer_blob_->mutable_mlu_data();
  MLU_CHECK(cnmlComputePluginYolov3DetectionOutputOpForward(yolo_op_ptr_,
        mlutensor_input_ptrs,
        bottom_size,
        mlutensor_output_ptrs,
        top_size,
        Caffe::forward_param(),
        Caffe::queue()));
}

INSTANTIATE_CLASS(MLUYolov3DetectionLayer);

}  // namespace caffe
#endif  // USE_MLU
