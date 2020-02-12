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
#include <iostream>
#include "caffe/layers/mlu_detection_out_layer.hpp"
#define ALIGN_SIZE 256

namespace caffe {
template <typename Dtype>
void MLUDetectionOutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
  DetectionOutLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void MLUDetectionOutLayer<Dtype>::Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                                                 const vector<Blob<Dtype>*>& top) {
  BaseDataType mlu_dtype = bottom[0]->mlu_type();
  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;

  int coords = this->layer_param_.detection_out_param().coords();
  int classes = this->layer_param_.detection_out_param().num_classes();
  int num = this->layer_param_.detection_out_param().num_box();  // Number of bias
  int AHW = num * bottom[0]->height() * bottom[0]->width();  // 5*13*13

  vector<int> top_shape(4, 1);
  top_shape[0] = bottom[0]->num();  // 1
  top_shape[1] = 256;  // 5*13*13
  top_shape[2] = 1;
  top_shape[3] = 7;  // number of output in each bbox
  top[0]->Reshape(top_shape, cpu_dtype, mlu_dtype, CNML_TENSOR);

  bbox_shufflechannel_shape.resize(4);
  bbox_shufflechannel_shape[0] = bottom[0]->num();
  bbox_shufflechannel_shape[1] = bottom[0]->channels();
  bbox_shufflechannel_shape[2] = bottom[0]->height();
  bbox_shufflechannel_shape[3] = bottom[0]->width();
  bbox_shufflechannel_blob.Reshape(bbox_shufflechannel_shape);

  bbox_transpose_shape.resize(4);
  bbox_transpose_shape[0] = bottom[0]->num();
  bbox_transpose_shape[1] = bottom[0]->width();
  bbox_transpose_shape[2] = bottom[0]->channels();
  bbox_transpose_shape[3] = bottom[0]->height();
  bbox_transpose_blob.Reshape(bbox_transpose_shape);

  vector<int> biases_shape(4, 1);
  biases_shape[0] = 1;
  biases_shape[1] = AHW;
  biases_shape[2] = 1;
  biases_shape[3] = coords;
  // biases_blob.Reshape(biases_shape, cpu_dtype, mlu_dtype, CNML_CONST);
  biases_blob.Reshape(biases_shape, cpu_dtype, DT_FLOAT32, CNML_CONST);

  /* used to occupy gdram memory to temparialy save bbox data before nms */
  int mlu_align_size = 256;
  vector<int> temp_buffer_shape(4, 1);
  if (Caffe::simpleFlag())
    temp_buffer_shape[0] = Caffe::batchsize();
  else
    temp_buffer_shape[0] = 1;
  temp_buffer_shape[1] = (classes + coords + 1) * classes;
  temp_buffer_shape[2] = 1;
  temp_buffer_shape[3] = ((AHW - 1) / mlu_align_size + 1) * mlu_align_size;
  temp_buffer_blob.Reshape(temp_buffer_shape);
}

template <typename Dtype>
void MLUDetectionOutLayer<Dtype>::MLUCreateOpBindData(const vector<Blob<Dtype>*>& bottom,
                                                      const vector<Blob<Dtype>*>& top) {
  int num_classes = this->layer_param_.detection_out_param().num_classes();
  int num_box = this->layer_param_.detection_out_param().num_box();
  int coords = this->layer_param_.detection_out_param().coords();
  float confidence_threshold =
                  this->layer_param_.detection_out_param().confidence_threshold();
  float nms_threshold = this->layer_param_.detection_out_param().nms_threshold();
  biases.resize(this->layer_param_.detection_out_param().biases_size());
  for (int i = 0; i < biases.size(); i++)
    biases[i] = this->layer_param_.detection_out_param().biases(i);

  int h = bottom[0]->height();
  int w = bottom[0]->width();
  int batch = bottom[0]->num();
  int n = num_classes + coords + 1;  // number of parameter in each output box
  int int8_mode = false;
  /* create biases */
  int idx = 0;
  for (int k = 0; k < num_box; k++) {
    for (int j = 0; j < h; j++) {
      for (int i = 0; i < w; i++) {
        biases_blob.mutable_cpu_data()[idx * 4 + 0] = i;
        biases_blob.mutable_cpu_data()[idx * 4 + 1] = j;
        biases_blob.mutable_cpu_data()[idx * 4 + 2] = biases[2*k + 0];
        biases_blob.mutable_cpu_data()[idx * 4 + 3] = biases[2*k + 1];
        idx++;
      }
    }
  }

  cnmlTensor_t mluTensor_input[1];
  cnmlTensor_t mluTensor_output[1];
  mluTensor_input[0] = bottom[0]->mlu_tensor();
  mluTensor_output[0] = bbox_shufflechannel_blob.mlu_tensor();
  MLU_CHECK(cnmlCreateShuffleChannelOp(&shufflechannel_op_ptr_,
            mluTensor_input,
            mluTensor_output,
            num_box));

  int dim_order[4] = {0, 3, 1, 2};
  MLU_CHECK(cnmlCreateNdTransposeOpParam(&bbox_transpose_param_,
                                         dim_order,
                                         4));
  MLU_CHECK(cnmlCreateNdTransposeProOp(&bbox_transpose_ptr_,
                                       bbox_shufflechannel_blob.mlu_tensor(),
                                       bbox_transpose_blob.mlu_tensor(),
                                       bbox_transpose_param_));

  // data re-organize
  cnmlTensor_t mlutensor_inputs[1];
  mlutensor_inputs[0] = bbox_transpose_blob.mlu_tensor();
  cnmlTensor_t mlutensor_outputs[2];
  mlutensor_outputs[0] = top[0]->mlu_tensor();
  mlutensor_outputs[1] = temp_buffer_blob.mlu_tensor();

  MLU_CHECK(cnmlCreatePluginYolov2DetectionOutputOpParam(&yolov2_ptr_param_,
        w,
        h,
        num_classes,
        num_box,
        coords,
        n,
        batch,
        int8_mode,
        confidence_threshold,
        nms_threshold,
        Caffe::rt_core(),
        reinterpret_cast<float*>(biases_blob.sync_data())));

  MLU_CHECK(cnmlCreatePluginYolov2DetectionOutputOp(&detection_out_op_ptr_,
                     yolov2_ptr_param_,
                     mlutensor_inputs,
                     mlutensor_outputs));
}

template <typename Dtype>
void MLUDetectionOutLayer<Dtype>::MLUCompileOp() {
  MLU_CHECK(cnmlCompileBaseOp(shufflechannel_op_ptr_,
                              Caffe::rt_core(),
                              Caffe::core_number()));
  MLU_CHECK(cnmlCompileBaseOp(bbox_transpose_ptr_,
                               Caffe::rt_core(),
                               Caffe::core_number()));
  MLU_CHECK(cnmlCompileBaseOp(detection_out_op_ptr_,
                               Caffe::rt_core(),
                               Caffe::core_number()));
}

template <typename Dtype>
void MLUDetectionOutLayer<Dtype>::Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                                              const vector<Blob<Dtype>*>& top) {
  void* inputs[1];
  void* outputs[1];
  inputs[0] = bottom[0]->mutable_mlu_data();
  outputs[0] = bbox_shufflechannel_blob.mutable_mlu_data();
  // shuffle 1*(5*25)*13*13
  MLU_CHECK(cnmlComputeShuffleChannelOpForward_V3(shufflechannel_op_ptr_,
            inputs,
            outputs,
            Caffe::forward_param(),
            Caffe::queue()));
  // transpose 1*(5*13*13)*1*25
  MLU_CHECK(cnmlComputeNdTransposeProOpForward(bbox_transpose_ptr_,
            bbox_shufflechannel_blob.mutable_mlu_data(),
            bbox_transpose_blob.mutable_mlu_data(),
            Caffe::forward_param(),
            Caffe::queue()));

  void* mlutensor_input_ptrs[1];
  mlutensor_input_ptrs[0] = bbox_transpose_blob.mutable_mlu_data();
  void* mlutensor_output_ptrs[2];
  mlutensor_output_ptrs[0] = top[0]->mutable_mlu_data();
  mlutensor_output_ptrs[1] = temp_buffer_blob.mutable_mlu_data();
  MLU_CHECK(cnmlComputePluginYolov2DetectionOutputOpForward_V2(detection_out_op_ptr_,
                                          mlutensor_input_ptrs,
                                          1,
                                          mlutensor_output_ptrs,
                                          2,
                                          Caffe::forward_param(),
                                          Caffe::queue()));
}

template <typename Dtype>
void MLUDetectionOutLayer<Dtype>::fuse(MFusion<Dtype>* fuser) {
  fuser->fuse(shufflechannel_op_ptr_);
  fuser->fuse(bbox_transpose_ptr_);
  fuser->fuse(detection_out_op_ptr_);
}

template <typename Dtype>
void MLUDetectionOutLayer<Dtype>::MLUDestroyOp() {
  if (shufflechannel_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&shufflechannel_op_ptr_));
    shufflechannel_op_ptr_ = nullptr;
  }
  if (bbox_transpose_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&bbox_transpose_ptr_));
    bbox_transpose_ptr_ = nullptr;
  }
  if (bbox_transpose_param_ != nullptr) {
    MLU_CHECK(cnmlDestroyNdTransposeOpParam(&bbox_transpose_param_));
    bbox_transpose_param_ = nullptr;
  }
  if (detection_out_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&detection_out_op_ptr_));
    detection_out_op_ptr_ = nullptr;
  }
  if (yolov2_ptr_param_ != nullptr) {
    MLU_CHECK(cnmlDestroyPluginYolov2DetectionOutputOpParam(&yolov2_ptr_param_));
    yolov2_ptr_param_ = nullptr;
  }
}

INSTANTIATE_CLASS(MLUDetectionOutLayer);
}  // namespace caffe

#endif   // USE_MLU
