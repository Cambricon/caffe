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
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <fstream>
#include <iostream>
#include "caffe/layers/mlu_image_detect_layer.hpp"

namespace caffe {
template <typename Dtype>
void MLUImageDetectLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
  ImageDetectLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void MLUImageDetectLayer<Dtype>::Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                                                const vector<Blob<Dtype>*>& top) {
  const int bbox_C = bottom[0]->channels();  // (num of class) * 4
  const int BATCH_SIZE = bottom[2]->num();  // num of batch_size
  const int ROIS_NUM = bottom[2]->channels();  // num of rois
  const int CLASS = bottom[1]->channels();  // num of class
  top[0]->Reshape(BATCH_SIZE, ROIS_NUM * (bbox_C / 4), 1, 6);

  bbox_reshape_shape.resize(4);
  bbox_reshape_shape[0] = BATCH_SIZE;
  bbox_reshape_shape[1] = bbox_C;
  bbox_reshape_shape[2] = ROIS_NUM;
  bbox_reshape_shape[3] = 1;
  bbox_reshape_blob.Reshape(bbox_reshape_shape);
  score_reshape_shape.resize(4);
  score_reshape_shape[0] = BATCH_SIZE;
  score_reshape_shape[1] = CLASS;
  score_reshape_shape[2] = ROIS_NUM;
  score_reshape_shape[3] = 1;
  score_reshape_blob.Reshape(score_reshape_shape);

  buffer_reshape_shape.resize(4, 1);
  buffer_reshape_shape[1] = 64;
  buffer_blob.Reshape(buffer_reshape_shape);

  vector<int> bbox_transpose_shape(4, 1);
  bbox_transpose_shape[0] = BATCH_SIZE;
  bbox_transpose_shape[1] = ROIS_NUM;
  bbox_transpose_shape[2] = 1;
  bbox_transpose_shape[3] = bbox_C;
  bbox_transpose_blob.Reshape(bbox_transpose_shape);
  vector<int> score_transpose_shape(4, 1);
  score_transpose_shape[0] = BATCH_SIZE;
  score_transpose_shape[1] = ROIS_NUM;
  score_transpose_shape[2] = 1;
  score_transpose_shape[3] = CLASS;
  score_transpose_blob.Reshape(score_transpose_shape);
}

template <typename Dtype>
void MLUImageDetectLayer<Dtype>::MLUCreateOpBindData(
                        const vector<Blob<Dtype>*>& bottom,
                        const vector<Blob<Dtype>*>& top) {
  int box_dim[4] = {bbox_reshape_blob.mlu_shape()[0], bbox_reshape_blob.mlu_shape()[1],
                    bbox_reshape_blob.mlu_shape()[2], bbox_reshape_blob.mlu_shape()[3]};
  MLU_CHECK(cnmlCreateNdReshapeOpParam(&bbox_reshape_param,
                                       box_dim,
                                       4));
  MLU_CHECK(cnmlCreateReshapeOp(&bbox_reshape_ptr,
                                 bbox_reshape_param,
                                 bottom[0]->mlu_tensor(),
                                 bbox_reshape_blob.mlu_tensor()));
  int dim_trans[4] = {0, 2, 3, 1};
  MLU_CHECK(cnmlCreateNdTransposeOpParam(&bbox_transpose_param,
                                         dim_trans,
                                         4));
  MLU_CHECK(cnmlCreateNdTransposeProOp(&bbox_transpose_ptr,
                                       bbox_reshape_blob.mlu_tensor(),
                                       bbox_transpose_blob.mlu_tensor(),
                                       bbox_transpose_param));

  int score_dim[4] =
        {score_reshape_blob.mlu_shape()[0], score_reshape_blob.mlu_shape()[1],
         score_reshape_blob.mlu_shape()[2], score_reshape_blob.mlu_shape()[3]};
  MLU_CHECK(cnmlCreateNdReshapeOpParam(&score_reshape_param,
                                     score_dim,
                                     4));
  MLU_CHECK(cnmlCreateReshapeOp(&score_reshape_ptr,
                                 score_reshape_param,
                                 bottom[1]->mlu_tensor(),
                                 score_reshape_blob.mlu_tensor()));

  MLU_CHECK(cnmlCreateNdTransposeOpParam(&score_transpose_param,
                                         dim_trans,
                                         4));
  MLU_CHECK(cnmlCreateNdTransposeProOp(&score_transpose_ptr,
                                     score_reshape_blob.mlu_tensor(),
                                     score_transpose_blob.mlu_tensor(),
                                     score_transpose_param));

  int rois_num_ = bottom[2]->channels();
  bool int8_ctx = false;
  int batch_size = bottom[2]->num();

  MLU_CHECK(cnmlCreatePluginFasterRcnnDetectionOutputOpParam(&faster_rcnn_opt_param,
        batch_size,
        rois_num_,
        this->num_class_,
        this->im_h_,
        this->im_w_,
        int8_ctx,
        Caffe::rt_core(),
        this->scale_,
        this->nms_thresh_,
        this->score_thresh_));
  MLU_CHECK(cnmlCreatePluginFasterRcnnDetectionOutputOp(&detection_op_ptr_,
        bbox_transpose_blob.mlu_tensor(),
        score_transpose_blob.mlu_tensor(),
        bottom[2]->mlu_tensor(),
        top[0]->mlu_tensor(),
        buffer_blob.mlu_tensor(),
        faster_rcnn_opt_param));
}

template <typename Dtype>
void MLUImageDetectLayer<Dtype>::MLUCompileOp() {
  MLU_CHECK(cnmlCompileBaseOp(bbox_reshape_ptr,
                              Caffe::rt_core(),
                              Caffe::core_number()));
  MLU_CHECK(cnmlCompileBaseOp(bbox_transpose_ptr,
                              Caffe::rt_core(),
                              Caffe::core_number()));
  MLU_CHECK(cnmlCompileBaseOp(score_reshape_ptr,
                              Caffe::rt_core(),
                              Caffe::core_number()));
  MLU_CHECK(cnmlCompileBaseOp(score_transpose_ptr,
                              Caffe::rt_core(),
                              Caffe::core_number()));
  MLU_CHECK(cnmlCompileBaseOp(detection_op_ptr_,
                              Caffe::rt_core(),
                              Caffe::core_number()));
}

template <typename Dtype>
void MLUImageDetectLayer<Dtype>::fuse(MFusion<Dtype>* fuser) {
  fuser->fuse(bbox_reshape_ptr);
  fuser->fuse(bbox_transpose_ptr);
  fuser->fuse(score_reshape_ptr);
  fuser->fuse(score_transpose_ptr);
  fuser->fuse(detection_op_ptr_);
}

template <typename Dtype>
void MLUImageDetectLayer<Dtype>::Forward_mlu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  MLU_CHECK(cnmlComputeReshapeOpForward_V3(bbox_reshape_ptr,
                                           bottom[0]->mutable_mlu_data(),
                                           bbox_reshape_blob.mutable_mlu_data(),
                                           Caffe::forward_param(),
                                           Caffe::queue()));
  MLU_CHECK(cnmlComputeNdTransposeProOpForward(bbox_transpose_ptr,
                                 bbox_reshape_blob.mutable_mlu_data(),
                                 bbox_transpose_blob.mutable_mlu_data(),
                                 Caffe::forward_param(),
                                 Caffe::queue()));

  MLU_CHECK(cnmlComputeReshapeOpForward_V3(score_reshape_ptr,
                                           bottom[1]->mutable_mlu_data(),
                                           score_reshape_blob.mutable_mlu_data(),
                                           Caffe::forward_param(),
                                           Caffe::queue()));
  MLU_CHECK(cnmlComputeNdTransposeProOpForward(score_transpose_ptr,
                                 score_reshape_blob.mutable_mlu_data(),
                                 score_transpose_blob.mutable_mlu_data(),
                                 Caffe::forward_param(),
                                 Caffe::queue()));

  void* mlutensor_input_data[bottom.size()];
  mlutensor_input_data[0] = bbox_transpose_blob.mutable_mlu_data();
  mlutensor_input_data[1] = score_transpose_blob.mutable_mlu_data();
  mlutensor_input_data[2] = bottom[2]->mutable_mlu_data();

  void* mlutensor_output_data[top.size() + 1];
  for (int i = 0; i < top.size(); i++) {
    mlutensor_output_data[i] = top[i]->mutable_mlu_data();
  }
  mlutensor_output_data[top.size()] = buffer_blob.mutable_mlu_data();


  MLU_CHECK(cnmlComputePluginFasterRcnnDetectionOutputOpForward(detection_op_ptr_,
                                          mlutensor_input_data,
                                          bottom.size(),
                                          mlutensor_output_data,
                                          top.size() + 1,
                                          Caffe::forward_param(),
                                          Caffe::queue()));
}

template <typename Dtype>
void MLUImageDetectLayer<Dtype>::MLUDestroyOp() {
  if (bbox_reshape_param != nullptr) {
    MLU_CHECK(cnmlDestroyReshapeOpParam(&bbox_reshape_param));
    bbox_reshape_param = nullptr;
  }
  if (score_reshape_param != nullptr) {
    MLU_CHECK(cnmlDestroyReshapeOpParam(&score_reshape_param));
    score_reshape_param = nullptr;
  }
  if (bbox_reshape_ptr != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&bbox_reshape_ptr));
    bbox_reshape_ptr = nullptr;
  }
  if (score_reshape_ptr != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&score_reshape_ptr));
    score_reshape_ptr = nullptr;
  }
  if (bbox_transpose_param != nullptr) {
    MLU_CHECK(cnmlDestroyNdTransposeOpParam(&bbox_transpose_param));
    bbox_transpose_param = nullptr;
  }
  if (bbox_transpose_ptr != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&bbox_transpose_ptr));
    bbox_transpose_ptr = nullptr;
  }
  if (score_transpose_param != nullptr) {
    MLU_CHECK(cnmlDestroyNdTransposeOpParam(&score_transpose_param));
    score_transpose_param = nullptr;
  }
  if (score_transpose_ptr != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&score_transpose_ptr));
    score_transpose_ptr = nullptr;
  }
  if (detection_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&detection_op_ptr_));
    detection_op_ptr_ = nullptr;
  }
  if (faster_rcnn_opt_param != nullptr) {
    MLU_CHECK(cnmlDestroyPluginFasterRcnnDetectionOutputOpParam(&faster_rcnn_opt_param));
    faster_rcnn_opt_param = nullptr;
  }
}

template <typename Dtype>
MLUImageDetectLayer<Dtype>::~MLUImageDetectLayer() {
  MLUDestroyOp();
}

INSTANTIATE_CLASS(MLUImageDetectLayer);
}  // namespace caffe
#endif
