/*
All modification made by Cambricon Corporation: © 2018 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2018, the respective contributors
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
#include "caffe/layers/mlu_detection_pose_output_layer.hpp"
namespace caffe {

template <typename Dtype>
void MLUDetectionPoseOutputLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  DetectionPoseOutputLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void MLUDetectionPoseOutputLayer<Dtype>::Reshape_tensor(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  DetectionPoseOutputLayer<Dtype>::Reshape(bottom, top);
  vector<int> mlu_shape(4, 1);
  mlu_shape[0] = bottom[0]->num();
  mlu_shape[1] = this->keep_top_k_;
  mlu_shape[2] = 1;
  mlu_shape[3] = 8;
  top[0]->Reshape(mlu_shape);
}

template <typename Dtype>
void MLUDetectionPoseOutputLayer<Dtype>::MLUCreateOpBindData(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  MLU_CHECK(cnmlCreateSsdDetectionPoseOutputOpParam(
      &detection_pose_output_op_param_,
      this->num_classes_,
      this->share_location_,
      this->background_label_id_,
      this->code_type_ - 1,
      this->variance_encoded_in_target_,
      this->confidence_threshold_,
      this->nms_threshold_,
      this->top_k_,
      this->keep_top_k_,
      this->share_pose_,
      this->num_poses_,
      CNML_NCHW));

  MLU_CHECK(cnmlCreateSsdDetectionPoseOutputOp(&detection_pose_output_op_ptr_,
      bottom[0]->mlu_tensor(),
      bottom[1]->mlu_tensor(),
      bottom[2]->mlu_tensor(),
      bottom[3]->mlu_tensor(),
      top[0]->mlu_tensor(),
      detection_pose_output_op_param_));
}

template <typename Dtype>
void MLUDetectionPoseOutputLayer<Dtype>::MLUCompileOp() {
    MLU_CHECK(cnmlCompileBaseOp(detection_pose_output_op_ptr_,
                                Caffe::rt_core(),
                                Caffe::model_parallel()));
}

template <typename Dtype>
void MLUDetectionPoseOutputLayer<Dtype>::fuse(MFusion<Dtype>* fuser) {
    fuser->fuse(detection_pose_output_op_ptr_);
}

template <typename Dtype>
void MLUDetectionPoseOutputLayer<Dtype>::MLUDestroyOp() {
  if (detection_pose_output_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&detection_pose_output_op_ptr_));
    detection_pose_output_op_ptr_ = nullptr;
  }
}

template <typename Dtype>
MLUDetectionPoseOutputLayer<Dtype>::~MLUDetectionPoseOutputLayer() {
  MLUDestroyOp();
}

template <typename Dtype>
void MLUDetectionPoseOutputLayer<Dtype>::Forward_mlu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  MLU_CHECK(cnmlComputeSsdDetectionPoseOutputOpForward_V3(
      detection_pose_output_op_ptr_,
      bottom[0]->mutable_mlu_data(),
      bottom[1]->mutable_mlu_data(),
      bottom[2]->mutable_mlu_data(),
      bottom[3]->mutable_mlu_data(),
      top[0]->mutable_mlu_data(),
      Caffe::forward_param(), Caffe::queue()));
}

INSTANTIATE_CLASS(MLUDetectionPoseOutputLayer);
}  // namespace caffe
#endif  // USE_MLU
