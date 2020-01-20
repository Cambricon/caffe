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
#include "caffe/layers/mlu_detection_output_layer.hpp"
namespace caffe {
template <typename Dtype>
void MLUDetectionOutputLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  DetectionOutputLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void MLUDetectionOutputLayer<Dtype>::Reshape_tensor(
        const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
  vector<int> mlu_shape(4, 1);
  mlu_shape[0] = bottom[0]->num();
  mlu_shape[1] = this->keep_top_k_ * 7 + 64;
  top[0]->Reshape(mlu_shape);

  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
  BaseDataType mlu_dtype = bottom[0]->mlu_type();

  num_preds_per_class = bottom[0]->shape(1) / 4;
  num_classes = this->num_classes_;
  int num = bottom[0]->shape(0);
  vector<int> loc_reshape_shape = {num, 4, 1, num_preds_per_class};
  vector<int> conf_reshape_shape = {num, num_classes, 1, num_preds_per_class};
  vector<int> prior_reshape_shape = {1, 2, num_preds_per_class, 4};

  loc_blob1_.reset(new Blob<Dtype>(loc_reshape_shape,
                                   cpu_dtype, mlu_dtype, CNML_TENSOR));
  conf_blob1_.reset(new Blob<Dtype>(conf_reshape_shape,
                                    cpu_dtype, mlu_dtype, CNML_TENSOR));
  priors_blob1_.reset(new Blob<Dtype>(prior_reshape_shape,
                                      cpu_dtype, mlu_dtype, CNML_TENSOR));

  vector<int> loc_trans_shape = {num, num_preds_per_class, 1, 4};
  vector<int> conf_trans_shape = {num, num_preds_per_class, 1, num_classes};
  vector<int> prior_trans_shape = {1, num_preds_per_class, 2, 4};
  loc_blob2_.reset(new Blob<Dtype>(loc_trans_shape,
                                   cpu_dtype, mlu_dtype, CNML_TENSOR));
  conf_blob2_.reset(new Blob<Dtype>(conf_trans_shape,
                                    cpu_dtype, mlu_dtype, CNML_TENSOR));
  priors_blob2_.reset(new Blob<Dtype>(prior_trans_shape,
                                      cpu_dtype, mlu_dtype, CNML_TENSOR));
  int align_size = 64;
  int channel = this->num_classes_ * 3;
  int channel_plus = ((channel - 1) / align_size + 1) * align_size;
  int num_preds_per_class_plus =
    ((num_preds_per_class - 1) / align_size + 1) * align_size;

  if (!Caffe::simpleFlag()) {
    vector<int> loc_trans_shape1 = {1, channel_plus, 1, num_preds_per_class_plus};
    tmp_blob_.reset(new Blob<Dtype>(loc_trans_shape1,
                                     cpu_dtype, mlu_dtype, CNML_TENSOR));
  } else {
    vector<int> loc_trans_shape1 = {Caffe::batchsize(),
                                    channel_plus , 1, num_preds_per_class_plus};
    tmp_blob_.reset(new Blob<Dtype>(loc_trans_shape1,
                                    cpu_dtype, mlu_dtype, CNML_TENSOR));
  }
}

template <typename Dtype>
void MLUDetectionOutputLayer<Dtype>::MLUCreateOpBindData(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  int loc_dim[4];
  loc_dim[0] = loc_blob1_->mlu_shape()[0];
  loc_dim[1] = loc_blob1_->mlu_shape()[1];
  loc_dim[2] = loc_blob1_->mlu_shape()[2];
  loc_dim[3] = loc_blob1_->mlu_shape()[3];
  MLU_CHECK(cnmlCreateNdReshapeOpParam(&loc_reshape_param_,
                                     loc_dim,
                                     4));
  MLU_CHECK(cnmlCreateReshapeOp(&loc_reshape_op_,
                                loc_reshape_param_,
                                bottom[0]->mlu_tensor(),
                                loc_blob1_->mlu_tensor()));
  int conf_dim[4];
  conf_dim[0] = conf_blob1_->mlu_shape()[0];
  conf_dim[1] = conf_blob1_->mlu_shape()[1];
  conf_dim[2] = conf_blob1_->mlu_shape()[2];
  conf_dim[3] = conf_blob1_->mlu_shape()[3];
  MLU_CHECK(cnmlCreateNdReshapeOpParam(&conf_reshape_param_,
                                     conf_dim,
                                     4));
  MLU_CHECK(cnmlCreateReshapeOp(&conf_reshape_op_,
                                conf_reshape_param_,
                                bottom[1]->mlu_tensor(),
                                conf_blob1_->mlu_tensor()));
  int prior_dim[4];
  prior_dim[0] = priors_blob1_->mlu_shape()[0];
  prior_dim[1] = priors_blob1_->mlu_shape()[1];
  prior_dim[2] = priors_blob1_->mlu_shape()[2];
  prior_dim[3] = priors_blob1_->mlu_shape()[3];
  MLU_CHECK(cnmlCreateNdReshapeOpParam(&prior_reshape_param_,
                                     prior_dim,
                                     4));
  MLU_CHECK(cnmlCreateReshapeOp(&prior_reshape_op_,
                                prior_reshape_param_,
                                bottom[2]->mlu_tensor(),
                                priors_blob1_->mlu_tensor()));

  int dim_order[4] = {0, 1, 3, 2};
  MLU_CHECK(cnmlCreateNdTransposeOpParam(&loc_transpose_param_,
                                       dim_order,
                                       4));

  MLU_CHECK(cnmlCreateNdTransposeProOp(&loc_transpose_op_,
                                     loc_blob1_->mlu_tensor(),
                                     loc_blob2_->mlu_tensor(),
                                     loc_transpose_param_));

  MLU_CHECK(cnmlCreateNdTransposeOpParam(&conf_transpose_param_,
                                       dim_order,
                                       4));

  MLU_CHECK(cnmlCreateNdTransposeProOp(&conf_transpose_op_,
                                     conf_blob1_->mlu_tensor(),
                                     conf_blob2_->mlu_tensor(),
                                     conf_transpose_param_));
  int dim_order_prior[4] = {0, 3, 2, 1};
  MLU_CHECK(cnmlCreateNdTransposeOpParam(&prior_transpose_param_,
                                       dim_order_prior,
                                       4));
  MLU_CHECK(cnmlCreateNdTransposeProOp(&prior_transpose_op_,
                                     priors_blob1_->mlu_tensor(),
                                     priors_blob2_->mlu_tensor(),
                                     prior_transpose_param_));

  int num_priors_ = num_preds_per_class;
  int code_type = this->code_type_ - 1;
  int batch_size = bottom[0]->num();
  int8_mode = this->int8_context;
  int clip = 0;
  int pad_size = 64;
  int pad_size_const = 64;
  cnmlTensor_t mlutensor_inputs[3];
  mlutensor_inputs[0] = loc_blob2_->mlu_tensor();
  mlutensor_inputs[1] = conf_blob2_->mlu_tensor();
  mlutensor_inputs[2] = priors_blob2_->mlu_tensor();
  cnmlTensor_t mlutensor_outputs[2];
  mlutensor_outputs[0] = top[0]->mlu_tensor();
  mlutensor_outputs[1] = tmp_blob_->mlu_tensor();

  MLU_CHECK(cnmlCreatePluginSsdDetectionOutputOpParam(
        &ssd_detection_param_,
        batch_size,
        num_priors_,
        this->num_classes_,
        this->share_location_,
        this->background_label_id_,
        code_type,
        this->variance_encoded_in_target_,
        clip,
        this->top_k_,
        this->keep_top_k_,
        0,
        pad_size,
        pad_size_const,
        this->confidence_threshold_,
        this->nms_threshold_,
        Caffe::rt_core()));
  MLU_CHECK(cnmlCreatePluginSsdDetectionOutputOp(&detection_plugin_op_ptr_,
                     ssd_detection_param_,
                     mlutensor_inputs,
                     mlutensor_outputs,
                     nullptr));

}

template <typename Dtype>
void MLUDetectionOutputLayer<Dtype>::MLUDestroyOp() {
  if (loc_reshape_op_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&loc_reshape_op_));
    loc_reshape_op_ = nullptr;
  }
  if (loc_reshape_param_ != nullptr) {
    MLU_CHECK(cnmlDestroyReshapeOpParam(&loc_reshape_param_));
    loc_reshape_param_ = nullptr;
  }
  if (conf_reshape_op_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&conf_reshape_op_));
    conf_reshape_op_ = nullptr;
  }
  if (conf_reshape_param_ != nullptr) {
    MLU_CHECK(cnmlDestroyReshapeOpParam(&conf_reshape_param_));
    conf_reshape_param_ = nullptr;
  }
  if (prior_reshape_op_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&prior_reshape_op_));
    prior_reshape_op_ = nullptr;
  }
  if (prior_reshape_param_ != nullptr) {
    MLU_CHECK(cnmlDestroyReshapeOpParam(&prior_reshape_param_));
    prior_reshape_param_ = nullptr;
  }
  if (loc_transpose_op_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&loc_transpose_op_));
    loc_transpose_op_ = nullptr;
  }
  if (loc_transpose_param_ != nullptr) {
    MLU_CHECK(cnmlDestroyNdTransposeOpParam(&loc_transpose_param_));
    loc_transpose_param_ = nullptr;
  }
  if (conf_transpose_op_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&conf_transpose_op_));
    conf_transpose_op_ = nullptr;
  }
  if (conf_transpose_param_ != nullptr) {
    MLU_CHECK(cnmlDestroyNdTransposeOpParam(&conf_transpose_param_));
    conf_transpose_param_ = nullptr;
  }
  if (prior_transpose_op_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&prior_transpose_op_));
    prior_transpose_op_ = nullptr;
  }
  if (prior_transpose_param_ != nullptr) {
    MLU_CHECK(cnmlDestroyNdTransposeOpParam(&prior_transpose_param_));
    prior_transpose_param_ = nullptr;
  }
  if (detection_plugin_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&detection_plugin_op_ptr_));
    detection_plugin_op_ptr_ = nullptr;
  }
  if (ssd_detection_param_ != nullptr) {
    MLU_CHECK(cnmlDestroyPluginSsdDetectionOutputOpParam(&ssd_detection_param_));
    ssd_detection_param_ = nullptr;
  }
}

template <typename Dtype>
MLUDetectionOutputLayer<Dtype>::~MLUDetectionOutputLayer() {
  MLUDestroyOp();
}

template <typename Dtype>
void MLUDetectionOutputLayer<Dtype>::MLUCompileOp() {
  MLU_CHECK(cnmlCompileBaseOp(loc_reshape_op_,
                              Caffe::rt_core(),
                              Caffe::core_number()));
  MLU_CHECK(cnmlCompileBaseOp(conf_reshape_op_,
                              Caffe::rt_core(),
                              Caffe::core_number()));
  MLU_CHECK(cnmlCompileBaseOp(prior_reshape_op_,
                              Caffe::rt_core(),
                              Caffe::core_number()));
  MLU_CHECK(cnmlCompileBaseOp(loc_transpose_op_,
                              Caffe::rt_core(),
                              Caffe::core_number()));
  MLU_CHECK(cnmlCompileBaseOp(conf_transpose_op_,
                              Caffe::rt_core(),
                              Caffe::core_number()));
  MLU_CHECK(cnmlCompileBaseOp(prior_transpose_op_,
                              Caffe::rt_core(),
                              Caffe::core_number()));
  MLU_CHECK(cnmlCompileBaseOp(detection_plugin_op_ptr_,
                              Caffe::rt_core(),
                              Caffe::core_number()));
}

template <typename Dtype>
void MLUDetectionOutputLayer<Dtype>::fuse(MFusion<Dtype>* fuser) {
  fuser->fuse(loc_reshape_op_);
  fuser->fuse(conf_reshape_op_);
  fuser->fuse(prior_reshape_op_);
  fuser->fuse(loc_transpose_op_);
  fuser->fuse(conf_transpose_op_);
  fuser->fuse(prior_transpose_op_);
  fuser->fuse(detection_plugin_op_ptr_);
}

template <typename Dtype>
void MLUDetectionOutputLayer<Dtype>::Forward_mlu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  MLU_CHECK(cnmlComputeReshapeOpForward_V3(loc_reshape_op_,
                   bottom[0]->mutable_mlu_data(),
                   loc_blob1_->mutable_mlu_data(),
                   Caffe::forward_param(), Caffe::queue()));

  MLU_CHECK(cnmlComputeReshapeOpForward_V3(conf_reshape_op_,
                   bottom[1]->mutable_mlu_data(),
                   conf_blob1_->mutable_mlu_data(),
                   Caffe::forward_param(), Caffe::queue()));

  MLU_CHECK(cnmlComputeReshapeOpForward_V3(prior_reshape_op_,
                   bottom[2]->mutable_mlu_data(),
                   priors_blob1_->mutable_mlu_data(),
                   Caffe::forward_param(), Caffe::queue()));

  MLU_CHECK(cnmlComputeNdTransposeProOpForward(loc_transpose_op_,
                              loc_blob1_->mutable_mlu_data(),
                              loc_blob2_->mutable_mlu_data(),
                              Caffe::forward_param(), Caffe::queue()));

  MLU_CHECK(cnmlComputeNdTransposeProOpForward(conf_transpose_op_,
                              conf_blob1_->mutable_mlu_data(),
                              conf_blob2_->mutable_mlu_data(),
                              Caffe::forward_param(), Caffe::queue()));

  MLU_CHECK(cnmlComputeNdTransposeProOpForward(prior_transpose_op_,
                              priors_blob1_->mutable_mlu_data(),
                              priors_blob2_->mutable_mlu_data(),
                              Caffe::forward_param(), Caffe::queue()));

  void* mlutensor_input_ptrs[3];
  mlutensor_input_ptrs[0] = loc_blob2_->mutable_mlu_data();
  mlutensor_input_ptrs[1] = conf_blob2_->mutable_mlu_data();
  mlutensor_input_ptrs[2] = priors_blob2_->mutable_mlu_data();
  void* mlutensor_output_ptrs[2];
  mlutensor_output_ptrs[0] = top[0]->mutable_mlu_data();
  mlutensor_output_ptrs[1] = tmp_blob_->mutable_mlu_data();

  MLU_CHECK(cnmlComputePluginSsdDetectionOutputOpForward(detection_plugin_op_ptr_,
                                 mlutensor_input_ptrs,
                                 3,
                                 mlutensor_output_ptrs,
                                 2,
                                 Caffe::forward_param(),
                                 Caffe::queue()));
}
INSTANTIATE_CLASS(MLUDetectionOutputLayer);
}  // namespace caffe
#endif
