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
#include "caffe/layer.hpp"
#include "caffe/layers/mlu_ssd_detection_layer.hpp"
#include "caffe/net.hpp"
#include "caffe/util/format.hpp"

namespace caffe {
template <typename Dtype>
void MLUSsdDetectionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
  const DetectionOutputParameter& detection_output_param =
      this->layer_param_.detection_output_param();
  CHECK(detection_output_param.has_num_classes()) << "Must specify num_classes";
  num_classes_ = detection_output_param.num_classes();
  share_location_ = detection_output_param.share_location();
  num_loc_classes_ = share_location_ ? 1 : num_classes_;
  background_label_id_ = detection_output_param.background_label_id();
  code_type_ = detection_output_param.code_type();
  variance_encoded_in_target_ =
      detection_output_param.variance_encoded_in_target();
  keep_top_k_ = detection_output_param.keep_top_k();
  confidence_threshold_ = detection_output_param.has_confidence_threshold() ?
      detection_output_param.confidence_threshold() : -1;
  nms_threshold_ = detection_output_param.nms_param().nms_threshold();
  CHECK_GE(nms_threshold_, 0.) << "nms_threshold must be non negative.";
  top_k_ = -1;
  if (detection_output_param.nms_param().has_top_k()) {
    top_k_ = detection_output_param.nms_param().top_k();
  }
  bottom_nums = bottom.size();
  bottom_each_size = (bottom_nums - 1)/3;
  loc_reshape_ops_.resize(bottom_each_size);
  conf_reshape_ops_.resize(bottom_each_size);
  loc_reshape_params_.resize(bottom_each_size);
  conf_reshape_params_.resize(bottom_each_size);

  for (int i = 0; i < bottom_each_size; i++) {
    num_preds_per_class += bottom[i]->shape(1) /
        4 * bottom[i]->shape(2) * bottom[i]->shape(3);
  }
}

template <typename Dtype>
void MLUSsdDetectionLayer<Dtype>::Reshape_tensor(
        const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
  BaseDataType mlu_dtype = bottom[0]->mlu_type();
  vector<int> mlu_shape(4, 1);
  mlu_shape[0] = bottom[0]->num();
  mlu_shape[1] = this->keep_top_k_ * 7 + 64;
  mlu_shape[2] = 1;
  mlu_shape[3] = 1;

  top[0]->Reshape(mlu_shape, cpu_dtype, mlu_dtype, CNML_TENSOR);

  /* reshape loc blob && conf blob   */
  loc_blobs_.clear();
  conf_blobs_.clear();
  loc_blobs_.resize(bottom_each_size);
  conf_blobs_.resize(bottom_each_size);
  for (int i = 0 ; i < bottom_each_size; i++) {
    vector<int> loc_shape = bottom[i]->shape();
    vector<int> conf_shape = bottom[i]->shape();
    int c = bottom[i]->shape(1) / 4;
    loc_shape[1] = 4;
    conf_shape[1] = this->num_classes_;
    loc_shape[2] = conf_shape[2] = 1;
    loc_shape[3] = conf_shape[3] = c * bottom[i]->shape(2) * bottom[i]->shape(3);
    loc_blobs_[i].reset(new Blob<Dtype>(loc_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));
    conf_blobs_[i].reset(new Blob<Dtype>(conf_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));
  }
  vector<int> concat_shape = loc_blobs_[0]->shape();
  concat_shape[3] = num_preds_per_class;
  loc_concat_blob.Reshape(concat_shape, cpu_dtype, mlu_dtype, CNML_TENSOR);
  concat_shape[1] = this->num_classes_;
  conf_concat_blob.Reshape(concat_shape, cpu_dtype, mlu_dtype, CNML_TENSOR);

  vector<int> loc_trans_shape = loc_concat_blob.shape();
  loc_trans_shape[1] = loc_trans_shape[3];
  loc_trans_shape[3] = 4;
  loc_trans_blob.Reshape(loc_trans_shape, cpu_dtype, mlu_dtype, CNML_TENSOR);
  loc_trans_shape[3] = this->num_classes_;
  conf_trans_blob.Reshape(loc_trans_shape, cpu_dtype, mlu_dtype, CNML_TENSOR);
  conf_softmax_blob.Reshape(loc_trans_shape, cpu_dtype, mlu_dtype, CNML_TENSOR);

  int align_size = 64;
  int channel = this->num_classes_ * 3;
  int channel_plus = ((channel - 1) / align_size + 1) * align_size;

  int num_preds_per_class_plus =
    ((num_preds_per_class - 1) / align_size + 1) * align_size;
  if (!Caffe::simpleFlag()) {
    vector<int> loc_trans_shape1 = {1, channel_plus , 1, num_preds_per_class_plus};
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
void MLUSsdDetectionLayer<Dtype>::MLUCreateOpBindData(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // create a net named priorbox_concat to caculate priorboxes for ssd
  NetParameter priorbox_concat;

  int bottom_nums = bottom.size();
  int bottom_priorbox_index = 2*(bottom_nums - 1)/3;
  for (int i = 0; i < (bottom_nums - 1)/3; i++) {
    LayerParameter* input_layer_param = priorbox_concat.add_layer();
    input_layer_param->set_type("Input");
    input_layer_param->set_name("input" + format_int(i));
    InputParameter* input_param = input_layer_param->mutable_input_param();
    input_layer_param->add_top("input" + format_int(i));
    input_layer_param->set_engine(caffe::Engine::CAFFE);
    BlobShape input_shape;
    // from bottom[12] to bottom[17] add_dim as input.
    for (int j = 0; j < bottom[bottom_priorbox_index + i]->num_axes(); ++j) {
      input_shape.add_dim(bottom[bottom_priorbox_index + i]->shape(j));
    }
    input_param->add_shape()->CopyFrom(input_shape);
  }

  {
    LayerParameter* input_layer_param = priorbox_concat.add_layer();
    input_layer_param->set_type("Input");
    input_layer_param->set_name("data");
    InputParameter* input_param = input_layer_param->mutable_input_param();
    input_layer_param->add_top("data");
    input_layer_param->set_engine(caffe::Engine::CAFFE);
    BlobShape input_shape;
    for (int j = 0; j < bottom[bottom_nums-1]->num_axes(); ++j) {
      input_shape.add_dim(bottom[bottom_nums-1]->shape(j));
    }
    input_param->add_shape()->CopyFrom(input_shape);
  }

  for (int i = 0; i < (bottom_nums - 1)/3; i++) {
    LayerParameter* layer_param = priorbox_concat.add_layer();
    layer_param->set_type("PriorBox");
    layer_param->add_bottom("input" + format_int(i));
    layer_param->add_bottom("data");
    layer_param->add_top("priorbox" + format_int(i));
    layer_param->set_name("priorbox" + format_int(i));
    layer_param->set_engine(caffe::Engine::CAFFE);
    *layer_param->mutable_prior_box_param() =
        this->layer_param_.priorbox_params(i);
  }

  {
    LayerParameter* layer_param = priorbox_concat.add_layer();
    layer_param->set_type("Concat");
    for (int i = 0; i < (bottom_nums - 1)/3; i++) {
      layer_param->add_bottom("priorbox" + format_int(i));
    }
    layer_param->add_top("priorbox_concat");
    layer_param->set_name("priorbox_concat");
    layer_param->set_engine(caffe::Engine::CAFFE);
    ConcatParameter concat_param;
    concat_param.set_axis(2);
    *layer_param->mutable_concat_param() =
        concat_param;
  }

  {
    LayerParameter* layer_param = priorbox_concat.add_layer();
    layer_param->set_type("Reshape");
    layer_param->add_bottom("priorbox_concat");
    layer_param->add_top("priorbox_concat_reshape");
    layer_param->set_name("priorbox_concat_reshape");
    layer_param->set_engine(caffe::Engine::CAFFE);
    ReshapeParameter* reshape_param = layer_param->mutable_reshape_param();
    BlobShape input_shape;
    input_shape.add_dim(0);
    input_shape.add_dim(0);
    input_shape.add_dim(-1);
    input_shape.add_dim(4);
    *reshape_param->mutable_shape() = input_shape;
  }

  {
    LayerParameter* layer_param = priorbox_concat.add_layer();
    layer_param->set_type("Permute");
    layer_param->add_bottom("priorbox_concat_reshape");
    layer_param->add_top("priorbox_concat_permute");
    layer_param->set_name("priorbox_concat_permute");
    layer_param->set_engine(caffe::Engine::CAFFE);
    PermuteParameter* permute_param = layer_param->mutable_permute_param();
    permute_param->add_order(0);
    permute_param->add_order(2);
    permute_param->add_order(1);
    permute_param->add_order(3);
  }
  // forward the priorbox concat net to get the blob priorbox_concat_permute
  Net<Dtype> priorbox_concat_net(priorbox_concat);
  priorbox_concat_net.ForwardFromTo_default(
      0,
      priorbox_concat_net.layers().size()-1);

  Blob<Dtype>* temp_blob =
      priorbox_concat_net.blob_by_name("priorbox_concat_permute").get();
  // reshape the temp_blob to priorbox_blob_, then bind the const data
  // and create SsdDetectionOp
  priorbox_blob_.Reshape(
      temp_blob->shape(),
      DT_FLOAT32,
      DT_FLOAT16,
      CNML_CONST);
  caffe_copy(temp_blob->count(),
      temp_blob->cpu_data(),
      priorbox_blob_.mutable_cpu_data());
  for (int i = 0; i < bottom_each_size; i++) {
    int loc_dim[4];
    loc_dim[0] = loc_blobs_[i]->mlu_shape()[0];
    loc_dim[1] = loc_blobs_[i]->mlu_shape()[1];
    loc_dim[2] = loc_blobs_[i]->mlu_shape()[2];
    loc_dim[3] = loc_blobs_[i]->mlu_shape()[3];
    MLU_CHECK(cnmlCreateNdReshapeOpParam(&loc_reshape_params_[i],
                                         loc_dim,
                                         4));
    MLU_CHECK(cnmlCreateReshapeOp(&loc_reshape_ops_[i],
                                  loc_reshape_params_[i],
                                  bottom[i]->mlu_tensor(),
                                  loc_blobs_[i]->mlu_tensor()));
    int conf_dim[4];
    conf_dim[0] = conf_blobs_[i]->mlu_shape()[0];
    conf_dim[1] = conf_blobs_[i]->mlu_shape()[1];
    conf_dim[2] = conf_blobs_[i]->mlu_shape()[2];
    conf_dim[3] = conf_blobs_[i]->mlu_shape()[3];
    MLU_CHECK(cnmlCreateNdReshapeOpParam(&conf_reshape_params_[i],
                                         conf_dim,
                                         4));
    MLU_CHECK(cnmlCreateReshapeOp(&conf_reshape_ops_[i],
                                  conf_reshape_params_[i],
                                  bottom[bottom_each_size + i]->mlu_tensor(),
                                  conf_blobs_[i]->mlu_tensor()));
  }

  cnmlTensor_t loc_inputs_tensor[bottom_each_size];
  for (int i = 0; i < bottom_each_size; i++) {
    loc_inputs_tensor[i] = loc_blobs_[i]->mlu_tensor();
  }
  cnmlTensor_t loc_outputs_tensor = loc_concat_blob.mlu_tensor();
  MLU_CHECK(cnmlCreateNdConcatOp(&loc_concat_op_,
                              2,
                              loc_inputs_tensor,
                              bottom_each_size,
                              &loc_outputs_tensor,
                              1));
  cnmlTensor_t conf_inputs_tensor[bottom_each_size];
  for (int i = 0; i < bottom_each_size; i++) {
    conf_inputs_tensor[i] = conf_blobs_[i]->mlu_tensor();
  }
  cnmlTensor_t conf_outputs_tensor = conf_concat_blob.mlu_tensor();
  MLU_CHECK(cnmlCreateNdConcatOp(&conf_concat_op_,
                              2,
                              conf_inputs_tensor,
                              bottom_each_size,
                              &conf_outputs_tensor,
                              1));
  int dim_order[4] = {0, 1, 3, 2};
  MLU_CHECK(cnmlCreateNdTransposeOpParam(&loc_transpose_param,
                                       dim_order,
                                       4));

  MLU_CHECK(cnmlCreateNdTransposeProOp(&loc_transpose_op_,
                                     loc_concat_blob.mlu_tensor(),
                                     loc_trans_blob.mlu_tensor(),
                                     loc_transpose_param));
  MLU_CHECK(cnmlCreateNdTransposeOpParam(&conf_transpose_param,
                                       dim_order,
                                       4));
  MLU_CHECK(cnmlCreateNdTransposeProOp(&conf_transpose_op_,
                                     conf_concat_blob.mlu_tensor(),
                                     conf_trans_blob.mlu_tensor(),
                                     conf_transpose_param));

  MLU_CHECK(cnmlCreateNdSoftmaxOp(&softmax_op_,
                               2,
                               conf_trans_blob.mlu_tensor(),
                               conf_softmax_blob.mlu_tensor()));

  int num_priors_ = num_preds_per_class;
  int code_type = this->code_type_ - 1;
  int batch_size = bottom[0]->num();
  int clip = 0;
  int pad_size = 64;
  int pad_size_const = 64;
  cnmlTensor_t mlutensor_inputs[2];
  mlutensor_inputs[0] = loc_trans_blob.mlu_tensor();
  mlutensor_inputs[1] = conf_softmax_blob.mlu_tensor();
  cnmlTensor_t mlutensor_outputs[2];
  mlutensor_outputs[0] = top[0]->mlu_tensor();
  mlutensor_outputs[1] = tmp_blob_->mlu_tensor();

  cnmlTensor_t mlutensor_static[1];
  mlutensor_static[0] = priorbox_blob_.mlu_tensor();
  MLU_CHECK(cnmlBindConstData_V2(priorbox_blob_.mlu_tensor(),
                                 priorbox_blob_.sync_data(),
                                 false));

  MLU_CHECK(cnmlCreatePluginSsdDetectionOutputOpParam(
        &ssd_detection_param,
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
        1,
        pad_size,
        pad_size_const,
        this->confidence_threshold_,
        this->nms_threshold_,
        Caffe::rt_core()));
  MLU_CHECK(cnmlCreatePluginSsdDetectionOutputOp(&detection_plugin_op_ptr_,
                     ssd_detection_param,
                     mlutensor_inputs,
                     mlutensor_outputs,
                     mlutensor_static));
}

template <typename Dtype>
void MLUSsdDetectionLayer<Dtype>::MLUDestroyOp() {
  for (int i = 0; i < loc_reshape_params_.size(); i++) {
    if (loc_reshape_params_[i] != nullptr) {
      MLU_CHECK(cnmlDestroyReshapeOpParam(&loc_reshape_params_[i]));
      loc_reshape_params_[i] = nullptr;
    }
    if (loc_reshape_ops_[i] != nullptr) {
      MLU_CHECK(cnmlDestroyBaseOp(&loc_reshape_ops_[i]));
      loc_reshape_ops_[i] = nullptr;
    }
    if (conf_reshape_params_[i] != nullptr) {
      MLU_CHECK(cnmlDestroyReshapeOpParam(&conf_reshape_params_[i]));
      conf_reshape_params_[i] = nullptr;
    }
    if (conf_reshape_ops_[i] != nullptr) {
      MLU_CHECK(cnmlDestroyBaseOp(&conf_reshape_ops_[i]));
      conf_reshape_ops_[i] = nullptr;
    }
  }
  if (loc_concat_op_ != nullptr) {
    cnmlDestroyBaseOp(&loc_concat_op_);
    loc_concat_op_ = nullptr;
  }
  if (conf_concat_op_ != nullptr) {
    cnmlDestroyBaseOp(&conf_concat_op_);
    conf_concat_op_ = nullptr;
  }
  if (loc_transpose_param != nullptr) {
    MLU_CHECK(cnmlDestroyNdTransposeOpParam(&loc_transpose_param));
    loc_transpose_param = nullptr;
  }
  if (loc_transpose_op_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&loc_transpose_op_));
    loc_transpose_op_ = nullptr;
  }
  if (conf_transpose_param != nullptr) {
    MLU_CHECK(cnmlDestroyNdTransposeOpParam(&conf_transpose_param));
    conf_transpose_param = nullptr;
  }
  if (conf_transpose_op_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&conf_transpose_op_));
    conf_transpose_op_ = nullptr;
  }
  if (softmax_op_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&softmax_op_));
    softmax_op_ = nullptr;
  }
  if (ssd_detection_param != nullptr) {
    MLU_CHECK(cnmlDestroyPluginSsdDetectionOutputOpParam(&ssd_detection_param));
    ssd_detection_param = nullptr;
  }
  if (detection_plugin_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&detection_plugin_op_ptr_));
    detection_plugin_op_ptr_ = nullptr;
  }
}

template <typename Dtype>
MLUSsdDetectionLayer<Dtype>::~MLUSsdDetectionLayer() {
  MLUDestroyOp();
}

template <typename Dtype>
void MLUSsdDetectionLayer<Dtype>::Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                                              const vector<Blob<Dtype>*>& top) {
  int bottom_priorbox_index = 2*(bottom_nums - 1)/3;
  std::vector<void*> locs_ptr;
  std::vector<void*> confs_ptr;
  for (int i = 0; i < bottom_priorbox_index; i++) {
    if (i < bottom_priorbox_index/2)
      locs_ptr.push_back(reinterpret_cast<void*>(bottom[i]->mutable_mlu_data()));
    else
      confs_ptr.push_back(reinterpret_cast<void*>(bottom[i]->mutable_mlu_data()));
  }

  for (int i = 0; i < bottom_each_size; i++) {
    MLU_CHECK(cnmlComputeReshapeOpForward_V3(loc_reshape_ops_[i],
                     bottom[i]->mutable_mlu_data(),
                     loc_blobs_[i]->mutable_mlu_data(),
                     Caffe::forward_param(), Caffe::queue()));

    MLU_CHECK(cnmlComputeReshapeOpForward_V3(conf_reshape_ops_[i],
                     bottom[bottom_each_size + i]->mutable_mlu_data(),
                     conf_blobs_[i]->mutable_mlu_data(),
                     Caffe::forward_param(), Caffe::queue()));
  }

  void* mlutensor_loc_input_ptrs[bottom_each_size];
  for (int i = 0; i < bottom_each_size; i++) {
    mlutensor_loc_input_ptrs[i] = loc_blobs_[i]->mutable_mlu_data();
  }
  void* mlutensor_loc_output_ptrs = loc_concat_blob.mutable_mlu_data();
  MLU_CHECK(cnmlComputeConcatOpForward_V3(loc_concat_op_,
                            mlutensor_loc_input_ptrs,
                            bottom_each_size,
                            &mlutensor_loc_output_ptrs,
                            1,
                            Caffe::forward_param(), Caffe::queue()));

  void* mlutensor_conf_input_ptrs[bottom_each_size];
  for (int i = 0; i < bottom_each_size; i++) {
    mlutensor_conf_input_ptrs[i] = conf_blobs_[i]->mutable_mlu_data();
  }
  void* mlutensor_conf_output_ptrs = conf_concat_blob.mutable_mlu_data();
  MLU_CHECK(cnmlComputeConcatOpForward_V3(conf_concat_op_,
                            mlutensor_conf_input_ptrs,
                            bottom_each_size,
                            &mlutensor_conf_output_ptrs,
                            1,
                            Caffe::forward_param(), Caffe::queue()));

  MLU_CHECK(cnmlComputeTransposeProOpForward_V3(loc_transpose_op_,
                              loc_concat_blob.mutable_mlu_data(),
                              loc_trans_blob.mutable_mlu_data(),
                              Caffe::forward_param(), Caffe::queue()));

  MLU_CHECK(cnmlComputeTransposeProOpForward_V3(conf_transpose_op_,
                              conf_concat_blob.mutable_mlu_data(),
                              conf_trans_blob.mutable_mlu_data(),
                              Caffe::forward_param(), Caffe::queue()));

  MLU_CHECK(cnmlComputeNdSoftmaxOpForward(softmax_op_,
                                    conf_trans_blob.mutable_mlu_data(),
                                    conf_softmax_blob.mutable_mlu_data(),
                                    Caffe::forward_param(),
                                    Caffe::queue()));

  void* mlutensor_input_ptrs[2];
  mlutensor_input_ptrs[0] = loc_trans_blob.mutable_mlu_data();
  mlutensor_input_ptrs[1] = conf_softmax_blob.mutable_mlu_data();
  void* mlutensor_output_ptrs[2];
  mlutensor_output_ptrs[0] = top[0]->mutable_mlu_data();
  mlutensor_output_ptrs[1] = tmp_blob_->mutable_mlu_data();

  MLU_CHECK(cnmlComputePluginSsdDetectionOutputOpForward(detection_plugin_op_ptr_,
                                 mlutensor_input_ptrs,
                                 2,
                                 mlutensor_output_ptrs,
                                 2,
                                 Caffe::forward_param(),
                                 Caffe::queue()));
}

template <typename Dtype>
void MLUSsdDetectionLayer<Dtype>::MLUCompileOp() {
  for (int i = 0; i < bottom_each_size; i++) {
     MLU_CHECK(cnmlCompileBaseOp(loc_reshape_ops_[i],
                                 Caffe::rt_core(),
                                 Caffe::core_number()));
     MLU_CHECK(cnmlCompileBaseOp(conf_reshape_ops_[i],
                                 Caffe::rt_core(),
                                 Caffe::core_number()));
  }
  MLU_CHECK(cnmlCompileBaseOp(loc_concat_op_,
                              Caffe::rt_core(),
                              Caffe::core_number()));
  MLU_CHECK(cnmlCompileBaseOp(conf_concat_op_,
                              Caffe::rt_core(),
                              Caffe::core_number()));
  MLU_CHECK(cnmlCompileBaseOp(loc_transpose_op_,
                              Caffe::rt_core(),
                              Caffe::core_number()));
  MLU_CHECK(cnmlCompileBaseOp(conf_transpose_op_,
                              Caffe::rt_core(),
                              Caffe::core_number()));
  MLU_CHECK(cnmlCompileBaseOp(softmax_op_,
                              Caffe::rt_core(),
                              Caffe::core_number()));
  MLU_CHECK(cnmlCompileBaseOp(detection_plugin_op_ptr_,
                              Caffe::rt_core(),
                              Caffe::core_number()));
}

template <typename Dtype>
void MLUSsdDetectionLayer<Dtype>::fuse(MFusion<Dtype>* fuser) {
  for (int i = 0; i < bottom_each_size; i++) {
    fuser->fuse(loc_reshape_ops_[i]);
    fuser->fuse(conf_reshape_ops_[i]);
  }
  fuser->fuse(loc_concat_op_);
  fuser->fuse(conf_concat_op_);
  fuser->fuse(loc_transpose_op_);
  fuser->fuse(conf_transpose_op_);
  fuser->fuse(softmax_op_);
  fuser->fuse(detection_plugin_op_ptr_);
}

INSTANTIATE_CLASS(MLUSsdDetectionLayer);

}  // namespace caffe
#endif  // USE_MLU
