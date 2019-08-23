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
}

template <typename Dtype>
void MLUSsdDetectionLayer<Dtype>::Reshape_tensor(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  vector<int> mlu_shape(4, 1);
  mlu_shape[0] = bottom[0]->num();
  mlu_shape[1] = this->keep_top_k_;
  mlu_shape[2] = 1;
  mlu_shape[3] = 6;
  top[0]->Reshape(mlu_shape);
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

  MLU_CHECK(cnmlBindConstData(priorbox_blob_.mlu_tensor(),
      priorbox_blob_.cpu_tensor(),
      (void*)priorbox_blob_.cpu_data())); //  NOLINT

  MLU_CHECK(cnmlCreateSsdDetectionOpParam(
      &ssd_detection_op_param_ptr_,
      this->num_classes_,
      this->share_location_,
      this->background_label_id_,
      this->code_type_ - 1,
      this->variance_encoded_in_target_,
      this->confidence_threshold_,
      this->nms_threshold_,
      this->top_k_,
      this->keep_top_k_,
      CNML_NCHW));
  std::vector<cnmlTensor_t> locs_tensor;
  std::vector<cnmlTensor_t> confs_tensor;
  for (int i = 0; i < bottom_priorbox_index; i++) {
      if (i < bottom_priorbox_index/2)
          locs_tensor.push_back(bottom[i]->mlu_tensor());
      else
          confs_tensor.push_back(bottom[i]->mlu_tensor());
  }
  MLU_CHECK(cnmlCreateSsdDetectionOpV2(
      &ssd_detection_op_ptr_,
      locs_tensor.data(),
      confs_tensor.data(),
      locs_tensor.size(),
      priorbox_blob_.mlu_tensor(),
      top[0]->mlu_tensor(),
      ssd_detection_op_param_ptr_));
}

template <typename Dtype>
void MLUSsdDetectionLayer<Dtype>::MLUDestroyOp() {
  if (ssd_detection_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&ssd_detection_op_ptr_));
    ssd_detection_op_ptr_ = nullptr;
  }
  if (ssd_detection_op_param_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroySsdDetectionOpParam(&ssd_detection_op_param_ptr_));
    ssd_detection_op_param_ptr_ = nullptr;
  }
}

template <typename Dtype>
MLUSsdDetectionLayer<Dtype>::~MLUSsdDetectionLayer() {
  MLUDestroyOp();
}

template <typename Dtype>
void MLUSsdDetectionLayer<Dtype>::Forward_mlu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int bottom_nums = bottom.size();
  int bottom_priorbox_index = 2*(bottom_nums - 1)/3;
  std::vector<void*> locs_ptr;
  std::vector<void*> confs_ptr;
  for (int i = 0; i < bottom_priorbox_index; i++) {
      if (i < bottom_priorbox_index/2)
          locs_ptr.push_back(reinterpret_cast<void*>(bottom[i]->mutable_mlu_data()));
      else
          confs_ptr.push_back(reinterpret_cast<void*>(bottom[i]->mutable_mlu_data()));
  }
  MLU_CHECK(cnmlComputeSsdDetectionOpForward_V3(
      ssd_detection_op_ptr_,
      locs_ptr.data(),
      confs_ptr.data(),
      locs_ptr.size(),
      nullptr,
      top[0]->mutable_mlu_data(),
      Caffe::forward_param(),
      Caffe::queue()));
}

template <typename Dtype>
void MLUSsdDetectionLayer<Dtype>::MLUCompileOp() {
  MLU_CHECK(cnmlCompileBaseOp(ssd_detection_op_ptr_,
                              Caffe::rt_core(),
                              Caffe::model_parallel()));
}

template <typename Dtype>
void MLUSsdDetectionLayer<Dtype>::fuse(MFusion<Dtype>* fuser) {
  fuser->fuse(ssd_detection_op_ptr_);
}


INSTANTIATE_CLASS(MLUSsdDetectionLayer);
}  // namespace caffe
#endif
