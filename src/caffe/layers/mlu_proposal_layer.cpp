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
#include "caffe/layers/mlu_proposal_layer.hpp"
#include "caffe/util/proposal_generate_anchors.hpp"

namespace caffe {
template <typename Dtype>
void MLUProposalLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  ProposalLayer<Dtype>::LayerSetUp(bottom, top);

  /* get parameter of anchors */
  anchor_scales.clear();
  anchor_ratios.clear();
  if ((this->layer_param_.proposal_param().anchor_scale_size() == 0) &&
      (this->layer_param_.proposal_param().anchor_ratio_size() == 0)) {
    anchor_scales.push_back(8);
    anchor_scales.push_back(16);
    anchor_scales.push_back(32);
    anchor_ratios.push_back(0.5);
    anchor_ratios.push_back(1);
    anchor_ratios.push_back(2);
  } else {
    for (int i = 0; i < this->layer_param_.proposal_param().anchor_scale_size(); i++) {
      anchor_scales.push_back(this->layer_param_.proposal_param().anchor_scale(i));
    }
    for (int i = 0; i < this->layer_param_.proposal_param().anchor_ratio_size(); i++) {
      anchor_ratios.push_back(this->layer_param_.proposal_param().anchor_ratio(i));
    }
  }
}

template <typename Dtype>
void MLUProposalLayer<Dtype>::Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
  BaseDataType mlu_dtype = bottom[0]->mlu_type();
  vector<int> top_shape(4, 1);
  top_shape[0] = bottom[1]->num();
  top_shape[1] = this->nms_num_;
  top_shape[2] = 1;
  top_shape[3] = 5;
  top[0]->Reshape(top_shape, cpu_dtype, mlu_dtype, CNML_TENSOR);

  // Reshape the tensor that used for Bang Op

  if (this->shuffle_channel_) {
    bbox_shufflechannel_shape.resize(4);
    bbox_shufflechannel_shape[0] = bottom[1]->num();
    bbox_shufflechannel_shape[1] = bottom[1]->channels();
    bbox_shufflechannel_shape[2] = bottom[1]->height();
    bbox_shufflechannel_shape[3] = bottom[1]->width();
    bbox_shufflechannel_blob.Reshape(bbox_shufflechannel_shape);
  }

  int AHW = this->A_ * bottom[1]->height() * bottom[1]->width();
  bbox_transpose_shape.resize(4);
  bbox_transpose_shape[0] = bottom[1]->num();
  bbox_transpose_shape[1] = bottom[1]->width();
  bbox_transpose_shape[2] = bottom[1]->channels();
  bbox_transpose_shape[3] = bottom[1]->height();
  bbox_transpose_blob.Reshape(bbox_transpose_shape);

  score_transpose_shape.resize(4);
  score_transpose_shape[0] = bottom[0]->num();
  score_transpose_shape[1] = bottom[0]->width();
  score_transpose_shape[2] = bottom[0]->channels();
  score_transpose_shape[3] = bottom[0]->height();
  score_transpose_blob.Reshape(score_transpose_shape);

  vector<int> anchor_shape(4, 1);
  anchor_shape[0] = 1;
  anchor_shape[1] = AHW;
  anchor_shape[2] = 1;
  anchor_shape[3] = 4;
  anchors_blob.Reshape(anchor_shape, cpu_dtype, mlu_dtype, CNML_CONST);

  /* used to occupy gdram memory to temparialy save bbox data before nms */
  int mlu_align_size = 64;
  vector<int> nms_data_shape(4, 1);
  if (!Caffe::simpleFlag())
    nms_data_shape[0] = 1;
  else
    nms_data_shape[0] = Caffe::batchsize();
  nms_data_shape[1] = 18;
  nms_data_shape[2] = 1;
  nms_data_shape[3] = ((AHW - 1) / mlu_align_size + 1) * mlu_align_size;
  nms_data_blob.Reshape(nms_data_shape);
}

template <typename Dtype>
void MLUProposalLayer<Dtype>::MLUCreateOpBindData(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  im_info_ = bottom[2];
  im_h = this->layer_param_.proposal_param().im_h();
  im_w = this->layer_param_.proposal_param().im_w();
  scale = this->layer_param_.proposal_param().scale();
  // If the value of im_info is passed through the third input,
  // the value of im_info will be overwritten.
  const Dtype* im_info = bottom[2]->cpu_data();
  if (bottom.size() == 3 && im_info[0] != 0) {
    im_h = im_info[0];
    im_w = im_info[1];
    scale = im_info[2];
  }

  int batch_size = bottom[1]->num();
  int H = bottom[0]->height();
  int W = bottom[0]->width();
  float min_h = this->layer_param_.proposal_param().im_min_h();
  float min_w = this->layer_param_.proposal_param().im_min_w();
  int nms_num = this->layer_param_.proposal_param().nms_num();
  float nms_scale = this->layer_param_.proposal_param().nms_scale();
  float stride = this->stride_;
  float nms_thresh = this->layer_param_.proposal_param().nms_thresh();
  int top_num = this->layer_param_.proposal_param().top_num();
  int base_size = this->layer_param_.proposal_param().base_size();
  int int8_mode = false;

  /* create anchors */
  vector<float> anchors_data(1 * (this->A_ * H * W) * 1 * 4);
  if (this->layer_param_.proposal_param().pvanet_mode()) {
    generate_anchor_box_pvanet(H, W, this->stride_, base_size,
                               anchor_scales, anchor_ratios, false,
                               anchors_data.data());
  } else {
    generate_anchor_box(H, W, this->stride_, base_size,
                        anchor_scales, anchor_ratios, false,
                        anchors_data.data());
  }

  for (int i = 0; i < this->A_ * H * W; i++) {
    anchors_blob.mutable_cpu_data()[i * 4 + 0] = anchors_data[i + 0 * this->A_ * H * W];
    anchors_blob.mutable_cpu_data()[i * 4 + 1] = anchors_data[i + 1 * this->A_ * H * W];
    anchors_blob.mutable_cpu_data()[i * 4 + 2] = anchors_data[i + 2 * this->A_ * H * W];
    anchors_blob.mutable_cpu_data()[i * 4 + 3] = anchors_data[i + 3 * this->A_ * H * W];
  }

  // if bbox shape is 1 x (anchor x 4) x h x w, use shufflechannel to change
  // it to 1 x (4 x anchor) x h x w
  if (this->shuffle_channel_) {
    cnmlTensor_t mluTensor_input[1];
    cnmlTensor_t mluTensor_output[1];

    mluTensor_input[0] = bottom[1]->mlu_tensor();
    mluTensor_output[0] = bbox_shufflechannel_blob.mlu_tensor();

    MLU_CHECK(cnmlCreateShuffleChannelOp(&bbox_shufflechannel_ptr,
                                         mluTensor_input,
                                         mluTensor_output,
                                         this->A_));
  }

  int dim_order[4] = {0, 3, 1, 2};
  MLU_CHECK(cnmlCreateNdTransposeOpParam(&bbox_transpose_param,
                                         dim_order,
                                         4));
  MLU_CHECK(cnmlCreateNdTransposeProOp(&bbox_transpose_ptr,
                                        this->shuffle_channel_ ?
                                        bbox_shufflechannel_blob.mlu_tensor() :
                                        bottom[1]->mlu_tensor(),
                                        bbox_transpose_blob.mlu_tensor(),
                                        bbox_transpose_param));
  MLU_CHECK(cnmlCreateNdTransposeOpParam(&score_transpose_param,
                                         dim_order,
                                         4));
  MLU_CHECK(cnmlCreateNdTransposeProOp(&score_transpose_ptr,
                                        bottom[0]->mlu_tensor(),
                                        score_transpose_blob.mlu_tensor(),
                                        score_transpose_param));

  cnmlTensor_t mlutensor_inputs[2];
  mlutensor_inputs[0] = bbox_transpose_blob.mlu_tensor();
  mlutensor_inputs[1] = score_transpose_blob.mlu_tensor();

  cnmlTensor_t mlutensor_outputs[top.size() + 1];  // NOLINT
  for (int i = 0; i < top.size(); i++) {
    mlutensor_outputs[i] = top[i]->mlu_tensor();
  }
  mlutensor_outputs[top.size()] = nms_data_blob.mlu_tensor();

  MLU_CHECK(cnmlCreatePluginProposalOpParam(&proposal_ptr_param,
        batch_size,
        H,
        W,
        this->A_,
        nms_num,
        top_num,
        min_h,
        min_w,
        nms_scale,
        stride,
        nms_thresh,
        im_h,
        im_w,
        scale,
        int8_mode,
        Caffe::rt_core(),
        reinterpret_cast<float*>(anchors_blob.sync_data())));
  MLU_CHECK(cnmlCreatePluginProposalOp(&proposal_op_ptr_,
        proposal_ptr_param,
        mlutensor_inputs,
        mlutensor_outputs));
  output_blob_ = top[0];
}

template <typename Dtype>
void MLUProposalLayer<Dtype>::MLUCompileOp() {
  if (this->shuffle_channel_) {
    MLU_CHECK(cnmlCompileBaseOp(bbox_shufflechannel_ptr,
                                Caffe::rt_core(),
                                Caffe::core_number()));
  }
  MLU_CHECK(cnmlCompileBaseOp(bbox_transpose_ptr,
                              Caffe::rt_core(),
                              Caffe::core_number()));
  MLU_CHECK(cnmlCompileBaseOp(score_transpose_ptr,
                              Caffe::rt_core(),
                              Caffe::core_number()));

  MLU_CHECK(cnmlCompileBaseOp(proposal_op_ptr_,
                              Caffe::rt_core(),
                              Caffe::core_number()));
}

template <typename Dtype>
void MLUProposalLayer<Dtype>::Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  // if bbox shape is 1 x (anchor x 4) x h x w, use shufflechannel to change
  // it to 1 x (4 x anchor) x h x w
  if (this->shuffle_channel_) {
    void* inputs[1];
    void* outputs[1];
    inputs[0] = bottom[1]->mutable_mlu_data();
    outputs[0] = bbox_shufflechannel_blob.mutable_mlu_data();

    MLU_CHECK(cnmlComputeShuffleChannelOpForward_V3(bbox_shufflechannel_ptr,
                                                    inputs,
                                                    outputs,
                                                    Caffe::forward_param(),
                                                    Caffe::queue()));
  }
  MLU_CHECK(cnmlComputeNdTransposeProOpForward(bbox_transpose_ptr,
                                             this->shuffle_channel_ ?
                                             bbox_shufflechannel_blob.mutable_mlu_data() :
                                             bottom[1]->mutable_mlu_data(),
                                             bbox_transpose_blob.mutable_mlu_data(),
                                             Caffe::forward_param(),
                                             Caffe::queue()));

  MLU_CHECK(cnmlComputeNdTransposeProOpForward(score_transpose_ptr,
                                               bottom[0]->mutable_mlu_data(),
                                               score_transpose_blob.mutable_mlu_data(),
                                               Caffe::forward_param(),
                                               Caffe::queue()));

  void* mlutensor_input_ptrs[2];
  mlutensor_input_ptrs[0] = bbox_transpose_blob.mutable_mlu_data();
  mlutensor_input_ptrs[1] = score_transpose_blob.mutable_mlu_data();

  void* mlutensor_output_ptrs[top.size() + 1];
  for (int i = 0; i < top.size(); i++) {
    mlutensor_output_ptrs[i] = top[i]->mutable_mlu_data();
  }
  mlutensor_output_ptrs[top.size()] = nms_data_blob.mutable_mlu_data();
  MLU_CHECK(cnmlComputePluginProposalOpForward(proposal_op_ptr_,
        mlutensor_input_ptrs,
        2,
        mlutensor_output_ptrs,
        top.size() + 1,
        Caffe::forward_param(),
        Caffe::queue()));
}

template <typename Dtype>
void MLUProposalLayer<Dtype>::fuse(MFusion<Dtype>* fuser) {
  fuser->addOutput(output_blob_);
  if (this->shuffle_channel_) {
    fuser->fuse(bbox_shufflechannel_ptr);
  }
  fuser->fuse(bbox_transpose_ptr);
  fuser->fuse(score_transpose_ptr);
  fuser->fuse(proposal_op_ptr_);
  // im_info(bottom[2]) does act as Proposal OP's input in CNML
  fuser->rmInput(im_info_);
}

template <typename Dtype>
void MLUProposalLayer<Dtype>::MLUDestroyOp() {
  if (proposal_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&proposal_op_ptr_));
    proposal_op_ptr_ = nullptr;
  }

  if (bbox_shufflechannel_ptr != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&bbox_shufflechannel_ptr));
    bbox_shufflechannel_ptr = nullptr;
  }
  if (bbox_transpose_ptr != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&bbox_transpose_ptr));
    bbox_transpose_ptr = nullptr;
  }
  if (score_transpose_ptr != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&score_transpose_ptr));
    score_transpose_ptr = nullptr;
  }
  if (bbox_transpose_param != nullptr) {
    MLU_CHECK(cnmlDestroyNdTransposeOpParam(&bbox_transpose_param));
    bbox_transpose_param = nullptr;
  }
  if (score_transpose_param != nullptr) {
    MLU_CHECK(cnmlDestroyNdTransposeOpParam(&score_transpose_param));
    score_transpose_param = nullptr;
  }
  if (proposal_ptr_param != nullptr) {
    MLU_CHECK(cnmlDestroyPluginProposalOpParam(&proposal_ptr_param));
    proposal_ptr_param = nullptr;
  }
}

INSTANTIATE_CLASS(MLUProposalLayer);

}  // namespace caffe
#endif  // USE_MLU
