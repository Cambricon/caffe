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

#ifndef INCLUDE_CAFFE_LAYERS_MLU_PROPOSAL_LAYER_HPP_
#define INCLUDE_CAFFE_LAYERS_MLU_PROPOSAL_LAYER_HPP_
#ifdef USE_MLU

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/neuron_layer.hpp"
#include "caffe/layers/proposal_layer.hpp"

namespace caffe {

/**
 * @brief MLU acceleration of ProposalLayer
 *
 *        Compute elementwise operations, such as product and sum,
 *        along multiple input Blobs.
 *
 * Applies the Region Proposal Network`s(RPN) predicted deltas to each
 * of the anchors,removes unsuitable boxes,and then ranks them by their
 * "objectness" scores. Non-maximimum suppression removes proposals of
 * the same object, and the top proposals are returned.
 */
template <typename Dtype>
class MLUProposalLayer : public ProposalLayer<Dtype> {
  public:
  explicit MLUProposalLayer(const LayerParameter& param)
      : ProposalLayer<Dtype>(param), proposal_op_ptr_(nullptr),
        bbox_transpose_param(nullptr), score_transpose_param(nullptr),
        bbox_shufflechannel_ptr(nullptr), bbox_transpose_ptr(nullptr),
        score_transpose_ptr(nullptr), proposal_ptr_param(nullptr) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                              const vector<Blob<Dtype>*>& top);
  virtual void fuse(MFusion<Dtype>* fuser);
  virtual inline bool mfus_supported() { return true; }
  virtual ~MLUProposalLayer() { MLUDestroyOp(); }

  protected:
  virtual void MLUDestroyOp();
  virtual void MLUCreateOpBindData(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top);
  virtual void MLUCompileOp();

  virtual void Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);

  cnmlBaseOp_t proposal_op_ptr_;
  Blob<Dtype>* output_blob_;  // to remove from fusion input
  Blob<Dtype>* im_info_;  // to remove from fusion input
  float im_h, im_w, scale;

  vector<int> bbox_shufflechannel_shape;
  vector<int> bbox_transpose_shape;
  vector<int> score_transpose_shape;

  vector<float> anchor_scales;
  vector<float> anchor_ratios;

  cnmlNdTransposeOpParam_t bbox_transpose_param;
  cnmlNdTransposeOpParam_t score_transpose_param;
  cnmlBaseOp_t bbox_shufflechannel_ptr;
  cnmlBaseOp_t bbox_transpose_ptr;
  cnmlBaseOp_t score_transpose_ptr;
  Blob<Dtype> bbox_shufflechannel_blob;
  Blob<Dtype> bbox_transpose_blob;
  Blob<Dtype> score_transpose_blob;
  Blob<Dtype> anchors_blob;
  Blob<Dtype> nms_data_blob;
  cnmlPluginProposalOpParam_t proposal_ptr_param;
};
}  // namespace caffe
#endif  // USE_MLU
#endif  // INCLUDE_CAFFE_LAYERS_MLU_PROPOSAL_LAYER_HPP_
