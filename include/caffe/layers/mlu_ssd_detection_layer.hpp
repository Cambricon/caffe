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

#ifndef INCLUDE_CAFFE_LAYERS_MLU_SSD_DETECTION_LAYER_HPP_
#define INCLUDE_CAFFE_LAYERS_MLU_SSD_DETECTION_LAYER_HPP_
#ifdef USE_MLU

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief CNML implementation of ssd_detection
 *        First get the priorboxies of input data.
 * Then generate the ssd_detection output based on location and confidence.
 *
 * NOTE: does not implement Backwards operation.
 */
template <typename Dtype>
class MLUSsdDetectionLayer : public Layer<Dtype> {
  public:
  explicit MLUSsdDetectionLayer(const LayerParameter& param)
      : Layer<Dtype>(param),
      detection_plugin_op_ptr_(nullptr),
      loc_transpose_op_(nullptr), conf_transpose_op_(nullptr),
      loc_concat_op_(nullptr), conf_concat_op_(nullptr),
      softmax_op_(nullptr), loc_transpose_param(nullptr),
      conf_transpose_param(nullptr), ssd_detection_param(nullptr),
      bottom_nums(0), bottom_each_size(0), num_preds_per_class(0){}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top) {}
  virtual void Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                              const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SsdDetection"; }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

  virtual inline bool mfus_supported() { return true; }
  virtual void fuse(MFusion<Dtype>* fuser);
  virtual ~MLUSsdDetectionLayer();

  protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top) {
    LOG(WARNING) << "NOT_IMPLEMENTED.";
  }
  // @brief Not implemented
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }
  virtual void MLUDestroyOp();
  virtual void MLUCreateOpBindData(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top);
  virtual void MLUCompileOp();
  virtual void Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);

  int num_classes_;
  bool share_location_;
  int num_loc_classes_;
  int background_label_id_;
  PriorBoxParameter_CodeType code_type_;
  bool variance_encoded_in_target_;
  int keep_top_k_;
  float confidence_threshold_;
  float nms_threshold_;
  int top_k_;

  vector<shared_ptr<Blob<Dtype> > > loc_blobs_;
  vector<shared_ptr<Blob<Dtype> > > conf_blobs_;
  vector<shared_ptr<Blob<Dtype> > > priors_blobs_;

  Blob<Dtype> priorbox_blob_;
  Blob<Dtype> loc_concat_blob;
  Blob<Dtype> conf_concat_blob;
  Blob<Dtype> loc_trans_blob;
  Blob<Dtype> conf_trans_blob;
  Blob<Dtype> conf_softmax_blob;
  shared_ptr<Blob<Dtype> >  tmp_blob_;

  vector<cnmlBaseOp_t> loc_reshape_ops_;
  vector<cnmlBaseOp_t> conf_reshape_ops_;
  vector<cnmlReshapeOpParam_t> loc_reshape_params_;
  vector<cnmlReshapeOpParam_t> conf_reshape_params_;

  cnmlBaseOp_t detection_plugin_op_ptr_;
  cnmlBaseOp_t loc_transpose_op_;
  cnmlBaseOp_t conf_transpose_op_;
  cnmlBaseOp_t loc_concat_op_;
  cnmlBaseOp_t conf_concat_op_;
  cnmlBaseOp_t softmax_op_;
  cnmlNdTransposeOpParam_t loc_transpose_param;
  cnmlNdTransposeOpParam_t conf_transpose_param;

  cnmlPluginSsdDetectionOutputOpParam_t ssd_detection_param;

  int bottom_nums;
  int bottom_each_size;
  int num_preds_per_class;
};
}  // namespace caffe
#endif  // USE_MLU
#endif  // INCLUDE_CAFFE_LAYERS_MLU_SSD_DETECTION_LAYER_HPP_
