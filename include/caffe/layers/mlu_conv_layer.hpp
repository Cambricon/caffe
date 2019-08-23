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

#ifndef INCLUDE_CAFFE_LAYERS_MLU_CONV_LAYER_HPP_
#define INCLUDE_CAFFE_LAYERS_MLU_CONV_LAYER_HPP_
#ifdef USE_MLU

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/conv_layer.hpp"

/*
 * @brief CNML implementation of ConvolutionLayer.
 *
 */

namespace caffe {
template <typename Dtype>
class MLUConvolutionLayer : public ConvolutionLayer<Dtype> {
  public:
  explicit MLUConvolutionLayer(const LayerParameter& param)
           : ConvolutionLayer<Dtype>(param),
             mlu_conv_op_ptrs_(nullptr), mlu_addpad_op_ptrs_(nullptr),
             mlu_slice_op_ptrs_(nullptr), mlu_concat_op_ptrs_(nullptr),
             mlu_conv_param_ptr_(nullptr), mlu_convf_param_ptr_(nullptr),
             mlu_addpad_param_ptr_(nullptr), mlu_slice_param_ptr_(nullptr),
             mlu_concat_param_ptr_(nullptr), add_pad_(false),
             depthwise_param_ptr_(nullptr), std_ptr_(nullptr) { }
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual ~MLUConvolutionLayer();
  virtual void Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                               const vector<Blob<Dtype>*>& top);
  virtual inline bool mfus_supported() { return true; }
  virtual void fuse(MFusion<Dtype>* fuser) {
    for (int i = 0; i < bottom_size_; i++) {
      if (!this->conv_first_ && add_pad_ && (is_depthwise_ || dilate())) {
        fuser->fuse(mlu_addpad_op_ptrs_[i]);
      }
      fuser->fuse(mlu_conv_op_ptrs_[i]);
    }
  }

  protected:
  bool dilate() {
    return (this->dilation_.cpu_data()[0] != 1 || this->dilation_.cpu_data()[1] != 1);
  }
  inline void SetBlobPosition(Blob<Dtype>* blob_in) {
    if (this->layer_param_.blobs_dtype_size() > 0 &&
        (this->layer_param_.blobs_dtype(0).position_size() ||
         this->layer_param_.blobs_dtype(0).scale_size())) {
      int pos_size = this->layer_param_.blobs_dtype(0).position_size();
      int scale_size = this->layer_param_.blobs_dtype(0).scale_size();
      vector<int> positions(pos_size);
      vector<float> scales(scale_size);
      for (int i = 0; i < pos_size; i++) {
        positions[i] = this->layer_param_.blobs_dtype(0).position(i);
      }
      for (int i = 0; i < scale_size; i++) {
        scales[i] = this->layer_param_.blobs_dtype(0).scale(i);
      }
      if (this->layer_param_.blobs_dtype(0).position_size()) {
        if (pos_size == 1)
          blob_in->set_mlu_position(positions[0]);
        else
          blob_in->set_mlu_positions(positions);
      }
      if (this->layer_param_.blobs_dtype(0).scale_size()) {
        if (scale_size == 1)
          blob_in->set_mlu_scale(scales[0]);
        else
          blob_in->set_mlu_scales(scales);
      }
    }
  }
  inline void SetBottomPosition(Blob<Dtype>* blob_in, int idx) {
    if (this->layer_param_.bottom_mlu_dtype_size() > idx) {
      if (this->layer_param_.bottom_mlu_dtype(idx).position_size()) {
        blob_in->set_mlu_position(this->layer_param_.bottom_mlu_dtype(idx).position(0));
      }
      if (this->layer_param_.bottom_mlu_dtype(idx).scale_size()) {
        blob_in->set_mlu_scale(this->layer_param_.bottom_mlu_dtype(idx).scale(0));
      }
    }
  }
  inline void SetMeanPosition(Blob<Dtype>* blob_in) {
    BlobDataType blob_dtype;
    blob_dtype = get_int8_info<Dtype>(*blob_in, this->layer_param_);
    blob_in->set_mlu_type(DT_INT8);
    blob_in->set_mlu_position(blob_dtype.position(0));
    blob_in->set_mlu_scale(blob_dtype.scale(0));
  }
  virtual void Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void MLUCreateOpBindData(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void MLUDestroyOp();
  virtual void MLUCompileOp();
  void ArrangeWeightRGB();
  void MLUCreateOpFirstConv(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top,
                           const int index);

  size_t bottom_size_;
  cnmlBaseOp_t* mlu_conv_op_ptrs_;
  cnmlBaseOp_t* mlu_addpad_op_ptrs_;
  cnmlBaseOp_t* mlu_slice_op_ptrs_;
  cnmlBaseOp_t* mlu_concat_op_ptrs_;
  cnmlConvOpParam_t mlu_conv_param_ptr_;
  cnmlConvFirstOpParam_t mlu_convf_param_ptr_;
  cnmlAddPadOpParam_t mlu_addpad_param_ptr_;
  cnmlSplitOpParam_t mlu_slice_param_ptr_;
  cnmlConcatOpParam_t mlu_concat_param_ptr_;
  vector<Blob<Dtype>*> addpad_;
  bool add_pad_;
  vector<vector<Blob<Dtype>*>> slice_;
  vector<vector<Blob<Dtype>*>> concat_;
  vector<Blob<Dtype>*> weight_blob_;
  Blob<Dtype> std_;
  cnmlSparseMode_t mluSparse_;
  // depthwise
  bool is_depthwise_;
  int multiplier_;
  cnmlConvDepthwiseOpParam_t depthwise_param_ptr_;
  float* std_ptr_;
};
}  // namespace caffe
#endif  // USE_MLU
#endif  // INCLUDE_CAFFE_LAYERS_MLU_CONV_LAYER_HPP_
