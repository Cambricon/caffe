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
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include "caffe/filler.hpp"
#include "caffe/common.hpp"
#include "caffe/util/io.hpp"
#include "caffe/layers/mlu_conv_depthwise_layer.hpp"

namespace caffe {

template <typename Dtype>
void MLUConvolutionDepthwiseLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                                     const vector<Blob<Dtype>*>& top) {
  BaseConvolutionLayer<Dtype>::LayerSetUp(bottom, top);
  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
  BaseDataType mlu_dtype = bottom[0]->mlu_type();
  this->blobs_[0].reset(new Blob<Dtype>(this->blobs_[0]->shape(),
        cpu_dtype, mlu_dtype, CNML_FILTER, CNML_CNHW));
  shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
  weight_filler->Fill(this->blobs_[0].get());
  if (this->bias_term_) {
    this->blobs_[1].reset(new Blob<Dtype>(this->blobs_[1]->shape(),
                                          cpu_dtype,
                                          mlu_dtype,
                                          CNML_CONST,
                                          CNML_CNHW));
    shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
      this->layer_param_.convolution_param().bias_filler()));
    bias_filler->Fill(this->blobs_[1].get());
  }
  bottom_size_ = bottom.size();
}

template <typename Dtype>
void MLUConvolutionDepthwiseLayer<Dtype>::Reshape_tensor(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  BaseConvolutionLayer<Dtype>::Reshape(bottom, top);
}

template <typename Dtype>
void MLUConvolutionDepthwiseLayer<Dtype>::MLUCreateOpBindData(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int stride_h = stride_data[0];
  const int stride_w = stride_data[1];
  // spread pad to 4 dimensions
  const int pad_htop = pad_data[0];
  const int pad_wleft = pad_data[1];
  const int pad_hbottom = pad_data[2];
  const int pad_wright = pad_data[3];

  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;

  bool pad = pad_htop || pad_hbottom || pad_wleft || pad_wright;
  add_pad_ = pad && !(pad_htop == pad_hbottom && pad_wleft == pad_wright);
  if (add_pad_) {
    MLU_CHECK(cnmlCreateAddPadOpParam_V2(&mlu_addpad_param_ptr_,
                                   pad_htop, pad_hbottom, pad_wleft, pad_wright, 0));
    mlu_addpad_op_ptrs_ = new cnmlBaseOp_t[bottom.size()];
    for (int i = 0; i < bottom.size(); i++) {
      vector<int> addpad_shape;
      addpad_shape.push_back(bottom[i]->shape(0));
      addpad_shape.push_back(bottom[i]->channels());
      addpad_shape.push_back(bottom[i]->height() + pad_htop + pad_hbottom);
      addpad_shape.push_back(bottom[i]->width() + pad_wleft + pad_wright);
      addpad_.push_back(new Blob<Dtype>(addpad_shape,
                                        cpu_dtype,
                                        DT_FLOAT16,
                                        CNML_TENSOR));
      MLU_CHECK(cnmlCreateAddPadOp(&mlu_addpad_op_ptrs_[i],
                                  mlu_addpad_param_ptr_,
                                  bottom[i]->mlu_tensor(),
                                  addpad_[i]->mlu_tensor()));
    }
  }

  MLU_CHECK(cnmlCreateConvDepthwiseOpParam_V2(&depthwise_param_ptr_,
          stride_h,
          stride_w,
          add_pad_? 0: pad_htop * 2,
          add_pad_? 0: pad_wleft * 2));
  conv_depthwise_op_ptrs_ = new cnmlBaseOp_t[bottom.size()];
  for (int i = 0; i < bottom.size(); i++) {
    MLU_CHECK(cnmlCreateConvDepthwiseOp(&conv_depthwise_op_ptrs_[i],
                             depthwise_param_ptr_,
                             add_pad_ ?
                             addpad_[i]->mlu_tensor() : bottom[i]->mlu_tensor(),
                             top[i]->mlu_tensor(),
                             this->blobs_[0]->mlu_tensor(),
                             this->bias_term_ ?
                             this->blobs_[1]->mlu_tensor() : nullptr));

    MLU_CHECK(cnmlBindConstData_V2(this->blobs_[0]->mlu_tensor(),
                                   this->blobs_[0]->sync_data(),
                                   false));
    if ( this->bias_term_ ) {
      MLU_CHECK(cnmlBindConstData_V2(this->blobs_[1]->mlu_tensor(),
                                     this->blobs_[1]->sync_data(),
                                     false));
    }
  }
}

template <typename Dtype>
void MLUConvolutionDepthwiseLayer<Dtype>::MLUCompileOp() {
  if (add_pad_) {
    for (int i = 0; i < bottom_size_; i++) {
        MLU_CHECK(cnmlCompileBaseOp(mlu_addpad_op_ptrs_[i],
                                    Caffe::rt_core(),
                                    Caffe::core_number()));
    }
  }
  for (int i = 0; i < bottom_size_; i++) {
      MLU_CHECK(cnmlCompileBaseOp(conv_depthwise_op_ptrs_[i],
                                  Caffe::rt_core(),
                                  Caffe::core_number()));
  }
}


template <typename Dtype>
void MLUConvolutionDepthwiseLayer<Dtype>::MLUDestroyOp() {
  if (conv_depthwise_op_ptrs_ != nullptr) {
    for (int i = 0; i < bottom_size_; i++)
      MLU_CHECK(cnmlDestroyBaseOp(&conv_depthwise_op_ptrs_[i]));
    delete [] conv_depthwise_op_ptrs_;
    conv_depthwise_op_ptrs_ = nullptr;
  }
  if (depthwise_param_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyConvDepthwiseOpParam(&depthwise_param_ptr_));
    depthwise_param_ptr_ = nullptr;
  }
  if (mlu_addpad_op_ptrs_ != nullptr) {
    for (int i = 0; i < bottom_size_; i++) {
      MLU_CHECK(cnmlDestroyBaseOp(&mlu_addpad_op_ptrs_[i]));
    }
    delete [] mlu_addpad_op_ptrs_;
    mlu_addpad_op_ptrs_ = nullptr;
  }
  if (mlu_addpad_param_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyAddPadOpParam(&mlu_addpad_param_ptr_));
    mlu_addpad_param_ptr_ = nullptr;
  }
  if (!addpad_.empty()) {
    for (int i = 0; i < addpad_.size(); i++) {
      delete addpad_[i];
    }
    addpad_.clear();
  }
}


template <typename Dtype>
MLUConvolutionDepthwiseLayer<Dtype>::~MLUConvolutionDepthwiseLayer() {
  MLUDestroyOp();
}

template <typename Dtype>
void MLUConvolutionDepthwiseLayer<Dtype>::Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                                                      const vector<Blob<Dtype>*>& top) {
  if (add_pad_) {
    for (int i = 0; i < bottom.size(); i++) {
      MLU_CHECK(cnmlComputeAddPadOpForward_V3(mlu_addpad_op_ptrs_[i],
                      bottom[i]->mutable_mlu_data(),
                      addpad_[i]->mutable_mlu_data(),
                      Caffe::forward_param(), Caffe::queue()));
    }
  }
  for (int i = 0; i < bottom.size(); i++) {
    MLU_CHECK(cnmlComputeConvDepthwiseOpForward_V3(conv_depthwise_op_ptrs_[i],
              add_pad_ ?
              addpad_[i]->mutable_mlu_data() : bottom[i]->mutable_mlu_data(),
              top[i]->mutable_mlu_data(),
              Caffe::forward_param(), Caffe::queue()));
  }
}

INSTANTIATE_CLASS(MLUConvolutionDepthwiseLayer);

}  // namespace caffe
#endif
