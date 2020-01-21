/*
All modification made by Cambricon Corporation: © 2018-2019 Cambricon Corporation
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
#include "caffe/layers/mlu_upsample_layer.hpp"

namespace caffe {

template <typename Dtype>
void MLUUpsampleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  UpsampleLayer<Dtype>::LayerSetUp(bottom, top);
  UpsampleParameter upsample_param = this->layer_param_.upsample_param();
  if (!this->NearestNeighbor_mode) {
    if (upsample_param.has_upsample_h() && upsample_param.has_upsample_w()) {
       if (this->upsample_h_ % bottom[0]->height() != 0) {
         LOG(FATAL) <<"it is suggested that upsample_h_"
                    <<"% bottom[0]->height() == 0.";
       }
       if (this->upsample_w_ % bottom[0]->width() != 0) {
         LOG(FATAL) <<"it is suggested that upsample_w_"
                    <<"% bottom[0]->width() == 0.";
       }
       this->scale_h_ = this->upsample_h_ / bottom[0]->height();
       this->scale_w_ = this->upsample_w_ / bottom[0]->width();
    } else if (!upsample_param.has_scale_h()) {
       this->scale_h_ = this->scale_w_ = upsample_param.scale();
       this->upsample_h_ = this->scale_h_ * bottom[0]->height();
       this->upsample_w_ = this->scale_w_ * bottom[0]->width();
    } else {
       this->scale_h_ = upsample_param.scale_h();
       this->scale_w_ = upsample_param.scale_w();
       this->upsample_h_ = this->scale_h_ * bottom[0]->height();
       this->upsample_w_ = this->scale_w_ * bottom[0]->width();
    }
  }
}

template <typename Dtype>
void MLUUpsampleLayer<Dtype>::Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape{bottom[0]->num(), bottom[0]->channels(),
                        this->upsample_h_, this->upsample_w_};
  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
  BaseDataType mlu_dtype = bottom[0]->mlu_type();
  top[0]->Reshape(top_shape, cpu_dtype, mlu_dtype, CNML_TENSOR);
}

template <typename Dtype>
void MLUUpsampleLayer<Dtype>::fuse(MFusion<Dtype>* fuser) {
  fuser->fuse(upsample_op_ptr_);
}

template <typename Dtype>
void MLUUpsampleLayer<Dtype>::MLUCompileOp() {
  MLU_CHECK(cnmlCompileBaseOp(upsample_op_ptr_,
                              Caffe::rt_core(),
                              Caffe::core_number()));
}

template <typename Dtype>
void MLUUpsampleLayer<Dtype>::MLUCreateOpBindData(
                              const vector<Blob<Dtype>*>& bottom,
                              const vector<Blob<Dtype>*>& top) {
  if (this->NearestNeighbor_mode) {
    MLU_CHECK(cnmlCreateNearestNeighborOpParam(&NearestNeighbor_param_ptr_,
                                               this->upsample_w_,
                                               this->upsample_h_));
    MLU_CHECK(cnmlCreateNearestNeighborOp(&upsample_op_ptr_,
                                          bottom[0]->mlu_tensor(),
                                          top[0]->mlu_tensor(),
                                          NearestNeighbor_param_ptr_));
  } else {
    cnmlUnpoolMode_t unpool_mode = CNML_MAXPOOLBP;
    MLU_CHECK(cnmlCreateUnpoolOpParam(&upsample_param_ptr_,
                                      this->scale_h_,
                                      this->scale_w_,
                                      this->scale_h_,
                                      this->scale_w_,
                                      unpool_mode));

    MLU_CHECK(cnmlCreateUnpoolOp(&upsample_op_ptr_,
                                  bottom[0]->mlu_tensor(),
                                  bottom[1]->mlu_tensor(),
                                  top[0]->mlu_tensor(),
                                  upsample_param_ptr_));
  }
}

template <typename Dtype>
void MLUUpsampleLayer<Dtype>::MLUDestroyOp() {
  if (upsample_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&upsample_op_ptr_));
    upsample_op_ptr_ = nullptr;
  }
  if (upsample_param_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyUnpoolOpParam(&upsample_param_ptr_));
    upsample_param_ptr_ = nullptr;
  }
  if (NearestNeighbor_param_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyNearestNeighborOpParam(&NearestNeighbor_param_ptr_));
    NearestNeighbor_param_ptr_ = nullptr;
  }
}

template <typename Dtype>
MLUUpsampleLayer<Dtype>::~MLUUpsampleLayer() {
  MLUDestroyOp();
}

template <typename Dtype>
void MLUUpsampleLayer<Dtype>::Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  if (this->NearestNeighbor_mode) {
    MLU_CHECK(cnmlComputeNearestNeighborOpForward_V3(upsample_op_ptr_,
                                                  bottom[0]->mutable_mlu_data(),
                                                  top[0]->mutable_mlu_data(),
                                                  Caffe::forward_param(),
                                                  Caffe::queue()));
  } else {
    MLU_CHECK(cnmlComputeUnpoolOpForward_V3(upsample_op_ptr_,
                                            bottom[0]->mutable_mlu_data(),
                                            bottom[1]->mutable_mlu_data(),
                                            top[0]->mutable_mlu_data(),
                                            Caffe::forward_param(),
                                            Caffe::queue()));
  }
}

INSTANTIATE_CLASS(MLUUpsampleLayer);

}  // namespace caffe
#endif  // USE_MLU
