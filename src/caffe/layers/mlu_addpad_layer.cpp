/*
All modification made by Cambricon Corporation: © 2019 Cambricon Corporation
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
#include "caffe/layers/mlu_addpad_layer.hpp"

namespace caffe {

template <typename Dtype>
void MLUAddPadLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
  AddPadParameter addpad_param = this->layer_param_.addpad_param();
  if (addpad_param.has_pad_h() || addpad_param.has_pad_w()) {
    CHECK_EQ(0, addpad_param.pad_size())
        << "Either pad or pad_h/w should be specified; not both.";
    pad_ht_ = addpad_param.pad_h();
    pad_hb_ = addpad_param.pad_h();
    pad_wl_ = addpad_param.pad_w();
    pad_wr_ = addpad_param.pad_w();
  } else {
    CHECK_EQ(1, addpad_param.pad_size())
        << "When using parameter pad, more than 1 are not supported.";
    pad_ht_ = addpad_param.pad(0);
    pad_hb_ = addpad_param.pad(0);
    pad_wl_ = addpad_param.pad(0);
    pad_wr_ = addpad_param.pad(0);
  }
}

template <typename Dtype>
void MLUAddPadLayer<Dtype>::Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {
  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
  BaseDataType mlu_dtype = bottom[0]->mlu_type();
  vector<int> mlu_shape(4, 1);
  mlu_shape[0] = bottom[0]->num();
  mlu_shape[1] = bottom[0]->channels();
  mlu_shape[2] = bottom[0]->height() + pad_ht_ + pad_hb_;
  mlu_shape[3] = bottom[0]->width() + pad_wl_ + pad_wr_;
  top[0]->Reshape(mlu_shape, cpu_dtype, mlu_dtype, CNML_TENSOR);
}

template <typename Dtype>
void MLUAddPadLayer<Dtype>::MLUCreateOpBindData(const vector<Blob<Dtype>*>& bottom,
                                                const vector<Blob<Dtype>*>& top) {
    MLU_CHECK(cnmlCreateAddPadOpParam_V2(&addpad_param_ptr_,
                                        pad_ht_,
                                        pad_hb_,
                                        pad_wl_,
                                        pad_wr_,
                                        0));
    MLU_CHECK(cnmlCreateAddPadOp(&addpad_op_ptr_,
                                addpad_param_ptr_,
                                bottom[0]->mlu_tensor(),
                                top[0]->mlu_tensor()));
}

template <typename Dtype>
void MLUAddPadLayer<Dtype>::MLUCompileOp() {
    MLU_CHECK(cnmlCompileBaseOp(addpad_op_ptr_,
                                Caffe::rt_core(),
                                Caffe::core_number()));
}

template <typename Dtype>
void MLUAddPadLayer<Dtype>::Forward_mlu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    MLU_CHECK(cnmlComputeAddPadOpForward_V3(addpad_op_ptr_,
                                bottom[0]->mutable_mlu_data(),
                                top[0]->mutable_mlu_data(),
                                Caffe::forward_param(),
                                Caffe::queue()));
}

template <typename Dtype>
void MLUAddPadLayer<Dtype>::fuse(MFusion<Dtype>* fuser) {
    fuser->fuse(addpad_op_ptr_);
}

template <typename Dtype>
void MLUAddPadLayer<Dtype>::MLUDestroyOp() {
  if (addpad_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&addpad_op_ptr_));
    addpad_op_ptr_ = nullptr;
  }
}

INSTANTIATE_CLASS(MLUAddPadLayer);

}  // namespace caffe
#endif
