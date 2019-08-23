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

#include "caffe/layers/mlu_reorg_layer.hpp"

namespace caffe {

template <typename Dtype>
void MLUReorgLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ReorgLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void MLUReorgLayer<Dtype>::Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  CHECK(bottom.size() == 1 && top.size() == 1);
  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
  BaseDataType mlu_dtype = DT_FLOAT16;
  top[0]->Reshape(bottom[0]->num(), this->reorged_channels_,
                  this->reorged_height_, this->reorged_width_,
                  cpu_dtype, mlu_dtype, CNML_TENSOR);
}

template <typename Dtype>
void MLUReorgLayer<Dtype>::MLUDestroyOp() {
  if (reorg_op_ptr_ != NULL) {
    MLU_CHECK(cnmlDestroyBaseOp(&reorg_op_ptr_));
    MLU_CHECK(cnmlDestroyReorgOpParam(&param));
    reorg_op_ptr_ = NULL;
  }
}

template <typename Dtype>
void MLUReorgLayer<Dtype>::MLUCreateOpBindData(
                                      const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
  MLU_CHECK(cnmlCreateReorgOpParam(&param,
                                   this->stride_,
                                   this->stride_,
                                   this->reverse_));

  MLU_CHECK(cnmlCreateReorgOp(&reorg_op_ptr_,
                              param,
                              bottom[0]->mlu_tensor(),
                              top[0]->mlu_tensor()));
  if (this->layer_param_.bottom_mlu_dtype_size() > 0) {
    if (this->layer_param_.bottom_mlu_dtype(0).position_size()) {
      bottom[0]->set_mlu_position(
        this->layer_param_.bottom_mlu_dtype(0).position(0));
    }
    if (this->layer_param_.bottom_mlu_dtype(0).scale_size()) {
      bottom[0]->set_mlu_scale(
        this->layer_param_.bottom_mlu_dtype(0).scale(0));
    }
    cnmlEnableReorgOpInt8Mode(reorg_op_ptr_,
      bottom[0]->mlu_position(),
      bottom[0]->mlu_scale());
  }
}

template <typename Dtype>
void MLUReorgLayer<Dtype>::MLUCompileOp() {
  MLU_CHECK(cnmlCompileBaseOp(reorg_op_ptr_,
                              Caffe::rt_core(),
                              Caffe::model_parallel()));
}

template <typename Dtype>
MLUReorgLayer<Dtype>::~MLUReorgLayer() {
  MLUDestroyOp();
}

template <typename Dtype>
void MLUReorgLayer<Dtype>::Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
  MLU_CHECK(cnmlComputeReorgOpForward_V3(reorg_op_ptr_,
                                      bottom[0]->mutable_mlu_data(),
                                      top[0]->mutable_mlu_data(),
                                      Caffe::forward_param(), Caffe::queue()));
}

INSTANTIATE_CLASS(MLUReorgLayer);

}  // namespace caffe
#endif
