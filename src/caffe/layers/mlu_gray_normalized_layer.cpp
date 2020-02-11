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
#include <iostream>
#include <vector>
#include "caffe/layers/mlu_gray_normalized_layer.hpp"

namespace caffe {

template <typename Dtype>
void MLUGrayNormalizedLayer<Dtype>::Reshape_tensor(
                                 const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top) {
  if (!(bottom[0]->shape(1) == 1)) {
     LOG(FATAL) << "YUV shape wrong.";
  }
  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
  BaseDataType mlu_dtype = DT_FLOAT16;
  top[0]->Reshape(bottom[0]->shape(), cpu_dtype, mlu_dtype, CNML_TENSOR);
}

template <typename Dtype>
void MLUGrayNormalizedLayer<Dtype>::fuse(MFusion<Dtype>* fuser) {
  fuser->fuse(yuv2gray_op_ptr_);
}

template <typename Dtype>
void MLUGrayNormalizedLayer<Dtype>::MLUDestroyOp() {
  if (yuv2gray_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&yuv2gray_op_ptr_));
    yuv2gray_op_ptr_ = nullptr;
  }
}

template <typename Dtype>
void MLUGrayNormalizedLayer<Dtype>::MLUCompileOp() {
  MLU_CHECK(cnmlCompileBaseOp(yuv2gray_op_ptr_,
                              Caffe::rt_core(),
                              Caffe::core_number()));
}

template <typename Dtype>
void MLUGrayNormalizedLayer<Dtype>::MLUCreateOpBindData(
                                const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top) {
  bottom[0]->set_preprocess(false);
  bottom[0]->set_cpu_type(DT_UINT8);
  bottom[0]->set_mlu_type(DT_UINT8);
  MLU_CHECK(cnmlCreateGrayNormalizeForKcfOp(&yuv2gray_op_ptr_,
                                 bottom[0]->mlu_tensor(),
                                 top[0]->mlu_tensor()));
}

template <typename Dtype>
void MLUGrayNormalizedLayer<Dtype>::Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  MLU_CHECK(cnmlComputeGrayNormalizeForKcfOpForward(yuv2gray_op_ptr_,
                                        bottom[0]->mlu_tensor(),
                                        bottom[0]->mutable_mlu_data(),
                                        top[0]->mlu_tensor(),
                                        top[0]->mutable_mlu_data(),
                                        Caffe::queue(), NULL));
}

INSTANTIATE_CLASS(MLUGrayNormalizedLayer);
REGISTER_LAYER_CLASS(MLUGrayNormalized);

}  // namespace caffe
#endif  // USE_MLU
