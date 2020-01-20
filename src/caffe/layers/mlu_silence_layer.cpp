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
#include "caffe/layers/mlu_silence_layer.hpp"

namespace caffe {

template <typename Dtype>
void MLUSilenceLayer<Dtype>::Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
  if (Caffe::mode() == Caffe::MFUS) {
    BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
    BaseDataType mlu_dtype = bottom[0]->mlu_type();
    top_temp_.clear();
    top_temp_.push_back(new Blob<Dtype>(bottom[0]->shape(),
                                        cpu_dtype,
                                        mlu_dtype,
                                        CNML_TENSOR));
  }
}

template <typename Dtype>
void MLUSilenceLayer<Dtype>::fuse(MFusion<Dtype>* fuser) {
  fuser->fuse(silence_op_ptr_);
}

template <typename Dtype>
MLUSilenceLayer<Dtype>::~MLUSilenceLayer() {
  MLUDestroyOp();
  if (Caffe::mode() == Caffe::MFUS) {
    if (silence_op_ptr_ != nullptr) {
      MLU_CHECK(cnmlDestroyBaseOp(&silence_op_ptr_));
      silence_op_ptr_ = nullptr;
    }
    for (auto flo : top_temp_) {
      free(flo);
    }
    top_temp_.clear();
  }
}

template <typename Dtype>
void MLUSilenceLayer<Dtype>::MLUCreateOpBindData(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  if (Caffe::mode() == Caffe::MFUS) {
    MLU_CHECK(cnmlCreateActiveOp(
              &silence_op_ptr_,
              cnmlActiveFunction_t::CNML_ACTIVE_NONE,
              bottom[0]->mlu_tensor(),
              top_temp_[0]->mlu_tensor()));
  }
}

INSTANTIATE_CLASS(MLUSilenceLayer);

}  // namespace caffe
#endif  // USE_MLU
