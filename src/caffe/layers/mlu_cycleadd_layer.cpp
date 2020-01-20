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
#include "caffe/layers/mlu_cycleadd_layer.hpp"
namespace caffe {

template <typename Dtype>
void MLUCycleAddLayer<Dtype>::Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[1]->num(), 1) << "number of bottom[1] not equal to 1";
  CHECK_EQ(bottom[1]->height(), 1) << "height of bottom[1] not be equal to 1";
  CHECK_EQ(bottom[1]->width(), 1) << "width of bottom[1] not equal to 1";
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels())
     << "channel of bottom[0] and botoom[1] not equal";

  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
  BaseDataType mlu_dtype = bottom[0]->mlu_type();
  top[0]->Reshape(bottom[0]->shape(), cpu_dtype, mlu_dtype, CNML_TENSOR);
}

template <typename Dtype>
void MLUCycleAddLayer<Dtype>::MLUCreateOpBindData(const vector<Blob<Dtype>*>& bottom,
                                                  const vector<Blob<Dtype>*>& top) {
  MLU_CHECK(cnmlCreateCycleAddOp(&cycleadd_op_ptr_,
                                 bottom[0]->mlu_tensor(),
                                 bottom[1]->mlu_tensor(),
                                 top[0]->mlu_tensor()));
}

template <typename Dtype>
void MLUCycleAddLayer<Dtype>::Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  MLU_CHECK(cnmlComputeCycleAddOpForward_V3(cycleadd_op_ptr_,
                                        bottom[0]->mutable_mlu_data(),
                                        bottom[1]->mutable_mlu_data(),
                                        top[0]->mutable_mlu_data(),
                                        Caffe::forward_param(),
                                        Caffe::queue()));
}

template <typename Dtype>
void MLUCycleAddLayer<Dtype>::fuse(MFusion<Dtype>* fuser) {
  fuser->fuse(cycleadd_op_ptr_);
}

template <typename Dtype>
void MLUCycleAddLayer<Dtype>::MLUDestroyOp() {
  if (cycleadd_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&cycleadd_op_ptr_));
    cycleadd_op_ptr_ = nullptr;
  }
}

INSTANTIATE_CLASS(MLUCycleAddLayer);

}  // namespace caffe
#endif
