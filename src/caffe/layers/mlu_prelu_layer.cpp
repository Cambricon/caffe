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
#include <memory>
#include <vector>
#include "caffe/filler.hpp"
#include "caffe/layers/mlu_prelu_layer.hpp"

namespace caffe {

template <typename Dtype>
void MLUPReLULayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "Number of axes of bottom blob must be >=2.";
  PReLUParameter prelu_param = this->layer_param().prelu_param();

  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
    BaseDataType mlu_dtype = this->layer_param_.blobs_dtype_size() > 0 ?
          this->layer_param_.blobs_dtype(0).type() : DT_FLOAT16;

    // CAUTION: mlu require input.c==param.c,that means mlu does not support
    // channel_shared. when split_num > 1,param's channels should equal 1.
    vector<int> param_shape(4, 1);
    if (prelu_param.channel_shared() == 1) {
      this->blobs_[0].reset(new Blob<Dtype>(
          param_shape, cpu_dtype, mlu_dtype, CNML_CONST));
    } else {
      param_shape[1]= bottom[0]->channels();
      this->blobs_[0].reset(new Blob<Dtype>(
         param_shape, cpu_dtype, mlu_dtype, CNML_CONST));
    }

    shared_ptr<Filler<Dtype> > filler;
    if (prelu_param.has_filler()) {
      filler.reset(GetFiller<Dtype>(prelu_param.filler()));
    } else {
      FillerParameter filler_param;
      filler_param.set_type("constant");
      filler_param.set_value(0.25);
      filler.reset(GetFiller<Dtype>(filler_param));
    }
    filler->Fill(this->blobs_[0].get());
  }

  if (prelu_param.channel_shared() == 1) {
    CHECK_EQ(this->blobs_[0]->count(), 1)
        << "Negative slope size is inconsistent with prototxt config";
  } else {
    CHECK_EQ(this->blobs_[0]->count(), bottom[0]->channels())
        << "Negative slope size is inconsistent with prototxt config";
  }
}

template <typename Dtype>
void MLUPReLULayer<Dtype>::Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "Number of axes of bottom blob must be >=2.";
  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
  BaseDataType mlu_dtype = bottom[0]->mlu_type();
  top[0]->Reshape(bottom[0]->shape(), cpu_dtype, mlu_dtype, CNML_TENSOR);
}

template <typename Dtype>
void MLUPReLULayer<Dtype>::MLUCreateOpBindData(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  /* PreluOp */
  MLU_CHECK(cnmlCreatePreluOp(&prelu_op_ptr_,
                              bottom[0]->mlu_tensor(),
                              top[0]->mlu_tensor(),
                              this->blobs_[0]->mlu_tensor()));
  MLU_CHECK(cnmlBindConstData_V2(this->blobs_[0]->mlu_tensor(),
            reinterpret_cast<float*>(this->blobs_[0]->sync_data()),
            false));
}

template <typename Dtype>
void MLUPReLULayer<Dtype>::MLUCompileOp() {
  MLU_CHECK(cnmlCompileBaseOp(prelu_op_ptr_,
                              Caffe::rt_core(),
                              Caffe::core_number()));
}

template <typename Dtype>
void MLUPReLULayer<Dtype>::Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
  MLU_CHECK(cnmlComputePreluOpForward_V3(prelu_op_ptr_,
                                         bottom[0]->mutable_mlu_data(),
                                         top[0]->mutable_mlu_data(),
                                         Caffe::forward_param(), Caffe::queue()));
}

template <typename Dtype>
void MLUPReLULayer<Dtype>::fuse(MFusion<Dtype>* fuser) {
  fuser->fuse(prelu_op_ptr_);
}

template <typename Dtype>
void MLUPReLULayer<Dtype>::MLUDestroyOp() {
  if (prelu_op_ptr_ != NULL) {
    MLU_CHECK(cnmlDestroyBaseOp(&prelu_op_ptr_));
    prelu_op_ptr_ = NULL;
  }
}

INSTANTIATE_CLASS(MLUPReLULayer);

}  // namespace caffe
#endif
