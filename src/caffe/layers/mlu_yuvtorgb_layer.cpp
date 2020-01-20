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
#include "caffe/layers/mlu_yuvtorgb_layer.hpp"

namespace caffe {

template <typename Dtype>
void MLUYUVtoRGBLayer<Dtype>::Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
  if (!((bottom[0]->shape(1) == 1) &&
      (bottom[1]->shape(1) == 1) &&
      (bottom[0]->shape(2) / bottom[1]->shape(2) == 2) &&
      (bottom[0]->shape(3) == bottom[1]->shape(3) &&
      (bottom[0]->shape(0) == bottom[1]->shape(0))))) {
     LOG(FATAL) << "YUV shape wrong.";
  }
  bottom[0]->set_preprocess(false);
  bottom[1]->set_preprocess(false);
  top[0]->set_preprocess(false);

  vector<int> out_shape(bottom[0]->shape());
  out_shape[1] = 4;
  BaseDataType cpu_dtype = DT_UINT8;
  BaseDataType mlu_dtype = DT_UINT8;
  bottom[0]->set_cpu_type(cpu_dtype);
  bottom[0]->set_mlu_type(mlu_dtype);
  bottom[1]->set_cpu_type(cpu_dtype);
  bottom[1]->set_mlu_type(mlu_dtype);
  top[0]->Reshape(out_shape, cpu_dtype, mlu_dtype, CNML_TENSOR);
}

template <typename Dtype>
void MLUYUVtoRGBLayer<Dtype>::fuse(MFusion<Dtype>* fuser) {
  fuser->fuse(yuv2rgb_op_ptr_);
}

template <typename Dtype>
void MLUYUVtoRGBLayer<Dtype>::MLUDestroyOp() {
  if (yuv2rgb_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&yuv2rgb_op_ptr_));
    yuv2rgb_op_ptr_ = nullptr;
  }
}

template <typename Dtype>
void MLUYUVtoRGBLayer<Dtype>::MLUCompileOp() {
  MLU_CHECK(cnmlCompileBaseOp(yuv2rgb_op_ptr_,
                              Caffe::rt_core(),
                              Caffe::core_number()));
}

template <typename Dtype>
void MLUYUVtoRGBLayer<Dtype>::MLUCreateOpBindData(
                                const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top) {
  if (this->layer_param_.yuvtorgb_param().input_format() == 0) {
    yuv_type = CNML_YUV420SP_NV12;
  } else {
    yuv_type = CNML_YUV420SP_NV21;
  }
  if (this->layer_param_.yuvtorgb_param().output_format() == 0) {
    rgb_type = CNML_RGB0;
  } else if (this->layer_param_.yuvtorgb_param().output_format() == 1) {
    rgb_type = CNML_BGR0;
  } else {
    rgb_type = CNML_ARGB;
  }
  MLU_CHECK(cnmlCreateYUVtoRGBProOp(&yuv2rgb_op_ptr_,
                                 bottom[0]->mlu_tensor(),
                                 bottom[1]->mlu_tensor(),
                                 top[0]->mlu_tensor(),
                                 yuv_type, rgb_type));
}

template <typename Dtype>
void MLUYUVtoRGBLayer<Dtype>::Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  MLU_CHECK(cnmlComputeYUVtoRGBProOpForward_V4(yuv2rgb_op_ptr_,
                                            NULL,
                                            bottom[0]->mutable_mlu_data(),
                                            NULL,
                                            bottom[1]->mutable_mlu_data(),
                                            NULL,
                                            top[0]->mutable_mlu_data(),
                                            Caffe::queue(),
                                            NULL));
}

INSTANTIATE_CLASS(MLUYUVtoRGBLayer);
REGISTER_LAYER_CLASS(MLUYUVtoRGB);

}  // namespace caffe
#endif  // USE_MLU
