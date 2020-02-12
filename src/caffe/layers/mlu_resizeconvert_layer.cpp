/*
All modification made by Cambricon Corporation: © 2018--2019 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2019, the respective contributors
All rights reserved.
For the list ofcontributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md
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

#include "caffe/layers/mlu_resizeconvert_layer.hpp"

namespace caffe {

template <typename Dtype>
void MLUResizeConvertLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  ResizeConvertParameter convert_param =
      this->layer_param().resize_convert_param();
  resize_h_ = convert_param.resize_h();
  resize_w_ = convert_param.resize_w();
  input_ = convert_param.input_type();
  output_ = convert_param.output_type();
}

template <typename Dtype>
void MLUResizeConvertLayer<Dtype>::Reshape_tensor(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape(bottom[0]->shape());
  top_shape[0] = bottom[0]->num();
  top_shape[1] = 4;
  top_shape[2] = resize_h_;
  top_shape[3] = resize_w_;
  top[0]->Reshape(top_shape, DT_UINT8, DT_UINT8, CNML_TENSOR, CNML_NCHW);

  for (int i = 0; i < bottom.size(); i++) {
    bottom[i]->set_preprocess(false);
    bottom[i]->set_cpu_type(DT_INT32);
    bottom[i]->set_mlu_type(DT_INT32);
  }
  top[0]->set_preprocess(false);
  top[0]->set_cpu_type(DT_UINT8);
  top[0]->set_mlu_type(DT_UINT8);

}

template <typename Dtype>
void MLUResizeConvertLayer<Dtype>::MLUDestroyOp() {
  if (convert_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&convert_op_ptr_));
    convert_op_ptr_ = nullptr;
  }
}

template <typename Dtype>
void MLUResizeConvertLayer<Dtype>::MLUCreateOpBindData(const vector<Blob<Dtype>*>& bottom,
                                                       const vector<Blob<Dtype>*>& top) {
  ResizeConvertParameter convert_param =
                         this->layer_param().resize_convert_param();
  int height = bottom[2]->cpu_data()[0];
  int width = bottom[2]->cpu_data()[1];
  int batchNum = bottom[0]->num();
  // roiParam
  int roi_x = convert_param.crop_x();
  int roi_y = convert_param.crop_y();
  int roi_w = convert_param.crop_w();
  int roi_h = convert_param.crop_h();
  int d_col = resize_w_;
  int d_row = resize_h_;
 
  ioParams mode;
  mode.color = YUV_TO_RGBA_NV21;
  cnmlCreatePluginResizeConvert16B16COpParam(&param,
      batchNum, height, width, d_row, d_col,
      roi_x, roi_y, roi_w, roi_h, mode, Caffe::rt_core());
  
  cnmlTensor_t mlu_rank_tensor[3];
  mlu_rank_tensor[0] = bottom[0]->mlu_tensor();
  mlu_rank_tensor[1] = bottom[1]->mlu_tensor();
  mlu_rank_tensor[2] = bottom[2]->mlu_tensor();
  cnmlTensor_t mlutensor_output_tensor[1];
  mlutensor_output_tensor[0] = top[0]->mlu_tensor();

  cnmlCreatePluginResizeConvert16B16COp(
      &convert_op_ptr_,
      param,
      mlu_rank_tensor,
      mlutensor_output_tensor);

}

template <typename Dtype>
void MLUResizeConvertLayer<Dtype>::MLUCompileOp() {
  MLU_CHECK(cnmlCompileBaseOp(convert_op_ptr_, Caffe::rt_core(),
                              Caffe::core_number()));
}

template <typename Dtype>
void MLUResizeConvertLayer<Dtype>::Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                                               const vector<Blob<Dtype>*>& top) {
  // bottom: Y + UV
  void* mlutensor_input_ptrs[3];
  mlutensor_input_ptrs[0] = bottom[0]->mutable_mlu_data();
  mlutensor_input_ptrs[1] = bottom[1]->mutable_mlu_data();
  mlutensor_input_ptrs[2] = bottom[2]->mutable_mlu_data();

  void* mlutensor_output_ptrs[1];
  mlutensor_output_ptrs[0] = top[0]->mutable_mlu_data();

  cnmlComputePluginResizeConvert16B6COpForward(
      convert_op_ptr_,
      param,
      mlutensor_input_ptrs,
      mlutensor_output_ptrs,
      *Caffe::forward_param(),
      Caffe::queue());
}

template <typename Dtype>
void MLUResizeConvertLayer<Dtype>::fuse(MFusion<Dtype>* fuser) {
  fuser->fuse(convert_op_ptr_);
}

INSTANTIATE_CLASS(MLUResizeConvertLayer);
REGISTER_LAYER_CLASS(MLUResizeConvert);

}  // namespace caffe
#endif  // USE_MLU
