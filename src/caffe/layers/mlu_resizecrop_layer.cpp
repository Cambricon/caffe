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
#include "caffe/layers/mlu_resizecrop_layer.hpp"
#include <vector>

namespace caffe {
typedef uint16_t half;

template <typename Dtype>
void MLUResizecropLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {
  ResizecropParameter resize_param = this->layer_param().resize_crop_param();
  resize_h_ = resize_param.resize_h();
  resize_w_ = resize_param.resize_w();
  crop_x_ = resize_param.crop_x();
  crop_y_ = resize_param.crop_y();
  crop_w_ = resize_param.crop_w();
  crop_h_ = resize_param.crop_h();
  if (crop_w_ <= 1) {
    crop_w_ = bottom[0]->width();
  }
  if (crop_h_ <= 1) {
    crop_h_ = bottom[0]->height();
  }
  CHECK_LE(crop_x_ + crop_w_, bottom[0]->width())
      << "crop_x + crop_w should less than bottom width";
  CHECK_LE(crop_y_ + crop_h_, bottom[0]->height())
      << "crop_y + crop_h should less than bottom height";
  bottom[0]->set_preprocess(false);
  top[0]->set_preprocess(false);
  if ((resize_h_ == 0 && resize_w_ == 0) ||
       (resize_h_ == crop_h_ && resize_w_ == crop_w_))
    resize_ = false;
  if (crop_h_ == bottom[0]->height() && crop_w_ == bottom[0]->width())
    crop_ = false;
}

template <typename Dtype>
void MLUResizecropLayer<Dtype>::Reshape_tensor(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BaseDataType cpu_dtype = DT_UINT8;
  BaseDataType mlu_dtype = DT_UINT8;
  vector<int> top_shape(4, 1);
  top_shape[0] = bottom[0]->num();
  top_shape[1] = 4;
  top_shape[2] = crop_h_;
  top_shape[3] = crop_w_;
  crop_blob_.Reshape(top_shape, cpu_dtype, mlu_dtype, CNML_TENSOR);
  top_shape[2] = resize_? resize_h_: crop_h_;
  top_shape[3] = resize_? resize_w_: crop_w_;
  top[0]->Reshape(top_shape, cpu_dtype, mlu_dtype, CNML_TENSOR);

  bottom[0]->set_cpu_type(cpu_dtype);
  bottom[0]->set_mlu_type(mlu_dtype);
  bottom[0]->set_preprocess(false);
  top[0]->set_preprocess(false);
  crop_blob_.set_preprocess(false);
}

template <typename Dtype>
void MLUResizecropLayer<Dtype>::MLUCreateOpBindData(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (crop_) {
    MLU_CHECK(cnmlCreateGrepOpParam(&crop_param_ptr_,
          0,
          crop_y_,
          crop_x_,
          0));
    MLU_CHECK(cnmlCreateGrepOp(&crop_op_ptr_,
          crop_param_ptr_,
          bottom[0]->mlu_tensor(),
          resize_? crop_blob_.mlu_tensor(): top[0]->mlu_tensor()));
  }
  if (resize_) {
    ioParams mode;
    mode.color = RGBA_TO_RGBA;
    mode.datatype = UINT8_TO_UINT8;
    cnmlCreatePluginResizeOpParam(
        &param_,
        crop_? crop_h_: bottom[0]->height(),
        crop_? crop_w_: bottom[0]->width(),
        top[0]->height(),
        top[0]->width(),
        mode,
        Caffe::rt_core());
    cnmlCreatePluginResizeOp(
        &resize_op_ptr_,
        param_,
        top[0]->mlu_tensor(),
        crop_? crop_blob_.mlu_tensor(): bottom[0]->mlu_tensor());
  }
}

template <typename Dtype>
void MLUResizecropLayer<Dtype>::MLUCompileOp() {
  if (crop_) {
    MLU_CHECK(cnmlCompileBaseOp(crop_op_ptr_, Caffe::rt_core(),
                          Caffe::core_number()));
  }
  if (resize_) {
    MLU_CHECK(cnmlCompileBaseOp(resize_op_ptr_, Caffe::rt_core(),
                             Caffe::core_number()));
  }
}

template <typename Dtype>
void MLUResizecropLayer<Dtype>::Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
  if (crop_) {
    MLU_CHECK(cnmlComputeGrepOpForward_V3(crop_op_ptr_,
        bottom[0]->mutable_mlu_data(),
        resize_? crop_blob_.mutable_mlu_data(): top[0]->mutable_mlu_data(),
        Caffe::forward_param(),
        Caffe::queue()));
  }
  if (resize_) {
     MLU_CHECK(cnmlComputePluginResizeOpForward(
           resize_op_ptr_,
           crop_? crop_blob_.mlu_tensor(): bottom[0]->mlu_tensor(),
           top[0]->mlu_tensor(),
           crop_? crop_blob_.mutable_mlu_data(): bottom[0]->mutable_mlu_data(),
           top[0]->mutable_mlu_data(),
           *Caffe::forward_param(),
           Caffe::queue()));
  }
}

template <typename Dtype>
void MLUResizecropLayer<Dtype>::fuse(MFusion<Dtype>* fuser) {
  if (crop_) {
    fuser->fuse(crop_op_ptr_);
  }
  if (resize_) {
    fuser->fuse(resize_op_ptr_);
  }
}

template <typename Dtype>
void MLUResizecropLayer<Dtype>::MLUDestroyOp() {
  if (resize_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&resize_op_ptr_));
    resize_op_ptr_ = nullptr;
  }
  if (crop_param_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyGrepOpParam(&crop_param_ptr_));
    crop_param_ptr_ = nullptr;
  }
  if (crop_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&crop_op_ptr_));
    crop_op_ptr_ = nullptr;
  }
  if (param_ != nullptr) {
    cnmlDestroyPluginResizeOpParam(&param_);
    param_ = nullptr;
  }
}

INSTANTIATE_CLASS(MLUResizecropLayer);

}  // namespace caffe
#endif  // USE_MLU
