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

#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include "caffe/filler.hpp"
#include "caffe/layers/mlu_conv_layer.hpp"
#include "caffe/common.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

template <typename Dtype>
void MLUConvolutionLayer<Dtype>::bindDataAndSetComputingDataType(
    shared_ptr<Blob<Dtype>> blob, cnmlBaseOp_t op, BaseDataType type) {
  MLU_CHECK(cnmlBindConstData_V2(blob->mlu_tensor(), blob->sync_data(), false));
  cnmlQuantizedParam_t param;
  if (blob->has_mlu_position()) {
    CNRT_CHECK(cnmlCreateQuantizedParam(&param, blob->mlu_position(),
          blob->mlu_scale(), 0));
    cnmlSetOperationComputingDataType(op, blob->mlu_tensor(),
        to_cnml_dtype(type), param);
    if (param != nullptr) {
      MLU_CHECK(cnmlDestroyQuantizedParam(&param));
      param = nullptr;
    }
  } else {
    LOG(FATAL) << "Quantized tensor should have position";
  }
}

template <typename Dtype>
void MLUConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
  ConvolutionLayer<Dtype>::LayerSetUp(bottom, top);
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();

  if (bottom[0]->shape(1) > 1 &&
      bottom[0]->shape(1) == this->group_ &&
    this->num_output_ % this->group_ == 0) {
    is_depthwise_ = true;
    multiplier_ = this->num_output_ / this->group_;
    assert(multiplier_ == 1);
  } else {
    is_depthwise_ = false;
  }
  // mlu tensor has more parameters, has to initialize them again
  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
  BaseDataType mlu_dtype = DT_FLOAT32;

  vector<int> weight_shape(2);
  int* kernel_shape_data = this->kernel_shape_.mutable_cpu_data();
  weight_shape[0] = this->conv_out_channels_;
  weight_shape[1] = this->conv_in_channels_ / this->group_;
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    weight_shape.push_back(kernel_shape_data[i]);
  }

  BaseDataType const_dtype = this->layer_param_.has_top_mlu_dtype() ?
         this->layer_param_.top_mlu_dtype(): DT_FLOAT16;
  if (is_depthwise_) {
    const_dtype = bottom[0]->mlu_type();
    mlu_dtype = bottom[0]->mlu_type();
    weight_shape[0] = 1;
    weight_shape[1] = this->conv_in_channels_ * conv_param.num_output() / this->group_;
  }
  // blobs_[0] mlu tensor will be created as set_mlu_postion is called
  // so it can't be reshaped any more.
  vector<int> stride_value(4, 0);
  stride_value[1] = 1;
  this->blobs_[0].reset(new Blob<Dtype>(weight_shape, cpu_dtype, mlu_dtype,
        CNML_FILTER, CNML_NCHW,
        (this->conv_first_ && bottom[0]->channels() == 3) ? &stride_value: nullptr));
  if (!is_depthwise_)
    SetBlobPosition(this->blobs_[0].get());
  shared_ptr<Filler<Dtype>> weight_filler(GetFiller<Dtype>(conv_param.weight_filler()));
  weight_filler->Fill(this->blobs_[0].get());

  // If necessary, initialize and fill the biases.
  if (this->bias_term_) {
    this->blobs_[1].reset(new Blob<Dtype>(this->blobs_[1]->shape(),
                          cpu_dtype,
                          const_dtype,
                          CNML_CONST,
                          CNML_CNHW));
    shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(conv_param.bias_filler()));
    bias_filler->Fill(this->blobs_[1].get());
  }
  this->kernel_dim_ = this->blobs_[0]->count(1);
  this->weight_offset_ = this->conv_out_channels_ * this->kernel_dim_ / this->group_;

  // so far neither conv_first nor depthwise supports dilation.
  if ((is_depthwise_ || this->conv_first_) && dilate()) {
    LOG(ERROR) << "dilate is set with convolution first or convolution depthwise ";
    LOG(ERROR) << "Please check your pt file";
  }
  if (this->conv_first_ && bottom[0]->channels() == 3) {
    // add the stride to the tensor.
    bottom[0]->set_dim_strides(stride_value);
    this->mean_.set_dim_strides(stride_value);
  }
  bottom_size_ = bottom.size();
  input_quant_params.resize(bottom_size_, nullptr);
}

template <typename Dtype>
void MLUConvolutionLayer<Dtype>::Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                                                const vector<Blob<Dtype>*>& top) {
  BaseConvolutionLayer<Dtype>::Reshape(bottom, top);
  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
  BaseDataType mlu_dtype = this->layer_param_.has_top_mlu_dtype() ?
         this->layer_param_.top_mlu_dtype(): DT_FLOAT16;
  if (is_depthwise_) {
     mlu_dtype = bottom[0]->mlu_type();
  }
  top[0]->Reshape(top[0]->shape(), cpu_dtype, mlu_dtype, CNML_TENSOR);
}

template <typename Dtype>
void MLUConvolutionLayer<Dtype>::MLUCreateOpBindData(const vector<Blob<Dtype>*>& bottom,
                                                     const vector<Blob<Dtype>*>& top) {
  const int* dilation_data = this->dilation_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int stride_h = stride_data[0];
  const int stride_w = stride_data[1];
  const int dilation_h = dilation_data[0];
  const int dilation_w = dilation_data[1];
  // spread pad to 4 dimensions
  const int pad_htop = pad_data[0];
  const int pad_wleft = pad_data[1];
  const int pad_hbottom = pad_data[2];
  const int pad_wright = pad_data[3];
  add_pad_ = pad_htop != 0 || pad_wleft != 0 || pad_hbottom !=0 || pad_wright !=0;
  if (is_depthwise_)
    add_pad_ = add_pad_ && !(pad_htop == pad_hbottom && pad_wleft == pad_wright);
  // ARGB input: reshape weights from 3 channels to ARGB
  if (this->conv_first_ && bottom[0]->channels() == 4) {
    MLUConvolutionLayer<Dtype>::ArrangeWeightRGB();
  }
  if (this->conv_first_ && !is_depthwise_) {  // depthwise doesn't support int8 for now
    for (int i = 0; i < bottom.size(); i++)
      bottom[i]->set_mlu_type(DT_UINT8);
  }

  // For now either add or dilate takes effect, not both. So for below scenarios
  // extra pad operation is needed. When this happens, pad blob should be the input
  // tensor to Op. Position and scale information is set upon it.
  //
  // Extra pad op is needed for:
  // Non-first-conv layer
  // Depthwise Convolution with pad enabled
  // both dilate and pad are set
  if (!this->conv_first_ && add_pad_ && (is_depthwise_ || dilate())) {
    BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
    BaseDataType mlu_dtype = bottom[0]->mlu_type();
    MLU_CHECK(cnmlCreateAddPadOpParam_V2(&mlu_addpad_param_ptr_,
                                      pad_htop, pad_hbottom, pad_wleft, pad_wright, 0));
    mlu_addpad_op_ptrs_ = new cnmlBaseOp_t[bottom.size()];
    for (int i = 0; i < bottom.size(); i++) {
      vector<int> pad_shape;
      pad_shape.push_back(bottom[i]->shape(0));
      pad_shape.push_back(bottom[i]->channels());
      pad_shape.push_back(bottom[i]->height() + pad_htop + pad_hbottom);
      pad_shape.push_back(bottom[i]->width() + pad_wleft + pad_wright);
      addpad_.push_back(new Blob<Dtype>(pad_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));
      MLU_CHECK(cnmlCreateAddPadOp(&mlu_addpad_op_ptrs_[i],
          mlu_addpad_param_ptr_,
          bottom[i]->mlu_tensor(),
          addpad_[i]->mlu_tensor()));
    }
  }

  if (this->conv_first_) {
    MLU_CHECK(cnmlCreateConvFirstOpParam(&mlu_convf_param_ptr_,
        stride_h,
        stride_w,
        this->pad_.mutable_cpu_data()[1],
        this->pad_.mutable_cpu_data()[3],
        this->pad_.mutable_cpu_data()[0],
        this->pad_.mutable_cpu_data()[2]));
  } else {
    if (is_depthwise_) {
      MLU_CHECK(cnmlCreateConvDepthwiseOpParam_V2(&depthwise_param_ptr_,
            stride_h,
            stride_w,
            add_pad_? 0: pad_htop * 2,
            add_pad_? 0: pad_wleft * 2));
    } else {
        MLU_CHECK(cnmlCreateConvOpParam(&mlu_conv_param_ptr_,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            (this->add_pad_ && dilate()) ? 0 : pad_htop + pad_hbottom,
            (this->add_pad_ && dilate()) ? 0 : pad_wleft + pad_wright));
    }
  }

  mlu_conv_op_ptrs_ = new cnmlBaseOp_t[bottom.size()];
  for (int i = 0; i < bottom.size(); i++) {
    if (this->conv_first_) {
      MLUCreateOpFirstConv(bottom, top, i);
      continue;
    }
    // deal with non-first covolution layer
    if (this->group_ == 1) {  // first scenario: ConvOp is used
      MLU_CHECK(cnmlCreateConvOp(&mlu_conv_op_ptrs_[i],
          mlu_conv_param_ptr_,
          (this->add_pad_ && dilate()) ?
          addpad_[i]->mlu_tensor() : bottom[i]->mlu_tensor(),
          top[i]->mlu_tensor(),
          this->blobs_[0]->mlu_tensor(),
          this->bias_term_ ? this->blobs_[1]->mlu_tensor() : nullptr));
      BaseDataType mlu_dtype = this->layer_param_.blobs_dtype_size() > 0 ?
                                    this->layer_param_.blobs_dtype(0).type() : DT_INT8;
      bindDataAndSetComputingDataType(this->blobs_[0], mlu_conv_op_ptrs_[i], mlu_dtype);
      SetBottomPositionScale(mlu_conv_op_ptrs_[i],
                             input_quant_params[i],
                             (this->add_pad_ && dilate()) ?
                             addpad_[i] : bottom[i], i);

      if ( this->bias_term_ ) {
          MLU_CHECK(cnmlBindConstData_V2(this->blobs_[1]->mlu_tensor(),
                                         this->blobs_[1]->sync_data(),
                                         false));
      }
    } else {
      if (is_depthwise_) {  // 2nd scenario: ConvDepthwiseOp is used
        MLU_CHECK(cnmlCreateConvDepthwiseOp(&mlu_conv_op_ptrs_[i],
            depthwise_param_ptr_,
            add_pad_ ? addpad_[i]->mlu_tensor() : bottom[i]->mlu_tensor(),
            top[i]->mlu_tensor(),
            this->blobs_[0]->mlu_tensor(),
            this->bias_term_ ? this->blobs_[1]->mlu_tensor() : nullptr));
        MLU_CHECK(cnmlBindConstData_V2(this->blobs_[0]->mlu_tensor(),
                                       this->blobs_[0]->sync_data(),
                                       false));
        if ( this->bias_term_ )
            MLU_CHECK(cnmlBindConstData_V2(this->blobs_[1]->mlu_tensor(),
                                           this->blobs_[1]->sync_data(),
                                           false));
      } else {  // default scenario: ConvGroupOp is used
        MLU_CHECK(cnmlCreateConvGroupOp(&mlu_conv_op_ptrs_[i],
            mlu_conv_param_ptr_,
            (this->add_pad_ && dilate()) ?
                addpad_[i]->mlu_tensor() : bottom[i]->mlu_tensor(),
            top[i]->mlu_tensor(),
            this->blobs_[0]->mlu_tensor(),
            this->bias_term_ ? this->blobs_[1]->mlu_tensor() : nullptr,
            this->group_));
        BaseDataType mlu_dtype = this->layer_param_.blobs_dtype_size() > 0 ?
                                 this->layer_param_.blobs_dtype(0).type() : DT_INT8;
        bindDataAndSetComputingDataType(this->blobs_[0],
            mlu_conv_op_ptrs_[i], mlu_dtype);

        SetBottomPositionScale(mlu_conv_op_ptrs_[i],
                               input_quant_params[i],
                               (this->add_pad_ && dilate()) ?
                               addpad_[i] : bottom[i], i);
        if ( this->bias_term_ )
            MLU_CHECK(cnmlBindConstData_V2(this->blobs_[1]->mlu_tensor(),
                                           this->blobs_[1]->sync_data(),
                                           false));
      }
    }
  }
}

template <typename Dtype>
void MLUConvolutionLayer<Dtype>::MLUCreateOpFirstConv(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    const int index) {
  vector<int> stride_value(4, 0);
  stride_value[1] = 1;
  bool std = this->layer_param().convolution_param().has_std();
  bool scale = this->layer_param().convolution_param().has_scale();
  if (std || scale) {
    BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
    BaseDataType const_dtype = this->layer_param_.has_top_mlu_dtype() ?
               this->layer_param_.top_mlu_dtype(): DT_FLOAT16;
    vector<int> std_shape(4, 1);
    std_shape[1] = bottom[0]->channels();
    if (bottom[0]->channels() == 3)
      std_.set_dim_strides(stride_value);
    std_.Reshape(std_shape, cpu_dtype, const_dtype, CNML_CONST);
    Dtype* std_ptr_ = std_.mutable_cpu_data();
    if (std) {
      for (int i = 0; i < bottom[index]->channels(); i++) {
        std_ptr_[i] = 1 / this->layer_param().convolution_param().std();
      }
    } else {
      for (int i = 0; i < bottom[index]->channels(); i++) {
        std_ptr_[i] = 1 / this->layer_param().convolution_param().scale();
      }
    }
  }
  if (std || scale)
    MLU_CHECK(cnmlBindConstData_V2(std_.mlu_tensor(),
                                   std_.sync_data(),
                                    false));

  MLU_CHECK(cnmlCreateConvFirstOp(&mlu_conv_op_ptrs_[index],
      mlu_convf_param_ptr_,
      bottom[index]->mlu_tensor(),
      this->mean_.mlu_tensor(),  // mean tensor
      top[index]->mlu_tensor(),
      this->blobs_[0]->mlu_tensor(),
      this->bias_term_ ? this->blobs_[1]->mlu_tensor() : nullptr,
      (std || scale) ? std_.mlu_tensor() : nullptr));

  SetBottomPositionScale(mlu_conv_op_ptrs_[index],
                         input_quant_params[index],
                         bottom[index], index);

  BaseDataType mlu_dtype = this->layer_param_.blobs_dtype_size() > 0 ?
                                    this->layer_param_.blobs_dtype(0).type() : DT_INT8;
  bindDataAndSetComputingDataType(this->blobs_[0], mlu_conv_op_ptrs_[index], mlu_dtype);
  if (this->bias_term_)
      MLU_CHECK(cnmlBindConstData_V2(this->blobs_[1]->mlu_tensor(),
                                     this->blobs_[1]->sync_data(),
                                     false));
  MLU_CHECK(cnmlBindConstData_V2(this->mean_.mlu_tensor(),
                                 this->mean_.sync_data(),
                                 false));
}

template <typename Dtype>
void MLUConvolutionLayer<Dtype>::Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < bottom.size(); i++) {
    if (this->conv_first_) {
      MLU_CHECK(cnmlComputeConvFirstOpForward_V3(mlu_conv_op_ptrs_[i],
          bottom[i]->mutable_mlu_data(),
          top[i]->mutable_mlu_data(),
          Caffe::forward_param(), Caffe::queue()));
      continue;
    }

    if (this->group_ == 1) {
      Dtype* data_ptr = bottom[i]->mutable_mlu_data();
      if (dilate() && add_pad_) {
        MLU_CHECK(cnmlComputeAddPadOpForward_V3(mlu_addpad_op_ptrs_[i],
            bottom[i]->mutable_mlu_data(),
            addpad_[i]->mutable_mlu_data(),
            Caffe::forward_param(), Caffe::queue()));
        data_ptr = addpad_[i]->mutable_mlu_data();
      }
      MLU_CHECK(cnmlComputeConvOpForward_V3(mlu_conv_op_ptrs_[i],
          data_ptr,
          top[i]->mutable_mlu_data(),
          Caffe::forward_param(), Caffe::queue()));
    } else {
      if (is_depthwise_) {
        Dtype* data_ptr = bottom[i]->mutable_mlu_data();
        if (add_pad_) {
          MLU_CHECK(cnmlComputeAddPadOpForward_V3(mlu_addpad_op_ptrs_[i],
              bottom[i]->mutable_mlu_data(),
              addpad_[i]->mutable_mlu_data(),
              Caffe::forward_param(), Caffe::queue()));
          data_ptr = addpad_[i]->mutable_mlu_data();
        }
        MLU_CHECK(cnmlComputeConvDepthwiseOpForward_V3(mlu_conv_op_ptrs_[i],
            data_ptr,
            top[i]->mutable_mlu_data(),
            Caffe::forward_param(), Caffe::queue()));
      } else {
        Dtype* data_ptr = bottom[i]->mutable_mlu_data();
        if (add_pad_ && dilate()) {
          MLU_CHECK(cnmlComputeAddPadOpForward_V3(mlu_addpad_op_ptrs_[i],
              bottom[i]->mutable_mlu_data(),
              addpad_[i]->mutable_mlu_data(),
              Caffe::forward_param(), Caffe::queue()));
          data_ptr = addpad_[i]->mutable_mlu_data();
        }
        MLU_CHECK(cnmlComputeConvGroupOpForward_V3(mlu_conv_op_ptrs_[i],
            data_ptr,
            top[i]->mutable_mlu_data(),
            Caffe::forward_param(), Caffe::queue()));
      }
    }
  }
}

template <typename Dtype>
void MLUConvolutionLayer<Dtype>::MLUCompileOp() {
  const int core_number = Caffe::core_number();  // avoid line wrap
  for (int i = 0; i < bottom_size_; i++) {
    if (!this->conv_first_ && add_pad_ && (is_depthwise_ || dilate())) {
      MLU_CHECK(cnmlCompileBaseOp(mlu_addpad_op_ptrs_[i], Caffe::rt_core(), core_number));
      MLU_CHECK(cnmlCompileBaseOp(mlu_conv_op_ptrs_[i], Caffe::rt_core(), core_number));
    } else {
      MLU_CHECK(cnmlCompileBaseOp(mlu_conv_op_ptrs_[i], Caffe::rt_core(), core_number));
    }
  }
}

template <typename Dtype>
void MLUConvolutionLayer<Dtype>::ArrangeWeightRGB() {
  int num_output = this->layer_param_.convolution_param().num_output();
  // Weight only has 3 channels, and input has 4 channels in pt
  Dtype* source_data = this->blobs_[0]->mutable_cpu_data();
  Dtype* data = this->blobs_[0]->mutable_cpu_data();
  source_data += (num_output * 3 - 3) *
      this->kernel_shape_.cpu_data()[0] *
      this->kernel_shape_.cpu_data()[1];
  data += (num_output * 4 - 4) *
      this->kernel_shape_.cpu_data()[0] *
      this->kernel_shape_.cpu_data()[1];
  int space_size = this->kernel_shape_.cpu_data()[0] *
                   this->kernel_shape_.cpu_data()[1];
  Dtype* tmp = new Dtype[space_size * 4];
  caffe_set(space_size * 4, Dtype(0), tmp);
  for (int i = 0; i < num_output; i++) {
    // BGR to RGB0 or 0RGB
    // tmp space to avoid the overlap between source_data and data
    if (this->layer_param_.convolution_param().input_format() ==
        ConvolutionParameter_InputFormat_ARGB) {
      caffe_copy(space_size, source_data + 0 * space_size, tmp + 3 * space_size);
      caffe_copy(space_size, source_data + 1 * space_size, tmp + 2 * space_size);
      caffe_copy(space_size, source_data + 2 * space_size, tmp + 1 * space_size);
    } else if (this->layer_param_.convolution_param().input_format() ==
        ConvolutionParameter_InputFormat_ABGR) {
      caffe_copy(space_size, source_data + 0 * space_size, tmp + 1 * space_size);
      caffe_copy(space_size, source_data + 1 * space_size, tmp + 2 * space_size);
      caffe_copy(space_size, source_data + 2 * space_size, tmp + 3 * space_size);
    } else {
      caffe_copy(space_size, source_data + 0 * space_size, tmp + 2 * space_size);
      caffe_copy(space_size, source_data + 1 * space_size, tmp + 1 * space_size);
      caffe_copy(space_size, source_data + 2 * space_size, tmp + 0 * space_size);
    }
    caffe_copy(4 * space_size, tmp, data);
    data -= 4 * space_size;
    source_data -= 3 * space_size;
  }
  delete [] tmp;
}

template <typename Dtype>
void MLUConvolutionLayer<Dtype>::MLUDestroyOp() {
  if (mlu_conv_op_ptrs_ != nullptr) {
    for (int i = 0; i < bottom_size_; i++)
      MLU_CHECK(cnmlDestroyBaseOp(&mlu_conv_op_ptrs_[i]));
    delete [] mlu_conv_op_ptrs_;
    mlu_conv_op_ptrs_ = nullptr;
  }

  if (mlu_addpad_op_ptrs_ != nullptr) {
    for (int i = 0; i < bottom_size_; i++) {
      MLU_CHECK(cnmlDestroyBaseOp(&mlu_addpad_op_ptrs_[i]));
    }
    delete [] mlu_addpad_op_ptrs_;
    mlu_addpad_op_ptrs_ = nullptr;
  }

  if (mlu_conv_param_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyConvOpParam(&mlu_conv_param_ptr_));
    mlu_conv_param_ptr_ = nullptr;
  }

  if (mlu_addpad_param_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyAddPadOpParam(&mlu_addpad_param_ptr_));
    mlu_addpad_param_ptr_ = nullptr;
  }

  if (mlu_convf_param_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyConvFirstOpParam(&mlu_convf_param_ptr_));
    mlu_convf_param_ptr_ = nullptr;
  }

  if (depthwise_param_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyConvDepthwiseOpParam(&depthwise_param_ptr_));
    depthwise_param_ptr_ = nullptr;
  }
  for (int i = 0; i < bottom_size_; i++) {
    if (input_quant_params[i] != nullptr) {
      MLU_CHECK(cnmlDestroyQuantizedParam(&input_quant_params[i]));
      input_quant_params[i] = nullptr;
    }
  }
  if (mean_quant_param != nullptr) {
    MLU_CHECK(cnmlDestroyQuantizedParam(&mean_quant_param));
    mean_quant_param = nullptr;
  }

  if (!addpad_.empty()) {
    for (int i = 0; i < addpad_.size(); i++)
      delete addpad_[i];
    addpad_.clear();
  }
}

template <typename Dtype>
MLUConvolutionLayer<Dtype>::~MLUConvolutionLayer() {
  MLUDestroyOp();
}

INSTANTIATE_CLASS(MLUConvolutionLayer);

}  // namespace caffe
#endif  // USE_MLU
