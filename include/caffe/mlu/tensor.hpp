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

#ifndef INCLUDE_CAFFE_MLU_TENSOR_HPP_
#define INCLUDE_CAFFE_MLU_TENSOR_HPP_
#ifdef USE_MLU

#include <vector>

#include "caffe/common.hpp"

namespace caffe {

class MLUTensorDesc {
  public:
  MLUTensorDesc()
      : mlu_tensor_(nullptr),
        tensor_type_(CNML_TENSOR),
        cpu_dtype_(DT_FLOAT32),
        mlu_dtype_(DT_FLOAT16),
        position_(0),
        has_position_(false),
        scale_(1),
        has_scale_(false),
        data_num_(0),
        dim_(4),
        has_dim_strides_(false),
        is_first_conv_input_tensor_(false),
        preprocess_(true) {}

  void remember(const vector<int>& shape, cnmlTensorType_t tensor_type,
                BaseDataType cpu_dtype, BaseDataType mlu_dtype,
                cnmlDataOrder_t shape_order,
                vector<int>* dim_strides = nullptr);
  void set_position(int position);
  void set_scale(float scale);
  void set_positions(const vector<int>& position);
  void set_scales(const vector<float>& scale);
  bool has_position() const { return has_position_; }
  bool has_scale() const { return has_scale_; }
  const int position() const { return position_; }
  const float scale() const { return scale_; }
  const vector<int>& positions() const { return positions_; }
  const vector<float>& scales() const { return scales_; }
  void set_cpu_type(BaseDataType cpu_dtype) { cpu_dtype_ = cpu_dtype; }
  const BaseDataType cpu_type() const { return cpu_dtype_; }
  void set_mlu_type(BaseDataType mlu_dtype) { mlu_dtype_ = mlu_dtype; }
  const BaseDataType mlu_type() const { return mlu_dtype_; }
  cnmlTensorType_t type() const { return tensor_type_; }
  const size_t data_num() const { return data_num_; }
  void Create();
  const cnmlTensor_t mlu() const;
  vector<int> cpu_shape() const;
  vector<int> mlu_shape() const { return shape_; }
  const int shape_dim() const { return dim_; }
  vector<int> dim_strides() const { return dim_strides_; }
  bool has_dim_strides() const { return has_dim_strides_; }
  void set_dim_strides(vector<int> dim_strides);
  bool is_first_conv_input_tensor() const { return is_first_conv_input_tensor_; }
  void set_preprocess(bool preprocess) { preprocess_ = preprocess; }
  bool is_preprocess() const {return preprocess_; }
  ~MLUTensorDesc();

  private:
  cnmlTensor_t mlu_tensor_;

  vector<int> shape_;
  cnmlTensorType_t tensor_type_;
  BaseDataType cpu_dtype_;
  BaseDataType mlu_dtype_;
  cnmlDataOrder_t data_order_;
  int position_;
  bool has_position_;
  float scale_;
  bool has_scale_;
  void Destroy();
  vector<int> positions_;
  vector<float> scales_;
  size_t data_num_;
  int dim_;
  vector<int> dim_strides_;
  bool has_dim_strides_;
  bool is_first_conv_input_tensor_;
  bool preprocess_;

  DISABLE_COPY_AND_ASSIGN(MLUTensorDesc);
  DISABLE_NEW_AND_DELETE();
};

}  // namespace caffe

#endif  // USE_MLU
#endif  // INCLUDE_CAFFE_MLU_TENSOR_HPP_
