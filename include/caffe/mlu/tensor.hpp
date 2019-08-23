/*
All modification made by Cambricon Corporation: © 2018--2019 Cambricon Corporation
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

#ifndef INCLUDE_CAFFE_MLU_TENSOR_HPP_
#define INCLUDE_CAFFE_MLU_TENSOR_HPP_
#ifdef USE_MLU

#include <vector>

#include "caffe/common.hpp"

namespace caffe {

class MLUTensorDesc {
  public:
  MLUTensorDesc()
      : cpu_tensor_(nullptr),
        mlu_tensor_(nullptr),
        tensor_type_(CNML_TENSOR),
        cpu_dtype_(DT_INVALID),
        mlu_dtype_(DT_INVALID),
        position_(0),
        has_position_(false),
        scale_(1),
        has_scale_(false),
        cpu_tensor_order_(CNML_NCHW) {}

  void remember(const vector<int>& shape, cnmlTensorType_t tensor_type,
                BaseDataType cpu_dtype, BaseDataType mlu_dtype,
                cnmlDataOrder_t shape_order);
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
  cnmlTensorType_t type() { return tensor_type_; }
  void cpuCreate();
  void mluCreate();
  const cnmlCpuTensor_t cpu() const;
  const cnmlTensor_t mlu() const;
  void setCpuTensorOrder(cnmlDataOrder_t order) { cpu_tensor_order_ = order; }
  ~MLUTensorDesc();

  private:
  cnmlCpuTensor_t cpu_tensor_;
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
  void cpuDestory();
  void mluDestory();
  vector<int> shapeWithoutParallel();
  vector<int> positions_;
  vector<float> scales_;
  cnmlDataOrder_t cpu_tensor_order_;

  DISABLE_COPY_AND_ASSIGN(MLUTensorDesc);
  DISABLE_NEW_AND_DELETE();
};

}  // namespace caffe

#endif  // USE_MLU
#endif  // INCLUDE_CAFFE_MLU_TENSOR_HPP_
