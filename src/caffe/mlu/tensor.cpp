/*
All modification made by Cambricon Corporation: © 2018 Cambricon Corporation
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

#ifdef USE_MLU
#include <glog/logging.h>
#include <vector>
#include "caffe/common.hpp"
#include "caffe/mlu/tensor.hpp"

namespace caffe {

static inline cnmlDataType_t to_mlu_dtype(BaseDataType type) {
  switch (type) {
  case DT_FLOAT16:
    return CNML_DATA_FLOAT16;
  case DT_FLOAT32:
    return CNML_DATA_FLOAT32;
  case DT_DOUBLE:
    return CNML_DATA_DOUBLE;
  case DT_INT8:
    return CNML_DATA_INT8;
  case DT_UINT8:
    return CNML_DATA_UINT8;
  case DT_INT16:
    return CNML_DATA_INT16;
  case DT_INT32:
    return CNML_DATA_INT32;
  case DT_QUANT8:
    return CNML_DATA_QUANT8;
  case DT_BINARY:
    return CNML_DATA_BINARY;
  default:
    return CNML_DATA_INVALID;
  }
}

vector<int> MLUTensorDesc::shapeWithoutParallel() {
  vector<int> s(shape_);
  // With the reshape op in scale layer, parallel can be
  // multiplied to either n or c
  // parallel can be multiplied to n, c, h, w in ReshapeLayer
  if (tensor_type_ == CNML_TENSOR) {
    if (s[0] % Caffe::data_parallel() == 0) {
      s[0] /= Caffe::data_parallel();
    } else if (s[1] % Caffe::data_parallel() == 0) {
      s[1] /= Caffe::data_parallel();
    } else if (s[2] % Caffe::data_parallel() == 0) {
      s[2] /= Caffe::data_parallel();
    } else if (s[3] % Caffe::data_parallel() == 0) {
      s[3] /= Caffe::data_parallel();
    }
  }
  return s;
}
void MLUTensorDesc::remember(const vector<int>& shape,
    cnmlTensorType_t tensor_type,
    BaseDataType cpu_dtype, BaseDataType mlu_dtype,
    cnmlDataOrder_t shape_order) {
  tensor_type_ = tensor_type;
  cpu_dtype_ = cpu_dtype;
  mlu_dtype_ = mlu_dtype;
  shape_.resize(4);
  switch (shape_order) {
    case CNML_NCHW:
      shape_[0] = shape.size() > 0 ? shape[0] : 1;
      shape_[1] = shape.size() > 1 ? shape[1] : 1;
      shape_[2] = shape.size() > 2 ? shape[2] : 1;
      shape_[3] = shape.size() > 3 ? shape[3] : 1;
    break;
    case CNML_NCWH:
      shape_[0] = shape.size() > 0 ? shape[0] : 1;
      shape_[1] = shape.size() > 1 ? shape[1] : 1;
      shape_[2] = shape.size() > 3 ? shape[3] : 1;
      shape_[3] = shape.size() > 2 ? shape[2] : 1;
    break;
    case CNML_NHWC:
      shape_[0] = shape.size() > 0 ? shape[0] : 1;
      shape_[1] = shape.size() > 3 ? shape[3] : 1;
      shape_[2] = shape.size() > 1 ? shape[1] : 1;
      shape_[3] = shape.size() > 2 ? shape[2] : 1;
    break;
    case CNML_NHCW:
      shape_[0] = shape.size() > 0 ? shape[0] : 1;
      shape_[1] = shape.size() > 2 ? shape[2] : 1;
      shape_[2] = shape.size() > 1 ? shape[1] : 1;
      shape_[3] = shape.size() > 3 ? shape[3] : 1;
    break;
    case CNML_NWCH:
      shape_[0] = shape.size() > 0 ? shape[0] : 1;
      shape_[1] = shape.size() > 2 ? shape[2] : 1;
      shape_[2] = shape.size() > 3 ? shape[3] : 1;
      shape_[3] = shape.size() > 1 ? shape[1] : 1;
    break;
    case CNML_NWHC:
      shape_[0] = shape.size() > 0 ? shape[0] : 1;
      shape_[1] = shape.size() > 3 ? shape[3] : 1;
      shape_[2] = shape.size() > 2 ? shape[2] : 1;
      shape_[3] = shape.size() > 1 ? shape[1] : 1;
    break;
    case CNML_CNHW:
      shape_[0] = shape.size() > 1 ? shape[1] : 1;
      shape_[1] = shape.size() > 0 ? shape[0] : 1;
      shape_[2] = shape.size() > 2 ? shape[2] : 1;
      shape_[3] = shape.size() > 3 ? shape[3] : 1;
    break;
    case CNML_CNWH:
      shape_[0] = shape.size() > 1 ? shape[1] : 1;
      shape_[1] = shape.size() > 0 ? shape[0] : 1;
      shape_[2] = shape.size() > 3 ? shape[3] : 1;
      shape_[3] = shape.size() > 2 ? shape[2] : 1;
    break;
    case CNML_CHWN:
      shape_[0] = shape.size() > 3 ? shape[3] : 1;
      shape_[1] = shape.size() > 0 ? shape[0] : 1;
      shape_[2] = shape.size() > 1 ? shape[1] : 1;
      shape_[3] = shape.size() > 2 ? shape[2] : 1;
    break;
    case CNML_CHNW:
      shape_[0] = shape.size() > 2 ? shape[2] : 1;
      shape_[1] = shape.size() > 0 ? shape[0] : 1;
      shape_[2] = shape.size() > 1 ? shape[1] : 1;
      shape_[3] = shape.size() > 3 ? shape[3] : 1;
    break;
    case CNML_CWNH:
      shape_[0] = shape.size() > 2 ? shape[2] : 1;
      shape_[1] = shape.size() > 0 ? shape[0] : 1;
      shape_[2] = shape.size() > 3 ? shape[3] : 1;
      shape_[3] = shape.size() > 1 ? shape[1] : 1;
    break;
    case CNML_CWHN:
      shape_[0] = shape.size() > 3 ? shape[3] : 1;
      shape_[1] = shape.size() > 0 ? shape[0] : 1;
      shape_[2] = shape.size() > 2 ? shape[2] : 1;
      shape_[3] = shape.size() > 1 ? shape[1] : 1;
    break;
    case CNML_HNCW:
      shape_[0] = shape.size() > 1 ? shape[1] : 1;
      shape_[1] = shape.size() > 2 ? shape[2] : 1;
      shape_[2] = shape.size() > 0 ? shape[0] : 1;
      shape_[3] = shape.size() > 3 ? shape[3] : 1;
    break;
    case CNML_HNWC:
      shape_[0] = shape.size() > 1 ? shape[1] : 1;
      shape_[1] = shape.size() > 3 ? shape[3] : 1;
      shape_[2] = shape.size() > 0 ? shape[0] : 1;
      shape_[3] = shape.size() > 2 ? shape[2] : 1;
    break;
    case CNML_HCWN:
      shape_[0] = shape.size() > 3 ? shape[3] : 1;
      shape_[1] = shape.size() > 1 ? shape[1] : 1;
      shape_[2] = shape.size() > 0 ? shape[0] : 1;
      shape_[3] = shape.size() > 2 ? shape[2] : 1;
    break;
    case CNML_HCNW:
      shape_[0] = shape.size() > 2 ? shape[2] : 1;
      shape_[1] = shape.size() > 1 ? shape[1] : 1;
      shape_[2] = shape.size() > 0 ? shape[0] : 1;
      shape_[3] = shape.size() > 3 ? shape[3] : 1;
    break;
    case CNML_HWNC:
      shape_[0] = shape.size() > 2 ? shape[2] : 1;
      shape_[1] = shape.size() > 3 ? shape[3] : 1;
      shape_[2] = shape.size() > 0 ? shape[0] : 1;
      shape_[3] = shape.size() > 1 ? shape[1] : 1;
    break;
    case CNML_HWCN:
      shape_[0] = shape.size() > 3 ? shape[3] : 1;
      shape_[1] = shape.size() > 2 ? shape[2] : 1;
      shape_[2] = shape.size() > 0 ? shape[0] : 1;
      shape_[3] = shape.size() > 1 ? shape[1] : 1;
    break;
    case CNML_WNCH:
      shape_[0] = shape.size() > 1 ? shape[1] : 1;
      shape_[1] = shape.size() > 2 ? shape[2] : 1;
      shape_[2] = shape.size() > 3 ? shape[3] : 1;
      shape_[3] = shape.size() > 0 ? shape[0] : 1;
    break;
    case CNML_WNHC:
      shape_[0] = shape.size() > 1 ? shape[1] : 1;
      shape_[1] = shape.size() > 3 ? shape[3] : 1;
      shape_[2] = shape.size() > 2 ? shape[2] : 1;
      shape_[3] = shape.size() > 0 ? shape[0] : 1;
    break;
    case CNML_WCHN:
      shape_[0] = shape.size() > 3 ? shape[3] : 1;
      shape_[1] = shape.size() > 1 ? shape[1] : 1;
      shape_[2] = shape.size() > 2 ? shape[2] : 1;
      shape_[3] = shape.size() > 0 ? shape[0] : 1;
    break;
    case CNML_WCNH:
      shape_[0] = shape.size() > 2 ? shape[2] : 1;
      shape_[1] = shape.size() > 1 ? shape[1] : 1;
      shape_[2] = shape.size() > 3 ? shape[3] : 1;
      shape_[3] = shape.size() > 0 ? shape[0] : 1;
    break;
    case CNML_WHNC:
      shape_[0] = shape.size() > 2 ? shape[2] : 1;
      shape_[1] = shape.size() > 3 ? shape[3] : 1;
      shape_[2] = shape.size() > 1 ? shape[1] : 1;
      shape_[3] = shape.size() > 0 ? shape[0] : 1;
    break;
    case CNML_WHCN:
      shape_[0] = shape.size() > 3 ? shape[3] : 1;
      shape_[1] = shape.size() > 2 ? shape[2] : 1;
      shape_[2] = shape.size() > 1 ? shape[1] : 1;
      shape_[3] = shape.size() > 0 ? shape[0] : 1;
    break;
    default:
      LOG(FATAL) << "Unsupported mluDataOrder! " << int(shape_order);
    break;
  }
}

void MLUTensorDesc::cpuCreate() {
  if (cpu_tensor_ == nullptr) {
    auto shape = shapeWithoutParallel();
    // all cpu data order in Caffe set to NCHW
    // data type set to Dtype
    MLU_CHECK(cnmlCreateCpuTensor(&cpu_tensor_,
                                 tensor_type_,
                                 to_mlu_dtype(cpu_dtype_),
                                 cpu_tensor_order_,
                                 shape[0],
                                 shape[1],
                                 shape[2],
                                 shape[3]));
  }
}

void MLUTensorDesc::mluCreate() {
  if (mlu_tensor_ == nullptr) {
    auto shape = shapeWithoutParallel();
    MLU_CHECK(cnmlCreateTensor(&mlu_tensor_,
                              tensor_type_,
                              to_mlu_dtype(mlu_dtype_),
                              shape[0],
                              shape[1],
                              shape[2],
                              shape[3]));
  }
}

void MLUTensorDesc::cpuDestroy() {
  if (cpu_tensor_ != nullptr) {
    MLU_CHECK(cnmlDestroyCpuTensor(&cpu_tensor_));
    cpu_tensor_ = nullptr;
  }
}

void MLUTensorDesc::mluDestroy() {
  if (mlu_tensor_ != nullptr) {
    MLU_CHECK(cnmlDestroyTensor(&mlu_tensor_));
    mlu_tensor_ = nullptr;
  }
}

const cnmlCpuTensor_t MLUTensorDesc::cpu() const {
  return cpu_tensor_;
}

const cnmlTensor_t MLUTensorDesc::mlu() const {
  return mlu_tensor_;
}

void MLUTensorDesc::set_position(int position) {
  position_ = position;
  has_position_ = true;
  mluCreate();
  cnmlSetQuantizedPosition(mlu_tensor_, position_);
}

void MLUTensorDesc::set_scale(float scale) {
  scale_ = scale;
  has_scale_ = true;
  mluCreate();
  cnmlSetQuantizedScale(mlu_tensor_, scale_);
}

void MLUTensorDesc::set_positions(const vector<int>& positions) {
  positions_ = positions;
  has_position_ = true;
  mluCreate();
  cnmlSetQuantizedPositionByChannel(mlu_tensor_, &positions_[0], positions_.size());
}

void MLUTensorDesc::set_scales(const vector<float>& scales) {
  scales_ = scales;
  has_scale_ = true;
  mluCreate();
  cnmlSetQuantizedScaleByChannel(mlu_tensor_, &scales_[0], scales_.size());
}

MLUTensorDesc::~MLUTensorDesc() {
  cpuDestroy();
  mluDestroy();
}

}  // namespace caffe

#endif  // USE_MLU
