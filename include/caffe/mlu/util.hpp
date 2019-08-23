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

#ifndef INCLUDE_CAFFE_MLU_UTIL_HPP_
#define INCLUDE_CAFFE_MLU_UTIL_HPP_
#ifdef USE_MLU

#include <cmath>
#include <vector>
#include "caffe/common.hpp"
#include "caffe/blob.hpp"
#include "cnrt.h"  // NOLINT

#include "glog/logging.h"

namespace caffe {

template <typename Dtype>
void uniquePushBack(std::vector<Dtype>* vec, const Dtype& item) {
  for ( auto &e : *vec ) {
    if (e == item) {
      return;
    }
  }
  vec->push_back(item);
}

template <typename Dtype>
void rmOneFromVector(std::vector<Dtype>* vec, const Dtype& item) {
  for (auto it = vec->begin(); it != vec->end();) {
    if (*it == item) {
      vec->erase(it);
    } else {
      it++;
    }
  }
}

template<typename Dtype>
void sparseFilter(const std::vector<int> shape, const Dtype* fromData,
    std::vector<Dtype>* toData, float sparsity) {
  if (sparsity < 0 || sparsity >= 1)
    LOG(FATAL) << "Illegal sparsity value";

  if (toData->size() == 1) {
    (*toData)[0] = fromData[0];
    return;
  }

  std::vector<Dtype> tempData(toData->size() / 2, 0);
  int stride = 1;
  if (shape.size() > 2) {
    for (int i = 2; i < shape.size(); i++) {
      stride *= shape[i];
    }
  }

  for (int i = 0; i < toData->size() / 2 / stride; i++) {
    for (int j = 0; j < stride; j++) {
      tempData[i * stride + j] = std::max(fabs(fromData[i * 2 * stride + j]),
          fabs(fromData[(i * 2 + 1) * stride + j]));
    }
  }

  sort(tempData.begin(), tempData.end());

  size_t location = size_t(sparsity * tempData.size());
  float threshold = tempData[location];
  for (size_t i = 0; i < toData->size() / 2 / stride; i++) {
    for (int j = 0; j < stride; j++) {
      if (fabs(fromData[i * 2 * stride + j]) < threshold &&
          fabs(fromData[(i * 2 + 1) * stride + j]) < threshold) {
        (*toData)[i * 2 * stride + j] = 0;
        (*toData)[(i * 2 + 1) * stride + j] = 0;
      } else {
        (*toData)[i * 2 * stride + j] = fromData[i * 2 * stride + j];
        (*toData)[(i * 2 + 1) * stride + j] = fromData[(i * 2 + 1) * stride + j];
      }
    }
  }
}

/**
 *  @brief Get a BlobDataType item containing int8 quantization information
 *
 *  @param blob blob that contains the data from which to calculate
 *  position and scale
 *  e.g for convolution layer, both its bottom or weights can be quantified,
 *  u have to specify which one u'd like to calculate on.
 *
 *  @param layer_param the parameter of the layer you are calculating,
 *  this is necessary because some information about the layer is needed,
 *  e.g. for multi-batch convolution quantization, we need to know its kernel
 *  height and width to get the step(offset) for each batch
 *
 *  @param mode "common": calculate position and scale
 *              "int8_channel": channel quantization, scale only
 *              "scale": scale only
 *
 *  @param max_value this is used for multi-batch images in generate_int8_pt,
 *  leave it for the default nullptr in most cases
 */
template <typename Dtype>
BlobDataType get_int8_info(const Blob<Dtype>& blob,
                 const LayerParameter& layer_param,
                 // this is for multi-batch images in generate_int8_pt
                 // for bottoms only
                 map<string, Dtype>* const max_value = nullptr,
                 // absmax should be squared for normalize bottom 0
                 // not normalize bottom 1
                 bool is_first_normalize = false) {
  vector<Dtype> max(1, 0), min(1, 0), abs_max(1, 0), position(1, 0);
  int channel = blob.channels();
  int length = blob.count();
  string layer_type = layer_param.type();
  bool lrn = layer_type == "LRN";
  bool conv = layer_type == "Convolution";
  bool mlp = layer_type == "InnerProduct";
  string key = layer_param.name();
  const Dtype* data = blob.cpu_data();
  min[0] = max[0] = data[0];
  for (int j = 0; j < length; ++j) {
    max[0] = std::max(data[j], max[0]);
    min[0] = std::min(data[j], min[0]);
  }
  abs_max[0] = std::max(std::abs(min[0]), std::abs(max[0]));
  if (max_value != nullptr) {
    for (int i = 0; i < abs_max.size(); i++) {
      auto iter = max_value->find(key);
      if (iter != max_value->end()) {
        if (abs_max[i] > iter->second) {
          (*max_value)[key] = abs_max[i];
        } else {
          abs_max[i] = (*max_value)[key];
        }
      } else {
        max_value->insert(pair<string, Dtype>(key, abs_max[i]));
      }
    }
  }
  if (lrn) {
    abs_max[0] = abs_max[0] * abs_max[0] * layer_param.lrn_param().alpha();
  } else if (is_first_normalize) {
    abs_max[0] *= abs_max[0];
  }
  BlobDataType blob_dtype;
  for (int i = 0; i < abs_max.size(); i++) {
    if (abs_max[i] == 0) {
      position[i] = 0;
    } else {
      position[i] = log2(abs_max[i] / 127);
      position[i] += position[i] > 0 ? 1 : 0;
    }
    if (position[i] > 32) position[i] = 32;
    if (position[i] < -32) position[i] = -32;
    blob_dtype.set_type(DT_INT8);
    blob_dtype.add_position(position[i]);
  }
  return blob_dtype;
}

}  // namespace caffe

#endif  // USE_MLU
#endif  // INCLUDE_CAFFE_MLU_UTIL_HPP_
