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

#include "caffe/mlu/data_trans.hpp"

namespace caffe {

vector<int> to_cpu_shape(const vector<int>& mlu_shape) {
  // shape: N(D)HWC --> NC(D)HW
  vector<int> cpu_shape(mlu_shape.size(), 1);
  int channel = mlu_shape[mlu_shape.size() - 1];
  for (int i = 2; i < mlu_shape.size(); i++) {
    cpu_shape[i] = mlu_shape[i - 1];
  }
  cpu_shape[0] = mlu_shape[0];
  cpu_shape[1] = channel;
  return cpu_shape;
}

vector<int> to_mlu_shape(const vector<int>& cpu_shape) {
  // shape : NC(D)HW --> N(D)HWC
  vector<int> mlu_shape(cpu_shape.size(), 1);
  int channel = cpu_shape[1];
  for (int i = 1; i < cpu_shape.size() - 1; i++) {
    mlu_shape[i] = cpu_shape[i + 1];
  }
  mlu_shape[0] = cpu_shape[0];
  mlu_shape[mlu_shape.size() - 1] = channel;
  return mlu_shape;
}

void transAndCast(void* src_ptr, cnrtDataType_t src_dtype,
    void* dst_ptr, cnrtDataType_t dst_dtype, void* tmp_ptr,
    const vector<int>& src_dim_values, bool is_first_conv,
    string trans_direction) {
  vector<int> src_dim_values_non_const = src_dim_values;
  vector<int> dst_dim_values;
  vector<int> dim_strides(src_dim_values.size(), 0);
  vector<int> dim_order(src_dim_values.size());
  vector<int> dim_order_tmp(src_dim_values.size());
  for (int i = 0; i < src_dim_values.size(); i++) {
    dim_order_tmp[i] = i;
  }

  if ("CPU2MLU" == trans_direction) {
    dst_dim_values = to_mlu_shape(src_dim_values);
    dim_order = to_mlu_shape(dim_order_tmp);
    if (is_first_conv) {
      dim_strides[dim_strides.size() - 1] = 1;
    }
  } else if ("MLU2CPU" == trans_direction) {
    dst_dim_values = to_cpu_shape(src_dim_values);
    dim_order = to_cpu_shape(dim_order_tmp);
  } else {
    LOG(FATAL) << "Unsupport trans direction.";
  }

  if (src_dtype != dst_dtype) {
    if (is_first_conv) {
      CNRT_CHECK(cnrtTransOrderAndCast(reinterpret_cast<void*>(src_ptr),
                                       src_dtype,
                                       reinterpret_cast<void*>(tmp_ptr),
                                       dst_dtype,
                                       nullptr,
                                       src_dim_values.size(),
                                       src_dim_values_non_const.data(),
                                       dim_order.data()));
      CNRT_CHECK(cnrtAddDataStride(reinterpret_cast<void*>(tmp_ptr),
                                   dst_dtype,
                                   reinterpret_cast<void*>(dst_ptr),
                                   src_dim_values.size(),
                                   dst_dim_values.data(),
                                   dim_strides.data()));
    } else {
      CNRT_CHECK(cnrtTransOrderAndCast(reinterpret_cast<void*>(src_ptr),
                                       src_dtype,
                                       reinterpret_cast<void*>(dst_ptr),
                                       dst_dtype,
                                       nullptr,
                                       src_dim_values.size(),
                                       src_dim_values_non_const.data(),
                                       dim_order.data()));
    }
  } else {
    CNRT_CHECK(cnrtTransDataOrder(reinterpret_cast<void*>(src_ptr),
                                  src_dtype,
                                  reinterpret_cast<void*>(dst_ptr),
                                  src_dim_values.size(),
                                  src_dim_values_non_const.data(),
                                  dim_order.data()));
  }
}

}  // namespace caffe

#endif  // USE_MLU
