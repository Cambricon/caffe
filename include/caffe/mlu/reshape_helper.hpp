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

#ifndef INCLUDE_CAFFE_MLU_RESHAPE_HELPER_HPP_
#define INCLUDE_CAFFE_MLU_RESHAPE_HELPER_HPP_
#ifdef USE_MLU

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/mlu/netdata.hpp"

namespace caffe {

/**
 * @brief ReshapeHelper reduces Reshape operation which is heavy on MLU.
 */
template <typename Dtype>
class ReshapeHelper {
  public:
  explicit ReshapeHelper(shared_ptr<NetData<Dtype> > net)
      : init_caffe_mode_(Caffe::mode()),
        init_reshape_mode_(Caffe::reshapeMode()),
        already_reshaped_(false) {
  }
  bool needReshape();

  private:
  Caffe::Brew init_caffe_mode_;
  Caffe::ReshapeMode init_reshape_mode_;

  bool already_reshaped_;  // for SETUPONLY mode.


  /**
   * For reshape and subnet, we don't support change mode after caffe init
   * since global status could be inconsistent for such scenarios.
   * XXX: put this at the begining of all non-constructor interfaces.
   */
  inline void modeCheck() {
    CHECK(init_caffe_mode_ == Caffe::mode())
        << "Caffe mode has changed since init!";
    CHECK(init_reshape_mode_ == Caffe::reshapeMode())
        << "Reshape mode has changed since init!";
  }
};  // class ReshapeHelper

template <typename Dtype>
bool ReshapeHelper<Dtype>::needReshape() {
  modeCheck();
  if (already_reshaped_) {
    return false;
  } else {
    already_reshaped_ = true;
    return true;
  }
}
}  // namespace caffe

#endif  // USE_MLU
#endif  // INCLUDE_CAFFE_MLU_RESHAPE_HELPER_HPP_
