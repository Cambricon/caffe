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

#ifndef INCLUDE_CAFFE_LAYERS_MLU_CYCLEADD_LAYER_HPP_
#define INCLUDE_CAFFE_LAYERS_MLU_CYCLEADD_LAYER_HPP_
#ifdef USE_MLU
#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layers/cycleadd_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief MLU acceleration of CycleAddLayer
 *        CycleAddLayer computes bottom[0] + bottom[1] = top
 * the shape of bottom[0] is n * c * h * w
 * the shape of bottom[1] is 1 * c * 1 * 1
 * the shape of top is n * c * h * w
 * @param param provides LayerParameter. It has no parameters for now
 *
 */
template <typename Dtype>
class MLUCycleAddLayer : public CycleAddLayer<Dtype> {
  public:
  explicit MLUCycleAddLayer(const LayerParameter& param)
      : CycleAddLayer<Dtype>(param), cycleadd_op_ptr_(nullptr) {}
  virtual void Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                              const vector<Blob<Dtype>*>& top);
  virtual inline bool mfus_supported() { return true; }
  virtual void fuse(MFusion<Dtype>* fuser);
  virtual ~MLUCycleAddLayer() { MLUDestroyOp(); };

  protected:
  virtual void MLUDestroyOp();
  /**
   * @brief Create CycleAdd Op
   *
   */
  virtual void MLUCreateOpBindData(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top);
  /**
   * @brief Compile the Op to instructions
   *
   */
  virtual void MLUCompileOp() {
    MLU_CHECK(cnmlCompileBaseOp(cycleadd_op_ptr_,
                                Caffe::rt_core(),
                                Caffe::core_number()));
  }
  /**
   * @param bottom input Blob vector
   *  bottom[0]'s shape is N*C*H*W
   *  bottom[1]'s shape is 1*C*1*1
   * @param top output Blob vector
   *  top's shape is N*C*H*W
   *  result is saved to top[0]
   */
  virtual void Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  cnmlBaseOp_t cycleadd_op_ptr_;
};

}  // namespace caffe
#endif  // USE_MLU
#endif  // INCLUDE_CAFFE_LAYERS_MLU_CYCLEADD_LAYER_HPP_
