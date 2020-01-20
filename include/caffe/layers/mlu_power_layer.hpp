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

#ifndef INCLUDE_CAFFE_LAYERS_MLU_POWER_LAYER_HPP_
#define INCLUDE_CAFFE_LAYERS_MLU_POWER_LAYER_HPP_
#ifdef USE_MLU

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layers/power_layer.hpp"
#include "caffe/proto/caffe.pb.h"
namespace caffe {

/**
 * @brief Computes @f$ y = (\alpha x + \beta) ^ \gamma @f$,
 *        as specified by the scale @f$ \alpha @f$, shift @f$ \beta @f$ and
 *        power @f$ \gamma @f$.
 */
template <typename Dtype>
class MLUPowerLayer:public PowerLayer<Dtype>{
  public:
  /**
   * @param param provides PowerParameter power_param,
   *        with PowerLayer options:
   *      - scale (\b optional, default 1.0) the scale @f$ \alpha @f$
   *      - shift (\b optional, default 0.0) the shift @f$ \beta @f$
   *      - power (\b optional, default 1.0) the power @f$ \gamma @f$
   */
  explicit MLUPowerLayer(const LayerParameter& param)
    :PowerLayer<Dtype>(param), power_op_ptr_(nullptr),
     scale_op_ptr_(nullptr) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                              const vector<Blob<Dtype>*>& top);
  virtual inline bool mfus_supported() { return true; }
  virtual void fuse(MFusion<Dtype>* fuser);
  virtual ~MLUPowerLayer();

  protected:
  virtual void MLUDestroyOp();
  virtual void MLUCreateOpBindData(const vector<Blob<Dtype>*>& bottom,
                                  const vector<Blob<Dtype>*>& top);
  virtual void MLUCompileOp();
  virtual void Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
  cnmlBaseOp_t power_op_ptr_;
  cnmlBaseOp_t scale_op_ptr_;
  Blob<Dtype>* alpha_;  // alpha * x + beta
  Blob<Dtype>* beta_;
  Blob<Dtype>* temp_;
};
}    // namespace caffe
#endif   //  USE_MLU
#endif   //  INCLUDE_CAFFE_LAYERS_MLU_POWER_LAYER_HPP_
