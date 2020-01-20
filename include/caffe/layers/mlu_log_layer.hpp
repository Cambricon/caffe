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

#ifndef INCLUDE_CAFFE_LAYERS_MLU_LOG_LAYER_HPP_
#define INCLUDE_CAFFE_LAYERS_MLU_LOG_LAYER_HPP_
#ifdef USE_MLU

#include <vector>
#include "caffe/layers/log_layer.hpp"

namespace caffe {

/**
 * @brief Computes @f$ y = log_{\gamma}(\alpha x + \beta) @f$,
 *        as specified by the scale @f$ \alpha @f$, shift @f$ \beta @f$,
 *        and base @f$ \gamma @f$.
 */
template <typename Dtype>
class MLULogLayer : public LogLayer<Dtype> {
  public:
  /**
   * @param param provides LogParameter log_param,
   *     with LogLayer options:
   *   - scale (\b optional, default 1) the scale @f$ \alpha @f$
   *   - shift (\b optional, default 0) the shift @f$ \beta @f$
   *   - base (\b optional, default -1 for a value of @f$ e \approx 2.718 @f$)
   *         the base @f$ \gamma @f$
   */
  explicit MLULogLayer(const LayerParameter& param)
      : LogLayer<Dtype>(param), mlu_log_op_ptr_(nullptr) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                              const vector<Blob<Dtype>*>& top);
  virtual ~MLULogLayer();
  virtual inline bool mfus_supported() { return true; }
  virtual void fuse(MFusion<Dtype>* fuser) { fuser->fuse(mlu_log_op_ptr_); }

  protected:
  virtual void MLUDestroyOp();
  virtual void MLUCreateOpBindData(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top);
  virtual void MLUCompileOp() {
    MLU_CHECK(cnmlCompileBaseOp(mlu_log_op_ptr_,
                                Caffe::rt_core(),
                                Caffe::core_number()));
  }
  virtual void Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  cnmlBaseOp_t mlu_log_op_ptr_;
};

}  // namespace caffe
#endif  // USE_MLU
#endif  // INCLUDE_CAFFE_LAYERS_MLU_LOG_LAYER_HPP_
