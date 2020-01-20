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

#ifndef INCLUDE_CAFFE_LAYERS_MLU_DROPOUT_LAYER_HPP_
#define INCLUDE_CAFFE_LAYERS_MLU_DROPOUT_LAYER_HPP_
#ifdef USE_MLU
#include <vector>
#include "caffe/layers/dropout_layer.hpp"

namespace caffe {

/**
 * @brief MLU acceleration of DropoutLayer
 *
 * @param bottom input Blob vector (length 1)
 *   -# @f$ (N \times C \times H \times W) @f$
 *      the inputs @f$ x @f$
 * @param top output Blob vector (length 1)
 *   -# @f$ (N \times C \times H \times W) @f$
 *      the computed outputs @f$ y = |x| @f$
 */

template <typename Dtype>
class MLUDropoutLayer : public DropoutLayer<Dtype> {
  public:
    explicit MLUDropoutLayer(const LayerParameter& param)
        : DropoutLayer<Dtype>(param), mlu_dropout_op_ptr_(NULL) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
    virtual void Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);
    virtual ~MLUDropoutLayer();
    virtual inline bool mfus_supported() { return true; }
    virtual void fuse(MFusion<Dtype>* fuser);

  protected:
    /**
     * @brief destroy the Ops have been setup
     *
     */
    virtual void MLUDestroyOp();
    /**
     * @brief Create Dropout Op
     *
     */
    virtual void MLUCreateOpBindData(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top);
    /**
     * @brief Complile the Op to instructions
     *
     */
    virtual void MLUCompileOp();
    /**
     * @param bottom input Blob vector (length 1)
     *   -# @f$ (N \times C \times H \times W) @f$
     *      the inputs @f$ x @f$
     * @param top output Blob vector (length 1)
     *   -# @f$ (N \times C \times H \times W) @f$
     *      the computed outputs. At training time, we have @f$
     *      y_{\mbox{train}} = \left\{
     *         \begin{array}{ll}
     *            \frac{x}{1 - p} & \mbox{if } u > p \\
     *            0 & \mbox{otherwise}
     *         \end{array} \right.
     *      @f$, where @f$ u \sim U(0, 1)@f$ is generated independently for each
     *      input at each iteration. At test time, we simply have
     *      @f$ y_{\mbox{test}} = \mathbb{E}[y_{\mbox{train}}] = x @f$.
     */
    virtual void Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    cnmlBaseOp_t mlu_dropout_op_ptr_;
    Blob<Dtype> zero_bias_data_;
    Blob<Dtype> weight;
};
}  // namespace caffe

#endif  // USE_MLU
#endif  // INCLUDE_CAFFE_LAYERS_MLU_DROPOUT_LAYER_HPP_
