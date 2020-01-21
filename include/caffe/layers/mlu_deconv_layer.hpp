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

#ifndef INCLUDE_CAFFE_LAYERS_MLU_DECONV_LAYER_HPP_
#define INCLUDE_CAFFE_LAYERS_MLU_DECONV_LAYER_HPP_
#ifdef USE_MLU
#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/deconv_layer.hpp"

namespace caffe {

/**
 * @brief Convolve the input with a bank of learned filters, and (optionally)
 *        add biases, treating filters and convolution parameters in the
 *        opposite sense as ConvolutionLayer.
 *
 *   ConvolutionLayer computes each output value by dotting an input window with
 *   a filter; DeconvolutionLayer multiplies each input value by a filter
 *   elementwise, and sums over the resulting output windows. In other words,
 *   DeconvolutionLayer is ConvolutionLayer with the forward and backward passes
 *   reversed. DeconvolutionLayer reuses ConvolutionParameter for its
 *   parameters, but they take the opposite sense as in ConvolutionLayer (so
 *   padding is removed from the output rather than added to the input, and
 *   stride results in upsampling rather than downsampling).
 */
template <typename Dtype>
class MLUDeconvolutionLayer : public DeconvolutionLayer<Dtype> {
  public:
  explicit MLUDeconvolutionLayer(const LayerParameter& param)
      : DeconvolutionLayer<Dtype>(param),
        deconv_param_(nullptr),
        crop_param_(nullptr),
        group_param_(nullptr) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual ~MLUDeconvolutionLayer() { MLUDestroyOp(); }
  virtual inline bool mfus_supported() { return true; }
  virtual void fuse(MFusion<Dtype>* fuser);
  virtual void set_optimization_level(int level) {
    optimization_level_ = level;
  }

  virtual void Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                              const vector<Blob<Dtype>*>& top);

  protected:
  virtual void Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void MLUDestroyOp();
  virtual void MLUCreateOpBindData(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top);
  virtual void MLUCompileOp();

  /* when group > deconv_limit (71)
   * concat need: extra_concat_op_a,extra_concat_op_b,concat_output_op
   *              extra_concat_param_a,extra_concat_param_b
   * split need: extra_split_op_a,extra_split_op_b,split_input_op
   *             extra_split_param_a,extra_split_param_b
   * how it work:
   *                  split_input_op
   *                 /            \
   *    extra_split_op_a      extra_split_op_b(if remainder != 0)
   *                 \            /
   *                   deconv_op
   *                  /       \
   *    extra_concat_op_a     extra_concat_op_b(if remainder != 0)
   *                  \       /
   *                  concat_output_op
   */

  vector<cnmlBaseOp_t> deconv_op_;
  vector<cnmlBaseOp_t> crop_op_;
  vector<cnmlBaseOp_t> split_input_op_;
  vector<cnmlBaseOp_t> concat_output_op_;

  vector<cnmlBaseOp_t> extra_concat_op_a_;
  vector<cnmlBaseOp_t> extra_concat_op_b_;
  vector<cnmlBaseOp_t> extra_split_op_a_;
  vector<cnmlBaseOp_t> extra_split_op_b_;

  cnmlDeconvOpParam_t deconv_param_;
  cnmlGrepOpParam_t crop_param_;
  cnmlDeconvOpParam_t group_param_;

  ///  @brief: container for output of split op
  vector<shared_ptr<Blob<Dtype>>> input_blobs_;
  ///  @brief: container for output of deconv op
  vector<shared_ptr<Blob<Dtype>>> output_blobs_;
  ///  @brief: container for grouped weight
  vector<shared_ptr<Blob<Dtype>>> weight_blobs_;
  ///  @brief: container for grouped bias
  vector<shared_ptr<Blob<Dtype>>> bias_blobs_;
  ///  @brief: contianer for extra concat blobs
  vector<shared_ptr<Blob<Dtype>>> extra_concat_blobs_;
  vector<shared_ptr<Blob<Dtype>>> extra_split_blobs_;
  // deconv's limit on max input size of concat op
  const int deconv_limit_ = 2048;
  // whether there should be extra concat op and split op
  bool extra_op_ = false;
  bool group_op_ = false;
  // two variables below are only used when there
  // is extra concat
  // quotient = group / concat_limit
  int quotient_ = 0;
  // remainder = group % concat_limit
  int remainder_ = 0;
  // offset = quotient + (remainder ? 1:0)
  int offset_ = 0;
  int optimization_level_ = 0;
};
}  // namespace caffe

#endif  // USE_MLU
#endif  // INCLUDE_CAFFE_LAYERS_MLU_DECONV_LAYER_HPP_
