
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

#ifndef INCLUDE_CAFFE_LAYERS_MLU_DETECTION_OUT_LAYER_HPP_
#define INCLUDE_CAFFE_LAYERS_MLU_DETECTION_OUT_LAYER_HPP_

#ifdef USE_MLU

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#include <boost/property_tree/json_parser.hpp>
#pragma GCC diagnostic pop
#include <boost/property_tree/ptree.hpp>
#include <boost/regex.hpp>

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/detection_out_layer.hpp"


namespace caffe {

/**
 * @brief MLU acceleration of DetectionOutLayer
 *        Compute elementwise operations, such as product and sum,
 *        along multiple input Blobs.
 */

template <typename Dtype>
class MLUDetectionOutLayer : public DetectionOutLayer<Dtype> {
  public:
  explicit MLUDetectionOutLayer(const LayerParameter& param)
      : DetectionOutLayer<Dtype>(param), detection_out_op_ptr_(nullptr),
        bbox_transpose_param_(nullptr), bbox_transpose_ptr_(nullptr),
        shufflechannel_op_ptr_(nullptr), yolov2_ptr_param_(nullptr) {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                               const vector<Blob<Dtype>*>& top);
  virtual void fuse(MFusion<Dtype>* fuser);
  virtual inline bool mfus_supported() { return true; }
  virtual ~MLUDetectionOutLayer() { MLUDestroyOp(); }

  protected:
  virtual void MLUDestroyOp();  // Destroy the Ops have been setup
  virtual void MLUCreateOpBindData(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top);
  virtual void MLUCompileOp();  // Compile the Op to instruction
  virtual void Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);

  cnmlBaseOp_t detection_out_op_ptr_;
  cnmlNdTransposeOpParam_t bbox_transpose_param_;
  cnmlBaseOp_t bbox_transpose_ptr_;
  cnmlBaseOp_t shufflechannel_op_ptr_;
  cnmlPluginYolov2DetectionOutputOpParam_t yolov2_ptr_param_;

  Blob<Dtype> bbox_transpose_blob;
  Blob<Dtype> bbox_shufflechannel_blob;
  Blob<Dtype> biases_blob;
  Blob<Dtype> temp_buffer_blob;

  vector<int> bbox_transpose_shape;
  vector<int> bbox_shufflechannel_shape;

  vector<float> biases;
};
}  // namespace caffe

#endif  // USE_MLU
#endif  // INCLUDE_CAFFE_LAYERS_MLU_DETECTION_OUT_LAYER_HPP_
