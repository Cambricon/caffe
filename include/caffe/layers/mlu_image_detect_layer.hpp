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

#ifndef INCLUDE_CAFFE_LAYERS_MLU_IMAGE_DETECT_LAYER_HPP_
#define INCLUDE_CAFFE_LAYERS_MLU_IMAGE_DETECT_LAYER_HPP_
#ifdef USE_MLU

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layers/image_detect_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {


/**
 * @brief First get the priorboxies of input data.
 * Then generate the ssd_detection output based on location and confidence.
 *
 * NOTE: does not implement Backwards operation.
 */

/* bottoms format */
/* take box_size = 304 and class_num = 21 for example */

/* bottom[0] bbox (output of origin faster-rcnn output layer bbox)
 * dimensions: 304 84
 *
 * [item0_center_x, item0_center_y, item0_width, item0_height, item1_center_x ... 21 items * 4 positions per item]
 * [item0_center_x, item0_center_y, item0_width, item0_height, item1_center_x ... 21 items * 4 positions per item]
 * ... 304 batches
 *
 * caffe: reshape dimensions to 1 304 1 84
 *
 * cnml: transpose dimensions to nhwc 1 1 84 304
 *
 *       [item0_center_x, item0_center_x ... 304 batches]
 *       [item0_center_y, item0_center_y ... 304 batches]
 *       [item0_width, item0_width ... 304 batches]
 *       [item0_height, item0_height ... 304 batches]
 *       [item1_center_x, item1_center_x ... 304 batches]
 *       [item1_center_y, item1_center_y ... 304 batches]
 *       [item1_width, item1_width ... 304 batches]
 *       [item1_height, item1_height ... 304 batches]
 *       ... 21 items * 4 positions per item
 *
 *       c dimension split
 *       Seg_size = 256
 *       Rem_size = 304 - 256 = 48
 *       S = 304 / 256 = 1
 *
 *       batch 0 -> batch 255
 *       [item0_center_x, item0_center_x ... 256 batches]
 *       [item0_center_y, item0_center_y ... 256 batches]
 *       [item0_width, item0_width ... 256 batches]
 *       [item0_height, item0_height ... 256 batches]
 *       [item1_center_x, item1_center_x ... 256 batches]
 *       [item1_center_y, item1_center_y ... 256 batches]
 *       [item1_width, item1_width ... 256 batches]
 *       [item1_height, item1_height ... 256 batches]
 *       ... 21 items * 4 positions per item
 *
 *       batch 256 -> batch 303
 *       [item0_center_x, item0_center_x ... 48 batches]
 *       [item0_center_y, item0_center_y ... 48 batches]
 *       [item0_width, item0_width ... 48 batches]
 *       [item0_height, item0_height ... 48 batches]
 *       [item1_center_x, item1_center_x ... 48 batches]
 *       [item1_center_y, item1_center_y ... 48 batches]
 *       [item1_width, item1_width ... 48 batches]
 *       [item1_height, item1_height ... 48 batches]
 *       ... 21 items * 4 positions per item
 *
 *
 * bottom[1] scores
 * 304 21
 * [item0_score_batch0, item1_score_batch0 ... 21 items]
 * [item0_score_batch1, item1_score_batch1 ... 21 items]
 * ... 304 (batch_size)
 * caffe: reshape dimensions to 1 304 1 21
 *
 * cnml: transpose dimensions to nhwc 1 1 21 304
 *       [item0_score, item0_score ... 304 batches]
 *       [item1_score, item1_score ... 304 batches]
 *       ... 21 items
 *
 *      c dimension split
 *      batch 0 -> batch 255
 *      [item0_score, item0_score .. 256 batches]
 *      [item1_score, item1_score .. 256 batches]
 *      ... 21 items
 *
 *      batch 256 -> batch 303
 *      [item0_score, item0_score ... 48 batches]
 *      [item1_score, item1_score ... 48 batches]
 *      ... 21 items
 *
 * bottom[2] rois
 * 1 304 1 5
 * [x0, y0, x1, y1, MLU_garbage]
 * [x0, y0, x1, y1, MLU_garbage]
 * ... 304 batches
 *
 * cnml: transpose dimensions to nhwc 1 1 4 304
 *       [x0, x0, x0, ... 304 batches]
 *       [y0, y0, y0, ... 304 batches]
 *       [x1, x1, x1, ... 304 batches]
 *       [y1, y1, y1, ... 304 batches]
 *       [MLU_garbage, MLU_garbage, ... 304 batches]
 *
 *       c dimension split
 *       batch 0 -> batch 255
 *       [x0, x0, x0, ... 256 batches]
 *       [y0, y0, y0, ... 256 batches]
 *       [x1, x1, x1, ... 256 batches]
 *       [y1, y1, y1, ... 256 batches]
 *       [MLU_garbage, MLU_garbage, ... 304 batches]
 *       batch 256 -> batch 303
 *       [x0, x0, x0, ... 48 batches]
 *       [y0, y0, y0, ... 48 batches]
 *       [x1, x1, x1, ... 48 batches]
 *       [y1, y1, y1, ... 48 batches]
 *       [MLU_garbage, MLU_garbage, ... 304 batches]
 */
template <typename Dtype>
class MLUImageDetectLayer : public ImageDetectLayer<Dtype> {
  public:
  explicit MLUImageDetectLayer(const LayerParameter& param)
      : ImageDetectLayer<Dtype>(param), bbox_reshape_ptr(nullptr),
        score_reshape_ptr(nullptr), detection_op_ptr_(nullptr),
        bbox_reshape_param(nullptr), score_reshape_param(nullptr),
        bbox_transpose_param(nullptr), score_transpose_param(nullptr),
        bbox_transpose_ptr(nullptr), score_transpose_ptr(nullptr),
        faster_rcnn_opt_param(nullptr) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                              const vector<Blob<Dtype>*>& top);
  virtual void fuse(MFusion<Dtype>* fuser);
  virtual ~MLUImageDetectLayer();
  virtual inline bool mfus_supported() { return true; }

  protected:
  virtual void MLUDestroyOp();
  virtual void MLUCreateOpBindData(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top);
  virtual void MLUCompileOp();
  virtual void Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);

  cnmlBaseOp_t bbox_reshape_ptr;
  cnmlBaseOp_t score_reshape_ptr;
  cnmlBaseOp_t detection_op_ptr_;

  Blob<Dtype> bbox_reshape_blob;
  Blob<Dtype> score_reshape_blob;
  Blob<Dtype> buffer_blob;

  cnmlReshapeOpParam_t bbox_reshape_param;
  cnmlReshapeOpParam_t score_reshape_param;

  vector<int> bbox_reshape_shape;
  vector<int> score_reshape_shape;
  vector<int> buffer_reshape_shape;

  Blob<Dtype> bbox_transpose_blob;
  Blob<Dtype> score_transpose_blob;
  cnmlNdTransposeOpParam_t bbox_transpose_param;
  cnmlNdTransposeOpParam_t score_transpose_param;
  cnmlBaseOp_t bbox_transpose_ptr;
  cnmlBaseOp_t score_transpose_ptr;
  cnmlPluginFasterRcnnDetectionOutputOpParam_t faster_rcnn_opt_param;
};

}  // namespace caffe
#endif  // USE_MLU
#endif  // INCLUDE_CAFFE_LAYERS_MLU_IMAGE_DETECT_LAYER_HPP_
