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

#ifndef INCLUDE_CAFFE_LAYERS_MLU_DETECTION_POSE_OUTPUT_LAYER_HPP_
#define INCLUDE_CAFFE_LAYERS_MLU_DETECTION_POSE_OUTPUT_LAYER_HPP_
#ifdef USE_MLU

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layers/detection_pose_output_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class MLUDetectionPoseOutputLayer : public DetectionPoseOutputLayer<Dtype> {
  public:
  explicit MLUDetectionPoseOutputLayer(const LayerParameter& param)
      : DetectionPoseOutputLayer<Dtype>(param),
        detection_pose_output_op_ptr_(nullptr),
        detection_pose_output_op_param_(nullptr) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                              const vector<Blob<Dtype>*>& top);
  virtual inline bool mfus_supported() { return true; }
  virtual void fuse(MFusion<Dtype>* fuser);
  virtual ~MLUDetectionPoseOutputLayer();

  protected:
  /**
   *@brief destroy the Ops have been setup
   *
   */
  virtual void MLUDestroyOp();
  /**
   *@brief Create DetectionPoseOutput Op
   *
   */
  virtual void MLUCreateOpBindData(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top);
  /**
   *@brief Complile the Op to instructions
   *
   */
  virtual void MLUCompileOp();
  /**
   * @brief Do non maximum suppression (nms) on prediction results.
   *
   * @param bottom input Blob vector (at least 2)
   *   -# @f$ (N \times C1 \times 1 \times 1) @f$
   *      the location predictions with C1 predictions.
   *   -# @f$ (N \times C2 \times 1 \times 1) @f$
   *      the confidence predictions with C2 predictions.
   *   -# @f$ (N \times 2 \times C3 \times 1) @f$
   *      the prior bounding boxes with C3 values.
   * @param top output Blob vector (length 1)
   *   -# @f$ (1 \times 1 \times N \times 7) @f$
   *      N is the number of detections after nms, and each row is:
   *      [image_id, label, confidence, xmin, ymin, xmax, ymax]
   */
  virtual void Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  cnmlBaseOp_t detection_pose_output_op_ptr_;
  cnmlSsdDetectionPoseOutputOpParam_t detection_pose_output_op_param_;
};

}  // namespace caffe
#endif  // USE_MLU
#endif  // INCLUDE_CAFFE_LAYERS_MLU_DETECTION_POSE_OUTPUT_LAYER_HPP_
