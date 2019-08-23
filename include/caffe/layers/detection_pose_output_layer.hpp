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

#ifndef INCLUDE_CAFFE_LAYERS_DETECTION_POSE_OUTPUT_LAYER_HPP_
#define INCLUDE_CAFFE_LAYERS_DETECTION_POSE_OUTPUT_LAYER_HPP_

#include <boost/property_tree/json_parser.hpp>
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
#include "caffe/util/bbox_util.hpp"

using namespace boost::property_tree;  // NOLINT(build/namespaces)

namespace caffe {

/**
 * @brief Generate the detection output based on location and confidence
 * predictions by doing non maximum suppression.
 *
 * Intended for use with MultiBox detection method.
 *
 * NOTE: does not implement Backwards operation.
 */
template <typename Dtype>
class DetectionPoseOutputLayer : public Layer<Dtype> {
  public:
  /**
   *@param  param provides DetectionPoseOutputParameter
   * detectionposeoutput_param with DetectionPoseOutputLayer
   * - num_classes: Number of classes to be predicted.
   * - num_poses: Number of classed to be predicted.
   * if true, bounding box are shared among different classes.
   * - share_pose
   * - share_location
   * - background_label_id: if there is no background class set it as -1.
   * - nms_param:Parameters used for non maximum suppression.
   * - save_output_param: Parameters used for saving detection results.
   * - code_type:coding method for bbox.
   * - variance_encoded_in_target: if true, variance is encoded in target;
   *   otherwise we need to adjust the predicted offset accordingly.
   * - keep_top_k: Number of total bboxes to be kept per image after nms step.
   *   -1 means keeping all bboxes after nms step
   * - confidence_threshold:Only consider detections whose confidences are larger
   *   than a threshold.If not provided, consider all bbox.
   * - visualize: If true, visualize the detection results
   * - visualize_threshold:The threshold used to visualize the detection results.
   *
   */
  explicit DetectionPoseOutputLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "DetectionPoseOutput"; }
  virtual inline int MinBottomBlobs() const { return 4; }
  virtual inline int MaxBottomBlobs() const { return 5; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

  protected:
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
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /// @brief Not implemented
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int num_classes_;
  int num_poses_;
  bool share_location_;
  bool share_pose_;
  int num_loc_classes_;
  int num_pose_classes_;

  int background_label_id_;
  CodeType code_type_;
  bool variance_encoded_in_target_;
  int keep_top_k_;
  float confidence_threshold_;

  int num_;
  int num_priors_;

  float nms_threshold_;
  int top_k_;

  bool need_save_;
  string output_directory_;
  string output_name_prefix_;
  string output_format_;
  map<int, string> label_to_name_;
  map<int, string> label_to_display_name_;
  vector<string> names_;
  vector<pair<int, int> > sizes_;
  int num_test_image_;
  int name_count_;

  ptree detections_;

  bool visualize_;
  float visualize_threshold_;
  shared_ptr<DataTransformer<Dtype> > data_transformer_;
};

}  // namespace caffe

#endif  // INCLUDE_CAFFE_LAYERS_DETECTION_POSE_OUTPUT_LAYER_HPP_
