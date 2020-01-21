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

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/detection_output_layer.hpp"
#include "caffe/layers/mlu_detection_output_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "gtest/gtest.h"
#include "test_detection_output_input_data.hpp"

namespace caffe {

template <typename Dtype>
void compute_detect(const Dtype* loc_data, const Dtype* conf_data,
                    const Dtype* prior_data, vector<Dtype>* top_data) {
  const int num = 1;             // bottom[0]->num()
  const int num_priors_ = 7308;  // bottom[2]->height() / 4
  const int num_loc_classes_ = 1;
  bool share_location_ = true;
  bool variance_encoded_in_target_ = false;
  const int num_classes_ = 21;
  const int background_label_id_ = 0;
  const float confidence_threshold_ = 0.01;
  const float nms_threshold_ = 0.45;
  const int top_k_ = 400;
  const int keep_top_k_ = 200;
  PriorBoxParameter_CodeType code_type_ =
      PriorBoxParameter_CodeType_CENTER_SIZE;

  // Retrieve all location predictions.
  vector<LabelBBox> all_loc_preds;
  GetLocPredictions(loc_data, num, num_priors_, num_loc_classes_,
                    share_location_, &all_loc_preds);

  // Retrieve all confidences.
  vector<map<int, vector<float> > > all_conf_scores;
  GetConfidenceScores(conf_data, num, num_priors_, num_classes_,
                      &all_conf_scores);

  // Retrieve all prior bboxes. It is same within a batch since we assume all
  // images in a batch are of same dimension.
  vector<NormalizedBBox> prior_bboxes;
  vector<vector<float> > prior_variances;
  GetPriorBBoxes(prior_data, num_priors_, &prior_bboxes, &prior_variances);

  // Decode all loc predictions to bboxes.
  vector<LabelBBox> all_decode_bboxes;
  DecodeBBoxesAll(all_loc_preds, prior_bboxes, prior_variances, num,
                  share_location_, num_loc_classes_, background_label_id_,
                  code_type_, variance_encoded_in_target_, &all_decode_bboxes);

  int num_kept = 0;
  vector<map<int, vector<int> > > all_indices;
  for (int i = 0; i < num; ++i) {
    const LabelBBox& decode_bboxes = all_decode_bboxes[i];
    const map<int, vector<float> >& conf_scores = all_conf_scores[i];
    map<int, vector<int> > indices;
    int num_det = 0;
    for (int c = 0; c < num_classes_; ++c) {
      if (c == background_label_id_) {
        // Ignore background class.
        continue;
      }
      if (conf_scores.find(c) == conf_scores.end()) {
        // Something bad happened if there are no predictions for current label.
        LOG(FATAL) << "Could not find confidence predictions for label " << c;
      }
      const vector<float>& scores = conf_scores.find(c)->second;
      int label = share_location_ ? -1 : c;
      if (decode_bboxes.find(label) == decode_bboxes.end()) {
        // Something bad happened if there are no predictions for current label.
        LOG(FATAL) << "Could not find location predictions for label " << label;
        continue;
      }
      const vector<NormalizedBBox>& bboxes = decode_bboxes.find(label)->second;
      ApplyNMSFast(bboxes, scores, confidence_threshold_, nms_threshold_,
                   top_k_, &(indices[c]));
      num_det += indices[c].size();
    }
    if (keep_top_k_ > -1 && num_det > keep_top_k_) {
      vector<pair<float, pair<int, int> > > score_index_pairs;
      for (map<int, vector<int> >::iterator it = indices.begin();
           it != indices.end(); ++it) {
        int label = it->first;
        const vector<int>& label_indices = it->second;
        if (conf_scores.find(label) == conf_scores.end()) {
          // Something bad happened for current label.
          LOG(FATAL) << "Could not find location predictions for label" << label;
          continue;
        }
        const vector<float>& scores = conf_scores.find(label)->second;
        for (int j = 0; j < label_indices.size(); ++j) {
          int idx = label_indices[j];
          CHECK_LT(idx, scores.size());
          score_index_pairs.push_back(
              std::make_pair(scores[idx], std::make_pair(label, idx)));
        }
      }
      // Keep top k results per image.
      std::sort(score_index_pairs.begin(), score_index_pairs.end(),
                SortScorePairDescend<pair<int, int> >);
      score_index_pairs.resize(keep_top_k_);
      // Store the new indices.
      map<int, vector<int> > new_indices;
      for (int j = 0; j < score_index_pairs.size(); ++j) {
        int label = score_index_pairs[j].second.first;
        int idx = score_index_pairs[j].second.second;
        new_indices[label].push_back(idx);
      }
      all_indices.push_back(new_indices);
      num_kept += keep_top_k_;
    } else {
      all_indices.push_back(indices);
      num_kept += num_det;
    }
  }
  top_data->clear();
  if (num_kept == 0) {
    LOG(INFO) << "Couldn't find any detections";
    for (int i = 0; i < 7; i++) top_data->push_back(-1);
    return;
  }

  for (int i = 0; i < num; ++i) {
    const map<int, vector<float> >& conf_scores = all_conf_scores[i];
    const LabelBBox& decode_bboxes = all_decode_bboxes[i];
    for (map<int, vector<int> >::iterator it = all_indices[i].begin();
         it != all_indices[i].end(); ++it) {
      int label = it->first;
      if (conf_scores.find(label) == conf_scores.end()) {
        // Something bad happened if there are no predictions for current label.
        LOG(FATAL) << "Could not find confidence predictions for " << label;
        continue;
      }
      const vector<float>& scores = conf_scores.find(label)->second;
      int loc_label = share_location_ ? -1 : label;
      if (decode_bboxes.find(loc_label) == decode_bboxes.end()) {
        // Something bad happened if there are no predictions for current label.
        LOG(FATAL) << "Could not find location predictions for label" << loc_label;
        continue;
      }
      const vector<NormalizedBBox>& bboxes =
          decode_bboxes.find(loc_label)->second;
      vector<int>& indices = it->second;
      for (int j = 0; j < indices.size(); ++j) {
        int idx = indices[j];
        top_data->push_back(i);
        top_data->push_back(label);
        top_data->push_back(scores[idx]);
        NormalizedBBox clip_bbox;
        ClipBBox(bboxes[idx], &clip_bbox);
        top_data->push_back(clip_bbox.xmin());
        top_data->push_back(clip_bbox.ymin());
        top_data->push_back(clip_bbox.xmax());
        top_data->push_back(clip_bbox.ymax());
      }
    }
  }
}

template <typename TypeParam>
class DetectionOutputLayerTest : public CPUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  DetectionOutputLayerTest()
      : blob_bottom_loc_(new Blob<Dtype>(1, 29232, 1, 1)),
        blob_bottom_conf_(new Blob<Dtype>(1, 153468, 1, 1)),
        blob_bottom_prior_(new Blob<Dtype>(1, 2, 29232, 1)),
        blob_bottom_data_(new Blob<Dtype>(1, 3, 300, 300)),
        blob_top_(new Blob<Dtype>()) {}
  void SetUp() {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_loc_);
    filler.Fill(this->blob_bottom_conf_);
    filler.Fill(this->blob_bottom_prior_);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_loc_);
    blob_bottom_vec_.push_back(blob_bottom_conf_);
    blob_bottom_vec_.push_back(blob_bottom_prior_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~DetectionOutputLayerTest() {
    delete blob_bottom_loc_;
    delete blob_bottom_conf_;
    delete blob_bottom_prior_;
    delete blob_bottom_data_;
    delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_loc_;
  Blob<Dtype>* const blob_bottom_conf_;
  Blob<Dtype>* const blob_bottom_prior_;
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(DetectionOutputLayerTest, TestDtypesAndDevices);

TYPED_TEST(DetectionOutputLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  DetectionOutputParameter* detection_output_param =
      layer_param.mutable_detection_output_param();
  detection_output_param->set_num_classes(21);
  detection_output_param->set_share_location(true);
  detection_output_param->mutable_nms_param()->set_nms_threshold(0.45);
  DetectionOutputLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 7);
}

TYPED_TEST(DetectionOutputLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.clear();
  this->blob_top_vec_.clear();

  Dtype* input_data;
  // fill real loc data
  input_data = this->blob_bottom_loc_->mutable_cpu_data();
  for (unsigned int i = 0; i < this->blob_bottom_loc_->count(); ++i) {
    input_data[i] = detectionoutput_input_data::input_loc_cpu_data[i];
  }
  // fill real conf data
  input_data = this->blob_bottom_conf_->mutable_cpu_data();
  for (unsigned int i = 0; i < this->blob_bottom_conf_->count(); ++i) {
    input_data[i] = detectionoutput_input_data::input_conf_cpu_data[i];
  }
  // fill real prior data
  input_data = this->blob_bottom_prior_->mutable_cpu_data();
  for (unsigned int i = 0; i < this->blob_bottom_prior_->count(); ++i) {
    input_data[i] = detectionoutput_input_data::input_prior_cpu_data[i];
  }
  // fill real input data
  input_data = this->blob_bottom_data_->mutable_cpu_data();
  for (unsigned int i = 0; i < this->blob_bottom_data_->count(); ++i) {
    input_data[i] = detectionoutput_input_data::input_data[i];
  }
  this->blob_bottom_vec_.push_back(this->blob_bottom_loc_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conf_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_prior_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_data_);
  this->blob_top_vec_.push_back(this->blob_top_);

  LayerParameter layer_param;
  DetectionOutputParameter* detection_output_param =
      layer_param.mutable_detection_output_param();
  detection_output_param->set_num_classes(21);
  detection_output_param->set_share_location(true);
  detection_output_param->set_background_label_id(0);
  detection_output_param->set_code_type(PriorBoxParameter_CodeType_CENTER_SIZE);
  detection_output_param->set_keep_top_k(200);
  detection_output_param->set_confidence_threshold(0.01);
  detection_output_param->mutable_nms_param()->set_nms_threshold(0.45);
  detection_output_param->mutable_nms_param()->set_top_k(400);
  DetectionOutputLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* top_data = this->blob_top_->cpu_data();

  // calculate reference values
  vector<Dtype> reference_value;
  compute_detect(this->blob_bottom_loc_->cpu_data(),
                 this->blob_bottom_conf_->cpu_data(),
                 this->blob_bottom_prior_->cpu_data(), &reference_value);
  const Dtype* reference_data = reference_value.data();

  float err_sum = 0, sum = 0;
  ASSERT_EQ(this->blob_top_->count(), reference_value.size());
  for (int i = 0; i < this->blob_top_->count(); i++) {
    EXPECT_NEAR(top_data[i], reference_data[i], 5e-5);
    err_sum += std::abs(top_data[i] - reference_data[i]);
    sum += std::abs(top_data[i]);
  }
  EXPECT_LE(err_sum / sum, 1e-5);
}

#ifdef USE_MLU

template <typename TypeParam>
class MLUDetectionOutputLayerTest : public MLUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MLUDetectionOutputLayerTest()
      : blob_bottom_loc_(new Blob<Dtype>(1, 29232, 1, 1)),
        blob_bottom_conf_(new Blob<Dtype>(1, 153468, 1, 1)),
        blob_bottom_prior_(new Blob<Dtype>(1, 2, 29232, 1)),
        blob_bottom_data_(new Blob<Dtype>(1, 3, 300, 300)),
        blob_top_(new Blob<Dtype>()) {}
  void SetUp() {
    blob_bottom_vec_.push_back(blob_bottom_loc_);
    blob_bottom_vec_.push_back(blob_bottom_conf_);
    blob_bottom_vec_.push_back(blob_bottom_prior_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MLUDetectionOutputLayerTest() {
    delete blob_bottom_loc_;
    delete blob_bottom_conf_;
    delete blob_bottom_prior_;
    delete blob_bottom_data_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_loc_;
  Blob<Dtype>* const blob_bottom_conf_;
  Blob<Dtype>* const blob_bottom_prior_;
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MLUDetectionOutputLayerTest, TestMLUDevices);

TYPED_TEST(MLUDetectionOutputLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  DetectionOutputParameter* detection_output_param =
      layer_param.mutable_detection_output_param();
  detection_output_param->set_num_classes(21);
  detection_output_param->set_keep_top_k(200);
  detection_output_param->set_share_location(true);
  detection_output_param->mutable_nms_param()->set_nms_threshold(0.45);
  MLUDetectionOutputLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1464);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
}


TYPED_TEST(MLUDetectionOutputLayerTest, TestForwardByCnml) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.clear();
  this->blob_top_vec_.clear();
  if (Caffe::rt_core() > 4) return;
  Dtype* input_data;
  // fill real loc data
  input_data = this->blob_bottom_loc_->mutable_cpu_data();
  for (unsigned int i = 0; i < this->blob_bottom_loc_->count(); ++i) {
    input_data[i] = detectionoutput_input_data::input_loc_cpu_data[i];
  }
  // fill real conf data
  input_data = this->blob_bottom_conf_->mutable_cpu_data();
  for (unsigned int i = 0; i < this->blob_bottom_conf_->count(); ++i) {
    input_data[i] = detectionoutput_input_data::input_conf_cpu_data[i];
  }
  // fill real prior data
  input_data = this->blob_bottom_prior_->mutable_cpu_data();
  for (unsigned int i = 0; i < this->blob_bottom_prior_->count(); ++i) {
    input_data[i] = detectionoutput_input_data::input_prior_cpu_data[i];
  }
  // fill real input data
  input_data = this->blob_bottom_data_->mutable_cpu_data();
  for (unsigned int i = 0; i < this->blob_bottom_data_->count(); ++i) {
    input_data[i] = detectionoutput_input_data::input_data[i];
  }
  this->blob_bottom_vec_.push_back(this->blob_bottom_loc_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conf_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_prior_);
  this->blob_top_vec_.push_back(this->blob_top_);

  LayerParameter layer_param;
  DetectionOutputParameter* detection_output_param =
      layer_param.mutable_detection_output_param();
  detection_output_param->set_num_classes(21);
  detection_output_param->set_share_location(true);
  detection_output_param->set_background_label_id(0);
  detection_output_param->set_code_type(PriorBoxParameter_CodeType_CENTER_SIZE);
  detection_output_param->set_keep_top_k(200);
  detection_output_param->set_confidence_threshold(0.01);
  detection_output_param->mutable_nms_param()->set_nms_threshold(0.45);
  detection_output_param->mutable_nms_param()->set_top_k(400);
  MLUDetectionOutputLayer<Dtype> layer(layer_param);
  Caffe::setDetectOpMode(0);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* top_data = this->blob_top_->cpu_data();

  // calculate reference values
  vector<Dtype> reference_value;
  compute_detect(this->blob_bottom_loc_->cpu_data(),
                 this->blob_bottom_conf_->cpu_data(),
                 this->blob_bottom_prior_->cpu_data(), &reference_value);
  const Dtype* reference_data = reference_value.data();

  // select valid detecting dialog of reference data
  vector<vector<float> > detections_references;
  for (int i = 0; i < reference_value.size() / 7; i++) {
    if (reference_data[0] == -1) {
      // Skip invalid detection.
      reference_data += 7;
      continue;
    }
    vector<float> detection(reference_data, reference_data + 7);
    detections_references.push_back(detection);
    reference_data += 7;
  }

  // select valid detecting dialog of reference data
  vector<vector<float> > detections_mlu;
  for (int i = 0; i < this->blob_top_->count() / 6; i++) {
    if (top_data[4] == 0) {
      // the score(result[4]) must be 0 if invalid detection,
      // so skip it.
      top_data += 6;
      continue;
    }
    vector<float> detection(7, 0);
    detection[0] = 0;
    detection[1] = top_data[5];
    detection[2] = top_data[4];
    detection[3] = top_data[0];
    detection[4] = top_data[1];
    detection[5] = top_data[2];
    detection[6] = top_data[3];
    detections_mlu.push_back(detection);
    top_data += 6;
  }

  // check the number of detect dialog, this picture has 1 dialog
  EXPECT_EQ(detections_references.size(), detections_mlu.size());
  // check the label of detect dialog
  EXPECT_EQ(detections_references[0][1], detections_mlu[0][1]);
  // check the error rate
  float err_sum = 0, sum = 0;
  for (int i = 0; i < detections_references[0].size(); i++) {
    err_sum += std::abs(detections_mlu[0][i] - detections_references[0][i]);
    sum += std::abs(detections_mlu[0][i]);
  }
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_loc_->shape_string().c_str() << "\t"
    << "bottom2:" << this->blob_bottom_conf_->shape_string().c_str() << "\t"
    << "bottom2:" << this->blob_bottom_prior_->shape_string().c_str() << "\t"
    << "bottom2:" << this->blob_bottom_data_->shape_string().c_str();
  BOTTOM(stream);
  ERR_RATE(err_sum/sum);
  EVENT_TIME(layer.get_event_time());
}

TYPED_TEST(MLUDetectionOutputLayerTest, TestForwardByBang) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.clear();
  this->blob_top_vec_.clear();

  Dtype* input_data;
  // fill real loc data
  input_data = this->blob_bottom_loc_->mutable_cpu_data();
  for (unsigned int i = 0; i < this->blob_bottom_loc_->count(); ++i) {
    input_data[i] = detectionoutput_input_data::input_loc_cpu_data[i];
  }
  // fill real conf data
  input_data = this->blob_bottom_conf_->mutable_cpu_data();
  for (unsigned int i = 0; i < this->blob_bottom_conf_->count(); ++i) {
    input_data[i] = detectionoutput_input_data::input_conf_cpu_data[i];
  }
  // fill real prior data
  input_data = this->blob_bottom_prior_->mutable_cpu_data();
  for (unsigned int i = 0; i < this->blob_bottom_prior_->count(); ++i) {
    input_data[i] = detectionoutput_input_data::input_prior_cpu_data[i];
  }
  // fill real input data
  input_data = this->blob_bottom_data_->mutable_cpu_data();
  for (unsigned int i = 0; i < this->blob_bottom_data_->count(); ++i) {
    input_data[i] = detectionoutput_input_data::input_data[i];
  }
  this->blob_bottom_vec_.push_back(this->blob_bottom_loc_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conf_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_prior_);
  this->blob_top_vec_.push_back(this->blob_top_);

  LayerParameter layer_param;
  DetectionOutputParameter* detection_output_param =
      layer_param.mutable_detection_output_param();
  detection_output_param->set_num_classes(21);
  detection_output_param->set_share_location(true);
  detection_output_param->set_background_label_id(0);
  detection_output_param->set_code_type(PriorBoxParameter_CodeType_CENTER_SIZE);
  detection_output_param->set_keep_top_k(200);
  detection_output_param->set_confidence_threshold(0.01);
  detection_output_param->mutable_nms_param()->set_nms_threshold(0.45);
  detection_output_param->mutable_nms_param()->set_top_k(400);
  MLUDetectionOutputLayer<Dtype> layer(layer_param);
  Caffe::setDetectOpMode(1);
  layer.set_int8_context(0);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* top_data = this->blob_top_->cpu_data();

  // calculate reference values
  vector<Dtype> reference_value;
  compute_detect(this->blob_bottom_loc_->cpu_data(),
                 this->blob_bottom_conf_->cpu_data(),
                 this->blob_bottom_prior_->cpu_data(), &reference_value);
  const Dtype* reference_data = reference_value.data();

  // select valid detecting dialog of reference data
  vector<vector<float> > detections_references;
  for (int i = 0; i < reference_value.size() / 7; i++) {
    if (reference_data[0] == -1) {
      // Skip invalid detection.
      reference_data += 7;
      continue;
    }
    vector<float> detection(reference_data, reference_data + 7);
    detections_references.push_back(detection);
    reference_data += 7;
  }

  // select valid detecting dialog of reference data
  vector<vector<float> > detections_mlu;
  int output_num = top_data[0];
  top_data += 64;
  for (int i = 0; i < output_num; i++) {
    vector<float> detection(top_data, top_data + 7);
    detections_mlu.push_back(detection);
    top_data += 7;
  }

  // check the number of detect dialog, this picture has 1 dialog
  EXPECT_EQ(detections_references.size(), detections_mlu.size());
  // check the label of detect dialog
  EXPECT_EQ(detections_references[0][1], detections_mlu[0][1]);
  // check the error rate
  float err_sum = 0, sum = 0;
  for (int i = 0; i < detections_references[0].size(); i++) {
    err_sum += std::abs(detections_mlu[0][i] - detections_references[0][i]);
    sum += std::abs(detections_mlu[0][i]);
  }
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_loc_->shape_string().c_str() << "\t"
    << "bottom2:" << this->blob_bottom_conf_->shape_string().c_str() << "\t"
    << "bottom2:" << this->blob_bottom_prior_->shape_string().c_str() << "\t"
    << "bottom2:" << this->blob_bottom_data_->shape_string().c_str();
  BOTTOM(stream);
  ERR_RATE(err_sum/sum);
  EVENT_TIME(layer.get_event_time());
}

template <typename TypeParam>
class MFUSDetectionOutputLayerTest : public MFUSDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MFUSDetectionOutputLayerTest()
      : blob_bottom_loc_(new Blob<Dtype>(1, 29232, 1, 1)),
        blob_bottom_conf_(new Blob<Dtype>(1, 153468, 1, 1)),
        blob_bottom_prior_(new Blob<Dtype>(1, 2, 29232, 1)),
        blob_bottom_data_(new Blob<Dtype>(1, 3, 300, 300)),
        blob_top_(new Blob<Dtype>()) {}
  void SetUp() {
    blob_bottom_vec_.push_back(blob_bottom_loc_);
    blob_bottom_vec_.push_back(blob_bottom_conf_);
    blob_bottom_vec_.push_back(blob_bottom_prior_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MFUSDetectionOutputLayerTest() {
    delete blob_bottom_loc_;
    delete blob_bottom_conf_;
    delete blob_bottom_prior_;
    delete blob_bottom_data_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_loc_;
  Blob<Dtype>* const blob_bottom_conf_;
  Blob<Dtype>* const blob_bottom_prior_;
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MFUSDetectionOutputLayerTest, TestMFUSDevices);

TYPED_TEST(MFUSDetectionOutputLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  DetectionOutputParameter* detection_output_param =
      layer_param.mutable_detection_output_param();
  detection_output_param->set_num_classes(21);
  detection_output_param->set_keep_top_k(200);
  detection_output_param->set_share_location(true);
  detection_output_param->mutable_nms_param()->set_nms_threshold(0.45);
  MLUDetectionOutputLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1464);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
}


TYPED_TEST(MFUSDetectionOutputLayerTest, TestForwardByCnml) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.clear();
  this->blob_top_vec_.clear();
  if (Caffe::rt_core() > 4)return;
  Dtype* input_data;
  // fill real loc data
  input_data = this->blob_bottom_loc_->mutable_cpu_data();
  for (unsigned int i = 0; i < this->blob_bottom_loc_->count(); ++i) {
    input_data[i] = detectionoutput_input_data::input_loc_cpu_data[i];
  }
  // fill real conf data
  input_data = this->blob_bottom_conf_->mutable_cpu_data();
  for (unsigned int i = 0; i < this->blob_bottom_conf_->count(); ++i) {
    input_data[i] = detectionoutput_input_data::input_conf_cpu_data[i];
  }
  // fill real prior data
  input_data = this->blob_bottom_prior_->mutable_cpu_data();
  for (unsigned int i = 0; i < this->blob_bottom_prior_->count(); ++i) {
    input_data[i] = detectionoutput_input_data::input_prior_cpu_data[i];
  }
  // fill real input data
  input_data = this->blob_bottom_data_->mutable_cpu_data();
  for (unsigned int i = 0; i < this->blob_bottom_data_->count(); ++i) {
    input_data[i] = detectionoutput_input_data::input_data[i];
  }
  this->blob_bottom_vec_.push_back(this->blob_bottom_loc_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conf_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_prior_);
  this->blob_top_vec_.push_back(this->blob_top_);

  LayerParameter layer_param;
  DetectionOutputParameter* detection_output_param =
      layer_param.mutable_detection_output_param();
  detection_output_param->set_num_classes(21);
  detection_output_param->set_share_location(true);
  detection_output_param->set_background_label_id(0);
  detection_output_param->set_code_type(PriorBoxParameter_CodeType_CENTER_SIZE);
  detection_output_param->set_keep_top_k(200);
  detection_output_param->set_confidence_threshold(0.01);
  detection_output_param->mutable_nms_param()->set_nms_threshold(0.45);
  detection_output_param->mutable_nms_param()->set_top_k(400);
  MLUDetectionOutputLayer<Dtype> layer(layer_param);
  Caffe::setDetectOpMode(0);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_TRUE(layer.mfus_supported());

  MFusion<Dtype> fuser;
  fuser.reset();
  fuser.addInputs(this->blob_bottom_vec_);
  fuser.addOutputs(this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.fuse(&fuser);
  fuser.compile();
  fuser.forward();
  // calculate reference values
  const Dtype* top_data = this->blob_top_->cpu_data();
  vector<Dtype> reference_value;
  compute_detect(this->blob_bottom_loc_->cpu_data(),
                 this->blob_bottom_conf_->cpu_data(),
                 this->blob_bottom_prior_->cpu_data(), &reference_value);
  const Dtype* reference_data = reference_value.data();

  // select valid detecting dialog of reference data
  vector<vector<float> > detections_references;
  for (int i = 0; i < reference_value.size() / 7; i++) {
    if (reference_data[0] == -1) {
      // Skip invalid detection.
      reference_data += 7;
      continue;
    }
    vector<float> detection(reference_data, reference_data + 7);
    detections_references.push_back(detection);
    reference_data += 7;
  }

  // select valid detecting dialog of reference data
  vector<vector<float> > detections_mfus;
  for (int i = 0; i < this->blob_top_->count() / 6; i++) {
    if (top_data[4] == 0) {
      // the score(result[4]) must be 0 if invalid detection,
      // so skip it.
      top_data += 6;
      continue;
    }
    vector<float> detection(7, 0);
    detection[0] = 0;
    detection[1] = top_data[5];
    detection[2] = top_data[4];
    detection[3] = top_data[0];
    detection[4] = top_data[1];
    detection[5] = top_data[2];
    detection[6] = top_data[3];
    detections_mfus.push_back(detection);
    top_data += 6;
  }

  // check the number of detect dialog, this picture has 1 dialog
  EXPECT_EQ(detections_references.size(), detections_mfus.size());
  // check the label of detect dialog
  EXPECT_EQ(detections_references[0][1], detections_mfus[0][1]);
  // check the error rate
  float err_sum = 0, sum = 0;
  for (int i = 0; i < detections_references[0].size(); i++) {
    err_sum += std::abs(detections_mfus[0][i] - detections_references[0][i]);
    sum += std::abs(detections_mfus[0][i]);
  }
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_loc_->shape_string().c_str() << "\t"
    << "bottom2:" << this->blob_bottom_conf_->shape_string().c_str() << "\t"
    << "bottom3:" << this->blob_bottom_prior_->shape_string().c_str() << "\t"
    << "bottom4:" << this->blob_bottom_data_->shape_string().c_str();
  BOTTOM(stream);
  ERR_RATE(err_sum/sum);
  EVENT_TIME(fuser.get_event_time());
}



TYPED_TEST(MFUSDetectionOutputLayerTest, TestForwardByBang) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.clear();
  this->blob_top_vec_.clear();

  Dtype* input_data;
  // fill real loc data
  input_data = this->blob_bottom_loc_->mutable_cpu_data();
  for (unsigned int i = 0; i < this->blob_bottom_loc_->count(); ++i) {
    input_data[i] = detectionoutput_input_data::input_loc_cpu_data[i];
  }
  // fill real conf data
  input_data = this->blob_bottom_conf_->mutable_cpu_data();
  for (unsigned int i = 0; i < this->blob_bottom_conf_->count(); ++i) {
    input_data[i] = detectionoutput_input_data::input_conf_cpu_data[i];
  }
  // fill real prior data
  input_data = this->blob_bottom_prior_->mutable_cpu_data();
  for (unsigned int i = 0; i < this->blob_bottom_prior_->count(); ++i) {
    input_data[i] = detectionoutput_input_data::input_prior_cpu_data[i];
  }
  // fill real input data
  input_data = this->blob_bottom_data_->mutable_cpu_data();
  for (unsigned int i = 0; i < this->blob_bottom_data_->count(); ++i) {
    input_data[i] = detectionoutput_input_data::input_data[i];
  }
  this->blob_bottom_vec_.push_back(this->blob_bottom_loc_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conf_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_prior_);
  this->blob_top_vec_.push_back(this->blob_top_);

  LayerParameter layer_param;
  DetectionOutputParameter* detection_output_param =
      layer_param.mutable_detection_output_param();
  detection_output_param->set_num_classes(21);
  detection_output_param->set_share_location(true);
  detection_output_param->set_background_label_id(0);
  detection_output_param->set_code_type(PriorBoxParameter_CodeType_CENTER_SIZE);
  detection_output_param->set_keep_top_k(200);
  detection_output_param->set_confidence_threshold(0.01);
  detection_output_param->mutable_nms_param()->set_nms_threshold(0.45);
  detection_output_param->mutable_nms_param()->set_top_k(400);
  MLUDetectionOutputLayer<Dtype> layer(layer_param);
  Caffe::setDetectOpMode(1);
  layer.set_int8_context(0);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_TRUE(layer.mfus_supported());

  MFusion<Dtype> fuser;
  fuser.reset();
  fuser.addInputs(this->blob_bottom_vec_);
  fuser.addOutputs(this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.fuse(&fuser);
  fuser.compile();
  fuser.forward();
  // calculate reference values
  const Dtype* top_data = this->blob_top_->cpu_data();
  vector<Dtype> reference_value;
  compute_detect(this->blob_bottom_loc_->cpu_data(),
                 this->blob_bottom_conf_->cpu_data(),
                 this->blob_bottom_prior_->cpu_data(), &reference_value);
  const Dtype* reference_data = reference_value.data();

  // select valid detecting dialog of reference data
  vector<vector<float> > detections_references;
  for (int i = 0; i < reference_value.size() / 7; i++) {
    if (reference_data[0] == -1) {
      // Skip invalid detection.
      reference_data += 7;
      continue;
    }
    vector<float> detection(reference_data, reference_data + 7);
    detections_references.push_back(detection);
    reference_data += 7;
  }

  // select valid detecting dialog of reference data
  vector<vector<float> > detections_mfus;
  int output_num = top_data[0];
  top_data += 64;
  for (int i = 0; i < output_num; i++) {
    vector<float> detection(top_data, top_data + 7);
    detections_mfus.push_back(detection);
    top_data += 7;
  }

  // check the number of detect dialog, this picture has 1 dialog
  EXPECT_EQ(detections_references.size(), detections_mfus.size());
  // check the label of detect dialog
  EXPECT_EQ(detections_references[0][1], detections_mfus[0][1]);
  // check the error rate
  float err_sum = 0, sum = 0;
  for (int i = 0; i < detections_references[0].size(); i++) {
    err_sum += std::abs(detections_mfus[0][i] - detections_references[0][i]);
    sum += std::abs(detections_mfus[0][i]);
  }
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_loc_->shape_string().c_str() << "\t"
    << "bottom2:" << this->blob_bottom_conf_->shape_string().c_str() << "\t"
    << "bottom3:" << this->blob_bottom_prior_->shape_string().c_str() << "\t"
    << "bottom4:" << this->blob_bottom_data_->shape_string().c_str();
  BOTTOM(stream);
  ERR_RATE(err_sum/sum);
  EVENT_TIME(fuser.get_event_time());
}


#endif

}  // namespace caffe
