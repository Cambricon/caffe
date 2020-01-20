/*
All modification made by Cambricon Corporation: © 2018--2019 Cambricon Corporation
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

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/mlu_ssd_detection_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "detection_output_data.hpp"
#include "gtest/gtest.h"
#include "ssd_detection_input_data.hpp"

namespace caffe {

// ssd_detection should have the same result with detection_out
// using the same input data scrap from example/ssd detecting one picture
#ifdef USE_MLU
template <typename TypeParam>
class MLUSsdDetectionLayerTest : public MLUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MLUSsdDetectionLayerTest()
      : blob_bottom_conv4_3_norm_mbox_loc_(new Blob<Dtype>(1, 12, 38, 38)),
        blob_bottom_fc7_mbox_loc_(new Blob<Dtype>(1, 24, 19, 19)),
        blob_bottom_conv6_2_mbox_loc_(new Blob<Dtype>(1, 24, 10, 10)),
        blob_bottom_conv7_2_mbox_loc_(new Blob<Dtype>(1, 24, 5, 5)),
        blob_bottom_conv8_2_mbox_loc_(new Blob<Dtype>(1, 24, 3, 3)),
        blob_bottom_pool6_mbox_loc_(new Blob<Dtype>(1, 24, 1, 1)),
        blob_bottom_conv4_3_norm_mbox_conf_(new Blob<Dtype>(1, 63, 38, 38)),
        blob_bottom_fc7_mbox_conf_(new Blob<Dtype>(1, 126, 19, 19)),
        blob_bottom_conv6_2_mbox_conf_(new Blob<Dtype>(1, 126, 10, 10)),
        blob_bottom_conv7_2_mbox_conf_(new Blob<Dtype>(1, 126, 5, 5)),
        blob_bottom_conv8_2_mbox_conf_(new Blob<Dtype>(1, 126, 3, 3)),
        blob_bottom_pool6_mbox_conf_(new Blob<Dtype>(1, 126, 1, 1)),
        blob_bottom_conv4_3_norm_(new Blob<Dtype>(1, 512, 38, 38)),
        blob_bottom_fc7_xx_(new Blob<Dtype>(1, 1024, 19, 19)),
        blob_bottom_conv6_2_xx_(new Blob<Dtype>(1, 512, 10, 10)),
        blob_bottom_conv7_2_xx_(new Blob<Dtype>(1, 256, 5, 5)),
        blob_bottom_conv8_2_xx_(new Blob<Dtype>(1, 256, 3, 3)),
        blob_bottom_pool6_(new Blob<Dtype>(1, 256, 1, 1)),
        blob_bottom_data_(new Blob<Dtype>(1, 3, 300, 300)),
        blob_top_(new Blob<Dtype>()) {}
  virtual ~MLUSsdDetectionLayerTest() {
    delete blob_bottom_conv4_3_norm_mbox_loc_;
    delete blob_bottom_fc7_mbox_loc_;
    delete blob_bottom_conv6_2_mbox_loc_;
    delete blob_bottom_conv7_2_mbox_loc_;
    delete blob_bottom_conv8_2_mbox_loc_;
    delete blob_bottom_pool6_mbox_loc_;
    delete blob_bottom_conv4_3_norm_mbox_conf_;
    delete blob_bottom_fc7_mbox_conf_;
    delete blob_bottom_conv6_2_mbox_conf_;
    delete blob_bottom_conv7_2_mbox_conf_;
    delete blob_bottom_conv8_2_mbox_conf_;
    delete blob_bottom_pool6_mbox_conf_;
    delete blob_bottom_conv4_3_norm_;
    delete blob_bottom_fc7_xx_;
    delete blob_bottom_conv6_2_xx_;
    delete blob_bottom_conv7_2_xx_;
    delete blob_bottom_conv8_2_xx_;
    delete blob_bottom_pool6_;
    delete blob_bottom_data_;
    delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_conv4_3_norm_mbox_loc_;
  Blob<Dtype>* const blob_bottom_fc7_mbox_loc_;
  Blob<Dtype>* const blob_bottom_conv6_2_mbox_loc_;
  Blob<Dtype>* const blob_bottom_conv7_2_mbox_loc_;
  Blob<Dtype>* const blob_bottom_conv8_2_mbox_loc_;
  Blob<Dtype>* const blob_bottom_pool6_mbox_loc_;
  Blob<Dtype>* const blob_bottom_conv4_3_norm_mbox_conf_;
  Blob<Dtype>* const blob_bottom_fc7_mbox_conf_;
  Blob<Dtype>* const blob_bottom_conv6_2_mbox_conf_;
  Blob<Dtype>* const blob_bottom_conv7_2_mbox_conf_;
  Blob<Dtype>* const blob_bottom_conv8_2_mbox_conf_;
  Blob<Dtype>* const blob_bottom_pool6_mbox_conf_;
  Blob<Dtype>* const blob_bottom_conv4_3_norm_;
  Blob<Dtype>* const blob_bottom_fc7_xx_;
  Blob<Dtype>* const blob_bottom_conv6_2_xx_;
  Blob<Dtype>* const blob_bottom_conv7_2_xx_;
  Blob<Dtype>* const blob_bottom_conv8_2_xx_;
  Blob<Dtype>* const blob_bottom_pool6_;
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MLUSsdDetectionLayerTest, TestMLUDevices);

TYPED_TEST(MLUSsdDetectionLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;

  LayerParameter layer_param;
  DetectionOutputParameter* detection_output_param =
      layer_param.mutable_detection_output_param();
  detection_output_param->set_keep_top_k(200);
  detection_output_param->set_num_classes(21);
  detection_output_param->set_share_location(true);
  detection_output_param->mutable_nms_param()->set_nms_threshold(0.45);
  MLUSsdDetectionLayer<Dtype> layer(layer_param);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv4_3_norm_mbox_loc_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_fc7_mbox_loc_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv6_2_mbox_loc_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv7_2_mbox_loc_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv8_2_mbox_loc_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_pool6_mbox_loc_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv4_3_norm_mbox_conf_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_fc7_mbox_conf_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv6_2_mbox_conf_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv7_2_mbox_conf_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv8_2_mbox_conf_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_pool6_mbox_conf_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv4_3_norm_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_fc7_xx_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv6_2_xx_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv7_2_xx_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv8_2_xx_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_pool6_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_data_);
  this->blob_top_vec_.push_back(this->blob_top_);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1464);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
}


TYPED_TEST(MLUSsdDetectionLayerTest, TestForwardByCnml) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.clear();
  this->blob_top_vec_.clear();
  if (Caffe::rt_core() > 4 ) return;
  Dtype* input_data;
  int count;
  input_data = this->blob_bottom_conv4_3_norm_mbox_loc_->mutable_cpu_data();
  count = this->blob_bottom_conv4_3_norm_mbox_loc_->count();
  CHECK_EQ(count, ssd_detection_input_data::conv4_3_norm_mbox_loc.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::conv4_3_norm_mbox_loc[i];
  }
  input_data = this->blob_bottom_fc7_mbox_loc_->mutable_cpu_data();
  count = this->blob_bottom_fc7_mbox_loc_->count();
  CHECK_EQ(count, ssd_detection_input_data::fc7_mbox_loc.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::fc7_mbox_loc[i];
  }
  input_data = this->blob_bottom_conv6_2_mbox_loc_->mutable_cpu_data();
  count = this->blob_bottom_conv6_2_mbox_loc_->count();
  CHECK_EQ(count, ssd_detection_input_data::conv6_2_mbox_loc.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::conv6_2_mbox_loc[i];
  }
  input_data = this->blob_bottom_conv7_2_mbox_loc_->mutable_cpu_data();
  count = this->blob_bottom_conv7_2_mbox_loc_->count();
  CHECK_EQ(count, ssd_detection_input_data::conv7_2_mbox_loc.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::conv7_2_mbox_loc[i];
  }
  input_data = this->blob_bottom_conv8_2_mbox_loc_->mutable_cpu_data();
  count = this->blob_bottom_conv8_2_mbox_loc_->count();
  CHECK_EQ(count, ssd_detection_input_data::conv8_2_mbox_loc.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::conv8_2_mbox_loc[i];
  }
  input_data = this->blob_bottom_pool6_mbox_loc_->mutable_cpu_data();
  count = this->blob_bottom_pool6_mbox_loc_->count();
  CHECK_EQ(count, ssd_detection_input_data::pool6_mbox_loc.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::pool6_mbox_loc[i];
  }
  input_data = this->blob_bottom_conv4_3_norm_mbox_conf_->mutable_cpu_data();
  count = this->blob_bottom_conv4_3_norm_mbox_conf_->count();
  CHECK_EQ(count, ssd_detection_input_data::conv4_3_norm_mbox_conf.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::conv4_3_norm_mbox_conf[i];
  }
  input_data = this->blob_bottom_fc7_mbox_conf_->mutable_cpu_data();
  count = this->blob_bottom_fc7_mbox_conf_->count();
  CHECK_EQ(count, ssd_detection_input_data::fc7_mbox_conf.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::fc7_mbox_conf[i];
  }
  input_data = this->blob_bottom_conv6_2_mbox_conf_->mutable_cpu_data();
  count = this->blob_bottom_conv6_2_mbox_conf_->count();
  CHECK_EQ(count, ssd_detection_input_data::conv6_2_mbox_conf.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::conv6_2_mbox_conf[i];
  }
  input_data = this->blob_bottom_conv7_2_mbox_conf_->mutable_cpu_data();
  count = this->blob_bottom_conv7_2_mbox_conf_->count();
  CHECK_EQ(count, ssd_detection_input_data::conv7_2_mbox_conf.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::conv7_2_mbox_conf[i];
  }
  input_data = this->blob_bottom_conv8_2_mbox_conf_->mutable_cpu_data();
  count = this->blob_bottom_conv8_2_mbox_conf_->count();
  CHECK_EQ(count, ssd_detection_input_data::conv8_2_mbox_conf.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::conv8_2_mbox_conf[i];
  }
  input_data = this->blob_bottom_pool6_mbox_conf_->mutable_cpu_data();
  count = this->blob_bottom_pool6_mbox_conf_->count();
  CHECK_EQ(count, ssd_detection_input_data::pool6_mbox_conf.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::pool6_mbox_conf[i];
  }
  input_data = this->blob_bottom_conv4_3_norm_->mutable_cpu_data();
  count = this->blob_bottom_conv4_3_norm_->count();
  CHECK_EQ(count, ssd_detection_input_data::conv4_3_norm.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::conv4_3_norm[i];
  }
  input_data = this->blob_bottom_fc7_xx_->mutable_cpu_data();
  count = this->blob_bottom_fc7_xx_->count();
  CHECK_EQ(count, ssd_detection_input_data::fc7.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::fc7[i];
  }
  input_data = this->blob_bottom_conv6_2_xx_->mutable_cpu_data();
  count = this->blob_bottom_conv6_2_xx_->count();
  CHECK_EQ(count, ssd_detection_input_data::conv6_2.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::conv6_2[i];
  }
  input_data = this->blob_bottom_conv7_2_xx_->mutable_cpu_data();
  count = this->blob_bottom_conv7_2_xx_->count();
  CHECK_EQ(count, ssd_detection_input_data::conv7_2.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::conv7_2[i];
  }
  input_data = this->blob_bottom_conv8_2_xx_->mutable_cpu_data();
  count = this->blob_bottom_conv8_2_xx_->count();
  CHECK_EQ(count, ssd_detection_input_data::conv8_2.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::conv8_2[i];
  }
  input_data = this->blob_bottom_pool6_->mutable_cpu_data();
  count = this->blob_bottom_pool6_->count();
  CHECK_EQ(count, ssd_detection_input_data::pool6.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::pool6[i];
  }
  input_data = this->blob_bottom_data_->mutable_cpu_data();
  count = this->blob_bottom_data_->count();
  CHECK_EQ(count, ssd_detection_input_data::input.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::input[i];
  }
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv4_3_norm_mbox_loc_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_fc7_mbox_loc_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv6_2_mbox_loc_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv7_2_mbox_loc_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv8_2_mbox_loc_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_pool6_mbox_loc_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv4_3_norm_mbox_conf_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_fc7_mbox_conf_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv6_2_mbox_conf_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv7_2_mbox_conf_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv8_2_mbox_conf_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_pool6_mbox_conf_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv4_3_norm_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_fc7_xx_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv6_2_xx_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv7_2_xx_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv8_2_xx_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_pool6_);
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
  auto prior_box_1 = layer_param.add_priorbox_params();
  auto prior_box_2 = layer_param.add_priorbox_params();
  auto prior_box_3 = layer_param.add_priorbox_params();
  auto prior_box_4 = layer_param.add_priorbox_params();
  auto prior_box_5 = layer_param.add_priorbox_params();
  auto prior_box_6 = layer_param.add_priorbox_params();
  prior_box_1->add_min_size(30.);
  prior_box_1->add_aspect_ratio(3.);
  prior_box_1->set_flip(true);
  prior_box_1->set_clip(true);
  prior_box_1->add_variance(0.1);
  prior_box_1->add_variance(0.1);
  prior_box_1->add_variance(0.2);
  prior_box_1->add_variance(0.2);
  prior_box_2->add_min_size(60.);
  prior_box_2->add_max_size(114.);
  prior_box_2->add_aspect_ratio(2.);
  prior_box_2->add_aspect_ratio(3.);
  prior_box_2->set_flip(true);
  prior_box_2->set_clip(true);
  prior_box_2->add_variance(0.1);
  prior_box_2->add_variance(0.1);
  prior_box_2->add_variance(0.2);
  prior_box_2->add_variance(0.2);
  prior_box_3->add_min_size(114.);
  prior_box_3->add_max_size(168.);
  prior_box_3->add_aspect_ratio(2.);
  prior_box_3->add_aspect_ratio(3.);
  prior_box_3->set_flip(true);
  prior_box_3->set_clip(true);
  prior_box_3->add_variance(0.1);
  prior_box_3->add_variance(0.1);
  prior_box_3->add_variance(0.2);
  prior_box_3->add_variance(0.2);
  prior_box_4->add_min_size(168.);
  prior_box_4->add_max_size(222.);
  prior_box_4->add_aspect_ratio(2.);
  prior_box_4->add_aspect_ratio(3.);
  prior_box_4->set_flip(true);
  prior_box_4->set_clip(true);
  prior_box_4->add_variance(0.1);
  prior_box_4->add_variance(0.1);
  prior_box_4->add_variance(0.2);
  prior_box_4->add_variance(0.2);
  prior_box_5->add_min_size(222.);
  prior_box_5->add_max_size(276.);
  prior_box_5->add_aspect_ratio(2.);
  prior_box_5->add_aspect_ratio(3.);
  prior_box_5->set_flip(true);
  prior_box_5->set_clip(true);
  prior_box_5->add_variance(0.1);
  prior_box_5->add_variance(0.1);
  prior_box_5->add_variance(0.2);
  prior_box_5->add_variance(0.2);
  prior_box_6->add_min_size(276.);
  prior_box_6->add_max_size(330.);
  prior_box_6->add_aspect_ratio(2.);
  prior_box_6->add_aspect_ratio(3.);
  prior_box_6->set_flip(true);
  prior_box_6->set_clip(true);
  prior_box_6->add_variance(0.1);
  prior_box_6->add_variance(0.1);
  prior_box_6->add_variance(0.2);
  prior_box_6->add_variance(0.2);
  MLUSsdDetectionLayer<Dtype> layer(layer_param);
  Caffe::setDetectOpMode(0);
  layer.set_int8_context(0);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  const Dtype* top_data = this->blob_top_->cpu_data();

  float err_sum = 0, sum = 0;
  ASSERT_EQ(this->blob_top_->count(),
            detection_output_data::output_data.size());
  for (int i = 0; i < this->blob_top_->count() / 6; i++) {
    if (top_data[i * 6 + 3] == 0) continue;
    for (int j=0; j < 6; j++) {
      int index = i * 6 + j;
      EXPECT_NEAR(top_data[index], detection_output_data::output_data[index], 5e-3);
      err_sum += std::abs(top_data[index] - detection_output_data::output_data[index]);
      sum += std::abs(top_data[index]);
    }
  }
  EXPECT_LE(err_sum / sum, 1e-4);
  ERR_RATE(err_sum / sum);
  EVENT_TIME(layer.get_event_time());
}


TYPED_TEST(MLUSsdDetectionLayerTest, TestForwardByBang) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.clear();
  this->blob_top_vec_.clear();
  Dtype* input_data;
  int count;
  input_data = this->blob_bottom_conv4_3_norm_mbox_loc_->mutable_cpu_data();
  count = this->blob_bottom_conv4_3_norm_mbox_loc_->count();
  CHECK_EQ(count, ssd_detection_input_data::conv4_3_norm_mbox_loc.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::conv4_3_norm_mbox_loc[i];
  }
  input_data = this->blob_bottom_fc7_mbox_loc_->mutable_cpu_data();
  count = this->blob_bottom_fc7_mbox_loc_->count();
  CHECK_EQ(count, ssd_detection_input_data::fc7_mbox_loc.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::fc7_mbox_loc[i];
  }
  input_data = this->blob_bottom_conv6_2_mbox_loc_->mutable_cpu_data();
  count = this->blob_bottom_conv6_2_mbox_loc_->count();
  CHECK_EQ(count, ssd_detection_input_data::conv6_2_mbox_loc.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::conv6_2_mbox_loc[i];
  }
  input_data = this->blob_bottom_conv7_2_mbox_loc_->mutable_cpu_data();
  count = this->blob_bottom_conv7_2_mbox_loc_->count();
  CHECK_EQ(count, ssd_detection_input_data::conv7_2_mbox_loc.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::conv7_2_mbox_loc[i];
  }
  input_data = this->blob_bottom_conv8_2_mbox_loc_->mutable_cpu_data();
  count = this->blob_bottom_conv8_2_mbox_loc_->count();
  CHECK_EQ(count, ssd_detection_input_data::conv8_2_mbox_loc.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::conv8_2_mbox_loc[i];
  }
  input_data = this->blob_bottom_pool6_mbox_loc_->mutable_cpu_data();
  count = this->blob_bottom_pool6_mbox_loc_->count();
  CHECK_EQ(count, ssd_detection_input_data::pool6_mbox_loc.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::pool6_mbox_loc[i];
  }
  input_data = this->blob_bottom_conv4_3_norm_mbox_conf_->mutable_cpu_data();
  count = this->blob_bottom_conv4_3_norm_mbox_conf_->count();
  CHECK_EQ(count, ssd_detection_input_data::conv4_3_norm_mbox_conf.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::conv4_3_norm_mbox_conf[i];
  }
  input_data = this->blob_bottom_fc7_mbox_conf_->mutable_cpu_data();
  count = this->blob_bottom_fc7_mbox_conf_->count();
  CHECK_EQ(count, ssd_detection_input_data::fc7_mbox_conf.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::fc7_mbox_conf[i];
  }
  input_data = this->blob_bottom_conv6_2_mbox_conf_->mutable_cpu_data();
  count = this->blob_bottom_conv6_2_mbox_conf_->count();
  CHECK_EQ(count, ssd_detection_input_data::conv6_2_mbox_conf.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::conv6_2_mbox_conf[i];
  }
  input_data = this->blob_bottom_conv7_2_mbox_conf_->mutable_cpu_data();
  count = this->blob_bottom_conv7_2_mbox_conf_->count();
  CHECK_EQ(count, ssd_detection_input_data::conv7_2_mbox_conf.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::conv7_2_mbox_conf[i];
  }
  input_data = this->blob_bottom_conv8_2_mbox_conf_->mutable_cpu_data();
  count = this->blob_bottom_conv8_2_mbox_conf_->count();
  CHECK_EQ(count, ssd_detection_input_data::conv8_2_mbox_conf.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::conv8_2_mbox_conf[i];
  }
  input_data = this->blob_bottom_pool6_mbox_conf_->mutable_cpu_data();
  count = this->blob_bottom_pool6_mbox_conf_->count();
  CHECK_EQ(count, ssd_detection_input_data::pool6_mbox_conf.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::pool6_mbox_conf[i];
  }
  input_data = this->blob_bottom_conv4_3_norm_->mutable_cpu_data();
  count = this->blob_bottom_conv4_3_norm_->count();
  CHECK_EQ(count, ssd_detection_input_data::conv4_3_norm.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::conv4_3_norm[i];
  }
  input_data = this->blob_bottom_fc7_xx_->mutable_cpu_data();
  count = this->blob_bottom_fc7_xx_->count();
  CHECK_EQ(count, ssd_detection_input_data::fc7.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::fc7[i];
  }
  input_data = this->blob_bottom_conv6_2_xx_->mutable_cpu_data();
  count = this->blob_bottom_conv6_2_xx_->count();
  CHECK_EQ(count, ssd_detection_input_data::conv6_2.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::conv6_2[i];
  }
  input_data = this->blob_bottom_conv7_2_xx_->mutable_cpu_data();
  count = this->blob_bottom_conv7_2_xx_->count();
  CHECK_EQ(count, ssd_detection_input_data::conv7_2.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::conv7_2[i];
  }
  input_data = this->blob_bottom_conv8_2_xx_->mutable_cpu_data();
  count = this->blob_bottom_conv8_2_xx_->count();
  CHECK_EQ(count, ssd_detection_input_data::conv8_2.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::conv8_2[i];
  }
  input_data = this->blob_bottom_pool6_->mutable_cpu_data();
  count = this->blob_bottom_pool6_->count();
  CHECK_EQ(count, ssd_detection_input_data::pool6.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::pool6[i];
  }
  input_data = this->blob_bottom_data_->mutable_cpu_data();
  count = this->blob_bottom_data_->count();
  CHECK_EQ(count, ssd_detection_input_data::input.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::input[i];
  }
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv4_3_norm_mbox_loc_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_fc7_mbox_loc_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv6_2_mbox_loc_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv7_2_mbox_loc_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv8_2_mbox_loc_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_pool6_mbox_loc_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv4_3_norm_mbox_conf_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_fc7_mbox_conf_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv6_2_mbox_conf_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv7_2_mbox_conf_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv8_2_mbox_conf_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_pool6_mbox_conf_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv4_3_norm_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_fc7_xx_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv6_2_xx_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv7_2_xx_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv8_2_xx_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_pool6_);
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
  auto prior_box_1 = layer_param.add_priorbox_params();
  auto prior_box_2 = layer_param.add_priorbox_params();
  auto prior_box_3 = layer_param.add_priorbox_params();
  auto prior_box_4 = layer_param.add_priorbox_params();
  auto prior_box_5 = layer_param.add_priorbox_params();
  auto prior_box_6 = layer_param.add_priorbox_params();
  prior_box_1->add_min_size(30.);
  prior_box_1->add_aspect_ratio(3.);
  prior_box_1->set_flip(true);
  prior_box_1->set_clip(true);
  prior_box_1->add_variance(0.1);
  prior_box_1->add_variance(0.1);
  prior_box_1->add_variance(0.2);
  prior_box_1->add_variance(0.2);
  prior_box_2->add_min_size(60.);
  prior_box_2->add_max_size(114.);
  prior_box_2->add_aspect_ratio(2.);
  prior_box_2->add_aspect_ratio(3.);
  prior_box_2->set_flip(true);
  prior_box_2->set_clip(true);
  prior_box_2->add_variance(0.1);
  prior_box_2->add_variance(0.1);
  prior_box_2->add_variance(0.2);
  prior_box_2->add_variance(0.2);
  prior_box_3->add_min_size(114.);
  prior_box_3->add_max_size(168.);
  prior_box_3->add_aspect_ratio(2.);
  prior_box_3->add_aspect_ratio(3.);
  prior_box_3->set_flip(true);
  prior_box_3->set_clip(true);
  prior_box_3->add_variance(0.1);
  prior_box_3->add_variance(0.1);
  prior_box_3->add_variance(0.2);
  prior_box_3->add_variance(0.2);
  prior_box_4->add_min_size(168.);
  prior_box_4->add_max_size(222.);
  prior_box_4->add_aspect_ratio(2.);
  prior_box_4->add_aspect_ratio(3.);
  prior_box_4->set_flip(true);
  prior_box_4->set_clip(true);
  prior_box_4->add_variance(0.1);
  prior_box_4->add_variance(0.1);
  prior_box_4->add_variance(0.2);
  prior_box_4->add_variance(0.2);
  prior_box_5->add_min_size(222.);
  prior_box_5->add_max_size(276.);
  prior_box_5->add_aspect_ratio(2.);
  prior_box_5->add_aspect_ratio(3.);
  prior_box_5->set_flip(true);
  prior_box_5->set_clip(true);
  prior_box_5->add_variance(0.1);
  prior_box_5->add_variance(0.1);
  prior_box_5->add_variance(0.2);
  prior_box_5->add_variance(0.2);
  prior_box_6->add_min_size(276.);
  prior_box_6->add_max_size(330.);
  prior_box_6->add_aspect_ratio(2.);
  prior_box_6->add_aspect_ratio(3.);
  prior_box_6->set_flip(true);
  prior_box_6->set_clip(true);
  prior_box_6->add_variance(0.1);
  prior_box_6->add_variance(0.1);
  prior_box_6->add_variance(0.2);
  prior_box_6->add_variance(0.2);
  MLUSsdDetectionLayer<Dtype> layer(layer_param);
  Caffe::setDetectOpMode(1);
  layer.set_int8_context(0);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  const Dtype* top_data = this->blob_top_->cpu_data();

  float err_sum = 0, sum = 0;
  ASSERT_EQ(this->blob_top_->count() - 64,
            detection_output_data::output_data_bang.size());
  int output_num = top_data[0];
  top_data += 64;
  for (int i = 0; i < output_num; i++) {
    vector<float> detection(top_data, top_data + 7);
    for (int j = 0; j < 7; j++) {
      EXPECT_NEAR(detection[j], detection_output_data::output_data_bang[j], 5e-3);
      err_sum += std::abs(top_data[j] - detection_output_data::output_data_bang[j]);
      sum += std::abs(top_data[j]);
    }
    top_data += 7;
  }
  EXPECT_LE(err_sum / sum, 1e-3);
  ERR_RATE(err_sum / sum);
  EVENT_TIME(layer.get_event_time());
}

template <typename TypeParam>
class MFUSSsdDetectionLayerTest : public MFUSDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MFUSSsdDetectionLayerTest()
      : blob_bottom_conv4_3_norm_mbox_loc_(new Blob<Dtype>(1, 12, 38, 38)),
        blob_bottom_fc7_mbox_loc_(new Blob<Dtype>(1, 24, 19, 19)),
        blob_bottom_conv6_2_mbox_loc_(new Blob<Dtype>(1, 24, 10, 10)),
        blob_bottom_conv7_2_mbox_loc_(new Blob<Dtype>(1, 24, 5, 5)),
        blob_bottom_conv8_2_mbox_loc_(new Blob<Dtype>(1, 24, 3, 3)),
        blob_bottom_pool6_mbox_loc_(new Blob<Dtype>(1, 24, 1, 1)),
        blob_bottom_conv4_3_norm_mbox_conf_(new Blob<Dtype>(1, 63, 38, 38)),
        blob_bottom_fc7_mbox_conf_(new Blob<Dtype>(1, 126, 19, 19)),
        blob_bottom_conv6_2_mbox_conf_(new Blob<Dtype>(1, 126, 10, 10)),
        blob_bottom_conv7_2_mbox_conf_(new Blob<Dtype>(1, 126, 5, 5)),
        blob_bottom_conv8_2_mbox_conf_(new Blob<Dtype>(1, 126, 3, 3)),
        blob_bottom_pool6_mbox_conf_(new Blob<Dtype>(1, 126, 1, 1)),
        blob_bottom_conv4_3_norm_(new Blob<Dtype>(1, 512, 38, 38)),
        blob_bottom_fc7_xx_(new Blob<Dtype>(1, 1024, 19, 19)),
        blob_bottom_conv6_2_xx_(new Blob<Dtype>(1, 512, 10, 10)),
        blob_bottom_conv7_2_xx_(new Blob<Dtype>(1, 256, 5, 5)),
        blob_bottom_conv8_2_xx_(new Blob<Dtype>(1, 256, 3, 3)),
        blob_bottom_pool6_(new Blob<Dtype>(1, 256, 1, 1)),
        blob_bottom_data_(new Blob<Dtype>(1, 3, 300, 300)),
        blob_top_(new Blob<Dtype>()) {}
  virtual ~MFUSSsdDetectionLayerTest() {
    delete blob_bottom_conv4_3_norm_mbox_loc_;
    delete blob_bottom_fc7_mbox_loc_;
    delete blob_bottom_conv6_2_mbox_loc_;
    delete blob_bottom_conv7_2_mbox_loc_;
    delete blob_bottom_conv8_2_mbox_loc_;
    delete blob_bottom_pool6_mbox_loc_;
    delete blob_bottom_conv4_3_norm_mbox_conf_;
    delete blob_bottom_fc7_mbox_conf_;
    delete blob_bottom_conv6_2_mbox_conf_;
    delete blob_bottom_conv7_2_mbox_conf_;
    delete blob_bottom_conv8_2_mbox_conf_;
    delete blob_bottom_pool6_mbox_conf_;
    delete blob_bottom_conv4_3_norm_;
    delete blob_bottom_fc7_xx_;
    delete blob_bottom_conv6_2_xx_;
    delete blob_bottom_conv7_2_xx_;
    delete blob_bottom_conv8_2_xx_;
    delete blob_bottom_pool6_;
    delete blob_bottom_data_;
    delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_conv4_3_norm_mbox_loc_;
  Blob<Dtype>* const blob_bottom_fc7_mbox_loc_;
  Blob<Dtype>* const blob_bottom_conv6_2_mbox_loc_;
  Blob<Dtype>* const blob_bottom_conv7_2_mbox_loc_;
  Blob<Dtype>* const blob_bottom_conv8_2_mbox_loc_;
  Blob<Dtype>* const blob_bottom_pool6_mbox_loc_;
  Blob<Dtype>* const blob_bottom_conv4_3_norm_mbox_conf_;
  Blob<Dtype>* const blob_bottom_fc7_mbox_conf_;
  Blob<Dtype>* const blob_bottom_conv6_2_mbox_conf_;
  Blob<Dtype>* const blob_bottom_conv7_2_mbox_conf_;
  Blob<Dtype>* const blob_bottom_conv8_2_mbox_conf_;
  Blob<Dtype>* const blob_bottom_pool6_mbox_conf_;
  Blob<Dtype>* const blob_bottom_conv4_3_norm_;
  Blob<Dtype>* const blob_bottom_fc7_xx_;
  Blob<Dtype>* const blob_bottom_conv6_2_xx_;
  Blob<Dtype>* const blob_bottom_conv7_2_xx_;
  Blob<Dtype>* const blob_bottom_conv8_2_xx_;
  Blob<Dtype>* const blob_bottom_pool6_;
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MFUSSsdDetectionLayerTest, TestMFUSDevices);

TYPED_TEST(MFUSSsdDetectionLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;

  LayerParameter layer_param;
  DetectionOutputParameter* detection_output_param =
      layer_param.mutable_detection_output_param();
  detection_output_param->set_keep_top_k(200);
  detection_output_param->set_num_classes(21);
  detection_output_param->set_share_location(true);
  detection_output_param->mutable_nms_param()->set_nms_threshold(0.45);
  MLUSsdDetectionLayer<Dtype> layer(layer_param);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv4_3_norm_mbox_loc_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_fc7_mbox_loc_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv6_2_mbox_loc_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv7_2_mbox_loc_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv8_2_mbox_loc_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_pool6_mbox_loc_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv4_3_norm_mbox_conf_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_fc7_mbox_conf_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv6_2_mbox_conf_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv7_2_mbox_conf_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv8_2_mbox_conf_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_pool6_mbox_conf_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv4_3_norm_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_fc7_xx_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv6_2_xx_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv7_2_xx_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv8_2_xx_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_pool6_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_data_);
  this->blob_top_vec_.push_back(this->blob_top_);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1464);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
}


TYPED_TEST(MFUSSsdDetectionLayerTest, TestForwardByCnml) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.clear();
  this->blob_top_vec_.clear();
  if (Caffe::rt_core() > 4) return;
  Dtype* input_data;
  int count;
  input_data = this->blob_bottom_conv4_3_norm_mbox_loc_->mutable_cpu_data();
  count = this->blob_bottom_conv4_3_norm_mbox_loc_->count();
  CHECK_EQ(count, ssd_detection_input_data::conv4_3_norm_mbox_loc.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::conv4_3_norm_mbox_loc[i];
  }
  input_data = this->blob_bottom_fc7_mbox_loc_->mutable_cpu_data();
  count = this->blob_bottom_fc7_mbox_loc_->count();
  CHECK_EQ(count, ssd_detection_input_data::fc7_mbox_loc.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::fc7_mbox_loc[i];
  }
  input_data = this->blob_bottom_conv6_2_mbox_loc_->mutable_cpu_data();
  count = this->blob_bottom_conv6_2_mbox_loc_->count();
  CHECK_EQ(count, ssd_detection_input_data::conv6_2_mbox_loc.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::conv6_2_mbox_loc[i];
  }
  input_data = this->blob_bottom_conv7_2_mbox_loc_->mutable_cpu_data();
  count = this->blob_bottom_conv7_2_mbox_loc_->count();
  CHECK_EQ(count, ssd_detection_input_data::conv7_2_mbox_loc.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::conv7_2_mbox_loc[i];
  }
  input_data = this->blob_bottom_conv8_2_mbox_loc_->mutable_cpu_data();
  count = this->blob_bottom_conv8_2_mbox_loc_->count();
  CHECK_EQ(count, ssd_detection_input_data::conv8_2_mbox_loc.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::conv8_2_mbox_loc[i];
  }
  input_data = this->blob_bottom_pool6_mbox_loc_->mutable_cpu_data();
  count = this->blob_bottom_pool6_mbox_loc_->count();
  CHECK_EQ(count, ssd_detection_input_data::pool6_mbox_loc.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::pool6_mbox_loc[i];
  }
  input_data = this->blob_bottom_conv4_3_norm_mbox_conf_->mutable_cpu_data();
  count = this->blob_bottom_conv4_3_norm_mbox_conf_->count();
  CHECK_EQ(count, ssd_detection_input_data::conv4_3_norm_mbox_conf.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::conv4_3_norm_mbox_conf[i];
  }
  input_data = this->blob_bottom_fc7_mbox_conf_->mutable_cpu_data();
  count = this->blob_bottom_fc7_mbox_conf_->count();
  CHECK_EQ(count, ssd_detection_input_data::fc7_mbox_conf.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::fc7_mbox_conf[i];
  }
  input_data = this->blob_bottom_conv6_2_mbox_conf_->mutable_cpu_data();
  count = this->blob_bottom_conv6_2_mbox_conf_->count();
  CHECK_EQ(count, ssd_detection_input_data::conv6_2_mbox_conf.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::conv6_2_mbox_conf[i];
  }
  input_data = this->blob_bottom_conv7_2_mbox_conf_->mutable_cpu_data();
  count = this->blob_bottom_conv7_2_mbox_conf_->count();
  CHECK_EQ(count, ssd_detection_input_data::conv7_2_mbox_conf.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::conv7_2_mbox_conf[i];
  }
  input_data = this->blob_bottom_conv8_2_mbox_conf_->mutable_cpu_data();
  count = this->blob_bottom_conv8_2_mbox_conf_->count();
  CHECK_EQ(count, ssd_detection_input_data::conv8_2_mbox_conf.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::conv8_2_mbox_conf[i];
  }
  input_data = this->blob_bottom_pool6_mbox_conf_->mutable_cpu_data();
  count = this->blob_bottom_pool6_mbox_conf_->count();
  CHECK_EQ(count, ssd_detection_input_data::pool6_mbox_conf.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::pool6_mbox_conf[i];
  }
  input_data = this->blob_bottom_conv4_3_norm_->mutable_cpu_data();
  count = this->blob_bottom_conv4_3_norm_->count();
  CHECK_EQ(count, ssd_detection_input_data::conv4_3_norm.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::conv4_3_norm[i];
  }
  input_data = this->blob_bottom_fc7_xx_->mutable_cpu_data();
  count = this->blob_bottom_fc7_xx_->count();
  CHECK_EQ(count, ssd_detection_input_data::fc7.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::fc7[i];
  }
  input_data = this->blob_bottom_conv6_2_xx_->mutable_cpu_data();
  count = this->blob_bottom_conv6_2_xx_->count();
  CHECK_EQ(count, ssd_detection_input_data::conv6_2.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::conv6_2[i];
  }
  input_data = this->blob_bottom_conv7_2_xx_->mutable_cpu_data();
  count = this->blob_bottom_conv7_2_xx_->count();
  CHECK_EQ(count, ssd_detection_input_data::conv7_2.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::conv7_2[i];
  }
  input_data = this->blob_bottom_conv8_2_xx_->mutable_cpu_data();
  count = this->blob_bottom_conv8_2_xx_->count();
  CHECK_EQ(count, ssd_detection_input_data::conv8_2.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::conv8_2[i];
  }
  input_data = this->blob_bottom_pool6_->mutable_cpu_data();
  count = this->blob_bottom_pool6_->count();
  CHECK_EQ(count, ssd_detection_input_data::pool6.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::pool6[i];
  }
  input_data = this->blob_bottom_data_->mutable_cpu_data();
  count = this->blob_bottom_data_->count();
  CHECK_EQ(count, ssd_detection_input_data::input.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::input[i];
  }
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv4_3_norm_mbox_loc_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_fc7_mbox_loc_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv6_2_mbox_loc_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv7_2_mbox_loc_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv8_2_mbox_loc_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_pool6_mbox_loc_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv4_3_norm_mbox_conf_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_fc7_mbox_conf_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv6_2_mbox_conf_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv7_2_mbox_conf_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv8_2_mbox_conf_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_pool6_mbox_conf_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv4_3_norm_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_fc7_xx_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv6_2_xx_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv7_2_xx_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv8_2_xx_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_pool6_);
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
  auto prior_box_1 = layer_param.add_priorbox_params();
  auto prior_box_2 = layer_param.add_priorbox_params();
  auto prior_box_3 = layer_param.add_priorbox_params();
  auto prior_box_4 = layer_param.add_priorbox_params();
  auto prior_box_5 = layer_param.add_priorbox_params();
  auto prior_box_6 = layer_param.add_priorbox_params();
  prior_box_1->add_min_size(30.);
  prior_box_1->add_aspect_ratio(2.);
  prior_box_1->set_flip(true);
  prior_box_1->set_clip(true);
  prior_box_1->add_variance(0.1);
  prior_box_1->add_variance(0.1);
  prior_box_1->add_variance(0.2);
  prior_box_1->add_variance(0.2);
  prior_box_2->add_min_size(60.);
  prior_box_2->add_max_size(114.);
  prior_box_2->add_aspect_ratio(2.);
  prior_box_2->add_aspect_ratio(3.);
  prior_box_2->set_flip(true);
  prior_box_2->set_clip(true);
  prior_box_2->add_variance(0.1);
  prior_box_2->add_variance(0.1);
  prior_box_2->add_variance(0.2);
  prior_box_2->add_variance(0.2);
  prior_box_3->add_min_size(114.);
  prior_box_3->add_max_size(168.);
  prior_box_3->add_aspect_ratio(2.);
  prior_box_3->add_aspect_ratio(3.);
  prior_box_3->set_flip(true);
  prior_box_3->set_clip(true);
  prior_box_3->add_variance(0.1);
  prior_box_3->add_variance(0.1);
  prior_box_3->add_variance(0.2);
  prior_box_3->add_variance(0.2);
  prior_box_4->add_min_size(168.);
  prior_box_4->add_max_size(222.);
  prior_box_4->add_aspect_ratio(2.);
  prior_box_4->add_aspect_ratio(3.);
  prior_box_4->set_flip(true);
  prior_box_4->set_clip(true);
  prior_box_4->add_variance(0.1);
  prior_box_4->add_variance(0.1);
  prior_box_4->add_variance(0.2);
  prior_box_4->add_variance(0.2);
  prior_box_5->add_min_size(222.);
  prior_box_5->add_max_size(276.);
  prior_box_5->add_aspect_ratio(2.);
  prior_box_5->add_aspect_ratio(3.);
  prior_box_5->set_flip(true);
  prior_box_5->set_clip(true);
  prior_box_5->add_variance(0.1);
  prior_box_5->add_variance(0.1);
  prior_box_5->add_variance(0.2);
  prior_box_5->add_variance(0.2);
  prior_box_6->add_min_size(276.);
  prior_box_6->add_max_size(330.);
  prior_box_6->add_aspect_ratio(2.);
  prior_box_6->add_aspect_ratio(3.);
  prior_box_6->set_flip(true);
  prior_box_6->set_clip(true);
  prior_box_6->add_variance(0.1);
  prior_box_6->add_variance(0.1);
  prior_box_6->add_variance(0.2);
  prior_box_6->add_variance(0.2);
  MLUSsdDetectionLayer<Dtype> layer(layer_param);
  Caffe::setDetectOpMode(0);
  layer.set_int8_context(0);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_TRUE(layer.mfus_supported());
  MFusion<Dtype> fuser;
  fuser.reset();
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  int bottom_nums = this->blob_bottom_vec_.size();
  int erase_priorbox_index = 2*(bottom_nums - 1) / 3;
  this->blob_bottom_vec_.erase(this->blob_bottom_vec_.begin() + erase_priorbox_index,
                               this->blob_bottom_vec_.end());
  fuser.addInputs(this->blob_bottom_vec_);
  fuser.addOutputs(this->blob_top_vec_);
  layer.fuse(&fuser);
  fuser.compile();
  fuser.forward();

  const Dtype* top_data = this->blob_top_->cpu_data();
  float err_sum = 0, sum = 0;
  ASSERT_EQ(this->blob_top_->count(),
            detection_output_data::output_data.size());
  for (int i = 0; i < this->blob_top_->count() / 6; i++) {
    if (top_data[i * 6 + 3] == 0) continue;
    for (int j=0; j < 6; j++) {
      int index = i * 6 + j;
      EXPECT_NEAR(top_data[index], detection_output_data::output_data[index], 5e-3);
      err_sum += std::abs(top_data[index] - detection_output_data::output_data[index]);
      sum += std::abs(top_data[index]);
    }
  }
  EXPECT_LE(err_sum / sum, 1e-4);
  ERR_RATE(err_sum / sum);
  EVENT_TIME(layer.get_event_time());
}

TYPED_TEST(MFUSSsdDetectionLayerTest, TestForwardByBang) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.clear();
  this->blob_top_vec_.clear();
  Dtype* input_data;
  int count;
  input_data = this->blob_bottom_conv4_3_norm_mbox_loc_->mutable_cpu_data();
  count = this->blob_bottom_conv4_3_norm_mbox_loc_->count();
  CHECK_EQ(count, ssd_detection_input_data::conv4_3_norm_mbox_loc.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::conv4_3_norm_mbox_loc[i];
  }
  input_data = this->blob_bottom_fc7_mbox_loc_->mutable_cpu_data();
  count = this->blob_bottom_fc7_mbox_loc_->count();
  CHECK_EQ(count, ssd_detection_input_data::fc7_mbox_loc.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::fc7_mbox_loc[i];
  }
  input_data = this->blob_bottom_conv6_2_mbox_loc_->mutable_cpu_data();
  count = this->blob_bottom_conv6_2_mbox_loc_->count();
  CHECK_EQ(count, ssd_detection_input_data::conv6_2_mbox_loc.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::conv6_2_mbox_loc[i];
  }
  input_data = this->blob_bottom_conv7_2_mbox_loc_->mutable_cpu_data();
  count = this->blob_bottom_conv7_2_mbox_loc_->count();
  CHECK_EQ(count, ssd_detection_input_data::conv7_2_mbox_loc.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::conv7_2_mbox_loc[i];
  }
  input_data = this->blob_bottom_conv8_2_mbox_loc_->mutable_cpu_data();
  count = this->blob_bottom_conv8_2_mbox_loc_->count();
  CHECK_EQ(count, ssd_detection_input_data::conv8_2_mbox_loc.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::conv8_2_mbox_loc[i];
  }
  input_data = this->blob_bottom_pool6_mbox_loc_->mutable_cpu_data();
  count = this->blob_bottom_pool6_mbox_loc_->count();
  CHECK_EQ(count, ssd_detection_input_data::pool6_mbox_loc.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::pool6_mbox_loc[i];
  }
  input_data = this->blob_bottom_conv4_3_norm_mbox_conf_->mutable_cpu_data();
  count = this->blob_bottom_conv4_3_norm_mbox_conf_->count();
  CHECK_EQ(count, ssd_detection_input_data::conv4_3_norm_mbox_conf.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::conv4_3_norm_mbox_conf[i];
  }
  input_data = this->blob_bottom_fc7_mbox_conf_->mutable_cpu_data();
  count = this->blob_bottom_fc7_mbox_conf_->count();
  CHECK_EQ(count, ssd_detection_input_data::fc7_mbox_conf.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::fc7_mbox_conf[i];
  }
  input_data = this->blob_bottom_conv6_2_mbox_conf_->mutable_cpu_data();
  count = this->blob_bottom_conv6_2_mbox_conf_->count();
  CHECK_EQ(count, ssd_detection_input_data::conv6_2_mbox_conf.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::conv6_2_mbox_conf[i];
  }
  input_data = this->blob_bottom_conv7_2_mbox_conf_->mutable_cpu_data();
  count = this->blob_bottom_conv7_2_mbox_conf_->count();
  CHECK_EQ(count, ssd_detection_input_data::conv7_2_mbox_conf.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::conv7_2_mbox_conf[i];
  }
  input_data = this->blob_bottom_conv8_2_mbox_conf_->mutable_cpu_data();
  count = this->blob_bottom_conv8_2_mbox_conf_->count();
  CHECK_EQ(count, ssd_detection_input_data::conv8_2_mbox_conf.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::conv8_2_mbox_conf[i];
  }
  input_data = this->blob_bottom_pool6_mbox_conf_->mutable_cpu_data();
  count = this->blob_bottom_pool6_mbox_conf_->count();
  CHECK_EQ(count, ssd_detection_input_data::pool6_mbox_conf.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::pool6_mbox_conf[i];
  }
  input_data = this->blob_bottom_conv4_3_norm_->mutable_cpu_data();
  count = this->blob_bottom_conv4_3_norm_->count();
  CHECK_EQ(count, ssd_detection_input_data::conv4_3_norm.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::conv4_3_norm[i];
  }
  input_data = this->blob_bottom_fc7_xx_->mutable_cpu_data();
  count = this->blob_bottom_fc7_xx_->count();
  CHECK_EQ(count, ssd_detection_input_data::fc7.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::fc7[i];
  }
  input_data = this->blob_bottom_conv6_2_xx_->mutable_cpu_data();
  count = this->blob_bottom_conv6_2_xx_->count();
  CHECK_EQ(count, ssd_detection_input_data::conv6_2.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::conv6_2[i];
  }
  input_data = this->blob_bottom_conv7_2_xx_->mutable_cpu_data();
  count = this->blob_bottom_conv7_2_xx_->count();
  CHECK_EQ(count, ssd_detection_input_data::conv7_2.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::conv7_2[i];
  }
  input_data = this->blob_bottom_conv8_2_xx_->mutable_cpu_data();
  count = this->blob_bottom_conv8_2_xx_->count();
  CHECK_EQ(count, ssd_detection_input_data::conv8_2.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::conv8_2[i];
  }
  input_data = this->blob_bottom_pool6_->mutable_cpu_data();
  count = this->blob_bottom_pool6_->count();
  CHECK_EQ(count, ssd_detection_input_data::pool6.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::pool6[i];
  }
  input_data = this->blob_bottom_data_->mutable_cpu_data();
  count = this->blob_bottom_data_->count();
  CHECK_EQ(count, ssd_detection_input_data::input.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = ssd_detection_input_data::input[i];
  }
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv4_3_norm_mbox_loc_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_fc7_mbox_loc_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv6_2_mbox_loc_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv7_2_mbox_loc_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv8_2_mbox_loc_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_pool6_mbox_loc_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv4_3_norm_mbox_conf_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_fc7_mbox_conf_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv6_2_mbox_conf_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv7_2_mbox_conf_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv8_2_mbox_conf_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_pool6_mbox_conf_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv4_3_norm_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_fc7_xx_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv6_2_xx_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv7_2_xx_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv8_2_xx_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_pool6_);
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
  auto prior_box_1 = layer_param.add_priorbox_params();
  auto prior_box_2 = layer_param.add_priorbox_params();
  auto prior_box_3 = layer_param.add_priorbox_params();
  auto prior_box_4 = layer_param.add_priorbox_params();
  auto prior_box_5 = layer_param.add_priorbox_params();
  auto prior_box_6 = layer_param.add_priorbox_params();
  prior_box_1->add_min_size(30.);
  prior_box_1->add_aspect_ratio(2.);
  prior_box_1->set_flip(true);
  prior_box_1->set_clip(true);
  prior_box_1->add_variance(0.1);
  prior_box_1->add_variance(0.1);
  prior_box_1->add_variance(0.2);
  prior_box_1->add_variance(0.2);
  prior_box_2->add_min_size(60.);
  prior_box_2->add_max_size(114.);
  prior_box_2->add_aspect_ratio(2.);
  prior_box_2->add_aspect_ratio(3.);
  prior_box_2->set_flip(true);
  prior_box_2->set_clip(true);
  prior_box_2->add_variance(0.1);
  prior_box_2->add_variance(0.1);
  prior_box_2->add_variance(0.2);
  prior_box_2->add_variance(0.2);
  prior_box_3->add_min_size(114.);
  prior_box_3->add_max_size(168.);
  prior_box_3->add_aspect_ratio(2.);
  prior_box_3->add_aspect_ratio(3.);
  prior_box_3->set_flip(true);
  prior_box_3->set_clip(true);
  prior_box_3->add_variance(0.1);
  prior_box_3->add_variance(0.1);
  prior_box_3->add_variance(0.2);
  prior_box_3->add_variance(0.2);
  prior_box_4->add_min_size(168.);
  prior_box_4->add_max_size(222.);
  prior_box_4->add_aspect_ratio(2.);
  prior_box_4->add_aspect_ratio(3.);
  prior_box_4->set_flip(true);
  prior_box_4->set_clip(true);
  prior_box_4->add_variance(0.1);
  prior_box_4->add_variance(0.1);
  prior_box_4->add_variance(0.2);
  prior_box_4->add_variance(0.2);
  prior_box_5->add_min_size(222.);
  prior_box_5->add_max_size(276.);
  prior_box_5->add_aspect_ratio(2.);
  prior_box_5->add_aspect_ratio(3.);
  prior_box_5->set_flip(true);
  prior_box_5->set_clip(true);
  prior_box_5->add_variance(0.1);
  prior_box_5->add_variance(0.1);
  prior_box_5->add_variance(0.2);
  prior_box_5->add_variance(0.2);
  prior_box_6->add_min_size(276.);
  prior_box_6->add_max_size(330.);
  prior_box_6->add_aspect_ratio(2.);
  prior_box_6->add_aspect_ratio(3.);
  prior_box_6->set_flip(true);
  prior_box_6->set_clip(true);
  prior_box_6->add_variance(0.1);
  prior_box_6->add_variance(0.1);
  prior_box_6->add_variance(0.2);
  prior_box_6->add_variance(0.2);
  MLUSsdDetectionLayer<Dtype> layer(layer_param);
  Caffe::setDetectOpMode(1);
  layer.set_int8_context(0);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_TRUE(layer.mfus_supported());
  MFusion<Dtype> fuser;
  fuser.reset();
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  int bottom_nums = this->blob_bottom_vec_.size();
  int erase_priorbox_index = 2*(bottom_nums - 1) / 3;
  this->blob_bottom_vec_.erase(this->blob_bottom_vec_.begin() + erase_priorbox_index,
                               this->blob_bottom_vec_.end());
  fuser.addInputs(this->blob_bottom_vec_);
  fuser.addOutputs(this->blob_top_vec_);
  layer.fuse(&fuser);
  fuser.compile();
  fuser.forward();

  const Dtype* top_data = this->blob_top_->cpu_data();
  float err_sum = 0, sum = 0;
  ASSERT_EQ(this->blob_top_->count() - 64,
            detection_output_data::output_data_bang.size());

  int output_num = top_data[0];
  top_data += 64;
  for (int i = 0; i < output_num; i++) {
    vector<float> detection(top_data, top_data + 7);
    for (int j = 0; j < 7; j++) {
      EXPECT_NEAR(detection[j], detection_output_data::output_data_bang[j], 5e-3);
      err_sum += std::abs(top_data[j] - detection_output_data::output_data_bang[j]);
      sum += std::abs(top_data[j]);
    }
    top_data += 7;
  }
  EXPECT_LE(err_sum / sum, 1e-3);
  ERR_RATE(err_sum / sum);
  EVENT_TIME(fuser.get_event_time());
}

#endif

}  // namespace caffe
