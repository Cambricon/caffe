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
#include "caffe/layers/mlu_yolov3_detection_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "gtest/gtest.h"
#include "yolov3_detection_input_data.hpp"

namespace caffe {

#ifdef USE_MLU
// yolov3_detection should have the same result with detection_out
// using the same input data scrap from example/yolov3 detecting one picture

template <typename TypeParam>
class MLUYolov3DetectionLayerTest : public MLUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MLUYolov3DetectionLayerTest()
      : blob_bottom_conv59_(new Blob<Dtype>(1, 255, 13, 13)),
        blob_bottom_conv67_(new Blob<Dtype>(1, 255, 26, 26)),
        blob_bottom_conv75_(new Blob<Dtype>(1, 255, 52, 52)),
        blob_top_(new Blob<Dtype>(1, 7 * 1024, 1, 1)) {}
  virtual ~MLUYolov3DetectionLayerTest() {
    delete blob_bottom_conv59_;
    delete blob_bottom_conv67_;
    delete blob_bottom_conv75_;
    delete blob_top_;
  }
  vector<float> biases_ = {116, 90, 156, 198, 373, 326, 30, 61, 62,
                           45,  59, 119, 10,  13,  16,  30, 33, 23};
  Blob<Dtype>* const blob_bottom_conv59_;
  Blob<Dtype>* const blob_bottom_conv67_;
  Blob<Dtype>* const blob_bottom_conv75_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MLUYolov3DetectionLayerTest, TestMLUDevices);

TYPED_TEST(MLUYolov3DetectionLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;

  LayerParameter layer_param;
  Yolov3DetectionParameter* yolov3_param = layer_param.mutable_yolov3_param();
  int num_classes = 80;
  int num_box = 1024;
  float confidence_thresh = 0.5;
  float nms_thresh = 0.45;
  int anchor_num = 3;
  int im_h = 416;
  int im_w = 416;
  yolov3_param->set_num_classes(num_classes);
  yolov3_param->set_num_box(num_box);
  yolov3_param->set_confidence_threshold(confidence_thresh);
  yolov3_param->set_nms_threshold(nms_thresh);
  yolov3_param->set_anchor_num(anchor_num);
  yolov3_param->set_im_w(im_w);
  yolov3_param->set_im_h(im_h);
  for (int i = 0; i < this->biases_.size(); i++) {
    yolov3_param->add_biases(this->biases_[i]);
  }
  MLUYolov3DetectionLayer<Dtype> layer(layer_param);
  layer.set_int8_context(0);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv59_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv67_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv75_);
  this->blob_top_vec_.push_back(this->blob_top_);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_bottom_conv59_->num() * (7 * num_box + 64),
            this->blob_top_->channels());
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

TYPED_TEST(MLUYolov3DetectionLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.clear();
  this->blob_top_vec_.clear();
  Dtype* input_data;
  int count;
  input_data = this->blob_bottom_conv59_->mutable_cpu_data();
  count = this->blob_bottom_conv59_->count();
  CHECK_EQ(count, yolov3_detection_input_data::conv59.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = yolov3_detection_input_data::conv59[i];
  }
  input_data = this->blob_bottom_conv67_->mutable_cpu_data();
  count = this->blob_bottom_conv67_->count();
  CHECK_EQ(count, yolov3_detection_input_data::conv67.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = yolov3_detection_input_data::conv67[i];
  }
  input_data = this->blob_bottom_conv75_->mutable_cpu_data();
  count = this->blob_bottom_conv75_->count();
  CHECK_EQ(count, yolov3_detection_input_data::conv75.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = yolov3_detection_input_data::conv75[i];
  }
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv59_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv67_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv75_);
  this->blob_top_vec_.push_back(this->blob_top_);
  LayerParameter layer_param;
  Yolov3DetectionParameter* yolov3_param = layer_param.mutable_yolov3_param();
  int num_classes = 80;
  int num_box = 1024;
  float confidence_thresh = 0.5;
  float nms_thresh = 0.45;
  int anchor_num = 3;
  int im_h = 416;
  int im_w = 416;
  yolov3_param->set_num_classes(num_classes);
  yolov3_param->set_num_box(num_box);
  yolov3_param->set_confidence_threshold(confidence_thresh);
  yolov3_param->set_nms_threshold(nms_thresh);
  yolov3_param->set_anchor_num(anchor_num);
  yolov3_param->set_im_w(im_w);
  yolov3_param->set_im_h(im_h);
  for (int i = 0; i < this->biases_.size(); i++) {
    yolov3_param->add_biases(this->biases_[i]);
  }
  MLUYolov3DetectionLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // compare data error
  Blob<Dtype> tmp_top_;
  tmp_top_.ReshapeLike(*this->blob_top_);
  vector<Blob<Dtype>*> top_vec;
  top_vec.push_back(&tmp_top_);
  Yolov3DetectionLayer<Dtype> cpu_layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, top_vec);
  layer.Reshape_dispatch(this->blob_bottom_vec_, top_vec);
  layer.Forward(this->blob_bottom_vec_, top_vec);

  const Dtype* top_data = this->blob_top_->cpu_data();
  const Dtype* tmp_data = tmp_top_.cpu_data();
  // campare boxes detected
  for (int n = 0; n < tmp_top_.num(); n++) {
    int num_boxes = tmp_data[0];
    int cpu_num = tmp_data[0];
    CHECK_EQ(cpu_num, num_boxes);
    for (int i = 0; i < cpu_num; i++) {
      int index = 64 + i * 7;
      if (tmp_data[index] == -1) continue;
      CHECK_EQ(top_data[index], tmp_data[index]);
      CHECK_EQ(top_data[index + 1], tmp_data[index + 1]);
      EXPECT_NEAR(top_data[index + 2], tmp_data[index + 2], 5e-2);
      EXPECT_NEAR(top_data[index + 3], tmp_data[index + 3], 5e-2);
      EXPECT_NEAR(top_data[index + 4], tmp_data[index + 4], 5e-2);
      EXPECT_NEAR(top_data[index + 5], tmp_data[index + 5], 5e-2);
      EXPECT_NEAR(top_data[index + 6], tmp_data[index + 6], 5e-2);
    }
  }
}

template <typename TypeParam>
class MFUSYolov3DetectionLayerTest : public MFUSDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MFUSYolov3DetectionLayerTest()
      : blob_bottom_conv59_(new Blob<Dtype>(1, 255, 13, 13)),
        blob_bottom_conv67_(new Blob<Dtype>(1, 255, 26, 26)),
        blob_bottom_conv75_(new Blob<Dtype>(1, 255, 52, 52)),
        blob_top_(new Blob<Dtype>(1, 7 * 1024, 1, 1)) {}
  virtual ~MFUSYolov3DetectionLayerTest() {
    delete blob_bottom_conv59_;
    delete blob_bottom_conv67_;
    delete blob_bottom_conv75_;
    delete blob_top_;
  }

  vector<float> biases_ = {116, 90, 156, 198, 373, 326, 30, 61, 62,
                           45,  59, 119, 10,  13,  16,  30, 33, 23};
  Blob<Dtype>* const blob_bottom_conv59_;
  Blob<Dtype>* const blob_bottom_conv67_;
  Blob<Dtype>* const blob_bottom_conv75_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MFUSYolov3DetectionLayerTest, TestMFUSDevices);

TYPED_TEST(MFUSYolov3DetectionLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  Yolov3DetectionParameter* yolov3_param = layer_param.mutable_yolov3_param();
  int num_classes = 80;
  int num_box = 1024;
  float confidence_thresh = 0.5;
  float nms_thresh = 0.45;
  int anchor_num = 3;
  int im_h = 416;
  int im_w = 416;
  yolov3_param->set_num_classes(num_classes);
  yolov3_param->set_num_box(num_box);
  yolov3_param->set_confidence_threshold(confidence_thresh);
  yolov3_param->set_nms_threshold(nms_thresh);
  yolov3_param->set_anchor_num(anchor_num);
  yolov3_param->set_im_w(im_w);
  yolov3_param->set_im_h(im_h);
  for (int i = 0; i < this->biases_.size(); i++) {
    yolov3_param->add_biases(this->biases_[i]);
  }
  MLUYolov3DetectionLayer<Dtype> layer(layer_param);
  layer.set_int8_context(0);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv59_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv67_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv75_);
  this->blob_top_vec_.push_back(this->blob_top_);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_bottom_conv59_->num() * (7 * num_box + 64),
            this->blob_top_->channels());
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

TYPED_TEST(MFUSYolov3DetectionLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.clear();
  this->blob_top_vec_.clear();
  Dtype* input_data;
  int count;
  input_data = this->blob_bottom_conv59_->mutable_cpu_data();
  count = this->blob_bottom_conv59_->count();
  CHECK_EQ(count, yolov3_detection_input_data::conv59.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = yolov3_detection_input_data::conv59[i];
  }
  input_data = this->blob_bottom_conv67_->mutable_cpu_data();
  count = this->blob_bottom_conv67_->count();
  CHECK_EQ(count, yolov3_detection_input_data::conv67.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = yolov3_detection_input_data::conv67[i];
  }
  input_data = this->blob_bottom_conv75_->mutable_cpu_data();
  count = this->blob_bottom_conv75_->count();
  CHECK_EQ(count, yolov3_detection_input_data::conv75.size());
  for (unsigned int i = 0; i < count; ++i) {
    input_data[i] = yolov3_detection_input_data::conv75[i];
  }
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv59_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv67_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_conv75_);
  this->blob_top_vec_.push_back(this->blob_top_);
  LayerParameter layer_param;
  Yolov3DetectionParameter* yolov3_param = layer_param.mutable_yolov3_param();
  int num_classes = 80;
  int num_box = 1024;
  float confidence_thresh = 0.5;
  float nms_thresh = 0.45;
  int anchor_num = 3;
  int im_h = 416;
  int im_w = 416;
  yolov3_param->set_num_classes(num_classes);
  yolov3_param->set_num_box(num_box);
  yolov3_param->set_confidence_threshold(confidence_thresh);
  yolov3_param->set_nms_threshold(nms_thresh);
  yolov3_param->set_anchor_num(anchor_num);
  yolov3_param->set_im_w(im_w);
  yolov3_param->set_im_h(im_h);
  for (int i = 0; i < this->biases_.size(); i++) {
    yolov3_param->add_biases(this->biases_[i]);
  }
  MLUYolov3DetectionLayer<Dtype> layer(layer_param);
  layer.set_int8_context(0);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_TRUE(layer.mfus_supported());
  MFusion<Dtype> fuser;
  fuser.reset();
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  fuser.addInputs(this->blob_bottom_vec_);
  fuser.addOutputs(this->blob_top_vec_);
  layer.fuse(&fuser);
  fuser.compile();
  fuser.forward();

  Blob<Dtype> tmp_top_;
  tmp_top_.ReshapeLike(*this->blob_top_);
  vector<Blob<Dtype>*> top_vec;
  top_vec.push_back(&tmp_top_);
  Yolov3DetectionLayer<Dtype> cpu_layer(layer_param);
  cpu_layer.SetUp(this->blob_bottom_vec_, top_vec);
  cpu_layer.Reshape_dispatch(this->blob_bottom_vec_, top_vec);
  cpu_layer.Forward(this->blob_bottom_vec_, top_vec);

  const Dtype* top_data = this->blob_top_->cpu_data();
  const Dtype* tmp_data = tmp_top_.cpu_data();
  // campare boxes detected
  for (int n = 0; n < tmp_top_.num(); n++) {
    int num_boxes = tmp_data[0];
    int cpu_num = tmp_data[0];
    CHECK_EQ(cpu_num, num_boxes);
    for (int i = 0; i < cpu_num; i++) {
      int index = 64 + i * 7;
      if (tmp_data[index] == -1) continue;
      CHECK_EQ(top_data[index], tmp_data[index]);
      CHECK_EQ(top_data[index + 1], tmp_data[index + 1]);
      EXPECT_NEAR(top_data[index + 2], tmp_data[index + 2], 5e-2);
      EXPECT_NEAR(top_data[index + 3], tmp_data[index + 3], 5e-2);
      EXPECT_NEAR(top_data[index + 4], tmp_data[index + 4], 5e-2);
      EXPECT_NEAR(top_data[index + 5], tmp_data[index + 5], 5e-2);
      EXPECT_NEAR(top_data[index + 6], tmp_data[index + 6], 5e-2);
    }
  }
}

#endif

}  // namespace caffe
