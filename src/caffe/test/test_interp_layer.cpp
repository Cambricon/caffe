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

#include <cmath>
#include <vector>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/interp_layer.hpp"
#include "caffe/layers/mlu_interp_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/util/interp.hpp"
#include "caffe/util/math_functions.hpp"
#include "gtest/gtest.h"

namespace caffe {

template <typename TypeParam>
class InterpLayerTest : public CPUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  InterpLayerTest()
      : blob_bottom_(new Blob<Dtype>(1, 32, 14, 14)),
        blob_top_(new Blob<Dtype>()),
        blob_expected_(new Blob<Dtype>()) {}

  virtual ~InterpLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
    delete blob_expected_;
  }

  virtual void SetUp() {
    // Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
    blob_expected_vec_.push_back(blob_expected_);
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_expected_;
  vector<Blob<Dtype>*> blob_expected_vec_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(InterpLayerTest, TestDtypesAndDevices);

TYPED_TEST(InterpLayerTest, TestSetUpShrink) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  InterpParameter* interp_param = layer_param.mutable_interp_param();
  interp_param->set_shrink_factor(4);
  InterpLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 32);
  EXPECT_EQ(this->blob_top_->height(), 4);
  EXPECT_EQ(this->blob_top_->width(), 4);
}

TYPED_TEST(InterpLayerTest, TestSetUpZoom) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  InterpParameter* interp_param = layer_param.mutable_interp_param();
  interp_param->set_zoom_factor(3);
  InterpLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 32);
  EXPECT_EQ(this->blob_top_->height(), 40);
  EXPECT_EQ(this->blob_top_->width(), 40);
}

TYPED_TEST(InterpLayerTest, TestSetUpShrinkZoom) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  InterpParameter* interp_param = layer_param.mutable_interp_param();
  interp_param->set_zoom_factor(3);
  interp_param->set_shrink_factor(4);
  InterpLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 32);
  EXPECT_EQ(this->blob_top_->height(), 10);
  EXPECT_EQ(this->blob_top_->width(), 10);
}

TYPED_TEST(InterpLayerTest, TestSetUpHeightWidth) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  InterpParameter* interp_param = layer_param.mutable_interp_param();
  interp_param->set_height(3);
  interp_param->set_width(4);
  InterpLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 32);
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 4);
}

TYPED_TEST(InterpLayerTest, ForwardShrink) {
  typedef typename TypeParam::Dtype Dtype;
  for (int i = 0; i < this->blob_bottom_->count(); i++) {
    this->blob_bottom_->mutable_cpu_data()[i] = i / 10.0 + 0.1;
  }
  LayerParameter layer_param;
  InterpParameter* interp_param = layer_param.mutable_interp_param();
  interp_param->set_shrink_factor(4);
  InterpLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 32);
  EXPECT_EQ(this->blob_top_->height(), 4);
  EXPECT_EQ(this->blob_top_->width(), 4);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* top_data = this->blob_top_->cpu_data();
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const int count = this->blob_top_->count();
  this->blob_expected_vec_.clear();
  this->blob_expected_->ReshapeLike(*this->blob_top_);
  this->blob_expected_vec_.push_back(this->blob_expected_);
  Dtype* expected_data = this->blob_expected_->mutable_cpu_data();
  caffe_cpu_interp2<Dtype>(
      this->blob_bottom_->num() * this->blob_bottom_->channels(), bottom_data,
      0, 0, this->blob_bottom_->height(), this->blob_bottom_->width(),
      this->blob_bottom_->height(), this->blob_bottom_->width(), expected_data,
      0, 0, 4, 4, 4, 4, false);

  for (int i = 0; i < count; i++) {
    EXPECT_NEAR(top_data[i], expected_data[i], 5e-5);
  }
}

TYPED_TEST(InterpLayerTest, ForwardZoom) {
  typedef typename TypeParam::Dtype Dtype;
  for (int i = 0; i < this->blob_bottom_->count(); i++) {
    this->blob_bottom_->mutable_cpu_data()[i] = i / 10.0 + 0.1;
  }
  LayerParameter layer_param;
  InterpParameter* interp_param = layer_param.mutable_interp_param();
  interp_param->set_zoom_factor(3);
  InterpLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 32);
  EXPECT_EQ(this->blob_top_->height(), 40);
  EXPECT_EQ(this->blob_top_->width(), 40);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* top_data = this->blob_top_->cpu_data();
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const int count = this->blob_top_->count();
  this->blob_expected_vec_.clear();
  this->blob_expected_->ReshapeLike(*this->blob_top_);
  this->blob_expected_vec_.push_back(this->blob_expected_);
  Dtype* expected_data = this->blob_expected_->mutable_cpu_data();
  caffe_cpu_interp2<Dtype>(
      this->blob_bottom_->num() * this->blob_bottom_->channels(), bottom_data,
      0, 0, this->blob_bottom_->height(), this->blob_bottom_->width(),
      this->blob_bottom_->height(), this->blob_bottom_->width(), expected_data,
      0, 0, 40, 40, 40, 40, false);

  for (int i = 0; i < count; i++) {
    EXPECT_NEAR(top_data[i], expected_data[i], 5e-5);
  }
}

TYPED_TEST(InterpLayerTest, ForwardShrinkZoom) {
  typedef typename TypeParam::Dtype Dtype;
  for (int i = 0; i < this->blob_bottom_->count(); i++) {
    this->blob_bottom_->mutable_cpu_data()[i] = i / 10.0 + 0.1;
  }
  LayerParameter layer_param;
  InterpParameter* interp_param = layer_param.mutable_interp_param();
  interp_param->set_zoom_factor(3);
  interp_param->set_shrink_factor(4);
  InterpLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 32);
  EXPECT_EQ(this->blob_top_->height(), 10);
  EXPECT_EQ(this->blob_top_->width(), 10);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* top_data = this->blob_top_->cpu_data();
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const int count = this->blob_top_->count();
  this->blob_expected_vec_.clear();
  this->blob_expected_->ReshapeLike(*this->blob_top_);
  this->blob_expected_vec_.push_back(this->blob_expected_);
  Dtype* expected_data = this->blob_expected_->mutable_cpu_data();
  caffe_cpu_interp2<Dtype>(
      this->blob_bottom_->num() * this->blob_bottom_->channels(), bottom_data,
      0, 0, this->blob_bottom_->height(), this->blob_bottom_->width(),
      this->blob_bottom_->height(), this->blob_bottom_->width(), expected_data,
      0, 0, 10, 10, 10, 10, false);

  for (int i = 0; i < count; i++) {
    EXPECT_NEAR(top_data[i], expected_data[i], 5e-5);
  }
}

TYPED_TEST(InterpLayerTest, ForwardHeightWidth) {
  typedef typename TypeParam::Dtype Dtype;
  for (int i = 0; i < this->blob_bottom_->count(); i++) {
    this->blob_bottom_->mutable_cpu_data()[i] = i / 10.0 + 0.1;
  }
  LayerParameter layer_param;
  InterpParameter* interp_param = layer_param.mutable_interp_param();
  interp_param->set_height(3);
  interp_param->set_width(4);
  InterpLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 32);
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 4);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* top_data = this->blob_top_->cpu_data();
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const int count = this->blob_top_->count();
  this->blob_expected_vec_.clear();
  this->blob_expected_->ReshapeLike(*this->blob_top_);
  this->blob_expected_vec_.push_back(this->blob_expected_);
  Dtype* expected_data = this->blob_expected_->mutable_cpu_data();
  caffe_cpu_interp2<Dtype>(
      this->blob_bottom_->num() * this->blob_bottom_->channels(), bottom_data,
      0, 0, this->blob_bottom_->height(), this->blob_bottom_->width(),
      this->blob_bottom_->height(), this->blob_bottom_->width(), expected_data,
      0, 0, 3, 4, 3, 4, false);

  for (int i = 0; i < count; i++) {
    EXPECT_NEAR(top_data[i], expected_data[i], 5e-5);
  }
}

#ifdef USE_MLU

template <typename TypeParam>
class MLUInterpLayerTest : public MLUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MLUInterpLayerTest()
      : blob_bottom_(new Blob<Dtype>(1, 32, 14, 14)),
        blob_top_(new Blob<Dtype>()),
        blob_expected_(new Blob<Dtype>()) {}

  virtual ~MLUInterpLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
    delete blob_expected_;
  }

  virtual void SetUp() {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
    blob_expected_vec_.push_back(blob_expected_);
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_expected_;
  vector<Blob<Dtype>*> blob_expected_vec_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MLUInterpLayerTest, TestMLUDevices);

TYPED_TEST(MLUInterpLayerTest, TestSetUpShrink) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  InterpParameter* interp_param = layer_param.mutable_interp_param();
  interp_param->set_shrink_factor(4);
  MLUInterpLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 32);
  EXPECT_EQ(this->blob_top_->height(), 4);
  EXPECT_EQ(this->blob_top_->width(), 4);
}

TYPED_TEST(MLUInterpLayerTest, TestSetUpZoom) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  InterpParameter* interp_param = layer_param.mutable_interp_param();
  interp_param->set_zoom_factor(3);
  MLUInterpLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 32);
  EXPECT_EQ(this->blob_top_->height(), 40);
  EXPECT_EQ(this->blob_top_->width(), 40);
}

TYPED_TEST(MLUInterpLayerTest, TestSetUpShrinkZoom) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  InterpParameter* interp_param = layer_param.mutable_interp_param();
  interp_param->set_zoom_factor(3);
  interp_param->set_shrink_factor(4);
  MLUInterpLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 32);
  EXPECT_EQ(this->blob_top_->height(), 10);
  EXPECT_EQ(this->blob_top_->width(), 10);
}

TYPED_TEST(MLUInterpLayerTest, TestSetUpHeightWidth) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  InterpParameter* interp_param = layer_param.mutable_interp_param();
  interp_param->set_height(3);
  interp_param->set_width(4);
  MLUInterpLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 32);
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 4);
}

TYPED_TEST(MLUInterpLayerTest, ForwardShrink) {
  typedef typename TypeParam::Dtype Dtype;
  for (int i = 0; i < this->blob_bottom_->count(); i++) {
    this->blob_bottom_->mutable_cpu_data()[i] = i / 10.0 + 0.1;
  }
  LayerParameter layer_param;
  InterpParameter* interp_param = layer_param.mutable_interp_param();
  interp_param->set_shrink_factor(4);
  MLUInterpLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 32);
  EXPECT_EQ(this->blob_top_->height(), 4);
  EXPECT_EQ(this->blob_top_->width(), 4);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* top_data = this->blob_top_->cpu_data();
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const int count = this->blob_top_->count();
  this->blob_expected_vec_.clear();
  this->blob_expected_->ReshapeLike(*this->blob_top_);
  this->blob_expected_vec_.push_back(this->blob_expected_);
  Dtype* expected_data = this->blob_expected_->mutable_cpu_data();
  caffe_cpu_interp2<Dtype>(
      this->blob_bottom_->num() * this->blob_bottom_->channels(), bottom_data,
      0, 0, this->blob_bottom_->height(), this->blob_bottom_->width(),
      this->blob_bottom_->height(), this->blob_bottom_->width(), expected_data,
      0, 0, 4, 4, 4, 4, false);
  float err_sum = 0, sum = 0;

  for (int i = 0; i < count; i++) {
    EXPECT_NEAR(top_data[i], expected_data[i], 1.5);
    err_sum += std::abs(top_data[i] - expected_data[i]);
    sum += std::abs(top_data[i]);
  }
  EXPECT_LE(err_sum / sum, 3e-3);
  std::ostringstream stream, param;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  param << "shrink_factor:" << interp_param -> shrink_factor();
  BOTTOM(stream);
  PARAM(param);
  ERR_RATE(err_sum/sum);
  EVENT_TIME(layer.get_event_time());
}

TYPED_TEST(MLUInterpLayerTest, ForwardZoom) {
  typedef typename TypeParam::Dtype Dtype;
  for (int i = 0; i < this->blob_bottom_->count(); i++) {
    this->blob_bottom_->mutable_cpu_data()[i] = i / 10.0 + 0.1;
  }
  LayerParameter layer_param;
  InterpParameter* interp_param = layer_param.mutable_interp_param();
  interp_param->set_zoom_factor(3);
  MLUInterpLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 32);
  EXPECT_EQ(this->blob_top_->height(), 40);
  EXPECT_EQ(this->blob_top_->width(), 40);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* top_data = this->blob_top_->cpu_data();
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const int count = this->blob_top_->count();
  this->blob_expected_vec_.clear();
  this->blob_expected_->ReshapeLike(*this->blob_top_);
  this->blob_expected_vec_.push_back(this->blob_expected_);
  Dtype* expected_data = this->blob_expected_->mutable_cpu_data();
  caffe_cpu_interp2<Dtype>(
      this->blob_bottom_->num() * this->blob_bottom_->channels(), bottom_data,
      0, 0, this->blob_bottom_->height(), this->blob_bottom_->width(),
      this->blob_bottom_->height(), this->blob_bottom_->width(), expected_data,
      0, 0, 40, 40, 40, 40, false);
  float err_sum = 0, sum = 0;

  for (int i = 0; i < count; i++) {
    EXPECT_NEAR(top_data[i], expected_data[i], 1.5);
    err_sum += std::abs(top_data[i] - expected_data[i]);
    sum += std::abs(top_data[i]);
  }
  EXPECT_LE(err_sum / sum, 3e-3);
  std::ostringstream stream, param;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  param << "zoom_factor:" << interp_param -> zoom_factor();
  BOTTOM(stream);
  PARAM(param);
  ERR_RATE(err_sum/sum);
  EVENT_TIME(layer.get_event_time());
}

TYPED_TEST(MLUInterpLayerTest, ForwardShrinkZoom) {
  typedef typename TypeParam::Dtype Dtype;
  for (int i = 0; i < this->blob_bottom_->count(); i++) {
    this->blob_bottom_->mutable_cpu_data()[i] = i / 10.0 + 0.1;
  }
  LayerParameter layer_param;
  InterpParameter* interp_param = layer_param.mutable_interp_param();
  interp_param->set_zoom_factor(3);
  interp_param->set_shrink_factor(4);
  MLUInterpLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 32);
  EXPECT_EQ(this->blob_top_->height(), 10);
  EXPECT_EQ(this->blob_top_->width(), 10);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* top_data = this->blob_top_->cpu_data();
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const int count = this->blob_top_->count();
  this->blob_expected_vec_.clear();
  this->blob_expected_->ReshapeLike(*this->blob_top_);
  this->blob_expected_vec_.push_back(this->blob_expected_);
  Dtype* expected_data = this->blob_expected_->mutable_cpu_data();
  caffe_cpu_interp2<Dtype>(
      this->blob_bottom_->num() * this->blob_bottom_->channels(), bottom_data,
      0, 0, this->blob_bottom_->height(), this->blob_bottom_->width(),
      this->blob_bottom_->height(), this->blob_bottom_->width(), expected_data,
      0, 0, 10, 10, 10, 10, false);
  float err_sum = 0, sum = 0;

  for (int i = 0; i < count; i++) {
    EXPECT_NEAR(top_data[i], expected_data[i], 2);
    err_sum += std::abs(top_data[i] - expected_data[i]);
    sum += std::abs(top_data[i]);
  }
  EXPECT_LE(err_sum / sum, 3e-3);
  std::ostringstream stream, param;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  param << "zoom_factor:" << interp_param -> zoom_factor() << "\t"
        << "shrink_factor:" << interp_param -> shrink_factor();
  BOTTOM(stream);
  PARAM(param);
  ERR_RATE(err_sum/sum);
  EVENT_TIME(layer.get_event_time());
}

TYPED_TEST(MLUInterpLayerTest, ForwardHeightWidth) {
  typedef typename TypeParam::Dtype Dtype;
  for (int i = 0; i < this->blob_bottom_->count(); i++) {
    this->blob_bottom_->mutable_cpu_data()[i] = i / 10.0 + 0.1;
  }
  LayerParameter layer_param;
  InterpParameter* interp_param = layer_param.mutable_interp_param();
  interp_param->set_height(3);
  interp_param->set_width(4);
  MLUInterpLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 32);
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 4);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* top_data = this->blob_top_->cpu_data();
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const int count = this->blob_top_->count();
  this->blob_expected_vec_.clear();
  this->blob_expected_->ReshapeLike(*this->blob_top_);
  this->blob_expected_vec_.push_back(this->blob_expected_);
  Dtype* expected_data = this->blob_expected_->mutable_cpu_data();
  caffe_cpu_interp2<Dtype>(
      this->blob_bottom_->num() * this->blob_bottom_->channels(), bottom_data,
      0, 0, this->blob_bottom_->height(), this->blob_bottom_->width(),
      this->blob_bottom_->height(), this->blob_bottom_->width(), expected_data,
      0, 0, 3, 4, 3, 4, false);
  float err_sum = 0, sum = 0;

  for (int i = 0; i < count; i++) {
    EXPECT_NEAR(top_data[i], expected_data[i], 1);
    err_sum += std::abs(top_data[i] - expected_data[i]);
    sum += std::abs(top_data[i]);
  }
  EXPECT_LE(err_sum / sum, 3e-3);
  std::ostringstream stream, param;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  param << "height:" << interp_param -> height()
        << "width:" << interp_param ->width();
  BOTTOM(stream);
  PARAM(param);
  ERR_RATE(err_sum/sum);
  EVENT_TIME(layer.get_event_time());
}

template <typename TypeParam>
class MFUSInterpLayerTest : public MFUSDeviceTest<TypeParam> {
typedef typename TypeParam :: Dtype Dtype;

  protected:
    MFUSInterpLayerTest()
      : blob_bottom_(new Blob<Dtype>(1, 32, 14, 14)),
        blob_top_(new Blob<Dtype>()),
        blob_expected_(new Blob<Dtype>()) {}

    virtual ~MFUSInterpLayerTest() {
      delete blob_bottom_;
      delete blob_top_;
      delete blob_expected_;
    }

    virtual void SetUp() {
      FillerParameter filler_param;
      GaussianFiller<Dtype> filler(filler_param);
      filler.Fill(this->blob_bottom_);
      blob_bottom_vec_.push_back(blob_bottom_);
      blob_top_vec_.push_back(blob_top_);
      blob_expected_vec_.push_back(blob_expected_);
    }

    Blob<Dtype>* const blob_bottom_;
    Blob<Dtype>* const blob_top_;
    Blob<Dtype>* const blob_expected_;
    vector<Blob<Dtype>*> blob_expected_vec_;
    vector<Blob<Dtype>*> blob_bottom_vec_;
    vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MFUSInterpLayerTest, TestMFUSDevices);

TYPED_TEST(MFUSInterpLayerTest, TestSetUpShrink) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  InterpParameter* interp_param = layer_param.mutable_interp_param();
  interp_param->set_shrink_factor(4);
  MLUInterpLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 32);
  EXPECT_EQ(this->blob_top_->height(), 4);
  EXPECT_EQ(this->blob_top_->width(), 4);
}

TYPED_TEST(MFUSInterpLayerTest, TestSetUpZoom) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  InterpParameter* interp_param = layer_param.mutable_interp_param();
  interp_param->set_zoom_factor(3);
  MLUInterpLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 32);
  EXPECT_EQ(this->blob_top_->height(), 40);
  EXPECT_EQ(this->blob_top_->width(), 40);
}

TYPED_TEST(MFUSInterpLayerTest, TestSetUpShrinkZoom) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  InterpParameter* interp_param = layer_param.mutable_interp_param();
  interp_param->set_zoom_factor(3);
  interp_param->set_shrink_factor(4);
  MLUInterpLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 32);
  EXPECT_EQ(this->blob_top_->height(), 10);
  EXPECT_EQ(this->blob_top_->width(), 10);
}

TYPED_TEST(MFUSInterpLayerTest, TestSetUpHeightWidth) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  InterpParameter* interp_param = layer_param.mutable_interp_param();
  interp_param->set_height(3);
  interp_param->set_width(4);
  MLUInterpLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 32);
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 4);
}

TYPED_TEST(MFUSInterpLayerTest, ForwardShrink) {
  typedef typename TypeParam::Dtype Dtype;
  for (int i = 0; i < this->blob_bottom_->count(); i++) {
    this->blob_bottom_->mutable_cpu_data()[i] = i / 10.0 + 0.1;
  }
  LayerParameter layer_param;
  InterpParameter* interp_param = layer_param.mutable_interp_param();
  interp_param->set_shrink_factor(4);
  MLUInterpLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_TRUE(layer.mfus_supported());

  MFusion<Dtype> fuser;
  fuser.reset();
  fuser.addInputs(this->blob_bottom_vec_);
  fuser.addOutputs(this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 32);
  EXPECT_EQ(this->blob_top_->height(), 4);
  EXPECT_EQ(this->blob_top_->width(), 4);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.fuse(&fuser);
  fuser.compile();
  fuser.forward();
  const Dtype* top_data = this->blob_top_->cpu_data();
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const int count = this->blob_top_->count();
  this->blob_expected_vec_.clear();
  this->blob_expected_->ReshapeLike(*this->blob_top_);
  this->blob_expected_vec_.push_back(this->blob_expected_);
  Dtype* expected_data = this->blob_expected_->mutable_cpu_data();
  caffe_cpu_interp2<Dtype>(
      this->blob_bottom_->num() * this->blob_bottom_->channels(), bottom_data,
      0, 0, this->blob_bottom_->height(), this->blob_bottom_->width(),
      this->blob_bottom_->height(), this->blob_bottom_->width(), expected_data,
      0, 0, 4, 4, 4, 4, false);
  float err_sum = 0, sum = 0;

  for (int i = 0; i < count; i++) {
    EXPECT_NEAR(top_data[i], expected_data[i], 1.5);
    err_sum += std::abs(top_data[i] - expected_data[i]);
    sum += std::abs(top_data[i]);
  }
  EXPECT_LE(err_sum / sum, 3e-3);
  std::ostringstream stream, param;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  param << "shrink_factor:" << interp_param -> shrink_factor();
  BOTTOM(stream);
  PARAM(param);
  ERR_RATE(err_sum/sum);
  EVENT_TIME(fuser.get_event_time());
}

TYPED_TEST(MFUSInterpLayerTest, ForwardZoom) {
  typedef typename TypeParam::Dtype Dtype;
  for (int i = 0; i < this->blob_bottom_->count(); i++) {
    this->blob_bottom_->mutable_cpu_data()[i] = i / 10.0 + 0.1;
  }
  LayerParameter layer_param;
  InterpParameter* interp_param = layer_param.mutable_interp_param();
  interp_param->set_zoom_factor(3);
  MLUInterpLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_TRUE(layer.mfus_supported());

  MFusion<Dtype> fuser;
  fuser.reset();
  fuser.addInputs(this->blob_bottom_vec_);
  fuser.addOutputs(this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 32);
  EXPECT_EQ(this->blob_top_->height(), 40);
  EXPECT_EQ(this->blob_top_->width(), 40);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.fuse(&fuser);
  fuser.compile();
  fuser.forward();
  const Dtype* top_data = this->blob_top_->cpu_data();
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const int count = this->blob_top_->count();
  this->blob_expected_vec_.clear();
  this->blob_expected_->ReshapeLike(*this->blob_top_);
  this->blob_expected_vec_.push_back(this->blob_expected_);
  Dtype* expected_data = this->blob_expected_->mutable_cpu_data();
  caffe_cpu_interp2<Dtype>(
      this->blob_bottom_->num() * this->blob_bottom_->channels(), bottom_data,
      0, 0, this->blob_bottom_->height(), this->blob_bottom_->width(),
      this->blob_bottom_->height(), this->blob_bottom_->width(), expected_data,
      0, 0, 40, 40, 40, 40, false);
  float err_sum = 0, sum = 0;

  for (int i = 0; i < count; i++) {
    EXPECT_NEAR(top_data[i], expected_data[i], 1.5);
    err_sum += std::abs(top_data[i] - expected_data[i]);
    sum += std::abs(top_data[i]);
  }
  EXPECT_LE(err_sum / sum, 3e-3);
  std::ostringstream stream, param;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  param << "shrink_factor:" << interp_param -> shrink_factor();
  BOTTOM(stream);
  PARAM(param);
  ERR_RATE(err_sum/sum);
  EVENT_TIME(layer.get_event_time());
}

TYPED_TEST(MFUSInterpLayerTest, ForwardShrinkZoom) {
  typedef typename TypeParam::Dtype Dtype;
  for (int i = 0; i < this->blob_bottom_->count(); i++) {
    this->blob_bottom_->mutable_cpu_data()[i] = i / 10.0 + 0.1;
  }
  LayerParameter layer_param;
  InterpParameter* interp_param = layer_param.mutable_interp_param();
  interp_param->set_zoom_factor(3);
  interp_param->set_shrink_factor(4);
  MLUInterpLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_TRUE(layer.mfus_supported());

  MFusion<Dtype> fuser;
  fuser.reset();
  fuser.addInputs(this->blob_bottom_vec_);
  fuser.addOutputs(this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 32);
  EXPECT_EQ(this->blob_top_->height(), 10);
  EXPECT_EQ(this->blob_top_->width(), 10);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.fuse(&fuser);
  fuser.compile();
  fuser.forward();
  const Dtype* top_data = this->blob_top_->cpu_data();
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const int count = this->blob_top_->count();
  this->blob_expected_vec_.clear();
  this->blob_expected_->ReshapeLike(*this->blob_top_);
  this->blob_expected_vec_.push_back(this->blob_expected_);
  Dtype* expected_data = this->blob_expected_->mutable_cpu_data();
  caffe_cpu_interp2<Dtype>(
      this->blob_bottom_->num() * this->blob_bottom_->channels(), bottom_data,
      0, 0, this->blob_bottom_->height(), this->blob_bottom_->width(),
      this->blob_bottom_->height(), this->blob_bottom_->width(), expected_data,
      0, 0, 10, 10, 10, 10, false);
  float err_sum = 0, sum = 0;

  for (int i = 0; i < count; i++) {
    EXPECT_NEAR(top_data[i], expected_data[i], 2);
    err_sum += std::abs(top_data[i] - expected_data[i]);
    sum += std::abs(top_data[i]);
  }
  EXPECT_LE(err_sum / sum, 3e-3);
  std::ostringstream stream, param;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  param << "shrink_factor:" << interp_param -> shrink_factor()
        << "zoom_factor:" << interp_param -> zoom_factor();
  BOTTOM(stream);
  PARAM(param);
  ERR_RATE(err_sum/sum);
  EVENT_TIME(layer.get_event_time());
}

TYPED_TEST(MFUSInterpLayerTest, ForwardHeightWidth) {
  typedef typename TypeParam::Dtype Dtype;
  for (int i = 0; i < this->blob_bottom_->count(); i++) {
    this->blob_bottom_->mutable_cpu_data()[i] = i / 10.0 + 0.1;
  }
  LayerParameter layer_param;
  InterpParameter* interp_param = layer_param.mutable_interp_param();
  interp_param->set_height(3);
  interp_param->set_width(4);
  MLUInterpLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_TRUE(layer.mfus_supported());

  MFusion<Dtype> fuser;
  fuser.reset();
  fuser.addInputs(this->blob_bottom_vec_);
  fuser.addOutputs(this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 32);
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 4);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.fuse(&fuser);
  fuser.compile();
  fuser.forward();
  const Dtype* top_data = this->blob_top_->cpu_data();
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const int count = this->blob_top_->count();
  this->blob_expected_vec_.clear();
  this->blob_expected_->ReshapeLike(*this->blob_top_);
  this->blob_expected_vec_.push_back(this->blob_expected_);
  Dtype* expected_data = this->blob_expected_->mutable_cpu_data();
  caffe_cpu_interp2<Dtype>(
      this->blob_bottom_->num() * this->blob_bottom_->channels(), bottom_data,
      0, 0, this->blob_bottom_->height(), this->blob_bottom_->width(),
      this->blob_bottom_->height(), this->blob_bottom_->width(), expected_data,
      0, 0, 3, 4, 3, 4, false);
  float err_sum = 0, sum = 0;

  for (int i = 0; i < count; i++) {
    EXPECT_NEAR(top_data[i], expected_data[i], 1);
    err_sum += std::abs(top_data[i] - expected_data[i]);
    sum += std::abs(top_data[i]);
  }
  EXPECT_LE(err_sum / sum, 3e-3);
  std::ostringstream stream, param;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  param << "height:" << interp_param -> height()
        << "width:" << interp_param ->width();
  BOTTOM(stream);
  PARAM(param);
  ERR_RATE(err_sum/sum);
  EVENT_TIME(layer.get_event_time());
}

#endif

}  // namespace caffe
