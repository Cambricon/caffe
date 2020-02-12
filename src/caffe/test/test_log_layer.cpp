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
#include "caffe/layers/log_layer.hpp"
#include "caffe/layers/mlu_log_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/util/math_functions.hpp"
#include "gtest/gtest.h"

namespace caffe {

template <typename TypeParam>
class LogLayerTest : public CPUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  LogLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 5, 8)),
        blob_top_(new Blob<Dtype>(2, 3, 5, 8)),
        blob_expected_(new Blob<Dtype>(2, 3, 5, 8)) {}

  virtual ~LogLayerTest() {
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

TYPED_TEST_CASE(LogLayerTest, TestDtypesAndDevices);

TYPED_TEST(LogLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  LogLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width());
}

TYPED_TEST(LogLayerTest, ForwardLog) {
  typedef typename TypeParam::Dtype Dtype;
  for (int i = 0; i < this->blob_bottom_->count(); i++) {
    this->blob_bottom_->mutable_cpu_data()[i] = i / 10.0 + 0.1;
  }
  this->blob_expected_vec_.clear();
  this->blob_expected_vec_.push_back(this->blob_expected_);
  LayerParameter layer_param;
  LogParameter* log_param = layer_param.mutable_log_param();
  log_param->set_scale(1);
  log_param->set_shift(0);
  LogLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* top_data = this->blob_top_->cpu_data();
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const int count = this->blob_bottom_->count();
  Dtype* expected_data = this->blob_expected_->mutable_cpu_data();
  caffe_log(count, bottom_data, expected_data);
  for (int i = 0; i < count; i++) {
    EXPECT_NEAR(top_data[i], expected_data[i], 5e-5);
  }
}

TYPED_TEST(LogLayerTest, ForwardScale) {
  typedef typename TypeParam::Dtype Dtype;
  for (int i = 0; i < this->blob_bottom_->count(); i++) {
    this->blob_bottom_->mutable_cpu_data()[i] = i / 10.0 + 0.1;
  }
  this->blob_expected_vec_.clear();
  this->blob_expected_vec_.push_back(this->blob_expected_);
  LayerParameter layer_param;
  LogParameter* log_param = layer_param.mutable_log_param();
  log_param->set_scale(2);
  log_param->set_shift(0);
  LogLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* top_data = this->blob_top_->cpu_data();
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const int count = this->blob_bottom_->count();
  Dtype* expected_data = this->blob_expected_->mutable_cpu_data();
  caffe_copy(count, bottom_data, expected_data);
  caffe_scal(count, Dtype(2), expected_data);
  caffe_log(count, expected_data, expected_data);
  for (int i = 0; i < count; i++) {
    EXPECT_NEAR(top_data[i], expected_data[i], 5e-5);
  }
}

TYPED_TEST(LogLayerTest, ForwardShift) {
  typedef typename TypeParam::Dtype Dtype;
  for (int i = 0; i < this->blob_bottom_->count(); i++) {
    this->blob_bottom_->mutable_cpu_data()[i] = i / 10.0 + 0.1;
  }
  this->blob_expected_vec_.clear();
  this->blob_expected_vec_.push_back(this->blob_expected_);
  LayerParameter layer_param;
  LogParameter* log_param = layer_param.mutable_log_param();
  log_param->set_scale(1);
  log_param->set_shift(2);
  LogLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* top_data = this->blob_top_->cpu_data();
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const int count = this->blob_bottom_->count();
  Dtype* expected_data = this->blob_expected_->mutable_cpu_data();
  caffe_copy(count, bottom_data, expected_data);
  caffe_add_scalar(count, Dtype(2), expected_data);
  caffe_log(count, expected_data, expected_data);
  for (int i = 0; i < count; i++) {
    EXPECT_NEAR(top_data[i], expected_data[i], 5e-5);
  }
}

TYPED_TEST(LogLayerTest, ForwardBaseScale) {
  typedef typename TypeParam::Dtype Dtype;
  for (int i = 0; i < this->blob_bottom_->count(); i++) {
    this->blob_bottom_->mutable_cpu_data()[i] = i / 10.0 + 0.1;
  }
  this->blob_expected_vec_.clear();
  this->blob_expected_vec_.push_back(this->blob_expected_);
  LayerParameter layer_param;
  LogParameter* log_param = layer_param.mutable_log_param();
  log_param->set_scale(1);
  log_param->set_shift(0);
  log_param->set_base(2);
  LogLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* top_data = this->blob_top_->cpu_data();
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const int count = this->blob_bottom_->count();
  Dtype* expected_data = this->blob_expected_->mutable_cpu_data();
  Dtype base_scale = Dtype(1) / log(2);
  caffe_log(count, bottom_data, expected_data);
  caffe_scal(count, base_scale, expected_data);
  for (int i = 0; i < count; i++) {
    EXPECT_NEAR(top_data[i], expected_data[i], 5e-5);
  }
}

#ifdef USE_MLU

template <typename TypeParam>
class MLULogLayerTest : public MLUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MLULogLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 5, 8)),
        blob_top_(new Blob<Dtype>(2, 3, 5, 8)),
        blob_expected_(new Blob<Dtype>(2, 3, 5, 8)) {}

  virtual ~MLULogLayerTest() {
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

TYPED_TEST_CASE(MLULogLayerTest, TestMLUDevices);

TYPED_TEST(MLULogLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  LogLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width());
}

TYPED_TEST(MLULogLayerTest, ForwardLog) {
  typedef typename TypeParam::Dtype Dtype;
  for (int i = 0; i < this->blob_bottom_->count(); i++) {
    this->blob_bottom_->mutable_cpu_data()[i] = i / 10.0 + 0.1;
  }
  this->blob_expected_vec_.clear();
  this->blob_expected_vec_.push_back(this->blob_expected_);
  LayerParameter layer_param;
  LogParameter* log_param = layer_param.mutable_log_param();
  log_param->set_scale(1);
  log_param->set_shift(0);
  MLULogLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* top_data = this->blob_top_->cpu_data();
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const int count = this->blob_bottom_->count();
  Dtype* expected_data = this->blob_expected_->mutable_cpu_data();
  caffe_log(count, bottom_data, expected_data);
  float err_sum = 0, sum = 0;
  for (int i = 0; i < count; i++) {
    EXPECT_NEAR(top_data[i], expected_data[i], 2e-2);
    err_sum += std::abs(top_data[i] - expected_data[i]);
    sum += std::abs(top_data[i]);
  }
  EXPECT_LE(err_sum / sum, 5e-3);
  std::ostringstream stream, param;
  param << "scale:" << log_param->scale() << "\t"
        << "shift:" << log_param->shift();
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  PARAM(param);
  ERR_RATE(err_sum/sum);
  EVENT_TIME(layer.get_event_time());
}

template <typename TypeParam>
class MFUSLogLayerTest : public MFUSDeviceTest<TypeParam> {
typedef typename TypeParam :: Dtype Dtype;

  protected:
    MFUSLogLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 1, 1, 1)),
        blob_top_(new Blob<Dtype>(2, 1, 1, 1)),
        blob_expected_(new Blob<Dtype>(2, 1, 1, 1)) {}
    virtual ~MFUSLogLayerTest() {
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

TYPED_TEST_CASE(MFUSLogLayerTest, TestMFUSDevices);

TYPED_TEST(MFUSLogLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  LogLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width());
}

TYPED_TEST(MFUSLogLayerTest, ForwardLog) {
  typedef typename TypeParam::Dtype Dtype;
  for (int i = 0; i < this->blob_bottom_->count(); i++) {
    this->blob_bottom_->mutable_cpu_data()[i] = i / 10.0 + 0.1;
  }
  this->blob_expected_vec_.clear();
  this->blob_expected_vec_.push_back(this->blob_expected_);
  LayerParameter layer_param;
  LogParameter* log_param = layer_param.mutable_log_param();
  log_param->set_scale(1);
  log_param->set_shift(0);
  MLULogLayer<Dtype> layer(layer_param);
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
  const Dtype* top_data = this->blob_top_->cpu_data();
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const int count = this->blob_bottom_->count();
  Dtype* expected_data = this->blob_expected_->mutable_cpu_data();
  caffe_log(count, bottom_data, expected_data);
  float err_sum = 0, sum = 0;
  for (int i = 0; i < count; i++) {
    EXPECT_NEAR(top_data[i], expected_data[i], 1e-2);
    err_sum += std::abs(top_data[i] - expected_data[i]);
    sum += std::abs(top_data[i]);
  }
  EXPECT_LE(err_sum / sum, 4e-3);
  std::ostringstream stream, param;
  param << "scale:" << log_param->scale() << "\t"
        << "shift:" << log_param->shift();
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  PARAM(param);
  ERR_RATE(err_sum/sum);
  EVENT_TIME(fuser.get_event_time());
}
#endif

}  // namespace caffe
