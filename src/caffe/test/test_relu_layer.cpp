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

#include <algorithm>
#include <vector>

#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"

#include "caffe/layers/relu_layer.hpp"

#ifdef USE_MLU
#include "caffe/layers/mlu_relu_layer.hpp"
#endif

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class ReLULayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  ReLULayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype>()) {
    Caffe::set_random_seed(1701);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~ReLULayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ReLULayerTest, TestDtypesAndDevices);

TYPED_TEST(ReLULayerTest, TestReLU) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ReLULayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const Dtype* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_GE(top_data[i], 0.);
    EXPECT_TRUE(top_data[i] == 0 || top_data[i] == bottom_data[i]);
  }
}

TYPED_TEST(ReLULayerTest, TestReLUGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ReLULayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3, 1701, 0., 0.01);
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
                               this->blob_top_vec_);
}

TYPED_TEST(ReLULayerTest, TestReLUWithNegativeSlope) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CHECK(google::protobuf::TextFormat::ParseFromString(
      "relu_param { negative_slope: 0.01 }", &layer_param));
  ReLULayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const Dtype* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    if (top_data[i] >= 0) {
      EXPECT_FLOAT_EQ(top_data[i], bottom_data[i]);
    } else {
      EXPECT_FLOAT_EQ(top_data[i], bottom_data[i] * 0.01);
    }
  }
}

TYPED_TEST(ReLULayerTest, TestReLUGradientWithNegativeSlope) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CHECK(google::protobuf::TextFormat::ParseFromString(
      "relu_param { negative_slope: 0.01 }", &layer_param));
  ReLULayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3, 1701, 0., 0.01);
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
                               this->blob_top_vec_);
}

#ifdef USE_MLU
template <typename TypeParam>
class MLUReLULayerTest : public MLUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MLUReLULayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype>()) {
    Caffe::set_random_seed(1701);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MLUReLULayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MLUReLULayerTest, TestMLUDevices);

TYPED_TEST(MLUReLULayerTest, TestReLUMLU) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MLUReLULayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const Dtype* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_GE(top_data[i], 0.);
    EXPECT_TRUE(top_data[i] == 0 || top_data[i] == bottom_data[i]);
  }
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  EVENT_TIME(layer.get_event_time());
}

TYPED_TEST(MLUReLULayerTest, TestReLUWithNegativeSlopeMLU) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CHECK(google::protobuf::TextFormat::ParseFromString(
      "relu_param { negative_slope: 0.01 }", &layer_param));
  MLUReLULayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const Dtype* top_data = this->blob_top_->cpu_data();
  const Dtype min_precisioin = 1e-2;
  float err_sum = 0, sum = 0;
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    if (top_data[i] >= 0) {
      EXPECT_NEAR(top_data[i], bottom_data[i], min_precisioin);
      err_sum += std::abs(top_data[i] - bottom_data[i]);
      sum += std::abs(bottom_data[i]);
    } else {
      EXPECT_NEAR(top_data[i], bottom_data[i] * 0.01, min_precisioin);
      err_sum += std::abs(top_data[i] - bottom_data[i] * 0.01);
      sum += std::abs(bottom_data[i] * 0.01);
    }
  }
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  ERR_RATE(err_sum/sum);
  EVENT_TIME(layer.get_event_time());
}

template <typename TypeParam>
class MFUSReLULayerTest : public MFUSDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MFUSReLULayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype>()) {
    Caffe::set_random_seed(1701);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MFUSReLULayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MFUSReLULayerTest, TestMFUSDevices);

TYPED_TEST(MFUSReLULayerTest, TestReLUMFUS) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MLUReLULayer<Dtype> layer(layer_param);
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
  // Now, check values
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const Dtype* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_GE(top_data[i], 0.);
    EXPECT_TRUE(top_data[i] == 0 || top_data[i] == bottom_data[i]);
  }
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  EVENT_TIME(layer.get_event_time());
}

TYPED_TEST(MFUSReLULayerTest, TestReLUWithNegativeSlopeMFUS) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CHECK(google::protobuf::TextFormat::ParseFromString(
      "relu_param { negative_slope: 0.01 }", &layer_param));
  MLUReLULayer<Dtype> layer(layer_param);
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
  // Now, check values
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const Dtype* top_data = this->blob_top_->cpu_data();
  const Dtype min_precisioin = 1e-2;
  float err_sum = 0, sum = 0;
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    if (top_data[i] >= 0) {
      EXPECT_NEAR(top_data[i], bottom_data[i], min_precisioin);
      err_sum += std::abs(top_data[i] - bottom_data[i]);
      sum += std::abs(bottom_data[i]);
    } else {
      EXPECT_NEAR(top_data[i], bottom_data[i] * 0.01, min_precisioin);
      err_sum += std::abs(top_data[i] - bottom_data[i] * 0.01);
      sum += std::abs(bottom_data[i] * 0.01);
    }
  }
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  ERR_RATE(err_sum/sum);
  EVENT_TIME(fuser.get_event_time());
}
#endif

}  // namespace caffe
