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
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/mlu_power_layer.hpp"
#include "caffe/layers/power_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include "gtest/gtest.h"

namespace caffe {

template <typename TypeParam>
class PowerLayerTest : public CPUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  PowerLayerTest()
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
  virtual ~PowerLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  void TestForward(Dtype power, Dtype scale, Dtype shift) {
    LayerParameter layer_param;
    layer_param.mutable_power_param()->set_power(power);
    layer_param.mutable_power_param()->set_scale(scale);
    layer_param.mutable_power_param()->set_shift(shift);
    PowerLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    // Now, check values
    const Dtype* bottom_data = this->blob_bottom_->cpu_data();
    const Dtype* top_data = this->blob_top_->cpu_data();
    const Dtype min_precision = 1e-5;
    for (int i = 0; i < this->blob_bottom_->count(); ++i) {
      Dtype expected_value = pow(shift + scale * bottom_data[i], power);
      if (power == Dtype(0) || power == Dtype(1) || power == Dtype(2)) {
        EXPECT_FALSE(isnan(top_data[i]));
      }
      if (isnan(expected_value)) {
        EXPECT_TRUE(isnan(top_data[i]));
      } else {
        Dtype precision = std::max(
            Dtype(std::abs(expected_value * Dtype(1e-4))), min_precision);
        EXPECT_NEAR(expected_value, top_data[i], precision);
      }
    }
  }

  void TestBackward(Dtype power, Dtype scale, Dtype shift) {
    LayerParameter layer_param;
    layer_param.mutable_power_param()->set_power(power);
    layer_param.mutable_power_param()->set_scale(scale);
    layer_param.mutable_power_param()->set_shift(shift);
    PowerLayer<Dtype> layer(layer_param);
    if (power != Dtype(0) && power != Dtype(1) && power != Dtype(2)) {
      // Avoid NaNs by forcing (shift + scale * x) >= 0
      Dtype* bottom_data = this->blob_bottom_->mutable_cpu_data();
      Dtype min_value = -shift / scale;
      for (int i = 0; i < this->blob_bottom_->count(); ++i) {
        if (bottom_data[i] < min_value) {
          bottom_data[i] = min_value + (min_value - bottom_data[i]);
        }
      }
    }
    GradientChecker<Dtype> checker(1e-3, 1e-2, 1701, 0., 0.01);
    checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
                                 this->blob_top_vec_);
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(PowerLayerTest, TestDtypesAndDevices);

TYPED_TEST(PowerLayerTest, TestPower) {
  typedef typename TypeParam::Dtype Dtype;
  Dtype power = 0.37;
  Dtype scale = 0.83;
  Dtype shift = -2.4;
  this->TestForward(power, scale, shift);
}

TYPED_TEST(PowerLayerTest, TestPowerGradient) {
  typedef typename TypeParam::Dtype Dtype;
  Dtype power = 0.37;
  Dtype scale = 0.83;
  Dtype shift = -2.4;
  this->TestBackward(power, scale, shift);
}

TYPED_TEST(PowerLayerTest, TestPowerGradientShiftZero) {
  typedef typename TypeParam::Dtype Dtype;
  Dtype power = 0.37;
  Dtype scale = 0.83;
  Dtype shift = 0.0;
  this->TestBackward(power, scale, shift);
}

TYPED_TEST(PowerLayerTest, TestPowerZero) {
  typedef typename TypeParam::Dtype Dtype;
  Dtype power = 0.0;
  Dtype scale = 0.83;
  Dtype shift = -2.4;
  this->TestForward(power, scale, shift);
}

TYPED_TEST(PowerLayerTest, TestPowerZeroGradient) {
  typedef typename TypeParam::Dtype Dtype;
  Dtype power = 0.0;
  Dtype scale = 0.83;
  Dtype shift = -2.4;
  this->TestBackward(power, scale, shift);
}

TYPED_TEST(PowerLayerTest, TestPowerOne) {
  typedef typename TypeParam::Dtype Dtype;
  Dtype power = 1.0;
  Dtype scale = 0.83;
  Dtype shift = -2.4;
  this->TestForward(power, scale, shift);
}

TYPED_TEST(PowerLayerTest, TestPowerOneGradient) {
  typedef typename TypeParam::Dtype Dtype;
  Dtype power = 1.0;
  Dtype scale = 0.83;
  Dtype shift = -2.4;
  this->TestBackward(power, scale, shift);
}

TYPED_TEST(PowerLayerTest, TestPowerTwo) {
  typedef typename TypeParam::Dtype Dtype;
  Dtype power = 2.0;
  Dtype scale = 0.34;
  Dtype shift = -2.4;
  this->TestForward(power, scale, shift);
}

TYPED_TEST(PowerLayerTest, TestPowerTwoGradient) {
  typedef typename TypeParam::Dtype Dtype;
  Dtype power = 2.0;
  Dtype scale = 0.83;
  Dtype shift = -2.4;
  this->TestBackward(power, scale, shift);
}

TYPED_TEST(PowerLayerTest, TestPowerTwoScaleHalfGradient) {
  typedef typename TypeParam::Dtype Dtype;
  Dtype power = 2.0;
  Dtype scale = 0.5;
  Dtype shift = -2.4;
  this->TestBackward(power, scale, shift);
}

#ifdef USE_MLU

template <typename TypeParam>
class MLUPowerLayerTest : public MLUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MLUPowerLayerTest()
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
  virtual ~MLUPowerLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  void TestForward(Dtype power, Dtype scale, Dtype shift) {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    this->blob_bottom_vec_.clear();
    this->blob_top_vec_.clear();
    this->blob_bottom_vec_.push_back(this->blob_bottom_);
    this->blob_top_vec_.push_back(this->blob_top_);
    LayerParameter layer_param;
    layer_param.mutable_power_param()->set_power(power);
    layer_param.mutable_power_param()->set_scale(scale);
    layer_param.mutable_power_param()->set_shift(shift);
    MLUPowerLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    // Now, check values
    const Dtype* bottom_data = this->blob_bottom_->cpu_data();
    const Dtype* top_data = this->blob_top_->cpu_data();
    for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    // Dtype expected_value = pow(shift + scale * bottom_data[i], power);
      float sign = 1;
      float old_flag = power - 2*floor(power / 2);
      sign = ((shift + scale * bottom_data[i]) < 0?-1:1);
      Dtype expected_value = powf(fabs(shift + scale * bottom_data[i]), power);
      if (old_flag != 0) {
         expected_value = expected_value * sign;
      }
      if (power == Dtype(0) || power == Dtype(1) || power == Dtype(2)) {
        EXPECT_FALSE(isnan(top_data[i]));
      }
      Dtype err = std::abs((expected_value - top_data[i]) / expected_value);
      if (sign != -1) {
          EXPECT_LT(err, 5e-2);
          ERR_RATE(err);
      }
    }
    std::ostringstream stream, param;
    stream << "bottom1:" << blob_bottom_->shape_string().c_str();
    param << "power:" << power << "\t" << "scale:" << scale << "\t"
          << "shifit:" << shift;
    BOTTOM(stream);
    PARAM(param);
    EVENT_TIME(layer.get_event_time());
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MLUPowerLayerTest, TestMLUDevices);

TYPED_TEST(MLUPowerLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  this->blob_bottom_vec_.clear();
  this->blob_top_vec_.clear();
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  this->blob_top_vec_.push_back(this->blob_top_);
  LayerParameter layer_param;
  MLUPowerLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width());
}

TYPED_TEST(MLUPowerLayerTest, TestPower) {
  typedef typename TypeParam::Dtype Dtype;
  Dtype power = 0.37;
  Dtype scale = 0.83;
  Dtype shift = 1;
  this->TestForward(power, scale, shift);
}

TYPED_TEST(MLUPowerLayerTest, TestPowerZero) {
  typedef typename TypeParam::Dtype Dtype;
  Dtype power = 0.0;
  Dtype scale = 0.83;
  Dtype shift = 100;
  this->TestForward(power, scale, shift);
}

TYPED_TEST(MLUPowerLayerTest, TestPowerOne) {
  typedef typename TypeParam::Dtype Dtype;
  Dtype power = 1.0;
  Dtype scale = 0.83;
  Dtype shift = 0;
  this->TestForward(power, scale, shift);
}

TYPED_TEST(MLUPowerLayerTest, TestPowerTwo) {
  typedef typename TypeParam::Dtype Dtype;
  Dtype power = 2.0;
  Dtype scale = 0.34;
  Dtype shift = 2.4;
  this->TestForward(power, scale, shift);
}

template <typename TypeParam>
class MFUSPowerLayerTest : public MFUSDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MFUSPowerLayerTest()
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
  virtual ~MFUSPowerLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  void TestForward(Dtype power, Dtype scale, Dtype shift) {
    LayerParameter layer_param;
    layer_param.mutable_power_param()->set_power(power);
    layer_param.mutable_power_param()->set_scale(scale);
    layer_param.mutable_power_param()->set_shift(shift);
    MLUPowerLayer<Dtype> layer(layer_param);
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
    const Dtype min_precision = 1e-5;
    for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    // Dtype expected_value = pow(shift + scale * bottom_data[i], power);
      float sign = 1;
      float old_flag = power - 2*floor(power / 2);
      sign = ((shift + scale * bottom_data[i]) < 0?-1:1);
      Dtype expected_value = powf(fabs(shift + scale * bottom_data[i]), power);
      if (old_flag != 0) {
         expected_value = expected_value * sign;
      }
      if (power == Dtype(0) || power == Dtype(1) || power == Dtype(2)) {
        EXPECT_FALSE(isnan(top_data[i]));
      }
      Dtype err = std::abs((expected_value - top_data[i]) / expected_value);
      if (sign != -1) {
          EXPECT_LT(err, 5e-2);
          ERR_RATE(err);
      }
    }
    std::ostringstream stream, param;
    stream << "bottom1:" << blob_bottom_->shape_string().c_str();
    param << "power:" << power << "\t" << "scale:" << scale << "\t"
                << "shifit:" << shift;
    PARAM(param);
    BOTTOM(stream);
    EVENT_TIME(fuser.get_event_time());
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MFUSPowerLayerTest, TestMFUSDevices);

TYPED_TEST(MFUSPowerLayerTest, TestPower) {
  typedef typename TypeParam::Dtype Dtype;
  Dtype power = 0.37;
  Dtype scale = 0.83;
  Dtype shift = -2.4;
  this->TestForward(power, scale, shift);
}

TYPED_TEST(MFUSPowerLayerTest, TestPowerZero) {
  typedef typename TypeParam::Dtype Dtype;
  Dtype power = 0.0;
  Dtype scale = 0.83;
  Dtype shift = -2.4;
  this->TestForward(power, scale, shift);
}

TYPED_TEST(MFUSPowerLayerTest, TestPowerOne) {
  typedef typename TypeParam::Dtype Dtype;
  Dtype power = 1.0;
  Dtype scale = 0.83;
  Dtype shift = -2.4;
  this->TestForward(power, scale, shift);
}

TYPED_TEST(MFUSPowerLayerTest, TestPowerTwo) {
  typedef typename TypeParam::Dtype Dtype;
  Dtype power = 2.0;
  Dtype scale = 0.34;
  Dtype shift = 2.4;
  this->TestForward(power, scale, shift);
}

#endif

}  // namespace caffe
