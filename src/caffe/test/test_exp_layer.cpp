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
#include "caffe/layers/exp_layer.hpp"
#include "caffe/layers/mlu_exp_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "gtest/gtest.h"

namespace caffe {

template <typename Dtype>
float caffe_exp(const Blob<Dtype>* bottom, const Blob<Dtype>* top_blob,
    Dtype base, Dtype scale, Dtype shift, float errRate) {
  const Dtype* bottom_data = bottom->cpu_data();
  const Dtype* top_data = top_blob->cpu_data();
  float err_sum = 0, sum = 0;
  for (int i = 0; i < bottom->count(); ++i) {
    const Dtype bottom_val = bottom_data[i];
    const Dtype top_val = top_data[i];
    if (base == -1) {
      Dtype top = exp(shift + scale * bottom_val);
      EXPECT_NEAR(top_val, top, errRate);
      err_sum += std::abs(top_data[i]-top);
      sum+=std::abs(top);
    } else {
      Dtype top = pow(base, shift + scale * bottom_val);
      EXPECT_NEAR(top_val, pow(base, shift + scale * bottom_val), errRate);
      err_sum += std::abs(top_data[i]-top);
      sum+=std::abs(top);
    }
  }
  EXPECT_LE(err_sum/sum, 5e-3);
  return err_sum/sum;
}

template <typename TypeParam>
class ExpLayerTest : public CPUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  ExpLayerTest()
      : blob_bottom_(new Blob<Dtype>(3, 5, 8, 10)),
        blob_top_(new Blob<Dtype>()),
        base_(-1.0), scale_(1.0), shift_(0.0) {}
  void SetUp() {
    FillerParameter filler_param;
    filler_param.set_value(0.4025);
    GaussianFiller<Dtype> filler(filler_param);
    this->blob_bottom_vec_.clear();
    this->blob_top_vec_.clear();
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~ExpLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  virtual void TestForward() {
    SetUp();
    LayerParameter layer_param;
    layer_param.mutable_exp_param()->set_base(base_);
    layer_param.mutable_exp_param()->set_scale(scale_);
    layer_param.mutable_exp_param()->set_shift(shift_);
    ExpLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    caffe_exp(blob_bottom_, blob_top_, base_, scale_, shift_, 5e-3);
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  Dtype base_;
  Dtype scale_;
  Dtype shift_;
};

TYPED_TEST_CASE(ExpLayerTest, TestDtypesAndDevices);

TYPED_TEST(ExpLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ExpLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width());
}

TYPED_TEST(ExpLayerTest, TestForward) {
  this->TestForward();
}

TYPED_TEST(ExpLayerTest, TestForward1Dim) {
  vector<int> shape;
  shape.push_back(51);
  this->blob_bottom_->Reshape(shape);
  this->TestForward();
}

TYPED_TEST(ExpLayerTest, TestForward2Dim) {
  vector<int> shape;
  shape.push_back(51);
  shape.push_back(31);
  this->blob_bottom_->Reshape(shape);
  this->TestForward();
}

TYPED_TEST(ExpLayerTest, TestForwardWithShift) {
  this->base_ = -1;
  this->scale_ = 1;
  this->shift_ = 1;
  this->TestForward();
}

TYPED_TEST(ExpLayerTest, TestForwardBase) {
  this->base_ = 2;
  this->scale_ = 1;
  this->shift_ = 0;
  this->TestForward();
}

TYPED_TEST(ExpLayerTest, TestForwardBaseShift) {
  this->base_ = 2;
  this->scale_ = 1;
  this->shift_ = 1;
  this->TestForward();
}

TYPED_TEST(ExpLayerTest, TestForwardBaseScale) {
  this->base_ = 2;
  this->scale_ = 3;
  this->shift_ = 0;
  this->TestForward();
}

TYPED_TEST(ExpLayerTest, TestForwardBaseShiftScale) {
  this->base_ = 2;
  this->scale_ = 3;
  this->shift_ = 1;
  this->TestForward();
}


#ifdef USE_MLU

template <typename TypeParam>
class MLUExpLayerTest : public MLUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MLUExpLayerTest()
      : blob_bottom_(new Blob<Dtype>(3, 5, 7, 11)),
        blob_top_(new Blob<Dtype>()),
        base_(-1.0), scale_(1.0), shift_(0.0) {}
  void SetUp() {
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    this->blob_bottom_vec_.clear();
    this->blob_top_vec_.clear();
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MLUExpLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  virtual void TestForward() {
    SetUp();
    LayerParameter layer_param;
    layer_param.mutable_exp_param()->set_base(base_);
    layer_param.mutable_exp_param()->set_scale(scale_);
    layer_param.mutable_exp_param()->set_shift(shift_);

    MLUEXPLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    layer.Reshape_dispatch(blob_bottom_vec_, blob_top_vec_);
    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    float rate = caffe_exp(blob_bottom_, blob_top_, base_, scale_, shift_, 3e-1);
    std::ostringstream stream, param;
    stream << "bottom1:" << blob_bottom_->shape_string().c_str();
    param << "base:" << layer_param.mutable_exp_param()->base() << "\t"
      << "scale:" << layer_param.mutable_exp_param()->scale() << "\t"
      << "shift:" << layer_param.mutable_exp_param()->shift();
    PARAM(param);
    BOTTOM(stream);
    ERR_RATE(rate);
    EVENT_TIME(layer.get_event_time());
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  Dtype base_;
  Dtype scale_;
  Dtype shift_;
};

TYPED_TEST_CASE(MLUExpLayerTest, TestMLUDevices);

TYPED_TEST(MLUExpLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MLUEXPLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width());
}

TYPED_TEST(MLUExpLayerTest, TestForward) {
  this->TestForward();
}

TYPED_TEST(MLUExpLayerTest, TestForward1Dim) {
  vector<int> shape;
  shape.push_back(51);
  this->blob_bottom_->Reshape(shape);
  this->TestForward();
}

TYPED_TEST(MLUExpLayerTest, TestForward2Dim) {
  vector<int> shape;
  shape.push_back(51);
  shape.push_back(31);
  this->blob_bottom_->Reshape(shape);
  this->TestForward();
}

TYPED_TEST(MLUExpLayerTest, TestForwardWithShift) {
  this->base_ = -1;
  this->scale_ = 1;
  this->shift_ = 1;
  this->TestForward();
}

TYPED_TEST(MLUExpLayerTest, TestForwardBase) {
  this->base_ = 2;
  this->scale_ = 1;
  this->shift_ = 0;
  this->TestForward();
}

TYPED_TEST(MLUExpLayerTest, TestForwardBaseShift) {
  this->base_ = 2;
  this->scale_ = 1;
  this->shift_ = 1;
  this->TestForward();
}

TYPED_TEST(MLUExpLayerTest, TestForwardBaseScale) {
  this->base_ = 2;
  this->scale_ = 3;
  this->shift_ = 0;
  this->TestForward();
}

TYPED_TEST(MLUExpLayerTest, TestForwardBaseShiftScale) {
  this->base_ = 2;
  this->scale_ = 3;
  this->shift_ = 1;
  this->TestForward();
}

template <typename TypeParam>
class MFUSExpLayerTest : public MFUSDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MFUSExpLayerTest()
      : blob_bottom_(new Blob<Dtype>(3, 5, 7, 11)),
        blob_top_(new Blob<Dtype>()),
        base_(-1.0), scale_(1.0), shift_(0.0) {}
  void SetUp() {
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    this->blob_bottom_vec_.clear();
    this->blob_top_vec_.clear();
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MFUSExpLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  virtual void TestForward() {
    SetUp();
    LayerParameter layer_param;
    layer_param.mutable_exp_param()->set_base(base_);
    layer_param.mutable_exp_param()->set_scale(scale_);
    layer_param.mutable_exp_param()->set_shift(shift_);
    MLUEXPLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    ASSERT_TRUE(layer.mfus_supported());

    MFusion<Dtype> fuser;
    fuser.reset();
    fuser.addInputs(this->blob_bottom_vec_);
    fuser.addOutputs(this->blob_top_vec_);
    layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.fuse(&fuser);
    fuser.compile();
    fuser.forward();
    float rate = caffe_exp(blob_bottom_, blob_top_, base_, scale_, shift_, 3e-1);
    std::ostringstream stream, param;
    stream << "bottom1:" << blob_bottom_->shape_string().c_str();
    param << "base:" << layer_param.mutable_exp_param()->base() << "\t"
          << "scale:" << layer_param.mutable_exp_param()->scale() << "\t"
          << "shift:" << layer_param.mutable_exp_param()->shift();
    PARAM(param);
    BOTTOM(stream);
    ERR_RATE(rate);
    EVENT_TIME(fuser.get_event_time());
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  Dtype base_;
  Dtype scale_;
  Dtype shift_;
};

TYPED_TEST_CASE(MFUSExpLayerTest, TestMLUDevices);

TYPED_TEST(MFUSExpLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MLUEXPLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width());
}

TYPED_TEST(MFUSExpLayerTest, TestForward) {
  this->TestForward();
}

TYPED_TEST(MFUSExpLayerTest, TestForward1Dim) {
  vector<int> shape;
  shape.push_back(51);
  this->blob_bottom_->Reshape(shape);
  this->TestForward();
}

TYPED_TEST(MFUSExpLayerTest, TestForward2Dim) {
  vector<int> shape;
  shape.push_back(51);
  shape.push_back(31);
  this->blob_bottom_->Reshape(shape);
  this->TestForward();
}

TYPED_TEST(MFUSExpLayerTest, TestForwardWithShift) {
  this->base_ = -1;
  this->scale_ = 1;
  this->shift_ = 1;
  this->TestForward();
}

TYPED_TEST(MFUSExpLayerTest, TestForwardBase) {
  this->base_ = 2;
  this->scale_ = 1;
  this->shift_ = 0;
  this->TestForward();
}

TYPED_TEST(MFUSExpLayerTest, TestForwardBaseShift) {
  this->base_ = 2;
  this->scale_ = 1;
  this->shift_ = 1;
  this->TestForward();
}

TYPED_TEST(MFUSExpLayerTest, TestForwardBaseScale) {
  this->base_ = 2;
  this->scale_ = 3;
  this->shift_ = 0;
  this->TestForward();
}

TYPED_TEST(MFUSExpLayerTest, TestForwardBaseShiftScale) {
  this->base_ = 2;
  this->scale_ = 3;
  this->shift_ = 1;
  this->TestForward();
}

#endif

}  // namespace caffe
