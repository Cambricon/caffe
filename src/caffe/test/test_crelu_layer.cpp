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
#include "caffe/layers/crelu_layer.hpp"
#include "caffe/layers/mlu_crelu_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "gtest/gtest.h"

namespace caffe {

template <typename Dtype>
float caffe_crelu(const Blob<Dtype>* bottom, const Blob<Dtype>* top_blob,
  Dtype negative_slope, int concat_axis,  float errRate) {
    const Dtype* bottom_data = bottom->cpu_data();
    const Dtype* top_data = top_blob->cpu_data();
    const int count = bottom->count(concat_axis);
    const int countAxis = bottom->count(0, concat_axis);
    float err_sum = 0, sum = 0;
    for (int i = 0; i < countAxis; ++i) {
      for (int j = 0; j < count; ++j) {
          Dtype top = std::max(bottom_data[j], Dtype(0)) +
          negative_slope * std::min(bottom_data[j], Dtype(0));
          EXPECT_NEAR(top_data[j], top, errRate);
          err_sum += std::abs(top_data[j]-top);
          sum += std::abs(top);
      }
      top_data += count;
      for (int j = 0; j < count; ++j) {
          Dtype top = std::max(-bottom_data[j], Dtype(0) +
          negative_slope * std::min(-bottom_data[j], Dtype(0)));
          EXPECT_NEAR(top_data[j], top, errRate);
          err_sum += std::abs(top_data[j]-top);
          sum += std::abs(top);
      }
      top_data += count;
      bottom_data += count;
    }
    EXPECT_LE(err_sum/sum, 5e-4);
    return err_sum/sum;
}

// CPUDeviceTest

template <typename TypeParam>
class CReLuLayerTest : public CPUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  CReLuLayerTest()
       : blob_bottom_(new Blob<Dtype>(3, 5, 8, 10)),
         blob_top_(new Blob<Dtype>()),
         negative_slope_(0),
         concat_axis_(1) {}
    void SetUp() {
      FillerParameter filler_param;
      GaussianFiller<Dtype> filler(filler_param);
      this->blob_bottom_vec_.clear();
      this->blob_top_vec_.clear();
      filler.Fill(this->blob_bottom_);
      blob_bottom_vec_.push_back(blob_bottom_);
      blob_top_vec_.push_back(blob_top_);
    }
    virtual ~CReLuLayerTest() {
      delete blob_bottom_;
      delete blob_top_;
    }
    virtual void TestForward() {
      SetUp();
      LayerParameter layer_param;
      CReLUParameter* crelu_param = layer_param.mutable_crelu_param();
      crelu_param->set_negative_slope(this->negative_slope_);
      crelu_param->set_concat_axis(this->concat_axis_);
      CReLULayer<Dtype> layer(layer_param);
      layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
      layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
      caffe_crelu(this->blob_bottom_, this->blob_top_,
        this->negative_slope_, this->concat_axis_,  1e-5);
    }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  Dtype negative_slope_;
  int concat_axis_;
};

TYPED_TEST_CASE(CReLuLayerTest, TestDtypesAndDevices);

TYPED_TEST(CReLuLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CReLULayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 3);
  EXPECT_EQ(this->blob_top_->channels(), 10);
  EXPECT_EQ(this->blob_top_->height(), 8);
  EXPECT_EQ(this->blob_top_->width(), 10);
}

TYPED_TEST(CReLuLayerTest, TestForward) { this->TestForward(); }

TYPED_TEST(CReLuLayerTest, TestForward1Batch) {
  this->blob_bottom_->Reshape(1, 2, 10, 10);
  this->TestForward();
}

TYPED_TEST(CReLuLayerTest, TestForwardNegativeSlope) {
  this->negative_slope_ = -0.5;
  this->TestForward();
}

TYPED_TEST(CReLuLayerTest, TestForwardNegativeSlope1) {
  this->negative_slope_ = 0.5;
  this->TestForward();
}

TYPED_TEST(CReLuLayerTest, TestSetupAxis) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CReLUParameter* crelu_param = layer_param.mutable_crelu_param();
  crelu_param->set_concat_axis(2);
  CReLULayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 3);
  EXPECT_EQ(this->blob_top_->channels(), 5);
  EXPECT_EQ(this->blob_top_->height(), 16);
  EXPECT_EQ(this->blob_top_->width(), 10);
}

TYPED_TEST(CReLuLayerTest, TestForwardAxis) {
  this->concat_axis_ = 0;
  this->TestForward();
}

TYPED_TEST(CReLuLayerTest, TestForwardAxisHeight) {
  this->concat_axis_ = 2;
  this->TestForward();
}

TYPED_TEST(CReLuLayerTest, TestForwardAxisWidth) {
  this->concat_axis_ = 3;
  this->TestForward();
}

#ifdef USE_MLU

template <typename TypeParam>
class MLUCReLuLayerTest : public MLUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MLUCReLuLayerTest()
       : blob_bottom_(new Blob<Dtype>(3, 5, 8, 10)),
         blob_top_(new Blob<Dtype>()),
         negative_slope_(0),
         concat_axis_(1) {}
    void SetUp() {
      FillerParameter filler_param;
      GaussianFiller<Dtype> filler(filler_param);
      this->blob_bottom_vec_.clear();
      this->blob_top_vec_.clear();
      filler.Fill(this->blob_bottom_);
      blob_bottom_vec_.push_back(blob_bottom_);
      blob_top_vec_.push_back(blob_top_);
    }
    virtual ~MLUCReLuLayerTest() {
      delete blob_bottom_;
      delete blob_top_;
    }
    virtual void TestForward() {
      SetUp();
      LayerParameter layer_param;
      CReLUParameter* crelu_param = layer_param.mutable_crelu_param();
      crelu_param->set_negative_slope(this->negative_slope_);
      crelu_param->set_concat_axis(this->concat_axis_);
      MLUCReLULayer<Dtype> layer(layer_param);
      layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
      layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
      layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
      // check values
      float rate = caffe_crelu(this->blob_bottom_, this->blob_top_,
          this->negative_slope_, this->concat_axis_,  3e-3);
      std::ostringstream stream;
      stream << "bottom1:" << blob_bottom_->shape_string().c_str();
      BOTTOM(stream);
      ERR_RATE(rate);
      EVENT_TIME(layer.get_event_time());
    }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  Dtype negative_slope_;
  int concat_axis_;
};

TYPED_TEST_CASE(MLUCReLuLayerTest, TestMLUDevices);

TYPED_TEST(MLUCReLuLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CReLULayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 3);
  EXPECT_EQ(this->blob_top_->channels(), 10);
  EXPECT_EQ(this->blob_top_->height(), 8);
  EXPECT_EQ(this->blob_top_->width(), 10);
}

TYPED_TEST(MLUCReLuLayerTest, TestForward) { this->TestForward(); }

TYPED_TEST(MLUCReLuLayerTest, TestForward1Batch) {
  this->blob_bottom_->Reshape(1, 2, 10, 10);
  this->TestForward();
}

TYPED_TEST(MLUCReLuLayerTest, TestForwardNegativeSlope) {
  this->negative_slope_ = -0.5;
  this->TestForward();
}

TYPED_TEST(MLUCReLuLayerTest, TestForwardNegativeSlope1) {
  this->negative_slope_ = 0.5;
  this->TestForward();
}

TYPED_TEST(MLUCReLuLayerTest, TestSetupAxis) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CReLUParameter* crelu_param = layer_param.mutable_crelu_param();
  crelu_param->set_concat_axis(2);
  CReLULayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 3);
  EXPECT_EQ(this->blob_top_->channels(), 5);
  EXPECT_EQ(this->blob_top_->height(), 16);
  EXPECT_EQ(this->blob_top_->width(), 10);
}

TYPED_TEST(MLUCReLuLayerTest, TestForwardAxis) {
  this->concat_axis_ = 0;
  this->TestForward();
}

TYPED_TEST(MLUCReLuLayerTest, TestForwardAxisHeight) {
  this->concat_axis_ = 2;
  this->TestForward();
}

TYPED_TEST(MLUCReLuLayerTest, TestForwardAxisWidth) {
  this->concat_axis_ = 3;
  this->TestForward();
}

template <typename TypeParam>
class MFUSCReLuLayerTest : public MFUSDeviceTest<TypeParam> {
typedef typename TypeParam::Dtype Dtype;

  protected:
  MFUSCReLuLayerTest()
       : blob_bottom_(new Blob<Dtype>(3, 5, 8, 10)),
         blob_top_(new Blob<Dtype>()),
         negative_slope_(0), concat_axis_(1) {}
    void SetUp() {
      FillerParameter filler_param;
      GaussianFiller<Dtype> filler(filler_param);
      this->blob_bottom_vec_.clear();
      this->blob_top_vec_.clear();
      filler.Fill(this->blob_bottom_);
      blob_bottom_vec_.push_back(blob_bottom_);
      blob_top_vec_.push_back(blob_top_);
    }
    virtual ~MFUSCReLuLayerTest() {
      delete blob_bottom_;
      delete blob_top_;
    }
    virtual void TestForward() {
      SetUp();
      LayerParameter layer_param;
      CReLUParameter* crelu_param = layer_param.mutable_crelu_param();
      crelu_param->set_negative_slope(this->negative_slope_);
      crelu_param->set_concat_axis(this->concat_axis_);
      MLUCReLULayer<Dtype> layer(layer_param);
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
      float rate = caffe_crelu(this->blob_bottom_, this->blob_top_,
        this->negative_slope_, this->concat_axis_,  3e-3);
      std::ostringstream stream;
      stream << "bottom1:" << blob_bottom_->shape_string().c_str();
      BOTTOM(stream);
      ERR_RATE(rate);
      EVENT_TIME(fuser.get_event_time());
    }

    Blob<Dtype>* const blob_bottom_;
    Blob<Dtype>* const blob_top_;
    vector<Blob<Dtype>*> blob_bottom_vec_;
    vector<Blob<Dtype>*> blob_top_vec_;
    Dtype negative_slope_;
    int concat_axis_;
};

TYPED_TEST_CASE(MFUSCReLuLayerTest, TestMLUDevices);

TYPED_TEST(MFUSCReLuLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MLUCReLULayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 3);
  EXPECT_EQ(this->blob_top_->channels(), 10);
  EXPECT_EQ(this->blob_top_->height(), 8);
  EXPECT_EQ(this->blob_top_->width(), 10);
}

TYPED_TEST(MFUSCReLuLayerTest, TestForward) {
  this->TestForward();
}

TYPED_TEST(MFUSCReLuLayerTest, TestForward1Batch) {
  this->blob_bottom_->Reshape(1, 2, 10, 10);
  this->TestForward();
}

TYPED_TEST(MFUSCReLuLayerTest, TestForwardNegativeSlope) {
  this->negative_slope_ = -0.5;
  this->TestForward();
}

TYPED_TEST(MFUSCReLuLayerTest, TestForwardNegativeSlope1) {
  this->negative_slope_ = 0.5;
  this->TestForward();
}

TYPED_TEST(MFUSCReLuLayerTest, TestSetupAxis) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CReLUParameter* crelu_param = layer_param.mutable_crelu_param();
  crelu_param->set_concat_axis(2);
  CReLULayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 3);
  EXPECT_EQ(this->blob_top_->channels(), 5);
  EXPECT_EQ(this->blob_top_->height(), 16);
  EXPECT_EQ(this->blob_top_->width(), 10);
}

TYPED_TEST(MFUSCReLuLayerTest, TestForwardAxis) {
  this->concat_axis_ = 0;
  this->TestForward();
}

TYPED_TEST(MFUSCReLuLayerTest, TestForwardAxisHeight) {
  this->concat_axis_ = 2;
  this->TestForward();
}

TYPED_TEST(MFUSCReLuLayerTest, TestForwardAxisWidth) {
  this->concat_axis_ = 3;
  this->TestForward();
}

#endif

}  // namespace caffe
