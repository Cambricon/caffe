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
#include "caffe/layers/dropout_layer.hpp"
#include "caffe/layers/mlu_dropout_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "gtest/gtest.h"

namespace caffe {

template <typename Dtype>
float caffe_dropout(const Blob<Dtype>* bottom, const Blob<Dtype>* top,
                   float errRate1, float errRate2) {
  const Dtype* bottom_data = bottom->cpu_data();
  const Dtype* top_data = top->cpu_data();
  float err_sum = 0, sum = 0;

  for (int i = 0; i < top->count(); i++) {
    Dtype top_expected = bottom_data[i] * 0.5;
    EXPECT_NEAR(top_data[i], top_expected, errRate1);
    err_sum += std::abs(top_data[i] - top_expected);
    sum += std::abs(top_data[i]);
  }
    EXPECT_LE(err_sum / sum, 1e-5);
    return err_sum/sum;
}

template <typename TypeParam>
class DropoutLayerTest : public CPUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  DropoutLayerTest()
      : blob_bottom_(new Blob<Dtype>(3, 5, 7, 11)),
        blob_top_(new Blob<Dtype>()) {}
  void SetUp() {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~DropoutLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  virtual void TestForward() {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    this->blob_bottom_vec_.clear();
    this->blob_top_vec_.clear();
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);

    LayerParameter layer_param;
    DropoutParameter* dropout_param = layer_param.mutable_dropout_param();
    dropout_param->set_dropout_ratio(0.5);
    dropout_param->set_scale_train(false);
    layer_param.set_phase(caffe::TEST);
    DropoutLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    caffe_dropout(this->blob_bottom_, this->blob_top_, 5e-5, 1e-5);
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(DropoutLayerTest, TestDtypesAndDevices);

TYPED_TEST(DropoutLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  DropoutLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width());
}

TYPED_TEST(DropoutLayerTest, TestForward) { this->TestForward(); }

TYPED_TEST(DropoutLayerTest, TestForward1Batch) {
  this->blob_bottom_->Reshape(1, 1, 28, 28);
  this->TestForward();
}

#ifdef USE_MLU

template <typename TypeParam>
class MLUDropoutLayerTest : public MLUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MLUDropoutLayerTest()
      : blob_bottom_(new Blob<Dtype>(3, 5, 7, 11)),
        blob_top_(new Blob<Dtype>()) {}
  void SetUp() {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MLUDropoutLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  virtual void TestForward() {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    this->blob_bottom_vec_.clear();
    this->blob_top_vec_.clear();
    this->blob_bottom_vec_.push_back(this->blob_bottom_);
    this->blob_top_vec_.push_back(this->blob_top_);

    LayerParameter layer_param;
    DropoutParameter* dropout_param = layer_param.mutable_dropout_param();
    dropout_param->set_dropout_ratio(0.5);
    dropout_param->set_scale_train(false);
    layer_param.set_phase(caffe::TEST);
    MLUDropoutLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    // Check values
    float rate = caffe_dropout(this->blob_bottom_, this->blob_top_, 5e-5, 1e-5);
    std::ostringstream stream, param;
    stream << "bottom1:" << blob_bottom_->shape_string().c_str();
    param << "dropout_ratio:" << dropout_param->dropout_ratio() << "\t"
          << "scale_train:" << dropout_param->scale_train();
    PARAM(param);
    BOTTOM(stream);
    ERR_RATE(rate);
    EVENT_TIME(layer.get_event_time());
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MLUDropoutLayerTest, TestMLUDevices);

TYPED_TEST(MLUDropoutLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MLUDropoutLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width());
}

TYPED_TEST(MLUDropoutLayerTest, TestForward) { this->TestForward(); }

TYPED_TEST(MLUDropoutLayerTest, TestForward1Batch) {
  this->blob_bottom_->Reshape(1, 1, 28, 28);
  this->TestForward();
}

template <typename TypeParam>
class MFUSDropoutLayerTest : public MFUSDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MFUSDropoutLayerTest()
      : blob_bottom_(new Blob<Dtype>(3, 5, 7, 11)),
        blob_top_(new Blob<Dtype>()) {}
  void SetUp() {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    this->blob_bottom_vec_.push_back(this->blob_bottom_);
    this->blob_top_vec_.push_back(this->blob_top_);
  }
  virtual ~MFUSDropoutLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  virtual void TestForward() {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    this->blob_bottom_vec_.clear();
    this->blob_top_vec_.clear();
    this->blob_bottom_vec_.push_back(this->blob_bottom_);
    this->blob_top_vec_.push_back(this->blob_top_);

    LayerParameter layer_param;
    DropoutParameter* dropout_param = layer_param.mutable_dropout_param();
    dropout_param->set_dropout_ratio(0.5);
    dropout_param->set_scale_train(false);
    layer_param.set_phase(caffe::TEST);
    MLUDropoutLayer<Dtype> layer(layer_param);
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
    // Check values
    float rate = caffe_dropout(this->blob_bottom_, this->blob_top_, 5e-5, 1e-5);
    std::ostringstream stream, param;
    stream << "bottom1:" << blob_bottom_->shape_string().c_str();
    param << "dropout_ratio:" << dropout_param->dropout_ratio() << "\t"
          << "scale_train:" << dropout_param->scale_train();
    PARAM(param);
    BOTTOM(stream);
    ERR_RATE(rate);
    EVENT_TIME(fuser.get_event_time());
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MFUSDropoutLayerTest, TestMFUSDevices);

TYPED_TEST(MFUSDropoutLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MLUDropoutLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width());
}

TYPED_TEST(MFUSDropoutLayerTest, TestForward) { this->TestForward(); }

TYPED_TEST(MFUSDropoutLayerTest, TestForward1Batch) {
  this->blob_bottom_->Reshape(1, 1, 28, 28);
  this->TestForward();
}

#endif
}  // namespace caffe
