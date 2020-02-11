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

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/elu_layer.hpp"
#include "caffe/layers/mlu_elu_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "gtest/gtest.h"

namespace caffe {

template <typename Dtype>
float caffe_elu(const Blob<Dtype>* bottom,
    const Blob<Dtype>* top_blob, Dtype alpha, float errRate ) {
  const Dtype* bottom_data = bottom->cpu_data();
  const Dtype* top_data = top_blob->cpu_data();
  float err_sum = 0, sum = 0;
  for (int i = 0; i < bottom->count(); i++) {
     Dtype top = std::max(bottom_data[i], Dtype(0)) +
        alpha * (exp(std::min(bottom_data[i], Dtype(0))) - Dtype(1));
     EXPECT_NEAR(top_data[i], top, errRate);
     err_sum += std::abs(top_data[i]-top);
     sum += std::abs(top);
  }
  EXPECT_LE(err_sum/sum, 5e-3);
  return err_sum/sum;
}

template <typename TypeParam>
class EluLayerTest : public CPUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  EluLayerTest()
      : blob_bottom_(new Blob<Dtype>(3, 6, 8, 10)),
        blob_top_(new Blob<Dtype>()),
        alpha_(1) {}
  void SetUp() {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~EluLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  virtual void TestForward() {
    LayerParameter layer_param;
    ELUParameter* elu_param = layer_param.mutable_elu_param();
    elu_param->set_alpha(this->alpha_);
    ELULayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    caffe_elu(this->blob_bottom_, this->blob_top_, this->alpha_, 1e-5);
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  Dtype alpha_;
};

TYPED_TEST_CASE(EluLayerTest, TestDtypesAndDevices);

TYPED_TEST(EluLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ELULayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width());
}

TYPED_TEST(EluLayerTest, TestForward) {
  this->alpha_ = 1;
  this->TestForward();
}

TYPED_TEST(EluLayerTest, TestForwardNegative) {
  this->alpha_ = -0.5;
  this->TestForward();
}

TYPED_TEST(EluLayerTest, TestForwardReLU) {
  this->alpha_ = 0;
  this->TestForward();
}

#ifdef USE_MLU

template <typename TypeParam>
class MLUEluLayerTest : public MLUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MLUEluLayerTest()
      : blob_bottom_(new Blob<Dtype>(3, 6, 8, 10)),
        blob_top_(new Blob<Dtype>()),
        alpha_(1) {}
  void SetUp() {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MLUEluLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  virtual void TestForward() {
    LayerParameter layer_param;
    ELUParameter* elu_param = layer_param.mutable_elu_param();
    elu_param->set_alpha(this->alpha_);
    MLUELULayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    float rate = caffe_elu(this->blob_bottom_, this->blob_top_, this->alpha_, 1e-2);
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
  Dtype alpha_;
};

TYPED_TEST_CASE(MLUEluLayerTest, TestMLUDevices);

TYPED_TEST(MLUEluLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ELULayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width());
}

TYPED_TEST(MLUEluLayerTest, TestForward) {
  this->alpha_ = 1;
  this->TestForward();
}

TYPED_TEST(MLUEluLayerTest, TestForwardNegative) {
  this->alpha_ = -0.5;
  this->TestForward();
}

TYPED_TEST(MLUEluLayerTest, TestForwardReLU) {
  this->alpha_ = 1;
  this->TestForward();
}

template <typename TypeParam>
class MFUSELULayerTest : public MFUSDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MFUSELULayerTest()
      : blob_bottom_(new Blob<Dtype>(3, 6, 8, 10)),
        blob_top_(new Blob<Dtype>()), alpha_(1) {}
  void SetUp() {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MFUSELULayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  virtual void TestForward() {
    LayerParameter layer_param;
    ELUParameter* elu_param = layer_param.mutable_elu_param();
    elu_param->set_alpha(this->alpha_);
    MLUELULayer<Dtype> layer(layer_param);
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
    float rate = caffe_elu(this->blob_bottom_, this->blob_top_, this->alpha_, 1);
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
  Dtype alpha_;
};

TYPED_TEST_CASE(MFUSELULayerTest, TestMFUSDevices);

TYPED_TEST(MFUSELULayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ELULayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width());
}

TYPED_TEST(MFUSELULayerTest, TestForward) {
  this->alpha_ = 1;
  this->TestForward();
}

TYPED_TEST(MFUSELULayerTest, TestForwardNegative) {
  this->alpha_ = -0.5;
  this->TestForward();
}

TYPED_TEST(MFUSELULayerTest, TestForwardReLU) {
  this->alpha_ = 0;
  this->TestForward();
}

#endif

}  // namespace caffe
