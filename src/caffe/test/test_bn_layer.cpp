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
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/bn_layer.hpp"
#ifdef USE_MLU
#include "caffe/layers/mlu_bn_layer.hpp"
#endif

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#ifdef USE_MLU

namespace caffe {

template <typename Dtype>
float caffe_bn(const Blob<Dtype>* cpu_top, const Blob<Dtype>* top_blob) {
  const Dtype* top = cpu_top->cpu_data();
  const Dtype* top_data = top_blob->cpu_data();
  float err_sum = 0, sum = 0;
  for (int i = 0; i < cpu_top->count(); i++) {
    err_sum += std::abs(top_data[i] - top[i]);
    sum += std::abs(top[i]);
  }
  EXPECT_LE(err_sum/sum, 1e-3);
  return err_sum/sum;
}

template <typename TypeParam>
class MLUBNLayerTest : public MLUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MLUBNLayerTest()
      : blob_bottom_(new Blob<Dtype>(1, 2, 4, 4)),
        blob_top_(new Blob<Dtype>()),
        cpu_blob_top_(new Blob<Dtype>()) {}
  void SetUp() {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.clear();
    blob_top_vec_.clear();
    cpu_blob_top_vec_.clear();
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
    cpu_blob_top_vec_.push_back(cpu_blob_top_);
  }
  virtual ~MLUBNLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
    delete cpu_blob_top_;
  }
  virtual void TestForwardBN() {
    SetUp();
    LayerParameter layer_param;
    layer_param.set_phase(TEST);

    BNParameter* param = layer_param.mutable_bn_param();
    param->mutable_slope_filler()->set_type("constant");
    param->mutable_slope_filler()->set_value(1);
    param->mutable_bias_filler()->set_type("constant");
    param->mutable_bias_filler()->set_value(0);
    param->set_eps(this->eps);

    BNLayer<Dtype> cpu_layer(layer_param);
    cpu_layer.SetUp(this->blob_bottom_vec_, this->cpu_blob_top_vec_);
    cpu_layer.Forward(this->blob_bottom_vec_, this->cpu_blob_top_vec_);

    MLUBNLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    caffe_bn(this->cpu_blob_top_, this->blob_top_);
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const cpu_blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  vector<Blob<Dtype>*> cpu_blob_top_vec_;
  float eps = 1e-5;
};

TYPED_TEST_CASE(MLUBNLayerTest, TestMLUDevices);

TYPED_TEST(MLUBNLayerTest, TestBNSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.set_phase(TEST);
  MLUBNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width());
}

TYPED_TEST(MLUBNLayerTest, TestPSPBNSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.set_phase(TEST);
  MLUBNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width());
}

TYPED_TEST(MLUBNLayerTest, TestBNForward0) {
  std::vector<int> shape = {1, 16, 24, 24};
  this->blob_bottom_->Reshape(shape);
  this->TestForwardBN();
}

TYPED_TEST(MLUBNLayerTest, TestBNForward1) {
  std::vector<int> shape = {1, 64, 237, 237};
  this->blob_bottom_->Reshape(shape);
  this->TestForwardBN();
}

TYPED_TEST(MLUBNLayerTest, TestBNForwardEPS) {
  this->eps = 0.1;
  this->TestForwardBN();
}

template <typename TypeParam>
class MFUSBNLayerTest : public MFUSDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MFUSBNLayerTest()
    : blob_bottom_(new Blob<Dtype>(1, 2, 4, 4)),
    blob_top_(new Blob<Dtype>()),
    cpu_blob_top_(new Blob<Dtype>()) {}
  void SetUp() {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.clear();
    blob_top_vec_.clear();
    cpu_blob_top_vec_.clear();
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
    cpu_blob_top_vec_.push_back(cpu_blob_top_);
  }
  virtual ~MFUSBNLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
    delete cpu_blob_top_;
  }
  virtual void TestForwardBN() {
    SetUp();
    LayerParameter layer_param;
    layer_param.set_phase(TEST);

    BNParameter* param = layer_param.mutable_bn_param();
    param->mutable_slope_filler()->set_type("constant");
    param->mutable_slope_filler()->set_value(1);
    param->mutable_bias_filler()->set_type("constant");
    param->mutable_bias_filler()->set_value(0);
    param->set_eps(this->eps);

    BNLayer<Dtype> cpu_layer(layer_param);
    cpu_layer.SetUp(this->blob_bottom_vec_, this->cpu_blob_top_vec_);
    cpu_layer.Forward(this->blob_bottom_vec_, this->cpu_blob_top_vec_);

    MLUBNLayer<Dtype> layer(layer_param);
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
    caffe_bn(this->cpu_blob_top_, this->blob_top_);
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const cpu_blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  vector<Blob<Dtype>*> cpu_blob_top_vec_;
  float eps = 1e-5;
};

TYPED_TEST_CASE(MFUSBNLayerTest, TestMFUSDevices);

TYPED_TEST(MFUSBNLayerTest, TestBNSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.set_phase(TEST);
  MLUBNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width());
}

TYPED_TEST(MFUSBNLayerTest, TestPSPBNSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.set_phase(TEST);
  MLUBNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width());
}

TYPED_TEST(MFUSBNLayerTest, TestBNForward0) {
  std::vector<int> shape = {1, 16, 24, 24};
  this->blob_bottom_->Reshape(shape);
  this->TestForwardBN();
}

TYPED_TEST(MFUSBNLayerTest, TestBNForward1) {
  std::vector<int> shape = {1, 64, 237, 237};
  this->blob_bottom_->Reshape(shape);
  this->TestForwardBN();
}

TYPED_TEST(MFUSBNLayerTest, TestBNForwardEPS) {
  this->eps = 0.1;
  this->TestForwardBN();
}

#endif

}  // namespace caffe
