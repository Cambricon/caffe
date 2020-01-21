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
#include <memory>
#include <vector>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/mlu_prelu_layer.hpp"
#include "caffe/layers/prelu_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "gtest/gtest.h"

namespace caffe {

template <typename TypeParam>
class PReLULayerTest : public CPUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  PReLULayerTest()
      : blob_bottom_(new Blob<Dtype>(10, 10, 20, 20)),
        blob_top_(new Blob<Dtype>()) {}
  void SetUp() {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~PReLULayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  virtual void TestForward(bool channel_shared_) {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    this->blob_bottom_vec_.clear();
    this->blob_top_vec_.clear();
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
    LayerParameter layer_param;
    PReLUParameter* prelu_param = layer_param.mutable_prelu_param();
    FillerParameter* prelu_filler_param_ = new FillerParameter();
    prelu_filler_param_->set_type("constant");
    prelu_filler_param_->set_value(0.25);
    prelu_param->set_allocated_filler(prelu_filler_param_);
    prelu_param->set_channel_shared(channel_shared_);
    PReLULayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    if (channel_shared_) {
      this->blob_.reset(new Blob<Dtype>(vector<int>(0)));
    } else {
      this->blob_.reset(
          new Blob<Dtype>(vector<int>(1, this->blob_bottom_->channels())));
    }
    this->prelu_filler.reset(GetFiller<Dtype>(*prelu_filler_param_));
    this->prelu_filler->Fill(blob_.get());
    // check values
    const Dtype* bottom_data = this->blob_bottom_->cpu_data();
    const Dtype* top_data = this->blob_top_->cpu_data();
    const int count = this->blob_bottom_->count();
    const int dim = this->blob_bottom_->count(2);
    const int channels = this->blob_bottom_->channels();
    const Dtype* slope_data = blob_->cpu_data();
    const int div_factor = channel_shared_ ? channels : 1;
    for (int i = 0; i < count; ++i) {
      int c = (i / dim) % channels / div_factor;
      Dtype top_data_ = std::max(bottom_data[i], Dtype(0)) +
                        slope_data[c] * std::min(bottom_data[i], Dtype(0));
      Dtype err = std::abs(top_data_ - top_data[i]);
      EXPECT_LT(err, 5e-3);
    }
  }
  virtual void TestSetup() {
    LayerParameter layer_param;
    PReLULayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    EXPECT_GE(this->blob_bottom_->num_axes(), 2);
    EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
    EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
    EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height());
    EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->height());
  }
  shared_ptr<Blob<Dtype> > blob_;
  shared_ptr<Filler<Dtype> > prelu_filler;
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(PReLULayerTest, TestDtypesAndDevices);

TYPED_TEST(PReLULayerTest, TestSetup) { this->TestSetup(); }

TYPED_TEST(PReLULayerTest, TestForward) {
  this->TestForward(false);
  this->TestForward(true);
}

#ifdef USE_MLU

template <typename TypeParam>
class MLUPReLULayerTest : public MLUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MLUPReLULayerTest()
      : blob_bottom_(new Blob<Dtype>(3, 4, 20, 20)),
        blob_top_(new Blob<Dtype>()) {}
  void SetUp() {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MLUPReLULayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  virtual void TestForward(bool channel_shared_) {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    this->blob_bottom_vec_.clear();
    this->blob_top_vec_.clear();
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);

    LayerParameter layer_param;
    PReLUParameter* prelu_param = layer_param.mutable_prelu_param();
    FillerParameter* prelu_filler_param_ = new FillerParameter();
    prelu_filler_param_->set_type("constant");
    prelu_filler_param_->set_value(0.25);
    prelu_param->set_allocated_filler(prelu_filler_param_);
    prelu_param->set_channel_shared(channel_shared_);
    MLUPReLULayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    if (channel_shared_) {
      this->blob_.reset(new Blob<Dtype>(vector<int>(0)));
    } else {
      this->blob_.reset(
          new Blob<Dtype>(vector<int>(1, this->blob_bottom_->channels())));
    }
    this->prelu_filler.reset(GetFiller<Dtype>(*prelu_filler_param_));
    this->prelu_filler->Fill(blob_.get());
    // check values
    const Dtype* bottom_data = this->blob_bottom_vec_[0]->cpu_data();
    const Dtype* top_data = this->blob_top_vec_[0]->cpu_data();
    const int count = this->blob_bottom_->count();
    const int dim = this->blob_bottom_->count(2);
    const int channels = this->blob_bottom_->channels();
    const Dtype* slope_data = blob_->cpu_data();
    const int div_factor = channel_shared_ ? channels : 1;
    float err_sum = 0, sum = 0;
    for (int i = 0; i < count; ++i) {
      int c = (i / dim) % channels / div_factor;
      Dtype top_data_ = std::max(bottom_data[i], Dtype(0)) +
                        slope_data[c] * std::min(bottom_data[i], Dtype(0));
      err_sum += std::abs(top_data_ - top_data[i]);
      sum += std::abs(top_data_);
    }
    EXPECT_LT(err_sum/sum, 5e-2);
    ERR_RATE(err_sum/sum);
    std::ostringstream stream, param;
    stream << "bottom1:" << blob_bottom_->shape_string().c_str();
    param << "channel_shared:" << channel_shared_;
    BOTTOM(stream);
    EVENT_TIME(layer.get_event_time());
  }
  virtual void TestSetup() {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    this->blob_bottom_vec_.clear();
    this->blob_top_vec_.clear();
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
    LayerParameter layer_param;
    PReLULayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    EXPECT_GE(this->blob_bottom_->num_axes(), 2);
    EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
    EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
    EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height());
    EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->height());
  }
  shared_ptr<Blob<Dtype> > blob_;
  shared_ptr<Filler<Dtype> > prelu_filler;
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MLUPReLULayerTest, TestMLUDevices);

TYPED_TEST(MLUPReLULayerTest, TestForward) {
  this->TestForward(false);
  this->TestForward(true);
}

TYPED_TEST(MLUPReLULayerTest, TestSetup) { this->TestSetup(); }

template <typename TypeParam>
class MFUSPReLULayerTest : public MFUSDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MFUSPReLULayerTest()
      : blob_bottom_(new Blob<Dtype>(3, 4, 20, 20)),
        blob_top_(new Blob<Dtype>()) {}
  void SetUp() {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MFUSPReLULayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  virtual void TestForward(bool channel_shared_) {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    this->blob_bottom_vec_.clear();
    this->blob_top_vec_.clear();
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);

    LayerParameter layer_param;
    PReLUParameter* prelu_param = layer_param.mutable_prelu_param();
    FillerParameter* prelu_filler_param_ = new FillerParameter();
    prelu_filler_param_->set_type("constant");
    prelu_filler_param_->set_value(0.25);
    prelu_param->set_allocated_filler(prelu_filler_param_);
    prelu_param->set_channel_shared(channel_shared_);
    MLUPReLULayer<Dtype> layer(layer_param);
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
    if (channel_shared_) {
      this->blob_.reset(new Blob<Dtype>(vector<int>(0)));
    } else {
      this->blob_.reset(
          new Blob<Dtype>(vector<int>(1, this->blob_bottom_->channels())));
    }
    this->prelu_filler.reset(GetFiller<Dtype>(*prelu_filler_param_));
    this->prelu_filler->Fill(blob_.get());
    // check values
    const Dtype* bottom_data = this->blob_bottom_vec_[0]->cpu_data();
    const Dtype* top_data = this->blob_top_vec_[0]->cpu_data();
    const int count = this->blob_bottom_->count();
    const int dim = this->blob_bottom_->count(2);
    const int channels = this->blob_bottom_->channels();
    const Dtype* slope_data = blob_->cpu_data();
    const int div_factor = channel_shared_ ? channels : 1;
    float err_sum = 0, sum = 0;
    for (int i = 0; i < count; ++i) {
      int c = (i / dim) % channels / div_factor;
      Dtype top_data_ = std::max(bottom_data[i], Dtype(0)) +
        slope_data[c] * std::min(bottom_data[i], Dtype(0));
      err_sum += std::abs(top_data_ - top_data[i]);
      sum += std::abs(top_data_);
    }
    EXPECT_LT(err_sum/sum, 5e-2);
    ERR_RATE(err_sum/sum);
    std::ostringstream stream, param;
    stream << "bottom1:" << blob_bottom_->shape_string().c_str();
    param << "channel_shared:" << channel_shared_;
    BOTTOM(stream);
    EVENT_TIME(fuser.get_event_time());
  }
  shared_ptr<Blob<Dtype> > blob_;
  shared_ptr<Filler<Dtype> > prelu_filler;
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MFUSPReLULayerTest, TestMFUSDevices);

TYPED_TEST(MFUSPReLULayerTest, TestForward) {
  this->TestForward(false);
  this->TestForward(true);
}
#endif
}  // namespace caffe
