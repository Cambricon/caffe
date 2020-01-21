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
#include "caffe/layers/axpy_layer.hpp"
#include "caffe/layers/mlu_axpy_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "gtest/gtest.h"


namespace caffe {

// Reference axpy for checking results:
template <typename Dtype>
float caffe_axpy(const Blob<Dtype>* bottom_a, const Blob<Dtype>* bottom_x,
  const Blob<Dtype>* bottom_y, const Blob<Dtype>* top_blob, float errRate) {
  int channel_dim = bottom_x->channels();
  int spatial_dim = bottom_x->count(2);
  const Dtype* scale_data = bottom_a->cpu_data();
  const Dtype* x_data = bottom_x->cpu_data();
  const Dtype* top_data = top_blob->cpu_data();
  const int count = bottom_y->count();
  Dtype* top_temp = static_cast<Dtype*>(malloc(count * sizeof(Dtype)));
  float err_sum = 0, sum = 0;
  caffe_copy(bottom_y->count(), bottom_y->cpu_data(), top_temp);
  for (int n = 0; n < bottom_x->num(); ++n) {
    for (int c = 0; c < channel_dim; ++c) {
      int scale_offset = n * channel_dim + c;
      caffe_axpy(spatial_dim, scale_data[scale_offset],
          x_data + scale_offset * spatial_dim,
          top_temp + scale_offset * spatial_dim);
    }
  }
  for (int i = 0; i < bottom_x->count(); i++) {
    EXPECT_NEAR(top_data[i], top_temp[i], errRate);
    err_sum += std::abs(top_data[i]-top_temp[i]);
    sum+=std::abs(top_temp[i]);
  }
  EXPECT_LE(err_sum/sum, 5e-3);
  free(top_temp);
  return err_sum/sum;
}

// CPUDeviceTest

template <typename TypeParam>
class AxpyLayerTest : public CPUDeviceTest<TypeParam> {
typedef typename TypeParam::Dtype Dtype;

  protected:
  AxpyLayerTest()
     : blob_bottom_a_(new Blob<Dtype>(3, 5, 1, 1)),
       blob_bottom_x_(new Blob<Dtype>(3, 5, 8, 10)),
       blob_bottom_y_(new Blob<Dtype>(3, 5, 8, 10)),
       blob_top_(new Blob<Dtype>()) {}
  void SetUp() {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    this->blob_bottom_vec_.clear();
    this->blob_top_vec_.clear();
    filler.Fill(this->blob_bottom_a_);
    filler.Fill(this->blob_bottom_x_);
    filler.Fill(this->blob_bottom_y_);
    blob_bottom_vec_.push_back(blob_bottom_a_);
    blob_bottom_vec_.push_back(blob_bottom_x_);
    blob_bottom_vec_.push_back(blob_bottom_y_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~AxpyLayerTest() {
    delete blob_bottom_a_;
    delete blob_bottom_x_;
    delete blob_bottom_y_;
    delete blob_top_;
  }
  virtual void TestForward() {
    SetUp();
    LayerParameter layer_param;
    AxpyLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    caffe_axpy(this->blob_bottom_a_, this->blob_bottom_x_,
      this->blob_bottom_y_,  this->blob_top_, 1e-5);
  }
  Blob<Dtype>* const blob_bottom_a_;
  Blob<Dtype>* const blob_bottom_x_;
  Blob<Dtype>* const blob_bottom_y_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(AxpyLayerTest, TestDtypesAndDevices);

TYPED_TEST(AxpyLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  AxpyLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 3);
  EXPECT_EQ(this->blob_top_->channels(), 5);
  EXPECT_EQ(this->blob_top_->height(), 8);
  EXPECT_EQ(this->blob_top_->width(), 10);
}

TYPED_TEST(AxpyLayerTest, TestForward) {
  this->TestForward();
}

#ifdef USE_MLU

// MLUDeviceTest

template <typename TypeParam>
class MLUAxpyLayerTest : public MLUDeviceTest<TypeParam> {
typedef typename TypeParam::Dtype Dtype;

  protected:
    MLUAxpyLayerTest()
       : blob_bottom_a_(new Blob<Dtype>(3, 5, 1, 1)),
         blob_bottom_x_(new Blob<Dtype>(3, 5, 8, 10)),
         blob_bottom_y_(new Blob<Dtype>(3, 5, 8, 10)),
         blob_top_(new Blob<Dtype>()) {}
    void SetUp() {
      FillerParameter filler_param;
      GaussianFiller<Dtype> filler(filler_param);
      filler.Fill(this->blob_bottom_a_);
      filler.Fill(this->blob_bottom_x_);
      filler.Fill(this->blob_bottom_y_);
      this->blob_bottom_vec_.clear();
      this->blob_top_vec_.clear();
      blob_bottom_vec_.push_back(blob_bottom_a_);
      blob_bottom_vec_.push_back(blob_bottom_x_);
      blob_bottom_vec_.push_back(blob_bottom_y_);
      blob_top_vec_.push_back(blob_top_);
    }
    virtual ~MLUAxpyLayerTest() {
      delete blob_bottom_a_;
      delete blob_bottom_x_;
      delete blob_bottom_y_;
      delete blob_top_;
    }
    virtual void TestForward() {
      SetUp();
      LayerParameter layer_param;
      MLUAxpyLayer<Dtype> layer(layer_param);
      layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
      layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
      layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
      float rate = caffe_axpy(this->blob_bottom_a_, this->blob_bottom_x_,
                this->blob_bottom_y_, this->blob_top_, 5e-2);
      OUTPUT("bottom1", this->blob_bottom_vec_[0]->shape_string().c_str());
      OUTPUT("bottom2", this->blob_bottom_vec_[1]->shape_string().c_str());
      OUTPUT("bottom3", this->blob_bottom_vec_[2]->shape_string().c_str());
      ERR_RATE(rate);
      EVENT_TIME(layer.get_event_time());
    }
    Blob<Dtype>* const blob_bottom_a_;
    Blob<Dtype>* const blob_bottom_x_;
    Blob<Dtype>* const blob_bottom_y_;
    Blob<Dtype>* const blob_top_;
    vector<Blob<Dtype>*> blob_bottom_vec_;
    vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MLUAxpyLayerTest, TestMLUDevices);

TYPED_TEST(MLUAxpyLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MLUAxpyLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 3);
  EXPECT_EQ(this->blob_top_->channels(), 5);
  EXPECT_EQ(this->blob_top_->height(), 8);
  EXPECT_EQ(this->blob_top_->width(), 10);
  OUTPUT("bottom1", this->blob_bottom_vec_[0]->shape_string().c_str());
  OUTPUT("bottom2", this->blob_bottom_vec_[1]->shape_string().c_str());
  OUTPUT("bottom3", this->blob_bottom_vec_[2]->shape_string().c_str());
}

TYPED_TEST(MLUAxpyLayerTest, TestForward) {
  this->TestForward();
}

// MFUSDeviceTest

template <typename TypeParam>
class MFUSAxpyLayerTest : public MFUSDeviceTest<TypeParam> {
typedef typename TypeParam::Dtype Dtype;

  protected:
    MFUSAxpyLayerTest()
       : blob_bottom_a_(new Blob<Dtype>(3, 5, 1, 1)),
         blob_bottom_x_(new Blob<Dtype>(3, 5, 8, 10)),
         blob_bottom_y_(new Blob<Dtype>(3, 5, 8, 10)),
         blob_top_(new Blob<Dtype>()) {}
  void SetUp() {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_a_);
    filler.Fill(this->blob_bottom_x_);
    filler.Fill(this->blob_bottom_y_);
    this->blob_bottom_vec_.clear();
    this->blob_top_vec_.clear();
    this->blob_bottom_vec_.push_back(this->blob_bottom_a_);
    this->blob_bottom_vec_.push_back(this->blob_bottom_x_);
    this->blob_bottom_vec_.push_back(this->blob_bottom_y_);
    this->blob_top_vec_.push_back(this->blob_top_);
  }
  virtual ~MFUSAxpyLayerTest() {
    delete blob_bottom_a_;
    delete blob_bottom_x_;
    delete blob_bottom_y_;
    delete blob_top_;
  }
  virtual void TestForward() {
    SetUp();
    LayerParameter layer_param;
    MLUAxpyLayer<Dtype> layer(layer_param);
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
    float rate = caffe_axpy(this->blob_bottom_a_, this->blob_bottom_x_,
        this->blob_bottom_y_, this->blob_top_, 5e-2);
    OUTPUT("bottom1", this->blob_bottom_vec_[0]->shape_string().c_str());
    OUTPUT("bottom2", this->blob_bottom_vec_[1]->shape_string().c_str());
    OUTPUT("bottom3", this->blob_bottom_vec_[2]->shape_string().c_str());
    ERR_RATE(rate);
    EVENT_TIME(fuser.get_event_time());
  }

  Blob<Dtype>* const blob_bottom_a_;
  Blob<Dtype>* const blob_bottom_x_;
  Blob<Dtype>* const blob_bottom_y_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MFUSAxpyLayerTest, TestMFUSDevices);

TYPED_TEST(MFUSAxpyLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MLUAxpyLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 3);
  EXPECT_EQ(this->blob_top_->channels(), 5);
  EXPECT_EQ(this->blob_top_->height(), 8);
  EXPECT_EQ(this->blob_top_->width(), 10);
  OUTPUT("bottom1", this->blob_bottom_vec_[0]->shape_string().c_str());
  OUTPUT("bottom2", this->blob_bottom_vec_[1]->shape_string().c_str());
  OUTPUT("bottom3", this->blob_bottom_vec_[2]->shape_string().c_str());
}

TYPED_TEST(MFUSAxpyLayerTest, TestForward) {
  this->TestForward();
}

#endif

}  // namespace caffe
