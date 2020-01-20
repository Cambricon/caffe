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

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/mlu_reshape_layer.hpp"
#include "caffe/layers/reshape_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class ReshapeLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
  protected:
  ReshapeLayerTest()
    : blob_bottom_(new Blob<Dtype>(2, 3, 6, 5)),
      blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~ReshapeLayerTest() { delete blob_bottom_; delete blob_top_; }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ReshapeLayerTest, TestDtypesAndDevices);

TYPED_TEST(ReshapeLayerTest, TestFlattenOutputSizes) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BlobShape* blob_shape = layer_param.mutable_reshape_param()->mutable_shape();
  blob_shape->add_dim(0);
  blob_shape->add_dim(-1);
  blob_shape->add_dim(1);
  blob_shape->add_dim(1);

  ReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3 * 6 * 5);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

TYPED_TEST(ReshapeLayerTest, TestFlattenValues) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BlobShape* blob_shape = layer_param.mutable_reshape_param()->mutable_shape();
  blob_shape->add_dim(0);
  blob_shape->add_dim(-1);
  blob_shape->add_dim(1);
  blob_shape->add_dim(1);
  ReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int c = 0; c < 3 * 6 * 5; ++c) {
    EXPECT_EQ(this->blob_top_->data_at(0, c, 0, 0),
        this->blob_bottom_->data_at(0, c / (6 * 5), (c / 5) % 6, c % 5));
    EXPECT_EQ(this->blob_top_->data_at(1, c, 0, 0),
        this->blob_bottom_->data_at(1, c / (6 * 5), (c / 5) % 6, c % 5));
  }
}

// Test whether setting output dimensions to 0 either explicitly or implicitly
// copies the respective dimension of the input layer.
TYPED_TEST(ReshapeLayerTest, TestCopyDimensions) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BlobShape* blob_shape = layer_param.mutable_reshape_param()->mutable_shape();
  blob_shape->add_dim(0);
  blob_shape->add_dim(0);
  blob_shape->add_dim(0);
  blob_shape->add_dim(0);
  ReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 6);
  EXPECT_EQ(this->blob_top_->width(), 5);
}

// When a dimension is set to -1, we should infer its value from the other
// dimensions (including those that get copied from below).
TYPED_TEST(ReshapeLayerTest, TestInferenceOfUnspecified) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BlobShape* blob_shape = layer_param.mutable_reshape_param()->mutable_shape();
  blob_shape->add_dim(0);
  blob_shape->add_dim(3);
  blob_shape->add_dim(10);
  blob_shape->add_dim(-1);

  // Count is 180, thus height should be 180 / (2*3*10) = 3.

  ReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 10);
  EXPECT_EQ(this->blob_top_->width(), 3);
}

TYPED_TEST(ReshapeLayerTest, TestInferenceOfUnspecifiedWithStartAxis) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_reshape_param()->set_axis(1);
  BlobShape* blob_shape = layer_param.mutable_reshape_param()->mutable_shape();
  blob_shape->add_dim(3);
  blob_shape->add_dim(10);
  blob_shape->add_dim(-1);

  ReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  ASSERT_EQ(this->blob_top_->num_axes(), 4);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 10);
  EXPECT_EQ(this->blob_top_->width(), 3);
}

TYPED_TEST(ReshapeLayerTest, TestFlattenMiddle) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_reshape_param()->set_axis(1);
  layer_param.mutable_reshape_param()->set_num_axes(2);
  BlobShape* blob_shape = layer_param.mutable_reshape_param()->mutable_shape();
  blob_shape->add_dim(-1);

  ReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  ASSERT_EQ(this->blob_top_->num_axes(), 3);
  EXPECT_EQ(this->blob_top_->shape(0), 2);
  EXPECT_EQ(this->blob_top_->shape(1), 3 * 6);
  EXPECT_EQ(this->blob_top_->shape(2), 5);
}

TYPED_TEST(ReshapeLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BlobShape* shape = layer_param.mutable_reshape_param()->mutable_shape();
  shape->add_dim(6);
  shape->add_dim(2);
  shape->add_dim(3);
  shape->add_dim(5);
  ReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_EQ(this->blob_top_->cpu_data()[i],
              this->blob_bottom_->cpu_data()[i]);
  }
}

TYPED_TEST(ReshapeLayerTest, TestForwardAfterReshape) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BlobShape* shape = layer_param.mutable_reshape_param()->mutable_shape();
  shape->add_dim(6);
  shape->add_dim(2);
  shape->add_dim(3);
  shape->add_dim(5);
  ReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // We know the above produced the correct result from TestForward.
  // Reshape the bottom and call layer.Reshape, then try again.
  vector<int> new_bottom_shape(1, 2 * 3 * 6 * 5);
  this->blob_bottom_->Reshape(new_bottom_shape);
  layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_EQ(this->blob_top_->cpu_data()[i],
              this->blob_bottom_->cpu_data()[i]);
  }
}

TYPED_TEST(ReshapeLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BlobShape* shape = layer_param.mutable_reshape_param()->mutable_shape();
  shape->add_dim(6);
  shape->add_dim(2);
  shape->add_dim(3);
  shape->add_dim(5);
  ReshapeLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

#ifdef USE_MLU

template <typename TypeParam>
class MLUReshapeLayerTest : public MLUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
  protected:
  MLUReshapeLayerTest()
    : blob_bottom_(new Blob<Dtype>(2, 3, 6, 5)),
      blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MLUReshapeLayerTest() {
     delete blob_bottom_;
     delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MLUReshapeLayerTest, TestMLUDevices);

TYPED_TEST(MLUReshapeLayerTest, TestFlattenOutputSizes) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BlobShape* blob_shape = layer_param.mutable_reshape_param()->mutable_shape();
  blob_shape->add_dim(0);
  blob_shape->add_dim(-1);
  blob_shape->add_dim(1);
  blob_shape->add_dim(1);

  MLUReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3 * 6 * 5);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

TYPED_TEST(MLUReshapeLayerTest, TestFlattenValues) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BlobShape* blob_shape = layer_param.mutable_reshape_param()->mutable_shape();
  blob_shape->add_dim(0);
  blob_shape->add_dim(-1);
  blob_shape->add_dim(1);
  blob_shape->add_dim(1);
  MLUReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int c = 0; c < 3 * 6 * 5; ++c) {
    EXPECT_EQ(this->blob_top_->data_at(0, c, 0, 0),
        this->blob_bottom_->data_at(0, c / (6 * 5), (c / 5) % 6, c % 5));
    EXPECT_EQ(this->blob_top_->data_at(1, c, 0, 0),
        this->blob_bottom_->data_at(1, c / (6 * 5), (c / 5) % 6, c % 5));
  }
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  EVENT_TIME(layer.get_event_time());
}

// Test whether setting output dimensions to 0 either explicitly or implicitly
// copies the respective dimension of the input layer.
TYPED_TEST(MLUReshapeLayerTest, TestCopyDimensions) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BlobShape* blob_shape = layer_param.mutable_reshape_param()->mutable_shape();
  blob_shape->add_dim(0);
  blob_shape->add_dim(0);
  blob_shape->add_dim(0);
  blob_shape->add_dim(0);
  MLUReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 6);
  EXPECT_EQ(this->blob_top_->width(), 5);
}

// When a dimension is set to -1, we should infer its value from the other
// dimensions (including those that get copied from below).
TYPED_TEST(MLUReshapeLayerTest, TestInferenceOfUnspecified) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BlobShape* blob_shape = layer_param.mutable_reshape_param()->mutable_shape();
  blob_shape->add_dim(0);
  blob_shape->add_dim(3);
  blob_shape->add_dim(10);
  blob_shape->add_dim(-1);

  // Count is 180, thus height should be 180 / (2*3*10) = 3.

  MLUReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 10);
  EXPECT_EQ(this->blob_top_->width(), 3);
}

TYPED_TEST(MLUReshapeLayerTest, TestInferenceOfUnspecifiedWithStartAxis) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_reshape_param()->set_axis(1);
  BlobShape* blob_shape = layer_param.mutable_reshape_param()->mutable_shape();
  blob_shape->add_dim(3);
  blob_shape->add_dim(10);
  blob_shape->add_dim(-1);

  MLUReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  ASSERT_EQ(this->blob_top_->num_axes(), 4);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 10);
  EXPECT_EQ(this->blob_top_->width(), 3);
}

TYPED_TEST(MLUReshapeLayerTest, TestFlattenMiddle) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_reshape_param()->set_axis(1);
  layer_param.mutable_reshape_param()->set_num_axes(2);
  BlobShape* blob_shape = layer_param.mutable_reshape_param()->mutable_shape();
  blob_shape->add_dim(-1);

  MLUReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  ASSERT_EQ(this->blob_top_->num_axes(), 3);
  EXPECT_EQ(this->blob_top_->shape(0), 2);
  EXPECT_EQ(this->blob_top_->shape(1), 3 * 6);
  EXPECT_EQ(this->blob_top_->shape(2), 5);
}

TYPED_TEST(MLUReshapeLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BlobShape* shape = layer_param.mutable_reshape_param()->mutable_shape();
  shape->add_dim(6);
  shape->add_dim(2);
  shape->add_dim(3);
  shape->add_dim(5);
  MLUReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_EQ(this->blob_top_->cpu_data()[i],
              this->blob_bottom_->cpu_data()[i]);
  }
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  EVENT_TIME(layer.get_event_time());
}

TYPED_TEST(MLUReshapeLayerTest, TestForwardAfterReshape) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BlobShape* shape = layer_param.mutable_reshape_param()->mutable_shape();
  shape->add_dim(6);
  shape->add_dim(2);
  shape->add_dim(3);
  shape->add_dim(5);
  MLUReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // We know the above produced the correct result from TestForward.
  // Reshape the bottom and call layer.Reshape, then try again.
  vector<int> new_bottom_shape(1, 2 * 3 * 6 * 5);
  this->blob_bottom_->Reshape(new_bottom_shape);
  layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_EQ(this->blob_top_->cpu_data()[i],
              this->blob_bottom_->cpu_data()[i]);
  }
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  EVENT_TIME(layer.get_event_time());
}

TYPED_TEST(MLUReshapeLayerTest, TestForwardBottomAxes2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BlobShape* shape = layer_param.mutable_reshape_param()->mutable_shape();
  shape->add_dim(6);
  shape->add_dim(2);
  shape->add_dim(3);
  shape->add_dim(5);
  vector<int> bottom_shape = {3, 60};
  this->blob_bottom_vec_[0]->Reshape(bottom_shape);
  MLUReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_EQ(this->blob_top_->cpu_data()[i],
              this->blob_bottom_->cpu_data()[i]);
  }
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  EVENT_TIME(layer.get_event_time());
}

TYPED_TEST(MLUReshapeLayerTest, TestForwardBottomAxes3) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BlobShape* shape = layer_param.mutable_reshape_param()->mutable_shape();
  shape->add_dim(6);
  shape->add_dim(2);
  shape->add_dim(3);
  shape->add_dim(5);
  vector<int> bottom_shape = {3, 10, 6};
  this->blob_bottom_vec_[0]->Reshape(bottom_shape);
  MLUReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_EQ(this->blob_top_->cpu_data()[i],
              this->blob_bottom_->cpu_data()[i]);
  }
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  EVENT_TIME(layer.get_event_time());
}

TYPED_TEST(MLUReshapeLayerTest, TestForwardBottomAxes5) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BlobShape* shape = layer_param.mutable_reshape_param()->mutable_shape();
  shape->add_dim(6);
  shape->add_dim(2);
  shape->add_dim(3);
  shape->add_dim(5);
  vector<int> bottom_shape = {3, 2, 5, 3, 2};
  this->blob_bottom_vec_[0]->Reshape(bottom_shape);
  MLUReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_EQ(this->blob_top_->cpu_data()[i],
              this->blob_bottom_->cpu_data()[i]);
  }
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  EVENT_TIME(layer.get_event_time());
}

TYPED_TEST(MLUReshapeLayerTest, TestForwardOutputAxes2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BlobShape* shape = layer_param.mutable_reshape_param()->mutable_shape();
  shape->add_dim(3);
  shape->add_dim(60);
  MLUReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_EQ(this->blob_top_->cpu_data()[i],
              this->blob_bottom_->cpu_data()[i]);
  }
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  EVENT_TIME(layer.get_event_time());
}

TYPED_TEST(MLUReshapeLayerTest, TestForwardOutputAxes3) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BlobShape* shape = layer_param.mutable_reshape_param()->mutable_shape();
  shape->add_dim(3);
  shape->add_dim(10);
  shape->add_dim(6);
  MLUReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_EQ(this->blob_top_->cpu_data()[i],
              this->blob_bottom_->cpu_data()[i]);
  }
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  EVENT_TIME(layer.get_event_time());
}

TYPED_TEST(MLUReshapeLayerTest, TestForwardOutputAxes5) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BlobShape* shape = layer_param.mutable_reshape_param()->mutable_shape();
  shape->add_dim(3);
  shape->add_dim(5);
  shape->add_dim(2);
  shape->add_dim(2);
  shape->add_dim(3);
  MLUReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_EQ(this->blob_top_->cpu_data()[i],
              this->blob_bottom_->cpu_data()[i]);
  }
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  EVENT_TIME(layer.get_event_time());
}
template <typename TypeParam>
class MFUSReshapeLayerTest : public MFUSDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
  protected:
  MFUSReshapeLayerTest()
    : blob_bottom_(new Blob<Dtype>(2, 3, 6, 5)),
      blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MFUSReshapeLayerTest() {
     delete blob_bottom_;
     delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MFUSReshapeLayerTest, TestMFUSDevices);

TYPED_TEST(MFUSReshapeLayerTest, TestFlattenOutputSizes) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BlobShape* blob_shape = layer_param.mutable_reshape_param()->mutable_shape();
  blob_shape->add_dim(0);
  blob_shape->add_dim(-1);
  blob_shape->add_dim(1);
  blob_shape->add_dim(1);

  MLUReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3 * 6 * 5);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

TYPED_TEST(MFUSReshapeLayerTest, TestFlattenValues) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BlobShape* blob_shape = layer_param.mutable_reshape_param()->mutable_shape();
  blob_shape->add_dim(0);
  blob_shape->add_dim(-1);
  blob_shape->add_dim(1);
  blob_shape->add_dim(1);
  MLUReshapeLayer<Dtype> layer(layer_param);
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
  for (int c = 0; c < 3 * 6 * 5; ++c) {
    EXPECT_EQ(this->blob_top_->data_at(0, c, 0, 0),
        this->blob_bottom_->data_at(0, c / (6 * 5), (c / 5) % 6, c % 5));
    EXPECT_EQ(this->blob_top_->data_at(1, c, 0, 0),
        this->blob_bottom_->data_at(1, c / (6 * 5), (c / 5) % 6, c % 5));
  }
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  EVENT_TIME(fuser.get_event_time());
}

// Test whether setting output dimensions to 0 either explicitly or implicitly
// copies the respective dimension of the input layer.
TYPED_TEST(MFUSReshapeLayerTest, TestCopyDimensions) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BlobShape* blob_shape = layer_param.mutable_reshape_param()->mutable_shape();
  blob_shape->add_dim(0);
  blob_shape->add_dim(0);
  blob_shape->add_dim(0);
  blob_shape->add_dim(0);
  MLUReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 6);
  EXPECT_EQ(this->blob_top_->width(), 5);
}

// When a dimension is set to -1, we should infer its value from the other
// dimensions (including those that get copied from below).
TYPED_TEST(MFUSReshapeLayerTest, TestInferenceOfUnspecified) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BlobShape* blob_shape = layer_param.mutable_reshape_param()->mutable_shape();
  blob_shape->add_dim(0);
  blob_shape->add_dim(3);
  blob_shape->add_dim(10);
  blob_shape->add_dim(-1);

  // Count is 180, thus height should be 180 / (2*3*10) = 3.

  MLUReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 10);
  EXPECT_EQ(this->blob_top_->width(), 3);
}

TYPED_TEST(MFUSReshapeLayerTest, TestInferenceOfUnspecifiedWithStartAxis) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_reshape_param()->set_axis(1);
  BlobShape* blob_shape = layer_param.mutable_reshape_param()->mutable_shape();
  blob_shape->add_dim(3);
  blob_shape->add_dim(10);
  blob_shape->add_dim(-1);

  MLUReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  ASSERT_EQ(this->blob_top_->num_axes(), 4);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 10);
  EXPECT_EQ(this->blob_top_->width(), 3);
}

TYPED_TEST(MFUSReshapeLayerTest, TestFlattenMiddle) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_reshape_param()->set_axis(1);
  layer_param.mutable_reshape_param()->set_num_axes(2);
  BlobShape* blob_shape = layer_param.mutable_reshape_param()->mutable_shape();
  blob_shape->add_dim(-1);

  MLUReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  ASSERT_EQ(this->blob_top_->num_axes(), 3);
  EXPECT_EQ(this->blob_top_->shape(0), 2);
  EXPECT_EQ(this->blob_top_->shape(1), 3 * 6);
  EXPECT_EQ(this->blob_top_->shape(2), 5);
}

TYPED_TEST(MFUSReshapeLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BlobShape* shape = layer_param.mutable_reshape_param()->mutable_shape();
  shape->add_dim(6);
  shape->add_dim(2);
  shape->add_dim(3);
  shape->add_dim(5);
  MLUReshapeLayer<Dtype> layer(layer_param);
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
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_EQ(this->blob_top_->cpu_data()[i],
              this->blob_bottom_->cpu_data()[i]);
  }
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  EVENT_TIME(fuser.get_event_time());
}

TYPED_TEST(MFUSReshapeLayerTest, TestForwardAfterReshape) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BlobShape* shape = layer_param.mutable_reshape_param()->mutable_shape();
  shape->add_dim(6);
  shape->add_dim(2);
  shape->add_dim(3);
  shape->add_dim(5);
  MLUReshapeLayer<Dtype> layer(layer_param);
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
  // We know the above produced the correct result from TestForward.
  // Reshape the bottom and call layer.Reshape, then try again.
  vector<int> new_bottom_shape(1, 2 * 3 * 6 * 5);
  this->blob_bottom_->Reshape(new_bottom_shape);
  layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  ASSERT_TRUE(layer.mfus_supported());

  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.fuse(&fuser);
  fuser.compile();
  fuser.forward();
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_EQ(this->blob_top_->cpu_data()[i],
              this->blob_bottom_->cpu_data()[i]);
  }
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  EVENT_TIME(fuser.get_event_time());
}

TYPED_TEST(MFUSReshapeLayerTest, TestForwardBottomAxes2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BlobShape* shape = layer_param.mutable_reshape_param()->mutable_shape();
  shape->add_dim(6);
  shape->add_dim(2);
  shape->add_dim(3);
  shape->add_dim(5);
  vector<int> bottom_shape = {3, 60};
  this->blob_bottom_vec_[0]->Reshape(bottom_shape);
  MLUReshapeLayer<Dtype> layer(layer_param);
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
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_EQ(this->blob_top_->cpu_data()[i],
              this->blob_bottom_->cpu_data()[i]);
  }
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  EVENT_TIME(fuser.get_event_time());
}

TYPED_TEST(MFUSReshapeLayerTest, TestForwardBottomAxes3) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BlobShape* shape = layer_param.mutable_reshape_param()->mutable_shape();
  shape->add_dim(6);
  shape->add_dim(2);
  shape->add_dim(3);
  shape->add_dim(5);
  vector<int> bottom_shape = {3, 10, 6};
  this->blob_bottom_vec_[0]->Reshape(bottom_shape);
  MLUReshapeLayer<Dtype> layer(layer_param);
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
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_EQ(this->blob_top_->cpu_data()[i],
              this->blob_bottom_->cpu_data()[i]);
  }
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  EVENT_TIME(fuser.get_event_time());
}

TYPED_TEST(MFUSReshapeLayerTest, TestForwardBottomAxes5) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BlobShape* shape = layer_param.mutable_reshape_param()->mutable_shape();
  shape->add_dim(6);
  shape->add_dim(2);
  shape->add_dim(3);
  shape->add_dim(5);
  vector<int> bottom_shape = {3, 2, 5, 3, 2};
  this->blob_bottom_vec_[0]->Reshape(bottom_shape);
  MLUReshapeLayer<Dtype> layer(layer_param);
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
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_EQ(this->blob_top_->cpu_data()[i],
              this->blob_bottom_->cpu_data()[i]);
  }
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  EVENT_TIME(fuser.get_event_time());
}

TYPED_TEST(MFUSReshapeLayerTest, TestForwardOutputAxes2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BlobShape* shape = layer_param.mutable_reshape_param()->mutable_shape();
  shape->add_dim(3);
  shape->add_dim(60);
  MLUReshapeLayer<Dtype> layer(layer_param);
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
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_EQ(this->blob_top_->cpu_data()[i],
              this->blob_bottom_->cpu_data()[i]);
  }
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  EVENT_TIME(fuser.get_event_time());
}

TYPED_TEST(MFUSReshapeLayerTest, TestForwardOutputAxes3) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BlobShape* shape = layer_param.mutable_reshape_param()->mutable_shape();
  shape->add_dim(3);
  shape->add_dim(10);
  shape->add_dim(6);
  MLUReshapeLayer<Dtype> layer(layer_param);
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
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_EQ(this->blob_top_->cpu_data()[i],
              this->blob_bottom_->cpu_data()[i]);
  }
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  EVENT_TIME(fuser.get_event_time());
}

TYPED_TEST(MFUSReshapeLayerTest, TestForwardOutputAxes5) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BlobShape* shape = layer_param.mutable_reshape_param()->mutable_shape();
  shape->add_dim(3);
  shape->add_dim(5);
  shape->add_dim(2);
  shape->add_dim(2);
  shape->add_dim(3);
  MLUReshapeLayer<Dtype> layer(layer_param);
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
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_EQ(this->blob_top_->cpu_data()[i],
              this->blob_bottom_->cpu_data()[i]);
  }
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  EVENT_TIME(fuser.get_event_time());
}
#endif

}  // namespace caffe
