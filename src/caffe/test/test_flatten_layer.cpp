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
#include "caffe/layers/flatten_layer.hpp"
#include "caffe/layers/mlu_flatten_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename Dtype>
float caffe_flatten(const Blob<Dtype>* bottom, const Blob<Dtype>* top_blob) {
  float err_sum = 0, sum = 0;
  for (int c = 0; c < 3 * 6 * 5; ++c) {
    EXPECT_EQ(top_blob->data_at(0, c, 0, 0),
              bottom->data_at(0, c / (6 * 5), (c / 5) % 6, c % 5));
    EXPECT_EQ(top_blob->data_at(1, c, 0, 0),
              bottom->data_at(1, c / (6 * 5), (c / 5) % 6, c % 5));
    err_sum += std::abs(top_blob->data_at(0, c, 0, 0) -
                      bottom->data_at(0, c / (6 * 5), (c / 5) % 6, c % 5));
    sum += std::abs(bottom->data_at(0, c / (6 * 5), (c / 5) % 6, c % 5));
  }
  return err_sum/sum;
}

template <typename Dtype>
float flatten_axis(const Blob<Dtype>* bottom, const Blob<Dtype>* top_blob) {
  float err_sum = 0, sum = 0;
  for (int c = 0; c < 3 * 6 * 5; ++c) {
    EXPECT_EQ(top_blob->data_at(0, c / (6 * 5), c % (5 * 6), 0),
              bottom->data_at(0, c / (6 * 5), (c / 5) % 6, c % 5));
    EXPECT_EQ(top_blob->data_at(1, c / (6 * 5), c % (5 * 6), 0),
              bottom->data_at(1, c / (6 * 5), (c / 5) % 6, c % 5));
    err_sum += std::abs(top_blob->data_at(1, c / (6 * 5), c % (5 * 6), 0) -
              bottom->data_at(1, c / (6 * 5), (c / 5) % 6, c % 5));
    sum += std::abs(bottom->data_at(1, c / (6 * 5), (c / 5) % 6, c % 5));
  }
  return err_sum/sum;
}

template <typename Dtype>
float flatten_endAxis(const Blob<Dtype>* bottom, const Blob<Dtype>* top_blob) {
  float err_sum = 0, sum = 0;
  for (int c = 0; c < 3 * 6 * 5; ++c) {
    EXPECT_EQ(top_blob->data_at(0, c / 5, c % 5, 0),
              bottom->data_at(0, c / (6 * 5), (c / 5) % 6, c % 5));
    EXPECT_EQ(top_blob->data_at(1, c / 5, c % 5, 0),
              bottom->data_at(1, c / (6 * 5), (c / 5) % 6, c % 5));
    err_sum += std::abs(top_blob->data_at(1, c / 5, c % 5, 0) -
        bottom->data_at(1, c / (6 * 5), (c / 5) % 6, c % 5));
    sum += std::abs(bottom->data_at(1, c / (6 * 5), (c / 5) % 6, c % 5));
  }
  return err_sum/sum;
}

template <typename Dtype>
float flatten_startEndAxis(const Blob<Dtype>* bottom,
                          const Blob<Dtype>* top_blob) {
  float err_sum = 0, sum = 0;
  for (int c = 0; c < 3 * 6 * 5; ++c) {
    EXPECT_EQ(top_blob->data_at(c / 5, c % 5, 0, 0),
              bottom->data_at(0, c / (6 * 5), (c / 5) % 6, c % 5));
    EXPECT_EQ(top_blob->data_at(c / 5 + 18, c % 5, 0, 0),
              bottom->data_at(1, c / (6 * 5), (c / 5) % 6, c % 5));
    err_sum += std::abs(top_blob->data_at(c / 5 + 18, c % 5, 0, 0) -
              bottom->data_at(1, c / (6 * 5), (c / 5) % 6, c % 5));
    sum += std::abs(bottom->data_at(1, c / (6 * 5), (c / 5) % 6, c % 5));
  }
  return err_sum/sum;
}

template <typename TypeParam>
class FlattenLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  FlattenLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 6, 5)),
        blob_top_(new Blob<Dtype>()) {
    Caffe::set_random_seed(1701);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    //   for(int i=0; i<this->blob_bottom_->count(); i++){
    //     this->blob_bottom_->mutable_cpu_data()[i]=i+1;
    //   }
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~FlattenLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(FlattenLayerTest, TestDtypesAndDevices);

TYPED_TEST(FlattenLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  FlattenLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_top_->num_axes(), 2);
  EXPECT_EQ(this->blob_top_->shape(0), 2);
  EXPECT_EQ(this->blob_top_->shape(1), 3 * 6 * 5);
}

TYPED_TEST(FlattenLayerTest, TestSetupWithAxis) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_flatten_param()->set_axis(2);
  FlattenLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_top_->num_axes(), 3);
  EXPECT_EQ(this->blob_top_->shape(0), 2);
  EXPECT_EQ(this->blob_top_->shape(1), 3);
  EXPECT_EQ(this->blob_top_->shape(2), 6 * 5);
}

TYPED_TEST(FlattenLayerTest, TestSetupWithEndAxis) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_flatten_param()->set_end_axis(-2);
  FlattenLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_top_->num_axes(), 3);
  EXPECT_EQ(this->blob_top_->shape(0), 2);
  EXPECT_EQ(this->blob_top_->shape(1), 3 * 6);
  EXPECT_EQ(this->blob_top_->shape(2), 5);
}

TYPED_TEST(FlattenLayerTest, TestSetupWithStartAndEndAxis) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_flatten_param()->set_axis(0);
  layer_param.mutable_flatten_param()->set_end_axis(-2);
  FlattenLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_top_->num_axes(), 2);
  EXPECT_EQ(this->blob_top_->shape(0), 2 * 3 * 6);
  EXPECT_EQ(this->blob_top_->shape(1), 5);
}

TYPED_TEST(FlattenLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  FlattenLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  caffe_flatten(this->blob_bottom_, this->blob_top_);
}

TYPED_TEST(FlattenLayerTest, TestForwardWithAxis) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_flatten_param()->set_axis(2);
  FlattenLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  flatten_axis(this->blob_bottom_, this->blob_top_);
}

TYPED_TEST(FlattenLayerTest, TestForwardWithEndAxis) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_flatten_param()->set_end_axis(-2);
  FlattenLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  flatten_endAxis(this->blob_bottom_, this->blob_top_);
}

TYPED_TEST(FlattenLayerTest, TestForwardWithStartEndAxis) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_flatten_param()->set_axis(0);
  layer_param.mutable_flatten_param()->set_end_axis(-2);
  FlattenLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  flatten_startEndAxis(this->blob_bottom_, this->blob_top_);
}

TYPED_TEST(FlattenLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  FlattenLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
                               this->blob_top_vec_);
}

#ifdef USE_MLU

template <typename TypeParam>
class MLUFlattenLayerTest : public MLUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
  protected:
  MLUFlattenLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 6, 5)),
        blob_top_(new Blob<Dtype>()) {}
  void SetUp() {
    Caffe::set_random_seed(1701);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MLUFlattenLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MLUFlattenLayerTest, TestMLUDevices);

TYPED_TEST(MLUFlattenLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MLUFlattenLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_top_->num_axes(), 2);
  EXPECT_EQ(this->blob_top_->shape(0), 2);
  EXPECT_EQ(this->blob_top_->shape(1), 3 * 6 * 5);
}

TYPED_TEST(MLUFlattenLayerTest, TestSetupWithAxis) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_flatten_param()->set_axis(2);
  MLUFlattenLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_top_->num_axes(), 3);
  EXPECT_EQ(this->blob_top_->shape(0), 2);
  EXPECT_EQ(this->blob_top_->shape(1), 3);
  EXPECT_EQ(this->blob_top_->shape(2), 6 * 5);
}

TYPED_TEST(MLUFlattenLayerTest, TestSetupWithEndAxis) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_flatten_param()->set_end_axis(-2);
  MLUFlattenLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_top_->num_axes(), 3);
  EXPECT_EQ(this->blob_top_->shape(0), 2);
  EXPECT_EQ(this->blob_top_->shape(1), 3 * 6);
  EXPECT_EQ(this->blob_top_->shape(2), 5);
}

TYPED_TEST(MLUFlattenLayerTest, TestSetupWithStartAndEndAxis) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_flatten_param()->set_axis(0);
  layer_param.mutable_flatten_param()->set_end_axis(-2);
  MLUFlattenLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_top_->num_axes(), 2);
  EXPECT_EQ(this->blob_top_->shape(0), 2 * 3 * 6);
  EXPECT_EQ(this->blob_top_->shape(1), 5);
}

TYPED_TEST(MLUFlattenLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MLUFlattenLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  float rate = caffe_flatten(this->blob_bottom_, this->blob_top_);
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  ERR_RATE(rate);
  EVENT_TIME(layer.get_event_time());
}

TYPED_TEST(MLUFlattenLayerTest, TestForwardWithAxis) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_flatten_param()->set_axis(2);
  MLUFlattenLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  float rate = flatten_axis(this->blob_bottom_, this->blob_top_);
  std::ostringstream stream, param;
  param << "axis:" << layer_param.mutable_flatten_param()->axis();
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  PARAM(param);
  BOTTOM(stream);
  ERR_RATE(rate);
  EVENT_TIME(layer.get_event_time());
}

TYPED_TEST(MLUFlattenLayerTest, TestForwardWithEndAxis) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_flatten_param()->set_end_axis(-2);
  MLUFlattenLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  float rate = flatten_endAxis(this->blob_bottom_, this->blob_top_);
  std::ostringstream stream, param;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  param << "axis:" << layer_param.mutable_flatten_param()->axis();
  PARAM(param);
  BOTTOM(stream);
  ERR_RATE(rate);
  EVENT_TIME(layer.get_event_time());
}

TYPED_TEST(MLUFlattenLayerTest, TestForwardWithStartEndAxis) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_flatten_param()->set_axis(0);
  layer_param.mutable_flatten_param()->set_end_axis(-2);
  MLUFlattenLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  float rate = flatten_startEndAxis(this->blob_bottom_, this->blob_top_);
  std::ostringstream stream, param;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  param << "axis:" << layer_param.mutable_flatten_param()->axis() << "\t"
        << "end_axis:" << layer_param.mutable_flatten_param()->end_axis();
  BOTTOM(stream);
  ERR_RATE(rate);
  EVENT_TIME(layer.get_event_time());
}

template <typename TypeParam>
class MFUSFlattenLayerTest : public MFUSDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MFUSFlattenLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 6, 5)),
        blob_top_(new Blob<Dtype>()),
        axis(1), endAxis(-1) {}
  void SetUp() {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.clear();
    blob_top_vec_.clear();
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MFUSFlattenLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  int axis, endAxis;
};

TYPED_TEST_CASE(MFUSFlattenLayerTest, TestMFUSDevices);

TYPED_TEST(MFUSFlattenLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_flatten_param()->set_axis(this->axis);
  layer_param.mutable_flatten_param()->set_end_axis(this->endAxis);
  MLUFlattenLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_TRUE(layer.mfus_supported());

  MFusion<Dtype> fuser;
  fuser.reset();
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  fuser.addInputs(this->blob_bottom_vec_);
  fuser.addOutputs(this->blob_top_vec_);
  layer.fuse(&fuser);
  fuser.compile();
  fuser.forward();
  float rate = caffe_flatten(this->blob_bottom_, this->blob_top_);
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  ERR_RATE(rate);
  EVENT_TIME(fuser.get_event_time());
}

TYPED_TEST(MFUSFlattenLayerTest, TestForwardWithAxis) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_flatten_param()->set_axis(2);
  layer_param.mutable_flatten_param()->set_end_axis(-1);
  MLUFlattenLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_TRUE(layer.mfus_supported());

  MFusion<Dtype> fuser;
  fuser.reset();
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  fuser.addInputs(this->blob_bottom_vec_);
  fuser.addOutputs(this->blob_top_vec_);
  layer.fuse(&fuser);
  fuser.compile();
  fuser.forward();
  float rate = flatten_axis(this->blob_bottom_, this->blob_top_);
  std::ostringstream stream, param;
  param << "axis:" << layer_param.mutable_flatten_param()->axis();
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  PARAM(param);
  BOTTOM(stream);
  ERR_RATE(rate);
  EVENT_TIME(fuser.get_event_time());
}

TYPED_TEST(MFUSFlattenLayerTest, TestForwardWithEndAxis) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_flatten_param()->set_axis(1);
  layer_param.mutable_flatten_param()->set_end_axis(-2);
  MLUFlattenLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_TRUE(layer.mfus_supported());

  MFusion<Dtype> fuser;
  fuser.reset();
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  fuser.addInputs(this->blob_bottom_vec_);
  fuser.addOutputs(this->blob_top_vec_);
  layer.fuse(&fuser);
  fuser.compile();
  fuser.forward();
  float rate = flatten_endAxis(this->blob_bottom_, this->blob_top_);
  std::ostringstream stream, param;
  param << "axis:" << layer_param.mutable_flatten_param()->axis();
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  PARAM(param);
  BOTTOM(stream);
  ERR_RATE(rate);
  EVENT_TIME(fuser.get_event_time());
}

TYPED_TEST(MFUSFlattenLayerTest, TestForwardWithStartEndAxis) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_flatten_param()->set_axis(0);
  layer_param.mutable_flatten_param()->set_end_axis(-2);
  MLUFlattenLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_TRUE(layer.mfus_supported());

  MFusion<Dtype> fuser;
  fuser.reset();
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  fuser.addInputs(this->blob_bottom_vec_);
  fuser.addOutputs(this->blob_top_vec_);
  layer.fuse(&fuser);
  fuser.compile();
  fuser.forward();
  float rate = flatten_startEndAxis(this->blob_bottom_, this->blob_top_);
  std::ostringstream stream, param;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  param << "axis:" << layer_param.mutable_flatten_param()->axis() << "\t"
    << "end_axis:" << layer_param.mutable_flatten_param()->end_axis();
  BOTTOM(stream);
  ERR_RATE(rate);
  EVENT_TIME(fuser.get_event_time());
}

#endif

}  // namespace caffe
