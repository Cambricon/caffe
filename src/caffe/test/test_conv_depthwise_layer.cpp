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

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/conv_depthwise_layer.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/mlu_conv_depthwise_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class ConvolutionDepthwiseLayerTest : public CPUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  ConvolutionDepthwiseLayerTest()
      : blob_bottom_(new Blob<Dtype>(1, 3, 6, 6)),
        blob_bottom_2_(new Blob<Dtype>(1, 3, 6, 6)),
        blob_top_(new Blob<Dtype>()),
        blob_top_2_(new Blob<Dtype>()),
        conv_top_(new Blob<Dtype>()),
        conv_top_2_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    FillerParameter filler_param;
    filler_param.set_value(1.);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_bottom_2_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
    conv_top_vec_.push_back(conv_top_);
  }

  virtual ~ConvolutionDepthwiseLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_2_;
    delete blob_top_;
    delete blob_top_2_;
    delete conv_top_;
    delete conv_top_2_;
  }

  virtual Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
    this->ref_blob_top_.reset(new Blob<Dtype>());
    this->ref_blob_top_->ReshapeLike(*top);
    return this->ref_blob_top_.get();
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_2_;
  Blob<Dtype>* const conv_top_;
  Blob<Dtype>* const conv_top_2_;
  shared_ptr<Blob<Dtype> > ref_blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  vector<Blob<Dtype>*> conv_top_vec_;
};

TYPED_TEST_CASE(ConvolutionDepthwiseLayerTest, TestDtypesAndDevices);

TYPED_TEST(ConvolutionDepthwiseLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.set_type("ConvolutionDepthwise");
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(2);
  convolution_param->set_num_output(3);
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  shared_ptr<Layer<Dtype> > layer(
      new ConvolutionDepthwiseLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 2);
  EXPECT_EQ(this->blob_top_->width(), 2);
  EXPECT_EQ(this->blob_top_2_->num(), 1);
  EXPECT_EQ(this->blob_top_2_->channels(), 3);
  EXPECT_EQ(this->blob_top_2_->height(), 2);
  EXPECT_EQ(this->blob_top_2_->width(), 2);
  // setting group should not change the shape
  convolution_param->set_num_output(6);
  layer.reset(new ConvolutionDepthwiseLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 6);
  EXPECT_EQ(this->blob_top_->height(), 2);
  EXPECT_EQ(this->blob_top_->width(), 2);
  EXPECT_EQ(this->blob_top_2_->num(), 1);
  EXPECT_EQ(this->blob_top_2_->channels(), 6);
  EXPECT_EQ(this->blob_top_2_->height(), 2);
  EXPECT_EQ(this->blob_top_2_->width(), 2);
}

TYPED_TEST(ConvolutionDepthwiseLayerTest, TestSimpleConvolution) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  this->conv_top_vec_.push_back(this->conv_top_2_);
  LayerParameter layer_param;
  layer_param.set_type("ConvolutionDepthwise");
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(2);
  convolution_param->set_num_output(6);
  convolution_param->mutable_weight_filler()->set_type("constant");
  convolution_param->mutable_weight_filler()->set_value(1e-3);
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);
  shared_ptr<Layer<Dtype> > layer(
      new ConvolutionDepthwiseLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  convolution_param->set_group(3);
  shared_ptr<Layer<Dtype> > conv_layer(
      new ConvolutionLayer<Dtype>(layer_param));
  conv_layer->SetUp(this->blob_bottom_vec_, this->conv_top_vec_);
  conv_layer->Forward(this->blob_bottom_vec_, this->conv_top_vec_);
  EXPECT_EQ(this->blob_top_->count(), this->conv_top_->count());
  float err_sum = 0, sum = 0;
  for (int i = 0; i < this->blob_top_->count(); i++) {
    err_sum += std::abs(this->blob_top_->cpu_data()[i] -
        this->conv_top_->cpu_data()[i]);
    sum += std::abs(this->conv_top_->cpu_data()[i]);
  }
  EXPECT_LE(err_sum / sum, 6e-2);
  EXPECT_EQ(this->blob_top_2_->count(), this->conv_top_2_->count());
  float err_sum2 = 0, sum2 = 0;
  for (int i = 0; i < this->blob_top_2_->count(); i++) {
    err_sum2 += std::abs(this->blob_top_2_->cpu_data()[i] -
        this->conv_top_2_->cpu_data()[i]);
    sum2 += std::abs(this->conv_top_2_->cpu_data()[i]);
  }
  EXPECT_LE(err_sum2 / sum2, 6e-2);
}

TYPED_TEST(ConvolutionDepthwiseLayerTest, TestDilatedConvolution) {
  typedef typename TypeParam::Dtype Dtype;
  vector<int> bottom_shape;
  bottom_shape.push_back(2);
  bottom_shape.push_back(3);
  bottom_shape.push_back(8);
  bottom_shape.push_back(7);
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  this->conv_top_vec_.push_back(this->conv_top_2_);
  for (int i = 0; i < this->blob_bottom_vec_.size(); ++i) {
    this->blob_bottom_vec_[i]->Reshape(bottom_shape);
  }
  LayerParameter layer_param;
  layer_param.set_type("ConvolutionDepthwise");
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(3);
  convolution_param->add_dilation(2);
  convolution_param->set_num_output(3);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);
  shared_ptr<Layer<Dtype> > layer(
      new ConvolutionDepthwiseLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  shared_ptr<Layer<Dtype> > conv_layer(
      new ConvolutionLayer<Dtype>(layer_param));
  conv_layer->SetUp(this->blob_bottom_vec_, this->conv_top_vec_);
  conv_layer->Forward(this->blob_bottom_vec_, this->conv_top_vec_);
  EXPECT_EQ(this->blob_top_->count(), this->conv_top_->count());
  float err_sum = 0, sum = 0;
  for (int i = 0; i < this->blob_top_->count(); i++) {
    err_sum += std::abs(this->blob_top_->cpu_data()[i] -
        this->conv_top_->cpu_data()[i]);
    sum += std::abs(this->conv_top_->cpu_data()[i]);
  }
  EXPECT_LE(err_sum / sum, 1e-4);
  EXPECT_EQ(this->blob_top_2_->count(), this->conv_top_2_->count());
  float err_sum2 = 0, sum2 = 0;
  for (int i = 0; i < this->blob_top_2_->count(); i++) {
    err_sum2 += std::abs(this->blob_top_2_->cpu_data()[i] -
        this->conv_top_2_->cpu_data()[i]);
    sum2 += std::abs(this->conv_top_2_->cpu_data()[i]);
  }
  EXPECT_LE(err_sum2 / sum2, 1e-4);
}

TYPED_TEST(ConvolutionDepthwiseLayerTest, Test1x1Convolution) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.set_type("ConvolutionDepthwise");
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(1);
  convolution_param->add_stride(1);
  convolution_param->set_num_output(3);
  convolution_param->mutable_weight_filler()->set_type("constant");
  convolution_param->mutable_weight_filler()->set_value(1e-3);
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);
  shared_ptr<Layer<Dtype> > layer(
      new ConvolutionDepthwiseLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  convolution_param->set_group(3);
  shared_ptr<Layer<Dtype> > conv_layer(
      new ConvolutionLayer<Dtype>(layer_param));
  conv_layer->SetUp(this->blob_bottom_vec_, this->conv_top_vec_);
  conv_layer->Forward(this->blob_bottom_vec_, this->conv_top_vec_);
  EXPECT_EQ(this->blob_top_->count(), this->conv_top_->count());
  float err_sum = 0, sum = 0;
  for (int i = 0; i < this->blob_top_->count(); i++) {
    err_sum += std::abs(this->blob_top_->cpu_data()[i] -
        this->conv_top_->cpu_data()[i]);
    sum += std::abs(this->conv_top_->cpu_data()[i]);
  }
  EXPECT_LE(err_sum / sum, 2e-2);
}

#ifdef USE_MLU
template <typename TypeParam>
class MLUConvolutionDepthwiseLayerTest : public MLUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MLUConvolutionDepthwiseLayerTest()
      : blob_bottom_(new Blob<Dtype>(1, 3, 6, 6)),
        blob_bottom_2_(new Blob<Dtype>(1, 3, 6, 6)),
        blob_top_(new Blob<Dtype>()),
        blob_top_2_(new Blob<Dtype>()),
        conv_top_(new Blob<Dtype>()),
        conv_top_2_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    blob_bottom_vec_.clear();
    blob_top_vec_.clear();
    conv_top_vec_.clear();
    FillerParameter filler_param;
    filler_param.set_value(1.);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_bottom_2_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
    conv_top_vec_.push_back(conv_top_);
  }

  virtual ~MLUConvolutionDepthwiseLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_2_;
    delete blob_top_;
    delete blob_top_2_;
    delete conv_top_;
    delete conv_top_2_;
  }

  virtual Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
    this->ref_blob_top_.reset(new Blob<Dtype>());
    this->ref_blob_top_->ReshapeLike(*top);
    return this->ref_blob_top_.get();
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_2_;
  Blob<Dtype>* const conv_top_;
  Blob<Dtype>* const conv_top_2_;
  shared_ptr<Blob<Dtype> > ref_blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  vector<Blob<Dtype>*> conv_top_vec_;
};

TYPED_TEST_CASE(MLUConvolutionDepthwiseLayerTest, TestMLUDevices);

TYPED_TEST(MLUConvolutionDepthwiseLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(2);
  convolution_param->set_num_output(3);
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  shared_ptr<Layer<Dtype> > layer(
      new MLUConvolutionDepthwiseLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 2);
  EXPECT_EQ(this->blob_top_->width(), 2);
  EXPECT_EQ(this->blob_top_2_->num(), 1);
  EXPECT_EQ(this->blob_top_2_->channels(), 3);
  EXPECT_EQ(this->blob_top_2_->height(), 2);
  EXPECT_EQ(this->blob_top_2_->width(), 2);
  convolution_param->set_num_output(6);
  layer.reset(new ConvolutionDepthwiseLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 6);
  EXPECT_EQ(this->blob_top_->height(), 2);
  EXPECT_EQ(this->blob_top_->width(), 2);
  EXPECT_EQ(this->blob_top_2_->num(), 1);
  EXPECT_EQ(this->blob_top_2_->channels(), 6);
  EXPECT_EQ(this->blob_top_2_->height(), 2);
  EXPECT_EQ(this->blob_top_2_->width(), 2);
}

TYPED_TEST(MLUConvolutionDepthwiseLayerTest, TestSimpleConvolution) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  this->conv_top_vec_.push_back(this->conv_top_2_);
  LayerParameter layer_param;
  layer_param.set_type("ConvolutionDepthwise");
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(2);
  convolution_param->set_num_output(6);
  convolution_param->mutable_weight_filler()->set_type("constant");
  convolution_param->mutable_weight_filler()->set_value(1e-3);
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(1e-3);
  shared_ptr<Layer<Dtype> > layer(
      new MLUConvolutionDepthwiseLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  convolution_param->set_group(3);
  shared_ptr<Layer<Dtype> > conv_layer(
      new ConvolutionLayer<Dtype>(layer_param));
  conv_layer->SetUp(this->blob_bottom_vec_, this->conv_top_vec_);
  conv_layer->Forward(this->blob_bottom_vec_, this->conv_top_vec_);
  EXPECT_EQ(this->blob_top_->count(), this->conv_top_->count());
  float err_sum = 0, sum = 0;
  for (int i = 0; i < this->blob_top_->count(); i++) {
    err_sum += std::abs(this->blob_top_->cpu_data()[i] -
        this->conv_top_->cpu_data()[i]);
    sum += std::abs(this->conv_top_->cpu_data()[i]);
  }
  EXPECT_EQ(this->blob_top_2_->count(), this->conv_top_2_->count());
  float err_sum2 = 0, sum2 = 0;
  for (int i = 0; i < this->blob_top_2_->count(); i++) {
    err_sum2 += std::abs(this->blob_top_2_->cpu_data()[i] -
        this->conv_top_2_->cpu_data()[i]);
    sum2 += std::abs(this->conv_top_2_->cpu_data()[i]);
  }
  EXPECT_LE(err_sum2 / sum2, 4e-2);
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  ERR_RATE(err_sum / sum);
  EVENT_TIME(layer->get_event_time());
}

TYPED_TEST(MLUConvolutionDepthwiseLayerTest, TestConvolutiondiffpad) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  this->conv_top_vec_.push_back(this->conv_top_2_);
  LayerParameter layer_param;
  layer_param.set_type("ConvolutionDepthwise");
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(2);
  convolution_param->set_num_output(6);
  convolution_param->add_pad(2);
  convolution_param->add_pad(2);
  convolution_param->add_pad(1);
  convolution_param->add_pad(1);
  convolution_param->mutable_weight_filler()->set_type("constant");
  convolution_param->mutable_weight_filler()->set_type("constant");
  convolution_param->mutable_weight_filler()->set_value(1e-3);
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(1e-3);
  shared_ptr<Layer<Dtype> > layer(
      new MLUConvolutionDepthwiseLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  convolution_param->set_group(3);
  shared_ptr<Layer<Dtype> > conv_layer(
      new ConvolutionLayer<Dtype>(layer_param));
  conv_layer->SetUp(this->blob_bottom_vec_, this->conv_top_vec_);
  conv_layer->Forward(this->blob_bottom_vec_, this->conv_top_vec_);
  EXPECT_EQ(this->blob_top_->count(), this->conv_top_->count());
  float err_sum = 0, sum = 0;
  for (int i = 0; i < this->blob_top_->count(); i++) {
    err_sum += std::abs(this->blob_top_->cpu_data()[i] -
        this->conv_top_->cpu_data()[i]);
    sum += std::abs(this->conv_top_->cpu_data()[i]);
  }
  EXPECT_EQ(this->blob_top_2_->count(), this->conv_top_2_->count());
  float err_sum2 = 0, sum2 = 0;
  for (int i = 0; i < this->blob_top_2_->count(); i++) {
    err_sum2 += std::abs(this->blob_top_2_->cpu_data()[i] -
        this->conv_top_2_->cpu_data()[i]);
    sum2 += std::abs(this->conv_top_2_->cpu_data()[i]);
  }
  EXPECT_LE(err_sum2 / sum2, 4e-2);
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  ERR_RATE(err_sum / sum);
  EVENT_TIME(layer->get_event_time());
}
TYPED_TEST(MLUConvolutionDepthwiseLayerTest, Test1x1Convolution) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(1);
  convolution_param->add_stride(1);
  convolution_param->set_num_output(3);
  layer_param.set_type("ConvolutionDepthwise");
  convolution_param->mutable_weight_filler()->set_type("constant");
  convolution_param->mutable_weight_filler()->set_value(1e-3);
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);
  shared_ptr<Layer<Dtype> > layer(
      new ConvolutionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  convolution_param->set_group(3);
  shared_ptr<Layer<Dtype> > conv_layer(
      new ConvolutionLayer<Dtype>(layer_param));
  conv_layer->SetUp(this->blob_bottom_vec_, this->conv_top_vec_);
  conv_layer->Forward(this->blob_bottom_vec_, this->conv_top_vec_);
  EXPECT_EQ(this->blob_top_->count(), this->conv_top_->count());
  float err_sum = 0, sum = 0;
  for (int i = 0; i < this->blob_top_->count(); i++) {
    err_sum += std::abs(this->blob_top_->cpu_data()[i] -
        this->conv_top_->cpu_data()[i]);
    sum += std::abs(this->conv_top_->cpu_data()[i]);
  }
  EXPECT_EQ(this->blob_top_2_->count(), this->conv_top_2_->count());
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  ERR_RATE(err_sum / sum);
  EVENT_TIME(layer->get_event_time());
}

template <typename TypeParam>
class MFUSConvolutionDepthwiseLayerTest : public MFUSDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MFUSConvolutionDepthwiseLayerTest()
      : blob_bottom_(new Blob<Dtype>(1, 3, 6, 6)),
        blob_bottom_2_(new Blob<Dtype>(1, 3, 6, 6)),
        blob_top_(new Blob<Dtype>()),
        blob_top_2_(new Blob<Dtype>()),
        conv_top_(new Blob<Dtype>()),
        conv_top_2_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(1.);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_bottom_2_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
    conv_top_vec_.push_back(conv_top_);
  }

  virtual ~MFUSConvolutionDepthwiseLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_2_;
    delete blob_top_;
    delete blob_top_2_;
    delete conv_top_;
    delete conv_top_2_;
  }

  virtual Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
    this->ref_blob_top_.reset(new Blob<Dtype>());
    this->ref_blob_top_->ReshapeLike(*top);
    return this->ref_blob_top_.get();
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_2_;
  Blob<Dtype>* const conv_top_;
  Blob<Dtype>* const conv_top_2_;
  shared_ptr<Blob<Dtype> > ref_blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  vector<Blob<Dtype>*> conv_top_vec_;
};

TYPED_TEST_CASE(MFUSConvolutionDepthwiseLayerTest, TestMFUSDevices);

TYPED_TEST(MFUSConvolutionDepthwiseLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.set_type("ConvolutionDepthwise");
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(2);
  convolution_param->set_num_output(3);
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  shared_ptr<Layer<Dtype> > layer(
      new MLUConvolutionDepthwiseLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 2);
  EXPECT_EQ(this->blob_top_->width(), 2);
  EXPECT_EQ(this->blob_top_2_->num(), 1);
  EXPECT_EQ(this->blob_top_2_->channels(), 3);
  EXPECT_EQ(this->blob_top_2_->height(), 2);
  EXPECT_EQ(this->blob_top_2_->width(), 2);
  convolution_param->set_num_output(6);
  layer.reset(new ConvolutionDepthwiseLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 6);
  EXPECT_EQ(this->blob_top_->height(), 2);
  EXPECT_EQ(this->blob_top_->width(), 2);
  EXPECT_EQ(this->blob_top_2_->num(), 1);
  EXPECT_EQ(this->blob_top_2_->channels(), 6);
  EXPECT_EQ(this->blob_top_2_->height(), 2);
  EXPECT_EQ(this->blob_top_2_->width(), 2);
}

TYPED_TEST(MFUSConvolutionDepthwiseLayerTest, TestSimpleConvolution) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  this->conv_top_vec_.push_back(this->conv_top_2_);
  LayerParameter layer_param;
  layer_param.set_type("ConvolutionDepthwise");
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(2);
  convolution_param->set_num_output(6);
  convolution_param->mutable_weight_filler()->set_type("constant");
  convolution_param->mutable_weight_filler()->set_value(1e-3);
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);
  shared_ptr<Layer<Dtype> > layer(
      new MLUConvolutionDepthwiseLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  MFusion<Dtype> fuser;
  fuser.reset();
  fuser.addInputs(this->blob_bottom_vec_);
  fuser.addOutputs(this->blob_top_vec_);
  layer->Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->fuse(&fuser);
  fuser.compile();
  fuser.forward();
  convolution_param->set_group(3);
  shared_ptr<Layer<Dtype> > conv_layer(
      new ConvolutionLayer<Dtype>(layer_param));
  conv_layer->SetUp(this->blob_bottom_vec_, this->conv_top_vec_);
  conv_layer->Forward(this->blob_bottom_vec_, this->conv_top_vec_);
  EXPECT_EQ(this->blob_top_->count(), this->conv_top_->count());
  float err_sum = 0, sum = 0;
  for (int i = 0; i < this->blob_top_->count(); i++) {
    err_sum += std::abs(this->blob_top_->cpu_data()[i] -
        this->conv_top_->cpu_data()[i]);
    sum += std::abs(this->conv_top_->cpu_data()[i]);
  }
  EXPECT_EQ(this->blob_top_2_->count(), this->conv_top_2_->count());
  float err_sum2 = 0, sum2 = 0;
  for (int i = 0; i < this->blob_top_2_->count(); i++) {
    err_sum2 += std::abs(this->blob_top_2_->cpu_data()[i] -
        this->conv_top_2_->cpu_data()[i]);
    sum2 += std::abs(this->conv_top_2_->cpu_data()[i]);
  }
  EXPECT_LE(err_sum2 / sum2, 4e-2);
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  ERR_RATE(err_sum / sum);
  EVENT_TIME(layer->get_event_time());
}
TYPED_TEST(MFUSConvolutionDepthwiseLayerTest, TestConvolutiondiffpad) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  this->conv_top_vec_.push_back(this->conv_top_2_);
  LayerParameter layer_param;
  layer_param.set_type("ConvolutionDepthwise");
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(2);
  convolution_param->set_num_output(6);
  convolution_param->add_pad(2);
  convolution_param->add_pad(2);
  convolution_param->add_pad(1);
  convolution_param->add_pad(1);
  convolution_param->mutable_weight_filler()->set_type("constant");
  convolution_param->mutable_weight_filler()->set_value(1e-3);
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);
  shared_ptr<Layer<Dtype> > layer(
      new MLUConvolutionDepthwiseLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  MFusion<Dtype> fuser;
  fuser.reset();
  fuser.addInputs(this->blob_bottom_vec_);
  fuser.addOutputs(this->blob_top_vec_);
  layer->Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->fuse(&fuser);
  fuser.compile();
  fuser.forward();
  convolution_param->set_group(3);
  shared_ptr<Layer<Dtype> > conv_layer(
      new ConvolutionLayer<Dtype>(layer_param));
  conv_layer->SetUp(this->blob_bottom_vec_, this->conv_top_vec_);
  conv_layer->Forward(this->blob_bottom_vec_, this->conv_top_vec_);
  EXPECT_EQ(this->blob_top_->count(), this->conv_top_->count());
  float err_sum = 0, sum = 0;
  for (int i = 0; i < this->blob_top_->count(); i++) {
    err_sum += std::abs(this->blob_top_->cpu_data()[i] -
        this->conv_top_->cpu_data()[i]);
    sum += std::abs(this->conv_top_->cpu_data()[i]);
  }
  EXPECT_EQ(this->blob_top_2_->count(), this->conv_top_2_->count());
  float err_sum2 = 0, sum2 = 0;
  for (int i = 0; i < this->blob_top_2_->count(); i++) {
    err_sum2 += std::abs(this->blob_top_2_->cpu_data()[i] -
        this->conv_top_2_->cpu_data()[i]);
    sum2 += std::abs(this->conv_top_2_->cpu_data()[i]);
  }
  EXPECT_LE(err_sum2 / sum2, 4e-2);
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  ERR_RATE(err_sum / sum);
  EVENT_TIME(layer->get_event_time());
}

TYPED_TEST(MFUSConvolutionDepthwiseLayerTest, Test1x1Convolution) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.set_type("ConvolutionDepthwise");
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(1);
  convolution_param->add_stride(1);
  convolution_param->set_num_output(3);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);
  shared_ptr<Layer<Dtype> > layer(
      new MLUConvolutionDepthwiseLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  MFusion<Dtype> fuser;
  fuser.reset();
  fuser.addInputs(this->blob_bottom_vec_);
  fuser.addOutputs(this->blob_top_vec_);
  layer->Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->fuse(&fuser);
  fuser.compile();
  fuser.forward();
  convolution_param->set_group(3);
  shared_ptr<Layer<Dtype> > conv_layer(
      new ConvolutionLayer<Dtype>(layer_param));
  conv_layer->SetUp(this->blob_bottom_vec_, this->conv_top_vec_);
  conv_layer->Forward(this->blob_bottom_vec_, this->conv_top_vec_);
  EXPECT_EQ(this->blob_top_->count(), this->conv_top_->count());
  float err_sum = 0, sum = 0;
  for (int i = 0; i < this->blob_top_->count(); i++) {
    err_sum += std::abs(this->blob_top_->cpu_data()[i] -
        this->conv_top_->cpu_data()[i]);
    sum += std::abs(this->conv_top_->cpu_data()[i]);
  }
  EXPECT_EQ(this->blob_top_2_->count(), this->conv_top_2_->count());
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  ERR_RATE(err_sum / sum);
  EVENT_TIME(layer->get_event_time());
}

#endif

}  // namespace caffe
