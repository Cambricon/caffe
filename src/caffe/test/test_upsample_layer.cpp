/*
All modification made by Cambricon Corporation: © 2018 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2018, the respective contributors
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

#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"

#include "caffe/test/test_gradient_check_util.hpp"

#include "caffe/layers/mlu_pooling_layer.hpp"
#include "caffe/layers/mlu_upsample_layer.hpp"
#include "caffe/layers/pooling_layer.hpp"
#include "caffe/layers/upsample_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
namespace caffe {

template <typename TypeParam>
class UpsampleLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  UpsampleLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()),
        blob_bottom_mask_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    blob_bottom_->Reshape(2, 3, 2, 2);
    blob_bottom_mask_->Reshape(2, 3, 2, 2);
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_mask_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~UpsampleLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
    delete blob_bottom_mask_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_bottom_mask_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

  void TestForwardSquare() {
    LayerParameter layer_param;
    UpsampleParameter* upsample_param = layer_param.mutable_upsample_param();
    upsample_param->set_scale(2);
    const int num = 4;
    const int channels = 3;
    blob_bottom_->Reshape(num, channels, 2, 2);
    blob_bottom_mask_->Reshape(num, channels, 2, 2);
    // Input: 4 * 3 channels of :
    //  [ 1  2 ]
    //  [ 9  4 ]
    for (int i = 0; i < 4 * num * channels; i += 4) {
      blob_bottom_->mutable_cpu_data()[i + 0] = 1;
      blob_bottom_->mutable_cpu_data()[i + 1] = 2;
      blob_bottom_->mutable_cpu_data()[i + 2] = 9;
      blob_bottom_->mutable_cpu_data()[i + 3] = 4;
    }
    // Input mask: 4 * 3 channels of:
    //  [  2  5 ]
    //  [ 12 14 ]
    for (int i = 0; i < 4 * num * channels; i += 4) {
      blob_bottom_mask_->mutable_cpu_data()[i + 0] = 2;
      blob_bottom_mask_->mutable_cpu_data()[i + 1] = 5;
      blob_bottom_mask_->mutable_cpu_data()[i + 2] = 12;
      blob_bottom_mask_->mutable_cpu_data()[i + 3] = 14;
    }
    UpsampleLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(blob_top_->num(), num);
    EXPECT_EQ(blob_top_->channels(), channels);
    EXPECT_EQ(blob_top_->height(), 4);
    EXPECT_EQ(blob_top_->width(), 4);
    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    // Expected output: 4 * 3 channels of:
    //  [ 0 0 1 0 ]
    //  [ 0 2 0 0 ]
    //  [ 0 0 0 0 ]
    //  [ 9 0 4 0 ]
    for (int i = 0; i < 16 * num * channels; i += 16) {
      EXPECT_EQ(blob_top_->cpu_data()[i + 0], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 1], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 2], 1);
      EXPECT_EQ(blob_top_->cpu_data()[i + 3], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 4], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 5], 2);
      EXPECT_EQ(blob_top_->cpu_data()[i + 6], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 7], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 8], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 9], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 10], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 11], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 12], 9);
      EXPECT_EQ(blob_top_->cpu_data()[i + 13], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 14], 4);
      EXPECT_EQ(blob_top_->cpu_data()[i + 15], 0);
    }
  }

  int MapIndexBottomToTop(int bottom_idx, int scale_w, int scale_h,
                          bool randomize) {
    const int input_width = bottom_idx % blob_bottom_->width();
    const int input_height = bottom_idx / blob_bottom_->width();
    const int top_w = scale_w * blob_bottom_->width();
    int out_w =
        scale_w * input_width + (randomize ? caffe_rng_rand() % scale_w : 0);
    int out_h =
        scale_h * input_height + (randomize ? caffe_rng_rand() % scale_h : 0);
    int out_idx = out_w + out_h * top_w;
    return out_idx;
  }

  void FillBottomMask(int scale_w, int scale_h, bool randomize = false) {
    Dtype* mask_data = blob_bottom_mask_->mutable_cpu_data();
    for (int n = 0; n < blob_bottom_->num(); ++n) {
      for (int c = 0; c < blob_bottom_->channels(); ++c) {
        for (int i = 0; i < blob_bottom_->height() * blob_bottom_->width();
             ++i) {
          int idx = MapIndexBottomToTop(i, scale_w, scale_h, randomize);
          mask_data[i] = idx;
        }
        mask_data += blob_bottom_mask_->offset(0, 1);
      }
    }
  }
};

TYPED_TEST_CASE(UpsampleLayerTest, TestDtypesAndDevices);

TYPED_TEST(UpsampleLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UpsampleParameter* upsample_param = layer_param.mutable_upsample_param();
  upsample_param->set_scale_h(2);
  upsample_param->set_scale_w(3);
  UpsampleLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height() * 2);
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width() * 3);
}

TYPED_TEST(UpsampleLayerTest, TestForward) { this->TestForwardSquare(); }

TYPED_TEST(UpsampleLayerTest, TestForwardFromPool) {
  typedef typename TypeParam::Dtype Dtype;
  int kernel_w = 2;
  int kernel_h = 2;
  Blob<Dtype>* input_blob = new Blob<Dtype>();
  input_blob->Reshape(2, 3, 4, 4);
  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(input_blob);
  std::vector<Blob<Dtype>*> pool_bottom_vec;
  pool_bottom_vec.push_back(input_blob);
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_kernel_h(kernel_h);
  pooling_param->set_kernel_w(kernel_w);
  pooling_param->set_stride(2);
  pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
  PoolingLayer<Dtype> pooling_layer(layer_param);
  pooling_layer.SetUp(pool_bottom_vec, this->blob_bottom_vec_);
  EXPECT_EQ(this->blob_bottom_->num(), 2);
  EXPECT_EQ(this->blob_bottom_->channels(), 3);
  EXPECT_EQ(this->blob_bottom_->height(), 2);
  EXPECT_EQ(this->blob_bottom_->width(), 2);
  EXPECT_EQ(this->blob_bottom_->num(), 2);
  EXPECT_EQ(this->blob_bottom_->channels(), 3);
  EXPECT_EQ(this->blob_bottom_->height(), 2);
  EXPECT_EQ(this->blob_bottom_->width(), 2);

  LayerParameter upsample_layer_param;
  UpsampleParameter* upsample_param =
      upsample_layer_param.mutable_upsample_param();
  upsample_param->set_upsample_h(4);
  upsample_param->set_upsample_w(4);
  UpsampleLayer<Dtype> upsample_layer(upsample_layer_param);
  upsample_layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 4);
  EXPECT_EQ(this->blob_top_->width(), 4);

  pooling_layer.Forward(pool_bottom_vec, this->blob_bottom_vec_);
  upsample_layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  const Dtype* top_data = this->blob_top_->cpu_data();
  const Dtype* pool_bottom_data = input_blob->cpu_data();
  int num_zeros = 0;
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    if (top_data[i] != 0) {
      EXPECT_EQ(top_data[i], pool_bottom_data[i]);
    } else {
      ++num_zeros;
    }
  }
  EXPECT_EQ(num_zeros, (16 - 4) * 2 * 3);
}

TYPED_TEST(UpsampleLayerTest, TestForwardFromPoolOddShape) {
  typedef typename TypeParam::Dtype Dtype;
  int kernel_w = 2;
  int kernel_h = 2;
  Blob<Dtype>* input_blob = new Blob<Dtype>();
  input_blob->Reshape(2, 3, 5, 4);
  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(input_blob);
  std::vector<Blob<Dtype>*> pool_bottom_vec;
  pool_bottom_vec.push_back(input_blob);
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_kernel_h(kernel_h);
  pooling_param->set_kernel_w(kernel_w);
  pooling_param->set_stride(2);
  pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
  PoolingLayer<Dtype> pooling_layer(layer_param);
  pooling_layer.SetUp(pool_bottom_vec, this->blob_bottom_vec_);
  EXPECT_EQ(this->blob_bottom_->num(), 2);
  EXPECT_EQ(this->blob_bottom_->channels(), 3);
  EXPECT_EQ(this->blob_bottom_->height(), 3);
  EXPECT_EQ(this->blob_bottom_->width(), 2);
  EXPECT_EQ(this->blob_bottom_mask_->num(), 2);
  EXPECT_EQ(this->blob_bottom_mask_->channels(), 3);
  EXPECT_EQ(this->blob_bottom_mask_->height(), 3);
  EXPECT_EQ(this->blob_bottom_mask_->width(), 2);

  LayerParameter upsample_layer_param;
  UpsampleParameter* upsample_param =
      upsample_layer_param.mutable_upsample_param();
  upsample_param->set_upsample_h(5);
  upsample_param->set_upsample_w(4);
  UpsampleLayer<Dtype> upsample_layer(upsample_layer_param);
  upsample_layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 5);
  EXPECT_EQ(this->blob_top_->width(), 4);

  pooling_layer.Forward(pool_bottom_vec, this->blob_bottom_vec_);
  upsample_layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* top_data = this->blob_top_->cpu_data();
  const Dtype* pool_bottom_data = input_blob->cpu_data();
  int num_zeros = 0;
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    if (top_data[i] != 0) {
      EXPECT_EQ(top_data[i], pool_bottom_data[i]);
    } else {
      ++num_zeros;
    }
  }
  EXPECT_EQ(num_zeros, (5 * 4 - 6) * 2 * 3);
}

TYPED_TEST(UpsampleLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  for (int scale_h = 2; scale_h <= 3; ++scale_h) {
    for (int scale_w = 2; scale_w <= 3; ++scale_w) {
      LayerParameter layer_param;
      UpsampleParameter* upsample_param = layer_param.mutable_upsample_param();
      upsample_param->set_scale_h(scale_h);
      upsample_param->set_scale_w(scale_w);
      this->FillBottomMask(scale_w, scale_h);
      UpsampleLayer<Dtype> layer(layer_param);
      layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
      GradientChecker<Dtype> checker(1e-4, 1e-2);
      checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                      this->blob_top_vec_, 0);
    }
  }
}

#ifdef USE_MLU

template <typename TypeParam>
class MLUUpsampleLayerTest : public MLUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MLUUpsampleLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()),
        blob_bottom_index_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    blob_bottom_->Reshape(2, 3, 2, 2);
    blob_bottom_index_->Reshape(2, 3, 2, 2);
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_index_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MLUUpsampleLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
    delete blob_bottom_index_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_bottom_index_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MLUUpsampleLayerTest, TestMLUDevices);

TYPED_TEST(MLUUpsampleLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UpsampleParameter* upsample_param = layer_param.mutable_upsample_param();
  upsample_param->set_scale_h(2);
  upsample_param->set_scale_w(3);
  MLUUpsampleLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height() * 2);
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width() * 3);
  OUTPUT("bottom1", this->blob_bottom_->shape_string().c_str());
  OUTPUT("bottom2", this->blob_bottom_index_->shape_string().c_str());
}

TYPED_TEST(MLUUpsampleLayerTest, TestForwardFromPool) {
  if(Caffe::rt_core() > 4)return;
  typedef typename TypeParam::Dtype Dtype;
  int kernel_w = 2;
  int kernel_h = 2;
  Blob<Dtype>* input_blob = new Blob<Dtype>();
  input_blob->Reshape(2, 3, 4, 4);
  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(input_blob);
  std::vector<Blob<Dtype>*> pool_bottom_vec;
  pool_bottom_vec.push_back(input_blob);
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_kernel_h(kernel_h);
  pooling_param->set_kernel_w(kernel_w);
  pooling_param->set_stride(2);
  pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
  MLUPoolingLayer<Dtype> pooling_layer(layer_param);
  pooling_layer.SetUp(pool_bottom_vec, this->blob_bottom_vec_);
  EXPECT_EQ(this->blob_bottom_->num(), 2);
  EXPECT_EQ(this->blob_bottom_->channels(), 3);
  EXPECT_EQ(this->blob_bottom_->height(), 2);
  EXPECT_EQ(this->blob_bottom_->width(), 2);
  EXPECT_EQ(this->blob_bottom_->num(), 2);
  EXPECT_EQ(this->blob_bottom_->channels(), 3);
  EXPECT_EQ(this->blob_bottom_->height(), 2);
  EXPECT_EQ(this->blob_bottom_->width(), 2);

  LayerParameter upsample_layer_param;
  UpsampleParameter* upsample_param =
      upsample_layer_param.mutable_upsample_param();
  upsample_param->set_upsample_h(4);
  upsample_param->set_upsample_w(4);
  MLUUpsampleLayer<Dtype> upsample_layer(upsample_layer_param);
  upsample_layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->width(), 4);
  EXPECT_EQ(this->blob_top_->height(), 4);

  pooling_layer.Reshape_dispatch(pool_bottom_vec, this->blob_bottom_vec_);
  pooling_layer.Forward(pool_bottom_vec, this->blob_bottom_vec_);
  upsample_layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  upsample_layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  const Dtype* top_data = this->blob_top_->cpu_data();
  const Dtype* pool_bottom_data = input_blob->cpu_data();
  int num_zeros = 0;
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    if (top_data[i] != 0) {
      EXPECT_EQ(top_data[i], pool_bottom_data[i]);
    } else {
      ++num_zeros;
    }
  }
  EXPECT_EQ(num_zeros, (16 - 4) * 2 * 3);
  EVENT_TIME(upsample_layer.get_event_time());
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str() << "\t"
    << "bottom2:" << this->blob_bottom_index_->shape_string().c_str();
  BOTTOM(stream);
  ERR_RATE(num_zeros/1.0 -((16 - 4) * 2 * 3));
}

TYPED_TEST(MLUUpsampleLayerTest, TestForwardNearestNeighbor) {
  typedef typename TypeParam::Dtype Dtype;
  int kernel_w = 2;
  int kernel_h = 2;
  Blob<Dtype>* input_blob = new Blob<Dtype>();
  input_blob->Reshape(2, 3, 2, 2);
  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(input_blob);
  this->blob_bottom_vec_.pop_back();
  this->blob_bottom_vec_.pop_back();
  this->blob_bottom_vec_.push_back(input_blob);
  LayerParameter upsample_layer_param;
  UpsampleParameter* upsample_param =
    upsample_layer_param.mutable_upsample_param();
  upsample_param->set_nearestneighbor_mode(true);
  upsample_param->set_upsample_h(4);
  upsample_param->set_upsample_w(4);
  MLUUpsampleLayer<Dtype> upsample_layer(upsample_layer_param);

  upsample_layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->width(), 4);
  EXPECT_EQ(this->blob_top_->height(), 4);

  upsample_layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  upsample_layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype cpu_top[2][3][4][4];
  const Dtype* top_data = this->blob_top_->cpu_data();
  const Dtype* bottom_data = input_blob->cpu_data();
  for (int i=0; i < input_blob->count();++i){
    int row = ( i % 4 ) / 2;
    int col = ( i % 4 ) % 2;
    cpu_top[i/12][(i%12)/4][row*kernel_h][col*kernel_w] = bottom_data[i];
    cpu_top[i/12][(i%12)/4][row*kernel_h][col*kernel_w+1] = bottom_data[i];
    cpu_top[i/12][(i%12)/4][row*kernel_h+1][col*kernel_w] = bottom_data[i];
    cpu_top[i/12][(i%12)/4][row*kernel_h+1][col*kernel_w+1] = bottom_data[i];
  }
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_EQ(top_data[i],cpu_top[i/48][(i%48)/16][(i%16)/4][(i%16)%4]);
  } 
  EVENT_TIME(upsample_layer.get_event_time());
}
TYPED_TEST(MLUUpsampleLayerTest, TestForwardFromPoolOddShape) {
  if(Caffe::rt_core() >4)return;
  typedef typename TypeParam::Dtype Dtype;
  int kernel_w = 2;
  int kernel_h = 2;
  Blob<Dtype>* input_blob = new Blob<Dtype>();
  input_blob->Reshape(3, 5, 8, 10);
  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(input_blob);
  std::vector<Blob<Dtype>*> pool_bottom_vec;
  pool_bottom_vec.push_back(input_blob);
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_kernel_h(kernel_h);
  pooling_param->set_kernel_w(kernel_w);
  pooling_param->set_stride(2);
  pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
  MLUPoolingLayer<Dtype> pooling_layer(layer_param);
  pooling_layer.SetUp(pool_bottom_vec, this->blob_bottom_vec_);
  EXPECT_EQ(this->blob_bottom_->num(), 3);
  EXPECT_EQ(this->blob_bottom_->channels(), 5);
  EXPECT_EQ(this->blob_bottom_->height(), 4);
  EXPECT_EQ(this->blob_bottom_->width(), 5);
  EXPECT_EQ(this->blob_bottom_index_->num(), 3);
  EXPECT_EQ(this->blob_bottom_index_->channels(), 5);
  EXPECT_EQ(this->blob_bottom_index_->height(), 4);
  EXPECT_EQ(this->blob_bottom_index_->width(), 5);

  LayerParameter upsample_layer_param;
  UpsampleParameter* upsample_param =
      upsample_layer_param.mutable_upsample_param();
  upsample_param->set_upsample_h(8);
  upsample_param->set_upsample_w(10);
  MLUUpsampleLayer<Dtype> upsample_layer(upsample_layer_param);
  upsample_layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 3);
  EXPECT_EQ(this->blob_top_->channels(), 5);
  EXPECT_EQ(this->blob_top_->height(), 8);
  EXPECT_EQ(this->blob_top_->width(), 10);

  pooling_layer.Reshape_dispatch(pool_bottom_vec, this->blob_bottom_vec_);
  pooling_layer.Forward(pool_bottom_vec, this->blob_bottom_vec_);
  upsample_layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  upsample_layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* top_data = this->blob_top_->cpu_data();
  const Dtype* pool_bottom_data = input_blob->cpu_data();
  int num_zeros = 0;
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    if (top_data[i] != 0) {
      EXPECT_EQ(top_data[i], pool_bottom_data[i]);
    } else {
      ++num_zeros;
    }
  }
  EXPECT_EQ(num_zeros, (8 * 10 - 4 * 5) * 3 * 5);
  EVENT_TIME(upsample_layer.get_event_time());
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str() << "\t"
    << "bottom2:" << this->blob_bottom_index_->shape_string().c_str();
  BOTTOM(stream);
  ERR_RATE(num_zeros/1.0 -((8 * 10 - 4 * 5) * 3 * 5));
}

TYPED_TEST(MLUUpsampleLayerTest, TestForwardNearestNeighborOddShape) {
  typedef typename TypeParam::Dtype Dtype;
  int kernel_w = 2;
  int kernel_h = 2;
  Blob<Dtype>* input_blob = new Blob<Dtype>();
  input_blob->Reshape(3, 5, 4, 5);
  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(input_blob);
  this->blob_bottom_vec_.pop_back();
  this->blob_bottom_vec_.pop_back();
  this->blob_bottom_vec_.push_back(input_blob);
  LayerParameter upsample_layer_param;
  UpsampleParameter* upsample_param =
      upsample_layer_param.mutable_upsample_param();
  upsample_param->set_upsample_h(8);
  upsample_param->set_upsample_w(10);
  upsample_param->set_nearestneighbor_mode(true);
  MLUUpsampleLayer<Dtype> upsample_layer(upsample_layer_param);
  upsample_layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 3);
  EXPECT_EQ(this->blob_top_->channels(), 5);
  EXPECT_EQ(this->blob_top_->height(), 8);
  EXPECT_EQ(this->blob_top_->width(), 10);
  upsample_layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  upsample_layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype cpu_top[3][5][8][10];
  const Dtype* top_data = this->blob_top_->cpu_data();
  const Dtype* bottom_data = input_blob->cpu_data();
  for (int i=0; i < input_blob->count();++i){
    int row = ( i % 20 ) / 5;
    int col = ( i % 20 ) % 5;
    cpu_top[i/100][(i%100)/20][row*kernel_h][col*kernel_w]=bottom_data[i];
    cpu_top[i/100][(i%100)/20][row*kernel_h][col*kernel_w+1]=bottom_data[i];
    cpu_top[i/100][(i%100)/20][row*kernel_h+1][col*kernel_w]=bottom_data[i];
    cpu_top[i/100][(i%100)/20][row*kernel_h+1][col*kernel_w+1]=bottom_data[i];
  }
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_EQ(top_data[i],cpu_top[i/400][(i%400)/80][(i%80)/10][(i%80)%10]);
  } 
  EVENT_TIME(upsample_layer.get_event_time());
}

template <typename TypeParam>
class MFUSUpsampleLayerTest : public MFUSDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MFUSUpsampleLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()),
        blob_bottom_index_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    blob_bottom_->Reshape(2, 3, 2, 2);
    blob_bottom_index_->Reshape(2, 3, 2, 2);
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_index_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MFUSUpsampleLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
    delete blob_bottom_index_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_bottom_index_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MFUSUpsampleLayerTest, TestMFUSDevices);

TYPED_TEST(MFUSUpsampleLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UpsampleParameter* upsample_param = layer_param.mutable_upsample_param();
  upsample_param->set_scale_h(2);
  upsample_param->set_scale_w(3);
  MLUUpsampleLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height() * 2);
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width() * 3);
  OUTPUT("bottom1", this->blob_bottom_->shape_string().c_str());
  OUTPUT("bottom2", this->blob_bottom_index_->shape_string().c_str());
}

TYPED_TEST(MFUSUpsampleLayerTest, TestForwardFromPool) {
  if( Caffe::rt_core() > 4)return;
  typedef typename TypeParam::Dtype Dtype;
  int kernel_w = 2;
  int kernel_h = 2;
  Blob<Dtype>* input_blob = new Blob<Dtype>();
  input_blob->Reshape(2, 3, 4, 4);
  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(input_blob);
  std::vector<Blob<Dtype>*> pool_bottom_vec;
  pool_bottom_vec.push_back(input_blob);
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_kernel_h(kernel_h);
  pooling_param->set_kernel_w(kernel_w);
  pooling_param->set_stride(2);
  pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
  MLUPoolingLayer<Dtype> pooling_layer(layer_param);
  pooling_layer.SetUp(pool_bottom_vec, this->blob_bottom_vec_);
  EXPECT_EQ(this->blob_bottom_->num(), 2);
  EXPECT_EQ(this->blob_bottom_->channels(), 3);
  EXPECT_EQ(this->blob_bottom_->height(), 2);
  EXPECT_EQ(this->blob_bottom_->width(), 2);
  EXPECT_EQ(this->blob_bottom_->num(), 2);
  EXPECT_EQ(this->blob_bottom_->channels(), 3);
  EXPECT_EQ(this->blob_bottom_->height(), 2);
  EXPECT_EQ(this->blob_bottom_->width(), 2);

  LayerParameter upsample_layer_param;
  UpsampleParameter* upsample_param =
      upsample_layer_param.mutable_upsample_param();
  upsample_param->set_upsample_h(4);
  upsample_param->set_upsample_w(4);
  MLUUpsampleLayer<Dtype> upsample_layer(upsample_layer_param);
  upsample_layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->width(), 4);
  EXPECT_EQ(this->blob_top_->height(), 4);

  ASSERT_TRUE(pooling_layer.mfus_supported());

  MFusion<Dtype> fuser_pool;
  fuser_pool.reset();
  fuser_pool.addInputs(pool_bottom_vec);
  fuser_pool.addOutputs(this->blob_bottom_vec_);
  pooling_layer.Reshape_dispatch(pool_bottom_vec, this->blob_bottom_vec_);
  pooling_layer.fuse(&fuser_pool);
  fuser_pool.compile();
  fuser_pool.forward();

  ASSERT_TRUE(upsample_layer.mfus_supported());

  MFusion<Dtype> fuser;
  fuser.reset();
  fuser.addInputs(this->blob_bottom_vec_);
  fuser.addOutputs(this->blob_top_vec_);
  upsample_layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  upsample_layer.fuse(&fuser);
  fuser.compile();
  fuser.forward();

  const Dtype* top_data = this->blob_top_->cpu_data();
  const Dtype* pool_bottom_data = input_blob->cpu_data();
  int num_zeros = 0;
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    if (top_data[i] != 0) {
      EXPECT_EQ(top_data[i], pool_bottom_data[i]);
    } else {
      ++num_zeros;
    }
  }
  EXPECT_EQ(num_zeros, (16 - 4) * 2 * 3);
  EVENT_TIME(fuser.get_event_time());
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str() << "\t"
    << "bottom2:" << this->blob_bottom_index_->shape_string().c_str();
  BOTTOM(stream);
  ERR_RATE(num_zeros/1.0 -((16 - 4) * 2 * 3));
}
TYPED_TEST(MFUSUpsampleLayerTest, TestForwardFromNearestNeighbor) {
  typedef typename TypeParam::Dtype Dtype;
  int kernel_w = 2;
  int kernel_h = 2;
  Blob<Dtype>* input_blob = new Blob<Dtype>();
  input_blob->Reshape(2, 3, 2, 2);
  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(input_blob);
  this->blob_bottom_vec_.pop_back();
  this->blob_bottom_vec_.pop_back();
  this->blob_bottom_vec_.push_back(input_blob);
  LayerParameter upsample_layer_param;
  UpsampleParameter* upsample_param =
      upsample_layer_param.mutable_upsample_param();
  upsample_param->set_nearestneighbor_mode(true);
  upsample_param->set_upsample_h(4);
  upsample_param->set_upsample_w(4);
  MLUUpsampleLayer<Dtype> upsample_layer(upsample_layer_param);
  upsample_layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->width(), 4);
  EXPECT_EQ(this->blob_top_->height(), 4);

  ASSERT_TRUE(upsample_layer.mfus_supported());

  MFusion<Dtype> fuser;
  fuser.reset();
  fuser.addInputs(this->blob_bottom_vec_);
  fuser.addOutputs(this->blob_top_vec_);
  upsample_layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  upsample_layer.fuse(&fuser);
  fuser.compile();
  fuser.forward();
  Dtype cpu_top[2][3][4][4];
  const Dtype* top_data = this->blob_top_->cpu_data();
  const Dtype* bottom_data = input_blob->cpu_data();
  for (int i=0; i < input_blob->count();++i){
    int row = ( i % 4 ) / 2;
    int col = ( i % 4 ) % 2;
    cpu_top[i/12][(i%12)/4][row*kernel_h][col*kernel_w]=bottom_data[i];
    cpu_top[i/12][(i%12)/4][row*kernel_h][col*kernel_w+1]=bottom_data[i];
    cpu_top[i/12][(i%12)/4][row*kernel_h+1][col*kernel_w]=bottom_data[i];
    cpu_top[i/12][(i%12)/4][row*kernel_h+1][col*kernel_w+1]=bottom_data[i];
  }
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_EQ(top_data[i],cpu_top[i/48][(i%48)/16][(i%16)/4][(i%16)%4]);
  }
  EVENT_TIME(fuser.get_event_time());
}

TYPED_TEST(MFUSUpsampleLayerTest, TestForwardFromPoolOddShape) {
  if(Caffe::rt_core() >4)return;
  typedef typename TypeParam::Dtype Dtype;
  int kernel_w = 2;
  int kernel_h = 2;
  Blob<Dtype>* input_blob = new Blob<Dtype>();
  input_blob->Reshape(3, 5, 8, 10);
  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(input_blob);
  std::vector<Blob<Dtype>*> pool_bottom_vec;
  pool_bottom_vec.push_back(input_blob);
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_kernel_h(kernel_h);
  pooling_param->set_kernel_w(kernel_w);
  pooling_param->set_stride(2);
  pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
  MLUPoolingLayer<Dtype> pooling_layer(layer_param);
  pooling_layer.SetUp(pool_bottom_vec, this->blob_bottom_vec_);
  EXPECT_EQ(this->blob_bottom_->num(), 3);
  EXPECT_EQ(this->blob_bottom_->channels(), 5);
  EXPECT_EQ(this->blob_bottom_->height(), 4);
  EXPECT_EQ(this->blob_bottom_->width(), 5);
  EXPECT_EQ(this->blob_bottom_index_->num(), 3);
  EXPECT_EQ(this->blob_bottom_index_->channels(), 5);
  EXPECT_EQ(this->blob_bottom_index_->height(), 4);
  EXPECT_EQ(this->blob_bottom_index_->width(), 5);

  LayerParameter upsample_layer_param;
  UpsampleParameter* upsample_param =
      upsample_layer_param.mutable_upsample_param();
  upsample_param->set_upsample_h(8);
  upsample_param->set_upsample_w(10);
  MLUUpsampleLayer<Dtype> upsample_layer(upsample_layer_param);
  upsample_layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 3);
  EXPECT_EQ(this->blob_top_->channels(), 5);
  EXPECT_EQ(this->blob_top_->height(), 8);
  EXPECT_EQ(this->blob_top_->width(), 10);
  ASSERT_TRUE(pooling_layer.mfus_supported());

  MFusion<Dtype> fuser_pool;
  fuser_pool.reset();
  fuser_pool.addInputs(pool_bottom_vec);
  fuser_pool.addOutputs(this->blob_bottom_vec_);
  pooling_layer.Reshape_dispatch(pool_bottom_vec, this->blob_bottom_vec_);
  pooling_layer.fuse(&fuser_pool);
  fuser_pool.compile();
  fuser_pool.forward();

  ASSERT_TRUE(upsample_layer.mfus_supported());

  MFusion<Dtype> fuser;
  fuser.reset();
  fuser.addInputs(this->blob_bottom_vec_);
  fuser.addOutputs(this->blob_top_vec_);
  upsample_layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  upsample_layer.fuse(&fuser);
  fuser.compile();
  fuser.forward();
  const Dtype* top_data = this->blob_top_->cpu_data();
  const Dtype* pool_bottom_data = input_blob->cpu_data();
  int num_zeros = 0;
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    if (top_data[i] != 0) {
      EXPECT_EQ(top_data[i], pool_bottom_data[i]);
    } else {
      ++num_zeros;
    }
  }
  EXPECT_EQ(num_zeros, (8 * 10 - 4 * 5) * 3 * 5);
  EVENT_TIME(upsample_layer.get_event_time());
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str() << "\t"
    << "bottom2:" << this->blob_bottom_index_->shape_string().c_str();
  BOTTOM(stream);
  ERR_RATE(num_zeros/1.0 -((8 * 10 - 4 * 5) * 3 * 5));
}
TYPED_TEST(MFUSUpsampleLayerTest, TestForwardNearestNeighborOddShape) {
  typedef typename TypeParam::Dtype Dtype;
  int kernel_w = 2;
  int kernel_h = 2;
  Blob<Dtype>* input_blob = new Blob<Dtype>();
  input_blob->Reshape(3, 5, 4, 5);
  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(input_blob);
  this->blob_bottom_vec_.pop_back();
  this->blob_bottom_vec_.pop_back();
  this->blob_bottom_vec_.push_back(input_blob);
  LayerParameter upsample_layer_param;
  UpsampleParameter* upsample_param =
      upsample_layer_param.mutable_upsample_param();
  upsample_param->set_upsample_h(8);
  upsample_param->set_upsample_w(10);
  upsample_param->set_nearestneighbor_mode(true);
  MLUUpsampleLayer<Dtype> upsample_layer(upsample_layer_param);
  upsample_layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 3);
  EXPECT_EQ(this->blob_top_->channels(), 5);
  EXPECT_EQ(this->blob_top_->height(), 8);
  EXPECT_EQ(this->blob_top_->width(), 10);
  ASSERT_TRUE(upsample_layer.mfus_supported());
  MFusion<Dtype> fuser;
  fuser.reset();
  fuser.addInputs(this->blob_bottom_vec_);
  fuser.addOutputs(this->blob_top_vec_);
  upsample_layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  upsample_layer.fuse(&fuser);
  fuser.compile();
  fuser.forward();
  Dtype cpu_top[3][5][8][10];
  const Dtype* top_data = this->blob_top_->cpu_data();
  const Dtype* bottom_data = input_blob->cpu_data();
  for (int i=0; i < input_blob->count();++i){
    int row = ( i % 20 ) / 5;
    int col = ( i % 20 ) % 5;
    cpu_top[i/100][(i%100)/20][row*kernel_h][col*kernel_w]=bottom_data[i];
    cpu_top[i/100][(i%100)/20][row*kernel_h][col*kernel_w+1]=bottom_data[i];
    cpu_top[i/100][(i%100)/20][row*kernel_h+1][col*kernel_w]=bottom_data[i];
    cpu_top[i/100][(i%100)/20][row*kernel_h+1][col*kernel_w+1]=bottom_data[i];
  }
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_EQ(top_data[i],cpu_top[i/400][(i%400)/80][(i%80)/10][(i%80)%10]);
  }
  EVENT_TIME(fuser.get_event_time());
}
#endif

}  // namespace caffe
