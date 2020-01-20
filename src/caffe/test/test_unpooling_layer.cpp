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

#include <vector>
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/unpooling_layer.hpp"

#ifdef USE_MLU
#include "caffe/layers/mlu_unpooling_layer.hpp"
#endif

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class UnPoolingLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  UnPoolingLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()),
        blob_bottom_mask_(new Blob<Dtype>()) {}

  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    blob_bottom_->Reshape(2, 3, 2, 4);
    blob_bottom_mask_->Reshape(2, 3, 2, 4);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_bottom_mask_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~UnPoolingLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
    delete blob_bottom_mask_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_bottom_mask_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(UnPoolingLayerTest, TestDtypesAndDevices);

TYPED_TEST(UnPoolingLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;

  LayerParameter layer_param;
  UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_out_kernel_size(2);
  unpooling_param->set_out_stride(1);
  unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_FIXED);
  UnPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 5);
}

TYPED_TEST(UnPoolingLayerTest, TestSetupPadded) {
  typedef typename TypeParam::Dtype Dtype;

  LayerParameter layer_param;
  UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_out_kernel_size(2);
  unpooling_param->set_out_stride(1);
  unpooling_param->set_out_pad(1);
  unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_FIXED);
  UnPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 3);
}

TYPED_TEST(UnPoolingLayerTest, TestForwardFixed) {
  typedef typename TypeParam::Dtype Dtype;

  LayerParameter layer_param;
  UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_out_kernel_size(2);
  unpooling_param->set_out_stride(1);
  unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_FIXED);
  const int num = 2;
  const int channels = 3;
  this->blob_bottom_->Reshape(num, channels, 2, 4);
  //     [9 5 5 8]
  //     [9 5 5 8]
  for (int i = 0; i < this->blob_bottom_->count(); i += 8) {
    this->blob_bottom_->mutable_cpu_data()[i + 0] = 9;
    this->blob_bottom_->mutable_cpu_data()[i + 1] = 5;
    this->blob_bottom_->mutable_cpu_data()[i + 2] = 5;
    this->blob_bottom_->mutable_cpu_data()[i + 3] = 8;
    this->blob_bottom_->mutable_cpu_data()[i + 4] = 9;
    this->blob_bottom_->mutable_cpu_data()[i + 5] = 5;
    this->blob_bottom_->mutable_cpu_data()[i + 6] = 5;
    this->blob_bottom_->mutable_cpu_data()[i + 7] = 8;
  }

  UnPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 5);

  for (int i = 0; i < this->blob_top_->count(); i += 15) {
    Dtype epsilon = 1e-8;
    // output:
    //     [ 9 5 5 8 0 ]
    //     [ 9 5 5 8 0 ]
    //     [ 0 0 0 0 0 ]
    EXPECT_NEAR(this->blob_top_->cpu_data()[0], 9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[1], 5, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[2], 5, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[3], 8, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[4], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[5], 9, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[6], 5, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[7], 5, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[8], 8, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[9], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[10], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[11], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[12], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[13], 0, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[14], 0, epsilon);
  }
}

TYPED_TEST(UnPoolingLayerTest, TestForward2x2Div) {
  typedef typename TypeParam::Dtype Dtype;

  LayerParameter layer_param;
  UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_out_kernel_size(2);
  unpooling_param->set_out_stride(2);
  unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_DIV);
  const int num = 2;
  const int channels = 3;
  this->blob_bottom_->Reshape(num, channels, 2, 2);
  //     [1 2]
  //     [3 4]
  for (int i = 0; i < this->blob_bottom_->count(); i += 4) {
    this->blob_bottom_->mutable_cpu_data()[i + 0] = 1;
    this->blob_bottom_->mutable_cpu_data()[i + 1] = 2;
    this->blob_bottom_->mutable_cpu_data()[i + 2] = 3;
    this->blob_bottom_->mutable_cpu_data()[i + 3] = 4;
  }

  UnPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), 4);
  EXPECT_EQ(this->blob_top_->width(), 4);

  for (int i = 0; i < this->blob_bottom_->count(); i += 16) {
    Dtype epsilon = 1e-8;
    // output:
    //     [0.25 0.25 0.50 0.50]
    //     [0.25 0.25 0.50 0.50]
    //     [0.75 0.75 1.00 1.00]
    //     [0.75 0.75 1.00 1.00]
    EXPECT_NEAR(this->blob_top_->cpu_data()[i + 0], 0.25, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i + 1], 0.25, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i + 2], 0.50, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i + 3], 0.50, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i + 4], 0.25, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i + 5], 0.25, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i + 6], 0.50, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i + 7], 0.50, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i + 8], 0.75, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i + 9], 0.75, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i + 10], 1.00, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i + 11], 1.00, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i + 12], 0.75, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i + 13], 0.75, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i + 14], 1.00, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i + 15], 1.00, epsilon);
  }
}

TYPED_TEST(UnPoolingLayerTest, TestForward2x2Rep) {
  typedef typename TypeParam::Dtype Dtype;

  LayerParameter layer_param;
  UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_out_kernel_size(2);
  unpooling_param->set_out_stride(2);
  unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_REP);
  const int num = 2;
  const int channels = 3;
  this->blob_bottom_->Reshape(num, channels, 2, 2);
  //     [1 2]
  //     [3 4]
  for (int i = 0; i < this->blob_bottom_->count(); i += 4) {
    this->blob_bottom_->mutable_cpu_data()[i + 0] = 1;
    this->blob_bottom_->mutable_cpu_data()[i + 1] = 2;
    this->blob_bottom_->mutable_cpu_data()[i + 2] = 3;
    this->blob_bottom_->mutable_cpu_data()[i + 3] = 4;
  }

  UnPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), 4);
  EXPECT_EQ(this->blob_top_->width(), 4);

  for (int i = 0; i < this->blob_bottom_->count(); i += 16) {
    Dtype epsilon = 1e-8;
    // output:
    //     [1 1 2 2]
    //     [1 1 2 2]
    //     [3 3 4 4]
    //     [3 3 4 4]
    EXPECT_NEAR(this->blob_top_->cpu_data()[i + 0], 1, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i + 1], 1, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i + 2], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i + 3], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i + 4], 1, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i + 5], 1, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i + 6], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i + 7], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i + 8], 3, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i + 9], 3, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i + 10], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i + 11], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i + 12], 3, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i + 13], 3, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i + 14], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i + 15], 4, epsilon);
  }
}

TYPED_TEST(UnPoolingLayerTest, TestForward2x2Fixed) {
  typedef typename TypeParam::Dtype Dtype;

  LayerParameter layer_param;
  UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_out_kernel_size(2);
  unpooling_param->set_out_stride(2);
  unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_FIXED);
  const int num = 2;
  const int channels = 3;
  this->blob_bottom_->Reshape(num, channels, 2, 2);
  //     [1 2]
  //     [3 4]
  for (int i = 0; i < this->blob_bottom_->count(); i += 4) {
    this->blob_bottom_->mutable_cpu_data()[i + 0] = 1;
    this->blob_bottom_->mutable_cpu_data()[i + 1] = 2;
    this->blob_bottom_->mutable_cpu_data()[i + 2] = 3;
    this->blob_bottom_->mutable_cpu_data()[i + 3] = 4;
  }

  UnPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), 4);
  EXPECT_EQ(this->blob_top_->width(), 4);

  for (int i = 0; i < this->blob_bottom_->count(); i += 16) {
    Dtype epsilon = 1e-8;
    // output:
    // * stands for 0 for clarity
    //     [1 * 2 *]
    //     [* * * *]
    //     [3 * 4 *]
    //     [* * * *]
    EXPECT_NEAR(this->blob_top_->cpu_data()[i + 0], 1, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i + 2], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i + 8], 3, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i + 10], 4, epsilon);
    for (int j = 0; j < 16; j++) {
      if (j != 0 && j != 2 && j != 8 && j != 10) {
        EXPECT_NEAR(this->blob_top_->cpu_data()[i + j], 0, epsilon);
      }
    }
  }
}

TYPED_TEST(UnPoolingLayerTest, TestForward3x3Fixed) {
  typedef typename TypeParam::Dtype Dtype;

  LayerParameter layer_param;
  UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_out_kernel_size(3);
  unpooling_param->set_out_stride(3);
  unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_FIXED);
  const int num = 2;
  const int channels = 3;
  this->blob_bottom_->Reshape(num, channels, 3, 3);
  //     [1 2 3]
  //     [4 5 6]
  //     [7 8 9]
  for (int i = 0; i < this->blob_bottom_->count(); i += 9) {
    this->blob_bottom_->mutable_cpu_data()[i + 0] = 1;
    this->blob_bottom_->mutable_cpu_data()[i + 1] = 2;
    this->blob_bottom_->mutable_cpu_data()[i + 2] = 3;
    this->blob_bottom_->mutable_cpu_data()[i + 3] = 4;
    this->blob_bottom_->mutable_cpu_data()[i + 4] = 5;
    this->blob_bottom_->mutable_cpu_data()[i + 5] = 6;
    this->blob_bottom_->mutable_cpu_data()[i + 6] = 7;
    this->blob_bottom_->mutable_cpu_data()[i + 7] = 8;
    this->blob_bottom_->mutable_cpu_data()[i + 8] = 9;
  }

  UnPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), 9);
  EXPECT_EQ(this->blob_top_->width(), 9);
  for (int i = 0; i < this->blob_bottom_->count(); i += 81) {
    Dtype epsilon = 1e-8;
    // output:
    // * stands for 0 for clarity
    //     [ * * * * * * * * *]
    //     [ * 1 * * 2 * * 3 *]
    //     [ * * * * * * * * *]
    //     [ * * * * * * * * *]
    //     [ * 4 * * 5 * * 6 *]
    //     [ * * * * * * * * *]
    //     [ * * * * * * * * *]
    //     [ * 7 * * 8 * * 9 *]
    //     [ * * * * * * * * *]
    EXPECT_NEAR(this->blob_top_->cpu_data()[i + 10], 1, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i + 13], 2, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i + 16], 3, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i + 37], 4, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i + 40], 5, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i + 43], 6, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i + 64], 7, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i + 67], 8, epsilon);
    EXPECT_NEAR(this->blob_top_->cpu_data()[i + 70], 9, epsilon);
    for (int j = 0; j < 81; j++) {
      if (j != 10 && j != 13 && j != 16 && j != 37 && j != 40 && j != 43 &&
          j != 64 && j != 67 && j != 70) {
        EXPECT_NEAR(this->blob_top_->cpu_data()[i + j], 0, epsilon);
      }
    }
  }
}

TYPED_TEST(UnPoolingLayerTest, TestForward2bottomFixed) {
  typedef typename TypeParam::Dtype Dtype;

  LayerParameter layer_param;
  UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_out_kernel_size(2);
  unpooling_param->set_out_stride(1);
  unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_FIXED);
  const int num = 2;
  const int channels = 3;
  this->blob_bottom_->Reshape(num, channels, 2, 4);
  //     [1 2 3 4]
  //     [5 6 7 8]
  int size = this->blob_bottom_->height() * this->blob_bottom_->width();
  for (int i = 0; i < this->blob_bottom_->count(); i += size) {
    for (int j = 0; j < size; j++) {
      this->blob_bottom_->mutable_cpu_data()[i + j] = j + 1;
    }
  }
  this->blob_bottom_mask_->Reshape(num, channels, 4, 4);
  //     [ 1  2  3  4]
  //     [ 5  6  7  8]
  //     [ 9 10 11 12]
  //     [13 14 15 16]
  size = this->blob_bottom_mask_->height() * this->blob_bottom_mask_->width();
  for (int i = 0; i < this->blob_bottom_->count(); i += size) {
    for (int j = 0; j < size; j++) {
      this->blob_bottom_mask_->mutable_cpu_data()[i + j] = j + 1;
    }
  }
  this->blob_bottom_vec_.push_back(this->blob_bottom_mask_);

  UnPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), 4);
  EXPECT_EQ(this->blob_top_->width(), 4);

  for (int i = 0; i < this->blob_top_->count(); i += 16) {
    Dtype epsilon = 1e-8;
    // output:
    //     [ 1 2 3 4 ]
    //     [ 0 0 0 0 ]
    //     [ 5 6 7 8 ]
    //     [ 0 0 0 0 ]
    for (int j = 0; j < 4; j++) {
      EXPECT_NEAR(this->blob_top_->cpu_data()[i + j], j + 1, epsilon);
    }
    for (int j = 4; j < 8; j++) {
      EXPECT_NEAR(this->blob_top_->cpu_data()[i + j], 0, epsilon);
    }
    for (int j = 8; j < 12; j++) {
      EXPECT_NEAR(this->blob_top_->cpu_data()[i + j], j - 3, epsilon);
    }
    for (int j = 12; j < 16; j++) {
      EXPECT_NEAR(this->blob_top_->cpu_data()[i + j], 0, epsilon);
    }
  }
}

TYPED_TEST(UnPoolingLayerTest, TestGradientDiv) {
  typedef typename TypeParam::Dtype Dtype;
  for (int kernel_h = 2; kernel_h <= 4; kernel_h++) {
    for (int kernel_w = 2; kernel_w <= 4; kernel_w++) {
      LayerParameter layer_param;
      UnPoolingParameter* unpooling_param =
          layer_param.mutable_unpooling_param();
      unpooling_param->set_out_kernel_h(kernel_h);
      unpooling_param->set_out_kernel_w(kernel_w);
      unpooling_param->set_out_stride(1);
      unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_DIV);
      UnPoolingLayer<Dtype> layer(layer_param);
      layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
      layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
      layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
      GradientChecker<Dtype> checker(1e-4, 1e-2);
      checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                      this->blob_top_vec_);
    }
  }
}

TYPED_TEST(UnPoolingLayerTest, TestGradientDivPadded) {
  typedef typename TypeParam::Dtype Dtype;
  for (int kernel_h = 2; kernel_h <= 4; kernel_h++) {
    for (int kernel_w = 2; kernel_w <= 4; kernel_w++) {
      LayerParameter layer_param;
      UnPoolingParameter* unpooling_param =
          layer_param.mutable_unpooling_param();
      unpooling_param->set_out_kernel_h(kernel_h);
      unpooling_param->set_out_kernel_w(kernel_w);
      unpooling_param->set_out_stride(1);
      unpooling_param->set_out_pad(1);
      unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_DIV);
      UnPoolingLayer<Dtype> layer(layer_param);
      layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
      layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
      layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
      GradientChecker<Dtype> checker(1e-4, 1e-2);
      checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                      this->blob_top_vec_);
    }
  }
}

TYPED_TEST(UnPoolingLayerTest, TestGradientRep) {
  typedef typename TypeParam::Dtype Dtype;
  for (int kernel_h = 2; kernel_h <= 4; kernel_h++) {
    for (int kernel_w = 2; kernel_w <= 4; kernel_w++) {
      LayerParameter layer_param;
      UnPoolingParameter* unpooling_param =
          layer_param.mutable_unpooling_param();
      unpooling_param->set_out_kernel_h(kernel_h);
      unpooling_param->set_out_kernel_w(kernel_w);
      unpooling_param->set_out_stride(1);
      unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_REP);
      UnPoolingLayer<Dtype> layer(layer_param);
      layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
      layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
      layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
      GradientChecker<Dtype> checker(1e-4, 1e-2);
      checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                      this->blob_top_vec_);
    }
  }
}

TYPED_TEST(UnPoolingLayerTest, TestGradientRepPadded) {
  typedef typename TypeParam::Dtype Dtype;
  for (int kernel_h = 2; kernel_h <= 4; kernel_h++) {
    for (int kernel_w = 2; kernel_w <= 4; kernel_w++) {
      LayerParameter layer_param;
      UnPoolingParameter* unpooling_param =
          layer_param.mutable_unpooling_param();
      unpooling_param->set_out_kernel_h(kernel_h);
      unpooling_param->set_out_kernel_w(kernel_w);
      unpooling_param->set_out_stride(1);
      unpooling_param->set_out_pad(1);
      unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_REP);
      UnPoolingLayer<Dtype> layer(layer_param);
      layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
      layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
      layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
      GradientChecker<Dtype> checker(1e-4, 1e-2);
      checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                      this->blob_top_vec_);
    }
  }
}

TYPED_TEST(UnPoolingLayerTest, TestGradientFixed) {
  typedef typename TypeParam::Dtype Dtype;
  for (int kernel_h = 2; kernel_h <= 4; kernel_h++) {
    for (int kernel_w = 2; kernel_w <= 4; kernel_w++) {
      LayerParameter layer_param;
      UnPoolingParameter* unpooling_param =
          layer_param.mutable_unpooling_param();
      unpooling_param->set_out_kernel_h(kernel_h);
      unpooling_param->set_out_kernel_w(kernel_w);
      unpooling_param->set_out_stride(1);
      unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_FIXED);
      UnPoolingLayer<Dtype> layer(layer_param);
      layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
      layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
      layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
      GradientChecker<Dtype> checker(1e-4, 1e-2);
      checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                      this->blob_top_vec_);
    }
  }
}

TYPED_TEST(UnPoolingLayerTest, TestGradientFixedPadded) {
  typedef typename TypeParam::Dtype Dtype;
  for (int kernel_h = 2; kernel_h <= 4; kernel_h++) {
    for (int kernel_w = 2; kernel_w <= 4; kernel_w++) {
      LayerParameter layer_param;
      UnPoolingParameter* unpooling_param =
          layer_param.mutable_unpooling_param();
      unpooling_param->set_out_kernel_h(kernel_h);
      unpooling_param->set_out_kernel_w(kernel_w);
      unpooling_param->set_out_stride(1);
      unpooling_param->set_out_pad(1);
      unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_FIXED);
      UnPoolingLayer<Dtype> layer(layer_param);
      layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
      layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
      layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
      GradientChecker<Dtype> checker(1e-4, 1e-2);
      checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                      this->blob_top_vec_);
    }
  }
}

#ifdef USE_MLU
template <typename TypeParam>
class MLUUnPoolingLayerTest : public MLUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MLUUnPoolingLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()),
        blob_bottom_mask_(new Blob<Dtype>()) {}

  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    blob_bottom_->Reshape(2, 3, 2, 4);
    blob_bottom_mask_->Reshape(2, 3, 2, 4);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_bottom_mask_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~MLUUnPoolingLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
    delete blob_bottom_mask_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_bottom_mask_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MLUUnPoolingLayerTest, TestMLUDevices);

TYPED_TEST(MLUUnPoolingLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;

  LayerParameter layer_param;
  UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_out_kernel_size(2);
  unpooling_param->set_out_stride(1);
  unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_FIXED);
  MLUUnPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 5);
  OUTPUT("bottom", this->blob_bottom_->shape_string().c_str());
}

template <typename TypeParam>
class MFUSUnPoolingLayerTest : public MFUSDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MFUSUnPoolingLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()),
        blob_bottom_mask_(new Blob<Dtype>()) {}

  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    blob_bottom_->Reshape(2, 3, 2, 4);
    blob_bottom_mask_->Reshape(2, 3, 2, 4);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_bottom_mask_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~MFUSUnPoolingLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
    delete blob_bottom_mask_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_bottom_mask_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MFUSUnPoolingLayerTest, TestMFUSDevices);

TYPED_TEST(MFUSUnPoolingLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;

  LayerParameter layer_param;
  UnPoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_out_kernel_size(2);
  unpooling_param->set_out_stride(1);
  unpooling_param->set_unpool(UnPoolingParameter_UnPoolMethod_FIXED);
  MLUUnPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 5);
  OUTPUT("bottom", this->blob_bottom_->shape_string().c_str());
}

#endif

}  // namespace caffe
