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
#include "caffe/layers/add_layer.hpp"
#include "caffe/layers/mlu_addpad_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "gtest/gtest.h"
#include "caffe/layer.hpp"

#ifdef USE_MLU
namespace caffe {
template <typename TypeParam>
class MLUAddPadLayerTest : public MLUDeviceTest<TypeParam> {
typedef typename TypeParam::Dtype Dtype;

  protected:
    MLUAddPadLayerTest()
       : blob_bottom_(new Blob<Dtype>(3, 5, 4, 4)),
         blob_top_(new Blob<Dtype>()) {}
    void SetUp() {
      FillerParameter filler_param;
      GaussianFiller<Dtype> filler(filler_param);
      this->blob_bottom_vec_.clear();
      this->blob_top_vec_.clear();
      filler.Fill(this->blob_bottom_);
      blob_bottom_vec_.push_back(blob_bottom_);
      blob_top_vec_.push_back(blob_top_);
    }
    virtual ~MLUAddPadLayerTest() {
      delete blob_bottom_;
      delete blob_top_;
    }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};
TYPED_TEST_CASE(MLUAddPadLayerTest, TestMLUDevices);

TYPED_TEST(MLUAddPadLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  AddPadParameter* addpad_param = layer_param.mutable_addpad_param();
  addpad_param->set_pad_h(1);
  addpad_param->set_pad_w(1);
  MLUAddPadLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 3);
  EXPECT_EQ(this->blob_top_->channels(), 5);
  EXPECT_EQ(this->blob_top_->height(), 6);
  EXPECT_EQ(this->blob_top_->width(), 6);
}
TYPED_TEST(MLUAddPadLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  AddPadParameter* addpad_param = layer_param.mutable_addpad_param();
  addpad_param->set_pad_h(1);
  addpad_param->set_pad_w(1);
  MLUAddPadLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  int count = 0;
  for (int i = 0; i < (this->blob_top_vec_)[0]->count(); i++) {
    if (this->blob_top_vec_[0]->cpu_data()[i] == 0) count++;
  }
  EXPECT_EQ(count, 20*3*5);
}

template <typename TypeParam>
class MFUSAddPadLayerTest : public MFUSDeviceTest<TypeParam> {
typedef typename TypeParam::Dtype Dtype;

  protected:
    MFUSAddPadLayerTest()
       : blob_bottom_(new Blob<Dtype>(3, 5, 4, 4)),
         blob_top_(new Blob<Dtype>()) {}
    void SetUp() {
      FillerParameter filler_param;
      GaussianFiller<Dtype> filler(filler_param);
      this->blob_bottom_vec_.clear();
      this->blob_top_vec_.clear();
      filler.Fill(this->blob_bottom_);
      blob_bottom_vec_.push_back(blob_bottom_);
      blob_top_vec_.push_back(blob_top_);
    }
    virtual ~MFUSAddPadLayerTest() {
      delete blob_bottom_;
      delete blob_top_;
    }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MFUSAddPadLayerTest, TestMLUDevices);

TYPED_TEST(MFUSAddPadLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  AddPadParameter* addpad_param = layer_param.mutable_addpad_param();
  addpad_param->set_pad_h(1);
  addpad_param->set_pad_w(1);
  MLUAddPadLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 3);
  EXPECT_EQ(this->blob_top_->channels(), 5);
  EXPECT_EQ(this->blob_top_->height(), 6);
  EXPECT_EQ(this->blob_top_->width(), 6);
}
TYPED_TEST(MFUSAddPadLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  AddPadParameter* addpad_param = layer_param.mutable_addpad_param();
  addpad_param->set_pad_h(1);
  addpad_param->set_pad_w(1);
  MLUAddPadLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  MFusion<Dtype> fuser;
  fuser.reset();
  fuser.addInputs(this->blob_bottom_vec_);
  fuser.addOutputs(this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.fuse(&fuser);
  fuser.compile();
  fuser.forward();
  int count = 0;
  for (int i = 0; i < (this->blob_top_vec_)[0]->count(); i++) {
    if (this->blob_top_vec_[0]->cpu_data()[i] == 0) count++;
  }
  EXPECT_EQ(count, 20*3*5);
}

#endif
}  // namespace caffe
