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

#include <cstring>
#include <vector>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/mlu_shufflechannel_layer.hpp"
#include "caffe/layers/shufflechannel_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "gtest/gtest.h"

namespace caffe {

template <typename TypeParam>
class ShuffleChannelLayerTest : public CPUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  ShuffleChannelLayerTest()
      : blob_bottom_(new Blob<Dtype>(3, 5, 7, 11)),
        blob_top_(new Blob<Dtype>()) {}
  void SetUp() {
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~ShuffleChannelLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  virtual void TestForward() {
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    this->blob_bottom_vec_.clear();
    this->blob_top_vec_.clear();
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);

    LayerParameter layer_param;
    layer_param.mutable_shuffle_channel_param()->set_group(5);
    ShuffleChannelLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    const Dtype* bottom_data = this->blob_bottom_->cpu_data();
    const Dtype* top_data = this->blob_top_->cpu_data();
    // check values
    const int num = blob_bottom_->shape(0);
    const int ch = blob_bottom_->shape(1);
    const int ht = blob_bottom_->shape(2);
    const int wt = blob_bottom_->shape(3);
    const int g = layer_param.shuffle_channel_param().group();
    const int g_col = static_cast<int>(ch / g);
    const int size = ch * ht * wt;
    const int len = ht * wt;
    EXPECT_EQ(ch, (g * g_col));

    for (int n = 0; n < num; n++) {
      for (int i = 0; i < g; i++) {
        for (int j = 0; j < g_col; j++) {
          const Dtype* src = bottom_data + n * size + (i * g_col + j) * len;
          const Dtype* dst = top_data + n * size + (j * g + i) * len;
          EXPECT_EQ(0, std::memcmp(dst, src, len));
        }
      }
    }
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ShuffleChannelLayerTest, TestDtypesAndDevices);

TYPED_TEST(ShuffleChannelLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ShuffleChannelLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width());
}

TYPED_TEST(ShuffleChannelLayerTest, TestForward) { this->TestForward(); }

TYPED_TEST(ShuffleChannelLayerTest, TestForward1Batch) {
  this->blob_bottom_->Reshape(1, 5, 7, 8);
  this->TestForward();
}

#ifdef USE_MLU

template <typename TypeParam>
class MLUShuffleChannelLayerTest : public MLUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MLUShuffleChannelLayerTest()
      : blob_bottom_(new Blob<Dtype>(3, 5, 7, 11)),
        blob_top_(new Blob<Dtype>()) {}
  void SetUp() {
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MLUShuffleChannelLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  virtual void TestForward() {
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    this->blob_bottom_vec_.clear();
    this->blob_top_vec_.clear();
    this->blob_bottom_vec_.push_back(this->blob_bottom_);
    this->blob_top_vec_.push_back(this->blob_top_);

    LayerParameter layer_param;
    layer_param.mutable_shuffle_channel_param()->set_group(5);
    MLUShuffleChannelLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype* bottom_data = this->blob_bottom_->cpu_data();
    const Dtype* top_data = this->blob_top_->cpu_data();
    // check values
    const int num = blob_bottom_->shape(0);
    const int ch = blob_bottom_->shape(1);
    const int ht = blob_bottom_->shape(2);
    const int wt = blob_bottom_->shape(3);
    const int g = layer_param.shuffle_channel_param().group();
    const int g_col = static_cast<int>(ch / g);
    const int size = ch * ht * wt;
    const int len = ht * wt;
    EXPECT_EQ(ch, (g * g_col));

    for (int n = 0; n < num; n++) {
      for (int i = 0; i < g; i++) {
        for (int j = 0; j < g_col; j++) {
          const Dtype* src = bottom_data + n * size + (i * g_col + j) * len;
          const Dtype* dst = top_data + n * size + (j * g + i) * len;
          EXPECT_EQ(0, std::memcmp(dst, src, len));
        }
      }
    }
    std::ostringstream stream;
    stream << "bottom1:" << blob_bottom_->shape_string().c_str();
    BOTTOM(stream);
    EVENT_TIME(layer.get_event_time());
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MLUShuffleChannelLayerTest, TestMLUDevices);

TYPED_TEST(MLUShuffleChannelLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MLUShuffleChannelLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width());
  OUTPUT("bottom1", this->blob_bottom_->shape_string().c_str());
}

TYPED_TEST(MLUShuffleChannelLayerTest, TestForwardgroupeqc) { 
  this->TestForward(); 
}
TYPED_TEST(MLUShuffleChannelLayerTest, TestForward) { 
  this->blob_bottom_->Reshape(3, 10, 7, 11);
  this->TestForward(); 
}

TYPED_TEST(MLUShuffleChannelLayerTest, TestForward1Batchgroupeqc) {
  this->blob_bottom_->Reshape(1, 5, 11, 13);
  this->TestForward();
}

TYPED_TEST(MLUShuffleChannelLayerTest, TestForward1Batch) {
  this->blob_bottom_->Reshape(1, 10, 11, 13);
  this->TestForward();
}

template <typename TypeParam>
class MFUSShuffleChannelLayerTest : public MFUSDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MFUSShuffleChannelLayerTest()
      : blob_bottom_(new Blob<Dtype>(3, 5, 7, 11)),
        blob_top_(new Blob<Dtype>()) {}
  void SetUp() {
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MFUSShuffleChannelLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  virtual void TestForward() {
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    this->blob_bottom_vec_.clear();
    this->blob_top_vec_.clear();
    this->blob_bottom_vec_.push_back(this->blob_bottom_);
    this->blob_top_vec_.push_back(this->blob_top_);

    LayerParameter layer_param;
    layer_param.mutable_shuffle_channel_param()->set_group(5);
    MLUShuffleChannelLayer<Dtype> layer(layer_param);
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
    const Dtype* bottom_data = this->blob_bottom_->cpu_data();
    const Dtype* top_data = this->blob_top_->cpu_data();
    // check values
    const int num = blob_bottom_->shape(0);
    const int ch = blob_bottom_->shape(1);
    const int ht = blob_bottom_->shape(2);
    const int wt = blob_bottom_->shape(3);
    const int g = layer_param.shuffle_channel_param().group();
    const int g_col = static_cast<int>(ch / g);
    const int size = ch * ht * wt;
    const int len = ht * wt;
    EXPECT_EQ(ch, (g * g_col));

    for (int n = 0; n < num; n++) {
      for (int i = 0; i < g; i++) {
        for (int j = 0; j < g_col; j++) {
          const Dtype* src = bottom_data + n * size + (i * g_col + j) * len;
          const Dtype* dst = top_data + n * size + (j * g + i) * len;
          EXPECT_EQ(0, std::memcmp(dst, src, len));
        }
      }
    }
    std::ostringstream stream;
    stream << "bottom1:" << blob_bottom_->shape_string().c_str();
    BOTTOM(stream);
    EVENT_TIME(fuser.get_event_time());
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MFUSShuffleChannelLayerTest, TestMFUSDevices);

TYPED_TEST(MFUSShuffleChannelLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MLUShuffleChannelLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width());
  OUTPUT("bottom1", this->blob_bottom_->shape_string().c_str());
}

TYPED_TEST(MFUSShuffleChannelLayerTest, TestForwardgroupeqc) { this->TestForward(); }

TYPED_TEST(MFUSShuffleChannelLayerTest, TestForward) {
  this->blob_bottom_->Reshape(3, 10, 7, 11);
  this->TestForward(); 
}

TYPED_TEST(MFUSShuffleChannelLayerTest, TestForward1Batchgroupeqc) {
  this->blob_bottom_->Reshape(1, 5, 11, 13);
  this->TestForward();
}
TYPED_TEST(MFUSShuffleChannelLayerTest, TestForward1Batch) {
  this->blob_bottom_->Reshape(1, 10, 11, 13);
  this->TestForward();
}
#endif

}  // namespace caffe
