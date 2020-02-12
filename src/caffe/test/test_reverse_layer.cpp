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
#include "caffe/layers/mlu_reverse_layer.hpp"
#include "caffe/layers/reverse_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "gtest/gtest.h"

namespace caffe {

template <typename TypeParam>
class ReverseLayerTest : public CPUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  ReverseLayerTest()
      : blob_bottom_(new Blob<Dtype>(3, 5, 8, 10)),
        blob_top_(new Blob<Dtype>()) {}
  void SetUp() {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~ReverseLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  virtual void TestForward() {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    this->blob_bottom_vec_.clear();
    this->blob_top_vec_.clear();
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);

    LayerParameter layer_param;
    ReverseLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    // check values
    const Dtype* bottom_data = this->blob_bottom_->cpu_data();
    const Dtype* top_data = this->blob_top_->cpu_data();
    const int num = this->blob_bottom_->num();
    const int channel = this->blob_bottom_->channels();
    const int height = this->blob_bottom_->height();
    const int width = this->blob_bottom_->width();
    int input_index = 0, out_index = 0;
    for (int n = 0; n < num; n++) {
      for (int c = 0; c < channel; c++) {
        for (int h = 0; h < height; h++) {
          for (int w = 0; w < width; w++) {
            out_index = n * channel * height * width +
                        (channel - c - 1) * height * width + h * width + w;
            input_index = n * channel * height * width + c * height * width +
                          h * width + w;
            Dtype top = bottom_data[input_index];
            EXPECT_EQ(top_data[out_index], top);
          }
        }
      }
    }
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ReverseLayerTest, TestDtypesAndDevices);

TYPED_TEST(ReverseLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ReverseLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 3);
  EXPECT_EQ(this->blob_top_->channels(), 5);
  EXPECT_EQ(this->blob_top_->height(), 8);
  EXPECT_EQ(this->blob_top_->width(), 10);
}

TYPED_TEST(ReverseLayerTest, TestForward) { this->TestForward(); }

#ifdef USE_MLU

template <typename TypeParam>
class MLUReverseLayerTest : public MLUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MLUReverseLayerTest()
      : blob_bottom_(new Blob<Dtype>(3, 5, 8, 10)),
        blob_top_(new Blob<Dtype>()) {}
  void SetUp() {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MLUReverseLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  virtual void TestForward() {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    this->blob_bottom_vec_.clear();
    this->blob_top_vec_.clear();
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);

    LayerParameter layer_param;
    MLUReverseLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    // check values
    const Dtype* bottom_data = this->blob_bottom_->cpu_data();
    const Dtype* top_data = this->blob_top_->cpu_data();
    const int num = this->blob_bottom_->num();
    const int channel = this->blob_bottom_->channels();
    const int height = this->blob_bottom_->height();
    const int width = this->blob_bottom_->width();
    int input_index = 0, out_index = 0;
    float err_sum = 0, sum = 0;
    for (int n = 0; n < num; n++) {
      for (int c = 0; c < channel; c++) {
        for (int h = 0; h < height; h++) {
          for (int w = 0; w < width; w++) {
            out_index = n * channel * height * width +
                        (channel - c - 1) * height * width + h * width + w;
            input_index = n * channel * height * width + c * height * width +
                          h * width + w;
            Dtype top = bottom_data[input_index];
            EXPECT_EQ(top_data[out_index], top);
            err_sum += std::abs(top_data[out_index] - top);
            sum += std::abs(top);
          }
        }
      }
    }
    std::ostringstream stream;
    stream << "bottom1:" << blob_bottom_->shape_string().c_str();
    BOTTOM(stream);
    ERR_RATE(err_sum/sum);
    EVENT_TIME(layer.get_event_time());
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MLUReverseLayerTest, TestMLUDevices);

TYPED_TEST(MLUReverseLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ReverseLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 3);
  EXPECT_EQ(this->blob_top_->channels(), 5);
  EXPECT_EQ(this->blob_top_->height(), 8);
  EXPECT_EQ(this->blob_top_->width(), 10);
}

TYPED_TEST(MLUReverseLayerTest, TestForward) { this->TestForward(); }

TYPED_TEST(MLUReverseLayerTest, TestForward1Batch) {
  this->blob_bottom_->Reshape(1, 8, 2, 2);
  this->TestForward();
}

template <typename TypeParam>
class MFUSReverseLayerTest : public MFUSDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MFUSReverseLayerTest()
      : blob_bottom_(new Blob<Dtype>(3, 5, 8, 10)),
        blob_top_(new Blob<Dtype>()) {}
  void SetUp() {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MFUSReverseLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  virtual void TestForward() {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    this->blob_bottom_vec_.clear();
    this->blob_top_vec_.clear();
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);

    LayerParameter layer_param;
    MLUReverseLayer<Dtype> layer(layer_param);
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
    // check values
    const Dtype* bottom_data = this->blob_bottom_->cpu_data();
    const Dtype* top_data = this->blob_top_->cpu_data();
    const int num = this->blob_bottom_->num();
    const int channel = this->blob_bottom_->channels();
    const int height = this->blob_bottom_->height();
    const int width = this->blob_bottom_->width();
    int input_index = 0, out_index = 0;
    float err_sum = 0, sum = 0;
    for (int n = 0; n < num; n++) {
      for (int c = 0; c < channel; c++) {
        for (int h = 0; h < height; h++) {
          for (int w = 0; w < width; w++) {
            out_index = n * channel * height * width +
                        (channel - c - 1) * height * width + h * width + w;
            input_index = n * channel * height * width + c * height * width +
                          h * width + w;
            Dtype top = bottom_data[input_index];
            EXPECT_EQ(top_data[out_index], top);
            err_sum += std::abs(top_data[out_index] - top);
            sum += std::abs(top);
          }
        }
      }
    }
    std::ostringstream stream;
    stream << "bottom1:" << blob_bottom_->shape_string().c_str();
    BOTTOM(stream);
    ERR_RATE(err_sum/sum);
    EVENT_TIME(fuser.get_event_time());
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MFUSReverseLayerTest, TestMFUSDevices);

TYPED_TEST(MFUSReverseLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ReverseLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 3);
  EXPECT_EQ(this->blob_top_->channels(), 5);
  EXPECT_EQ(this->blob_top_->height(), 8);
  EXPECT_EQ(this->blob_top_->width(), 10);
}

TYPED_TEST(MFUSReverseLayerTest, TestForward) {
  this->TestForward();
}

TYPED_TEST(MFUSReverseLayerTest, TestForward1Batch) {
  this->blob_bottom_->Reshape(1, 8, 2, 2);
  this->TestForward();
}
#endif

}  // namespace caffe
