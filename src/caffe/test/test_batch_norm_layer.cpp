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

#include <algorithm>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/batch_norm_layer.hpp"
#ifdef USE_MLU
#include "caffe/layers/mlu_batch_norm_layer.hpp"
#endif

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#define BATCH_SIZE 2
#define INPUT_DATA_SIZE 3

namespace caffe {

template <typename TypeParam>
class BatchNormLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  BatchNormLayerTest()
      : blob_bottom_(new Blob<Dtype>(5, 2, 3, 4)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~BatchNormLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(BatchNormLayerTest, TestDtypesAndDevices);

TYPED_TEST(BatchNormLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;

  BatchNormLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  // Test mean
  int num = this->blob_bottom_->num();
  int channels = this->blob_bottom_->channels();
  int height = this->blob_bottom_->height();
  int width = this->blob_bottom_->width();

  for (int j = 0; j < channels; ++j) {
    Dtype sum = 0, var = 0;
    for (int i = 0; i < num; ++i) {
      for (int k = 0; k < height; ++k) {
        for (int l = 0; l < width; ++l) {
          Dtype data = this->blob_top_->data_at(i, j, k, l);
          sum += data;
          var += data * data;
        }
      }
    }
    sum /= height * width * num;
    var /= height * width * num;

    const Dtype kErrorBound = 0.001;
    // expect zero mean
    EXPECT_NEAR(0, sum, kErrorBound);
    // expect unit variance
    EXPECT_NEAR(1, var, kErrorBound);
  }
}

TYPED_TEST(BatchNormLayerTest, TestForwardInplace) {
  typedef typename TypeParam::Dtype Dtype;
  Blob<Dtype> blob_inplace(5, 2, 3, 4);
  vector<Blob<Dtype>*> blob_bottom_vec;
  vector<Blob<Dtype>*> blob_top_vec;
  LayerParameter layer_param;
  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(&blob_inplace);
  blob_bottom_vec.push_back(&blob_inplace);
  blob_top_vec.push_back(&blob_inplace);

  BatchNormLayer<Dtype> layer(layer_param);
  layer.SetUp(blob_bottom_vec, blob_top_vec);
  layer.Forward(blob_bottom_vec, blob_top_vec);

  // Test mean
  int num = blob_inplace.num();
  int channels = blob_inplace.channels();
  int height = blob_inplace.height();
  int width = blob_inplace.width();

  for (int j = 0; j < channels; ++j) {
    Dtype sum = 0, var = 0;
    for (int i = 0; i < num; ++i) {
      for (int k = 0; k < height; ++k) {
        for (int l = 0; l < width; ++l) {
          Dtype data = blob_inplace.data_at(i, j, k, l);
          sum += data;
          var += data * data;
        }
      }
    }
    sum /= height * width * num;
    var /= height * width * num;

    const Dtype kErrorBound = 0.001;
    // expect zero mean
    EXPECT_NEAR(0, sum, kErrorBound);
    // expect unit variance
    EXPECT_NEAR(1, var, kErrorBound);
  }
}

TYPED_TEST(BatchNormLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;

  BatchNormLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-4);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                  this->blob_top_vec_);
}

#ifdef USE_MLU
template <typename TypeParam>
class MLUBatchNormLayerTest : public MLUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MLUBatchNormLayerTest()
      : blob_bottom_(new Blob<Dtype>(5, 2, 3, 4)),
        blob_top_(new Blob<Dtype>()),
        cpu_blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
    cpu_blob_top_vec_.push_back(cpu_blob_top_);
  }
  virtual ~MLUBatchNormLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
    delete cpu_blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const cpu_blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  vector<Blob<Dtype>*> cpu_blob_top_vec_;
};

TYPED_TEST_CASE(MLUBatchNormLayerTest, TestMLUDevices);

TYPED_TEST(MLUBatchNormLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.set_phase(TEST);

  BatchNormLayer<Dtype> cpu_layer(layer_param);
  cpu_layer.SetUp(this->blob_bottom_vec_, this->cpu_blob_top_vec_);
  cpu_layer.Forward(this->blob_bottom_vec_, this->cpu_blob_top_vec_);

  MLUBatchNormLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), this->cpu_blob_top_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->cpu_blob_top_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->cpu_blob_top_->height());
  EXPECT_EQ(this->blob_top_->width(), this->cpu_blob_top_->width());

  const Dtype* cpu_top_data = this->cpu_blob_top_->cpu_data();
  const Dtype* top_data = this->blob_top_->cpu_data();

  const Dtype precision = 0.01;
  float err_sum = 0, sum = 0;
  for (int i = 0; i < this->blob_top_->count(); i++) {
    Dtype tolerence = cpu_top_data[i] * precision;
    if (tolerence < 0) {
      tolerence = -tolerence;
    }
    EXPECT_NEAR(top_data[i], cpu_top_data[i], tolerence);
    err_sum += std::abs(top_data[i] - cpu_top_data[i]);
    sum += std::abs(cpu_top_data[i]);
  }
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  ERR_RATE(err_sum/sum);
  EVENT_TIME(layer.get_event_time());
}

template <typename TypeParam>
class MFUSBatchNormLayerTest : public MFUSDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MFUSBatchNormLayerTest()
      : blob_bottom_(new Blob<Dtype>(5, 2, 3, 4)),
        blob_top_(new Blob<Dtype>()),
        cpu_blob_top_(new Blob<Dtype>()) {}
  void SetUp() {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
    cpu_blob_top_vec_.push_back(cpu_blob_top_);
  }
  virtual ~MFUSBatchNormLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
    delete cpu_blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const cpu_blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  vector<Blob<Dtype>*> cpu_blob_top_vec_;
};

TYPED_TEST_CASE(MFUSBatchNormLayerTest, TestMFUSDevices);

TYPED_TEST(MFUSBatchNormLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.set_phase(TEST);

  BatchNormLayer<Dtype> cpu_layer(layer_param);
  cpu_layer.SetUp(this->blob_bottom_vec_, this->cpu_blob_top_vec_);
  cpu_layer.Forward(this->blob_bottom_vec_, this->cpu_blob_top_vec_);

  MLUBatchNormLayer<Dtype> layer(layer_param);
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

  EXPECT_EQ(this->blob_top_->num(), this->cpu_blob_top_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->cpu_blob_top_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->cpu_blob_top_->height());
  EXPECT_EQ(this->blob_top_->width(), this->cpu_blob_top_->width());

  const Dtype* cpu_top_data = this->cpu_blob_top_->cpu_data();
  const Dtype* top_data = this->blob_top_->cpu_data();

  const Dtype precision = 0.01;
  float err_sum = 0, sum = 0;
  for (int i = 0; i < this->blob_top_->count(); i++) {
    Dtype tolerence = cpu_top_data[i] * precision;
    if (tolerence < 0) {
      tolerence = -tolerence;
    }
    EXPECT_NEAR(top_data[i], cpu_top_data[i], tolerence);
    err_sum += std::abs(top_data[i] - cpu_top_data[i]);
    sum += std::abs(cpu_top_data[i]);
  }
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  ERR_RATE(err_sum/sum);
  EVENT_TIME(fuser.get_event_time());
}

#endif

}  // namespace caffe
