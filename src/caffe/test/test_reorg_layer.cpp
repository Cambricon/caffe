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
#include "caffe/layers/mlu_reorg_layer.hpp"
#include "caffe/layers/reorg_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class ReorgLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  ReorgLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 4, 2, 2)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~ReorgLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ReorgLayerTest, TestDtypesAndDevices);

TYPED_TEST(ReorgLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_reorg_param()->set_stride(1);
  ReorgLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 4);
  EXPECT_EQ(this->blob_top_->height(), 2);
  EXPECT_EQ(this->blob_top_->width(), 2);
  layer_param.mutable_reorg_param()->set_stride(2);
  ReorgLayer<Dtype> layer2(layer_param);
  layer2.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 16);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

TYPED_TEST(ReorgLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_reorg_param()->set_stride(1);
  ReorgLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_EQ(this->blob_top_->cpu_data()[i],
              this->blob_bottom_->cpu_data()[i]);
  }
  layer_param.mutable_reorg_param()->set_stride(2);
  ReorgLayer<Dtype> layer2(layer_param);
  layer2.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer2.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  vector<vector<int>> indexes = {
      {0, 0, 0, 0}, {0, 0, 1, 0}, {0, 2, 0, 0}, {0, 2, 1, 0}, {0, 0, 0, 1},
      {0, 0, 1, 1}, {0, 2, 0, 1}, {0, 2, 1, 1}, {0, 1, 0, 0}, {0, 1, 1, 0},
      {0, 3, 0, 0}, {0, 3, 1, 0}, {0, 1, 0, 1}, {0, 1, 1, 1}, {0, 3, 0, 1},
      {0, 3, 1, 1}, {1, 0, 0, 0}, {1, 0, 1, 0}, {1, 2, 0, 0}, {1, 2, 1, 0},
      {1, 0, 0, 1}, {1, 0, 1, 1}, {1, 2, 0, 1}, {1, 2, 1, 1}, {1, 1, 0, 0},
      {1, 1, 1, 0}, {1, 3, 0, 0}, {1, 3, 1, 0}, {1, 1, 0, 1}, {1, 1, 1, 1},
      {1, 3, 0, 1}, {1, 3, 1, 1}};
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_EQ(this->blob_bottom_->data_at(indexes[i]),
              this->blob_top_->cpu_data()[i]);
  }
}

#ifdef USE_MLU

template <typename TypeParam>
class MLUReorgLayerTest : public MLUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MLUReorgLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 4, 2, 2)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    ConstantFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MLUReorgLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MLUReorgLayerTest, TestMLUDevices);

TYPED_TEST(MLUReorgLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_reorg_param()->set_stride(1);
  MLUReorgLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 4);
  EXPECT_EQ(this->blob_top_->height(), 2);
  EXPECT_EQ(this->blob_top_->width(), 2);
  layer_param.mutable_reorg_param()->set_stride(2);
  MLUReorgLayer<Dtype> layer2(layer_param);
  layer2.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 16);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

TYPED_TEST(MLUReorgLayerTest, TestForwardInt8) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_reorg_param()->set_stride(1);
  BlobDataType blob_dtype;  // set position
  blob_dtype = get_quantized_info(*this->blob_bottom_,
                                layer_param, "common", DT_INT8);
  layer_param.add_bottom_mlu_dtype()->CopyFrom(blob_dtype);
  int position = -3;  // set weight position
  int scale = 1.5875;
  BlobDataType blobs_dtype;
  blobs_dtype.set_type(DT_INT8);
  blobs_dtype.add_position(position);
  blobs_dtype.add_scale(scale);
  layer_param.add_blobs_dtype()->CopyFrom(blobs_dtype);
  MLUReorgLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  float err_sum = 0, sum = 0;
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i],
                this->blob_bottom_->cpu_data()[i], 3e-2);
    err_sum += std::abs(this->blob_top_->cpu_data()[i] -
        this->blob_bottom_->cpu_data()[i]);
    sum += std::abs(this->blob_bottom_->cpu_data()[i]);
  }
  ERR_RATE(err_sum/sum);
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  EVENT_TIME(layer.get_event_time());
}

TYPED_TEST(MLUReorgLayerTest, TestForwardStrideInt8) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_reorg_param()->set_stride(2);
  BlobDataType blob_dtype;  // set position
  blob_dtype = get_quantized_info(*this->blob_bottom_,
                                layer_param, "common", DT_INT8);
  layer_param.add_bottom_mlu_dtype()->CopyFrom(blob_dtype);
  int position = -3;  // set weight position
  int scale = 1.5875;
  BlobDataType blobs_dtype;
  blobs_dtype.set_type(DT_INT8);
  blobs_dtype.add_position(position);
  blobs_dtype.add_scale(scale);
  layer_param.add_blobs_dtype()->CopyFrom(blobs_dtype);
  MLUReorgLayer<Dtype> layer2(layer_param);
  layer2.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer2.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer2.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  vector<vector<int>> indexes = {
      {0, 0, 0, 0}, {0, 0, 1, 0}, {0, 2, 0, 0}, {0, 2, 1, 0}, {0, 0, 0, 1},
      {0, 0, 1, 1}, {0, 2, 0, 1}, {0, 2, 1, 1}, {0, 1, 0, 0}, {0, 1, 1, 0},
      {0, 3, 0, 0}, {0, 3, 1, 0}, {0, 1, 0, 1}, {0, 1, 1, 1}, {0, 3, 0, 1},
      {0, 3, 1, 1}, {1, 0, 0, 0}, {1, 0, 1, 0}, {1, 2, 0, 0}, {1, 2, 1, 0},
      {1, 0, 0, 1}, {1, 0, 1, 1}, {1, 2, 0, 1}, {1, 2, 1, 1}, {1, 1, 0, 0},
      {1, 1, 1, 0}, {1, 3, 0, 0}, {1, 3, 1, 0}, {1, 1, 0, 1}, {1, 1, 1, 1},
      {1, 3, 0, 1}, {1, 3, 1, 1}};
  float err_sum = 0, sum = 0;
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_NEAR(this->blob_bottom_->data_at(indexes[i]),
                this->blob_top_->cpu_data()[i], 3e-2);
    err_sum += std::abs(this->blob_top_->cpu_data()[i]-
        this->blob_top_->cpu_data()[i]);
    sum += std::abs(this->blob_top_->cpu_data()[i]);
  }
  EVENT_TIME(layer2.get_event_time());
  ERR_RATE(err_sum/sum);
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
}

template <typename TypeParam>
class MFUSReorgLayerTest : public MFUSDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MFUSReorgLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 4, 2, 2)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    ConstantFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MFUSReorgLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MFUSReorgLayerTest, TestMFUSDevices);

TYPED_TEST(MFUSReorgLayerTest, TestForwardInt8) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_reorg_param()->set_stride(1);
  BlobDataType blob_dtype;  // set position
  blob_dtype = get_quantized_info(*this->blob_bottom_,
                                layer_param, "common", DT_INT8);
  layer_param.add_bottom_mlu_dtype()->CopyFrom(blob_dtype);
  int position = -3;  // set weight position
  int scale = 1.5875;
  BlobDataType blobs_dtype;
  blobs_dtype.set_type(DT_INT8);
  blobs_dtype.add_position(position);
  blobs_dtype.add_scale(scale);
  layer_param.add_blobs_dtype()->CopyFrom(blobs_dtype);
  MLUReorgLayer<Dtype> layer(layer_param);
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
  float err_sum = 0, sum = 0;
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i],
                this->blob_bottom_->cpu_data()[i], 3e-2);
    err_sum += std::abs(this->blob_top_->cpu_data()[i] -
                this->blob_bottom_->cpu_data()[i]);
    sum += std::abs(this->blob_bottom_->cpu_data()[i]);
  }
  EVENT_TIME(fuser.get_event_time());
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  ERR_RATE(err_sum/sum);
}

TYPED_TEST(MFUSReorgLayerTest, TestForwardStrideInt8) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_reorg_param()->set_stride(2);
  BlobDataType blob_dtype;  // set position
  blob_dtype = get_quantized_info(*this->blob_bottom_,
                                layer_param, "common", DT_INT8);
  layer_param.add_bottom_mlu_dtype()->CopyFrom(blob_dtype);
  int position = -3;  // set weight position
  int scale = 1.5875;
  BlobDataType blobs_dtype;
  blobs_dtype.set_type(DT_INT8);
  blobs_dtype.add_position(position);
  blobs_dtype.add_scale(scale);
  layer_param.add_blobs_dtype()->CopyFrom(blobs_dtype);
  MLUReorgLayer<Dtype> layer2(layer_param);
  layer2.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  ASSERT_TRUE(layer2.mfus_supported());

  MFusion<Dtype> fuser;
  fuser.reset();
  fuser.addInputs(this->blob_bottom_vec_);
  fuser.addOutputs(this->blob_top_vec_);
  layer2.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer2.fuse(&fuser);
  fuser.compile();
  fuser.forward();
  vector<vector<int>> indexes = {
      {0, 0, 0, 0}, {0, 0, 1, 0}, {0, 2, 0, 0}, {0, 2, 1, 0}, {0, 0, 0, 1},
      {0, 0, 1, 1}, {0, 2, 0, 1}, {0, 2, 1, 1}, {0, 1, 0, 0}, {0, 1, 1, 0},
      {0, 3, 0, 0}, {0, 3, 1, 0}, {0, 1, 0, 1}, {0, 1, 1, 1}, {0, 3, 0, 1},
      {0, 3, 1, 1}, {1, 0, 0, 0}, {1, 0, 1, 0}, {1, 2, 0, 0}, {1, 2, 1, 0},
      {1, 0, 0, 1}, {1, 0, 1, 1}, {1, 2, 0, 1}, {1, 2, 1, 1}, {1, 1, 0, 0},
      {1, 1, 1, 0}, {1, 3, 0, 0}, {1, 3, 1, 0}, {1, 1, 0, 1}, {1, 1, 1, 1},
      {1, 3, 0, 1}, {1, 3, 1, 1}};
  float err_sum = 0, sum = 0;
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_NEAR(this->blob_bottom_->data_at(indexes[i]),
                this->blob_top_->cpu_data()[i], 3e-2);
    err_sum += std::abs(this->blob_bottom_->data_at(indexes[i]) -
                this->blob_top_->cpu_data()[i]);
    sum += std::abs(this->blob_top_->cpu_data()[i]);
  }
  EVENT_TIME(fuser.get_event_time());
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  ERR_RATE(err_sum/sum);
}

#endif

}  // namespace caffe
