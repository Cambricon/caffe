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
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/mlu_permute_layer.hpp"
#include "caffe/layers/permute_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "gtest/gtest.h"

namespace caffe {
template <typename Dtype>
float caffe_permute(const Blob<Dtype>* bottom,
    const Blob<Dtype>* top_blob, vector<int> orders) {
  int num_axes_ = bottom->num_axes();
  const Dtype* bottom_data = bottom->cpu_data();
  const Dtype* top_data = top_blob->cpu_data();
  EXPECT_EQ(bottom->count(), top_blob->count());
  for (int i = 0; i < num_axes_; i++) {
    if (std::find(orders.begin(), orders.end(), i) == orders.end()) {
      orders.push_back(i);
    }
  }
  vector<int> top_shape;
  Blob<int> old_steps_;
  Blob<int> new_steps_;
  old_steps_.Reshape(num_axes_, 1, 1, 1);
  new_steps_.Reshape(num_axes_, 1, 1, 1);
  for (int i = 0; i < num_axes_; ++i) {
    if (i == num_axes_ - 1) {
      old_steps_.mutable_cpu_data()[i] = 1;
    } else {
      old_steps_.mutable_cpu_data()[i] = bottom->count(i + 1);
    }
  }
  int count = bottom->count();
  for (int i = 0; i < num_axes_; ++i) {
    if (i == num_axes_ - 1) {
      new_steps_.mutable_cpu_data()[i] = 1;
    } else {
      new_steps_.mutable_cpu_data()[i] =
        count / bottom->shape(orders[i]);
      count /= bottom->shape(orders[i]);
    }
  }
  const int* old_steps = old_steps_.cpu_data();
  const int* new_steps = new_steps_.cpu_data();
  float err_sum = 0, sum = 0;
  for (int i = 0; i < bottom->count(); ++i) {
    int old_idx = 0;
    int idx = i;
    for (int j = 0; j < num_axes_; ++j) {
      int order = orders[j];
      old_idx += (idx / new_steps[j]) * old_steps[order];
      idx %= new_steps[j];
    }
    EXPECT_EQ(top_data[i], bottom_data[old_idx]);
    err_sum += (top_data[i] - bottom_data[old_idx]);
    sum += bottom_data[old_idx];
  }
  return err_sum/sum;
}

template <typename TypeParam>
class PermuteLayerTest : public CPUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  PermuteLayerTest()
    : blob_bottom_(new Blob<Dtype>(1, 5, 3, 4)),
      blob_top_(new Blob<Dtype>()) {
      FillerParameter filler_param;
      GaussianFiller<Dtype> filler(filler_param);
      filler.Fill(this->blob_bottom_);
      blob_bottom_vec_.push_back(blob_bottom_);
      blob_top_vec_.push_back(blob_top_);
    }
  virtual ~PermuteLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  void TestForward(vector<int> orders) {
    LayerParameter layer_param;
    PermuteParameter* permute_param = layer_param.mutable_permute_param();
    for (int i = 0; i < orders.size(); i++) {
      permute_param->add_order(orders[i]);
    }
    PermuteLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    caffe_permute(this->blob_bottom_, this->blob_top_, orders);
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(PermuteLayerTest, TestDtypesAndDevices);

TYPED_TEST(PermuteLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PermuteParameter* permute_param = layer_param.mutable_permute_param();
  permute_param->add_order(1);
  permute_param->add_order(0);
  PermuteLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_bottom_vec_.size(), 1);
  vector<int> orders;
  int num_axes_ = this->blob_bottom_->num_axes();
  for (int i = 0; i < permute_param->order_size(); i++) {
    int order = permute_param->order(i);
    EXPECT_LT(order, num_axes_);
    EXPECT_TRUE(std::find(orders.begin(), orders.end(), order) == orders.end());
    orders.push_back(order);
  }
  for (int i = 0; i < this->blob_bottom_->num_axes(); i++) {
    if (std::find(orders.begin(), orders.end(), i) == orders.end()) {
      orders.push_back(i);
    }
  }
  EXPECT_EQ(num_axes_, orders.size());
  for (int i = 0; i < num_axes_; i++) {
    EXPECT_EQ(this->blob_top_->shape(i), this->blob_bottom_->shape(orders[i]));
  }
}

TYPED_TEST(PermuteLayerTest, TestForward) {
  vector<int> orders = {0, 1, 2, 3};
  this->TestForward(orders);
}

TYPED_TEST(PermuteLayerTest, TestForward1) {
  vector<int> orders = {0, 2, 3, 1};
  this->TestForward(orders);
}

TYPED_TEST(PermuteLayerTest, TestForward2) {
  vector<int> orders = {3, 1, 2, 0};
  this->TestForward(orders);
}

TYPED_TEST(PermuteLayerTest, TestForwardorders) {
  vector<int> orders = {3, 1};
  this->TestForward(orders);
}

#ifdef USE_MLU
template <typename TypeParam>
class MLUPermuteLayerTest : public MLUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MLUPermuteLayerTest()
    : blob_bottom_(new Blob<Dtype>(1, 5, 3, 4)),
    blob_top_(new Blob<Dtype>()) {
      FillerParameter filler_param;
      GaussianFiller<Dtype> filler(filler_param);
      filler.Fill(this->blob_bottom_);
      blob_bottom_vec_.push_back(blob_bottom_);
      blob_top_vec_.push_back(blob_top_);
    }
  virtual ~MLUPermuteLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  void TestForward(vector<int> orders) {
    LayerParameter layer_param;
    PermuteParameter* permute_param = layer_param.mutable_permute_param();
    for (int i = 0; i < orders.size(); i++) {
      permute_param->add_order(orders[i]);
    }
    MLUPermuteLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    float rate = caffe_permute(this->blob_bottom_, this->blob_top_, orders);
    std::ostringstream stream;
    stream << "bottom1:" << blob_bottom_->shape_string().c_str();
    BOTTOM(stream);
    ERR_RATE(rate);
    EVENT_TIME(layer.get_event_time());
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MLUPermuteLayerTest, TestMLUDevices);

TYPED_TEST(MLUPermuteLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PermuteParameter* permute_param = layer_param.mutable_permute_param();
  permute_param->add_order(1);
  permute_param->add_order(0);
  MLUPermuteLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_bottom_vec_.size(), 1);
  vector<int> orders;
  int num_axes_ = this->blob_bottom_->num_axes();
  for (int i = 0; i < permute_param->order_size(); i++) {
    int order = permute_param->order(i);
    EXPECT_LT(order, num_axes_);
    EXPECT_TRUE(std::find(orders.begin(), orders.end(), order) == orders.end());
    orders.push_back(order);
  }
  for (int i = 0; i < this->blob_bottom_->num_axes(); i++) {
    if (std::find(orders.begin(), orders.end(), i) == orders.end()) {
      orders.push_back(i);
    }
  }
  EXPECT_EQ(num_axes_, orders.size());
  for (int i = 0; i < num_axes_; i++) {
    EXPECT_EQ(this->blob_top_->shape(i), this->blob_bottom_->shape(orders[i]));
  }
}

TYPED_TEST(MLUPermuteLayerTest, TestForward) {
    vector<int> orders = {0, 1, 2, 3};
      this->TestForward(orders);
}

TYPED_TEST(MLUPermuteLayerTest, TestForward1) {
    vector<int> orders = {0, 2, 3, 1};
      this->TestForward(orders);
}

TYPED_TEST(MLUPermuteLayerTest, TestForward2) {
    vector<int> orders = {3, 1, 2, 0};
      this->TestForward(orders);
}

template <typename TypeParam>
class MFUSPermuteLayerTest : public MFUSDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MFUSPermuteLayerTest()
    : blob_bottom_(new Blob<Dtype>(1, 5, 3, 4)),
      blob_top_(new Blob<Dtype>()) {}
  void SetUp() {
      FillerParameter filler_param;
      GaussianFiller<Dtype> filler(filler_param);
      filler.Fill(this->blob_bottom_);
      blob_bottom_vec_.push_back(blob_bottom_);
      blob_top_vec_.push_back(blob_top_);
    }
  virtual ~MFUSPermuteLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  void TestForward(vector<int> orders) {
    LayerParameter layer_param;
    PermuteParameter* permute_param = layer_param.mutable_permute_param();
    for (int i = 0; i < orders.size(); i++) {
      permute_param->add_order(orders[i]);
    }
    MLUPermuteLayer<Dtype> layer(layer_param);
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
    float rate = caffe_permute(this->blob_bottom_, this->blob_top_, orders);
    std::ostringstream stream, param;
    stream << "bottom1:" << blob_bottom_->shape_string().c_str();
    param << "orders:";
    for (int i = 0; i < orders.size(); i++) {
      param << orders[i];
    }
    BOTTOM(stream);
    ERR_RATE(rate);
    PARAM(param);
    EVENT_TIME(fuser.get_event_time());
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MFUSPermuteLayerTest, TestMFUSDevices);

TYPED_TEST(MFUSPermuteLayerTest, TestForward) {
    vector<int> orders = {0, 1, 2, 3};
      this->TestForward(orders);
}

TYPED_TEST(MFUSPermuteLayerTest, TestForward1) {
    vector<int> orders = {0, 2, 3, 1};
      this->TestForward(orders);
}

TYPED_TEST(MFUSPermuteLayerTest, TestForward2) {
    vector<int> orders = {3, 1, 2, 0};
      this->TestForward(orders);
}

TYPED_TEST(MFUSPermuteLayerTest, TestForwardorders) {
  vector<int> orders = {3, 1};
  this->TestForward(orders);
}

#endif

}  // namespace caffe
