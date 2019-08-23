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
#include <cmath>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/silence_layer.hpp"
#include "caffe/layers/mlu_silence_layer.hpp"
#include "caffe/layers/mlu_relu_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "gtest/gtest.h"
#include "caffe/util/math_functions.hpp"


namespace caffe {
template <typename TypeParam>
class SilenceLayerTest : public CPUDeviceTest<TypeParam> {
  typedef typename TypeParam :: Dtype Dtype;

  protected:
    SilenceLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)) {}

    virtual ~SilenceLayerTest() {
      delete blob_bottom_;
    }

    virtual void SetUp() {
      FillerParameter filler_param;
      GaussianFiller<Dtype> filler(filler_param);
      filler.Fill(this->blob_bottom_);
      blob_bottom_vec_.push_back(blob_bottom_);
    }

    Blob<Dtype>* const blob_bottom_;
    vector<Blob<Dtype>*> blob_bottom_vec_;
    vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SilenceLayerTest, TestDtypesAndDevices);

TYPED_TEST(SilenceLayerTest, Forward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SilenceLayer<Dtype> layer(layer_param);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  CHECK_EQ(0, layer.ExactNumTopBlobs());
}

#ifdef USE_MLU

template <typename TypeParam>
class MLUSilenceLayerTest : public MLUDeviceTest<TypeParam> {
  typedef typename TypeParam :: Dtype Dtype;

  protected:
    MLUSilenceLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)) {}

    virtual ~MLUSilenceLayerTest() {
      delete blob_bottom_;
    }

    virtual void SetUp() {
      FillerParameter filler_param;
      GaussianFiller<Dtype> filler(filler_param);
      filler.Fill(this->blob_bottom_);
      blob_bottom_vec_.push_back(blob_bottom_);
    }

    Blob<Dtype>* const blob_bottom_;
    vector<Blob<Dtype>*> blob_bottom_vec_;
    vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MLUSilenceLayerTest, TestMLUDevices);

TYPED_TEST(MLUSilenceLayerTest, Forward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MLUSilenceLayer<Dtype> layer(layer_param);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  CHECK_EQ(0, layer.ExactNumTopBlobs());
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  EVENT_TIME(layer.get_event_time());
}

template <typename TypeParam>
class MFUSSilenceLayerTest : public MFUSDeviceTest<TypeParam> {
  typedef typename TypeParam :: Dtype Dtype;

  protected:
    MFUSSilenceLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype>()) {}

    virtual ~MFUSSilenceLayerTest() {
      delete blob_bottom_;
    }

    virtual void SetUp() {
      FillerParameter filler_param;
      GaussianFiller<Dtype> filler(filler_param);
      filler.Fill(this->blob_bottom_);
      blob_bottom_vec_.push_back(blob_bottom_);
      blob_top_relu_.push_back(blob_top_);
    }

    Blob<Dtype>* const blob_bottom_;
    Blob<Dtype>* const blob_top_;
    vector<Blob<Dtype>*> blob_bottom_vec_;
    vector<Blob<Dtype>*> blob_top_vec_;
    vector<Blob<Dtype>*> blob_top_relu_;
};

TYPED_TEST_CASE(MFUSSilenceLayerTest, TestMFUSDevices);

TYPED_TEST(MFUSSilenceLayerTest, Forward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MLUSilenceLayer<Dtype> layer(layer_param);
  MLUReLULayer<Dtype> relu(layer_param);
  relu.LayerSetUp(this->blob_bottom_vec_, this->blob_top_relu_);
  ASSERT_TRUE(layer.mfus_supported());
  ASSERT_TRUE(relu.mfus_supported());

  MFusion<Dtype> fuser;
  fuser.reset();
  fuser.addInputs(this->blob_bottom_vec_);
  fuser.addOutputs(this->blob_top_relu_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  relu.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_relu_);
  layer.fuse(&fuser);
  relu.fuse(&fuser);
  fuser.compile();
  fuser.forward();
  CHECK_EQ(0, layer.ExactNumTopBlobs());
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  EVENT_TIME(fuser.get_event_time());
}
#endif
}  // namespace caffe
