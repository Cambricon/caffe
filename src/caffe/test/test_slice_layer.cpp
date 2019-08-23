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
#include "caffe/layers/mlu_slice_layer.hpp"
#include "caffe/layers/slice_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class SliceLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  SliceLayerTest()
      : blob_bottom_(new Blob<Dtype>(6, 12, 2, 3)),
        blob_top_0_(new Blob<Dtype>()),
        blob_top_1_(new Blob<Dtype>()),
        blob_top_2_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_top_vec_0_.push_back(blob_top_0_);
    blob_top_vec_0_.push_back(blob_top_1_);
    blob_top_vec_1_.push_back(blob_top_0_);
    blob_top_vec_1_.push_back(blob_top_1_);
    blob_top_vec_1_.push_back(blob_top_2_);
    blob_bottom_vec_.push_back(blob_bottom_);
  }

  virtual void ReduceBottomBlobSize() {
    blob_bottom_->Reshape(4, 5, 2, 2);
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
  }

  virtual ~SliceLayerTest() {
    delete blob_top_0_; delete blob_top_1_;
    delete blob_top_2_; delete blob_bottom_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_0_;
  Blob<Dtype>* const blob_top_1_;
  Blob<Dtype>* const blob_top_2_;
  vector<Blob<Dtype>*> blob_top_vec_0_, blob_top_vec_1_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
};

TYPED_TEST_CASE(SliceLayerTest, TestDtypesAndDevices);

TYPED_TEST(SliceLayerTest, TestSetupNum) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_slice_param()->set_axis(0);
  SliceLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_1_);
  EXPECT_EQ(this->blob_bottom_->num(), 3 * this->blob_top_0_->num());
  EXPECT_EQ(this->blob_top_0_->num(), this->blob_top_1_->num());
  EXPECT_EQ(this->blob_top_0_->num(), this->blob_top_2_->num());
  EXPECT_EQ(this->blob_bottom_->channels(), this->blob_top_0_->channels());
  EXPECT_EQ(this->blob_bottom_->height(), this->blob_top_0_->height());
  EXPECT_EQ(this->blob_bottom_->width(), this->blob_top_0_->width());
}

TYPED_TEST(SliceLayerTest, TestSetupChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_slice_param()->add_slice_point(3);
  SliceLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_0_);
  EXPECT_EQ(this->blob_top_0_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_0_->channels(), 3);
  EXPECT_EQ(this->blob_top_1_->channels(), 9);
  EXPECT_EQ(this->blob_bottom_->channels(),
    this->blob_top_0_->channels() + this->blob_top_1_->channels());
  EXPECT_EQ(this->blob_bottom_->height(), this->blob_top_0_->height());
  EXPECT_EQ(this->blob_bottom_->width(), this->blob_top_0_->width());
}

TYPED_TEST(SliceLayerTest, TestTrivialSlice) {
  // Test the trivial (single output) "slice" operation --
  // should be the identity.
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SliceLayer<Dtype> layer(layer_param);
  this->blob_top_vec_0_.resize(1);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_0_);
  ASSERT_EQ(this->blob_bottom_->shape(), this->blob_top_0_->shape());
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_EQ(this->blob_bottom_->cpu_data()[i],
              this->blob_top_0_->cpu_data()[i]);
  }
}

TYPED_TEST(SliceLayerTest, TestSliceAcrossNum) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_slice_param()->set_axis(0);
  SliceLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_0_);
  const int top_num = this->blob_bottom_->num() / 2;
  ASSERT_EQ(top_num, this->blob_top_0_->num());
  ASSERT_EQ(top_num, this->blob_top_1_->num());
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_0_);
  for (int n = 0; n < top_num; ++n) {
    for (int c = 0; c < this->blob_top_0_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_->width(); ++w) {
          EXPECT_EQ(this->blob_bottom_->data_at(n, c, h, w),
                    this->blob_top_0_->data_at(n, c, h, w));
        }
      }
    }
    for (int c = 0; c < this->blob_top_1_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_->width(); ++w) {
          EXPECT_EQ(this->blob_bottom_->data_at(n + 3, c, h, w),
                    this->blob_top_1_->data_at(n, c, h, w));
        }
      }
    }
  }
}

TYPED_TEST(SliceLayerTest, TestSliceAcrossChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  // Slice at 2, 8: should produce output blobs with #channels 2, 6, 4.
  const int kSlicePoint0 = 2;
  const int kSlicePoint1 = 8;
  layer_param.mutable_slice_param()->add_slice_point(kSlicePoint0);
  layer_param.mutable_slice_param()->add_slice_point(kSlicePoint1);
  SliceLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_1_);
  ASSERT_EQ(kSlicePoint0, this->blob_top_0_->channels());
  ASSERT_EQ(kSlicePoint1 - kSlicePoint0, this->blob_top_1_->channels());
  ASSERT_EQ(this->blob_bottom_->channels() - kSlicePoint1,
            this->blob_top_2_->channels());
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_1_);
  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    for (int c = 0; c < this->blob_top_0_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_->width(); ++w) {
          EXPECT_EQ(this->blob_bottom_->data_at(n, c, h, w),
              this->blob_top_0_->data_at(n, c, h, w));
        }
      }
    }
    for (int c = 0; c < this->blob_top_1_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_->width(); ++w) {
          EXPECT_EQ(this->blob_bottom_->data_at(n, c + kSlicePoint0, h, w),
              this->blob_top_1_->data_at(n, c, h, w));
        }
      }
    }
    for (int c = 0; c < this->blob_top_2_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_->width(); ++w) {
          EXPECT_EQ(this->blob_bottom_->data_at(n, c + kSlicePoint1, h, w),
              this->blob_top_2_->data_at(n, c, h, w));
        }
      }
    }
  }
}

TYPED_TEST(SliceLayerTest, TestGradientTrivial) {
  // Test the trivial (single output) "slice" operation --
  // should be the identity.
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SliceLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  this->blob_top_vec_0_.resize(1);
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_0_);
}

TYPED_TEST(SliceLayerTest, TestGradientAcrossNum) {
  typedef typename TypeParam::Dtype Dtype;
  // Gradient checks are slow; reduce blob size.
  this->ReduceBottomBlobSize();
  LayerParameter layer_param;
  layer_param.mutable_slice_param()->set_axis(0);
  SliceLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
    this->blob_top_vec_0_);
}

TYPED_TEST(SliceLayerTest, TestGradientAcrossChannels) {
  typedef typename TypeParam::Dtype Dtype;
  // Gradient checks are slow; reduce blob size.
  this->ReduceBottomBlobSize();
  LayerParameter layer_param;
  const int kSlicePoint = 4;
  layer_param.mutable_slice_param()->add_slice_point(kSlicePoint);
  SliceLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
    this->blob_top_vec_0_);
}

#ifdef USE_MLU

template <typename TypeParam>
class MLUSliceLayerTest : public MLUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MLUSliceLayerTest()
      : blob_bottom_(new Blob<Dtype>(6, 12, 2, 3)),
        blob_top_0_(new Blob<Dtype>()),
        blob_top_1_(new Blob<Dtype>()),
        blob_top_2_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_top_vec_0_.push_back(blob_top_0_);
    blob_top_vec_0_.push_back(blob_top_1_);
    blob_top_vec_1_.push_back(blob_top_0_);
    blob_top_vec_1_.push_back(blob_top_1_);
    blob_top_vec_1_.push_back(blob_top_2_);
    blob_bottom_vec_.push_back(blob_bottom_);
  }

  virtual ~MLUSliceLayerTest() {
    delete blob_top_0_; delete blob_top_1_;
    delete blob_top_2_; delete blob_bottom_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_0_;
  Blob<Dtype>* const blob_top_1_;
  Blob<Dtype>* const blob_top_2_;
  vector<Blob<Dtype>*> blob_top_vec_0_, blob_top_vec_1_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
};

TYPED_TEST_CASE(MLUSliceLayerTest, TestMLUDevices);

TYPED_TEST(MLUSliceLayerTest, TestSetupNum) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_slice_param()->set_axis(0);
  MLUSliceLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_1_);
  EXPECT_EQ(this->blob_bottom_->num(), 3 * this->blob_top_0_->num());
  EXPECT_EQ(this->blob_top_0_->num(), this->blob_top_1_->num());
  EXPECT_EQ(this->blob_top_0_->num(), this->blob_top_2_->num());
  EXPECT_EQ(this->blob_bottom_->channels(), this->blob_top_0_->channels());
  EXPECT_EQ(this->blob_bottom_->height(), this->blob_top_0_->height());
  EXPECT_EQ(this->blob_bottom_->width(), this->blob_top_0_->width());
  OUTPUT("bottom1", this->blob_bottom_->shape_string().c_str());
}

TYPED_TEST(MLUSliceLayerTest, TestSetupChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_slice_param()->add_slice_point(3);
  MLUSliceLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_0_);
  EXPECT_EQ(this->blob_top_0_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_0_->channels(), 3);
  EXPECT_EQ(this->blob_top_1_->channels(), 9);
  EXPECT_EQ(this->blob_bottom_->channels(),
    this->blob_top_0_->channels() + this->blob_top_1_->channels());
  EXPECT_EQ(this->blob_bottom_->height(), this->blob_top_0_->height());
  EXPECT_EQ(this->blob_bottom_->width(), this->blob_top_0_->width());
  OUTPUT("bottom1", this->blob_bottom_->shape_string().c_str());
}

TYPED_TEST(MLUSliceLayerTest, TestSliceAcrossNum) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_slice_param()->set_axis(0);
  MLUSliceLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_0_);
  const int top_num = this->blob_bottom_->num() / 2;
  ASSERT_EQ(top_num, this->blob_top_0_->num());
  ASSERT_EQ(top_num, this->blob_top_1_->num());
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_0_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_0_);
  float err_sum = 0, sum = 0;
  for (int n = 0; n < top_num; ++n) {
    for (int c = 0; c < this->blob_top_0_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_->width(); ++w) {
          EXPECT_EQ(this->blob_bottom_->data_at(n, c, h, w),
                    this->blob_top_0_->data_at(n, c, h, w));
          err_sum += std::abs(this->blob_bottom_->data_at(n, c, h, w) -
                    this->blob_top_0_->data_at(n, c, h, w));
          sum += std::abs(this->blob_top_0_->data_at(n, c, h, w));
        }
      }
    }
    for (int c = 0; c < this->blob_top_1_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_->width(); ++w) {
          EXPECT_EQ(this->blob_bottom_->data_at(n + 3, c, h, w),
                    this->blob_top_1_->data_at(n, c, h, w));
          err_sum += std::abs(this->blob_bottom_->data_at(n, c, h, w) -
                                  this->blob_top_0_->data_at(n, c, h, w));
          sum += std::abs(this->blob_top_0_->data_at(n, c, h, w));
        }
      }
    }
  }
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  ERR_RATE(err_sum/sum);
  EVENT_TIME(layer.get_event_time());
}

TYPED_TEST(MLUSliceLayerTest, TestSliceAcrossChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  // Slice at 2, 8: should produce output blobs with #channels 2, 6, 4.
  const int kSlicePoint0 = 2;
  const int kSlicePoint1 = 8;
  layer_param.mutable_slice_param()->add_slice_point(kSlicePoint0);
  layer_param.mutable_slice_param()->add_slice_point(kSlicePoint1);
  MLUSliceLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_1_);
  ASSERT_EQ(kSlicePoint0, this->blob_top_0_->channels());
  ASSERT_EQ(kSlicePoint1 - kSlicePoint0, this->blob_top_1_->channels());
  ASSERT_EQ(this->blob_bottom_->channels() - kSlicePoint1,
            this->blob_top_2_->channels());
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_1_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_1_);
  float err_sum = 0, sum = 0;
  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    for (int c = 0; c < this->blob_top_0_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_->width(); ++w) {
          EXPECT_EQ(this->blob_bottom_->data_at(n, c, h, w),
              this->blob_top_0_->data_at(n, c, h, w));
          err_sum += std::abs(this->blob_bottom_->data_at(n, c, h, w) -
              this->blob_top_0_->data_at(n, c, h, w));
          sum += std::abs(this->blob_top_0_->data_at(n, c, h, w));
        }
      }
    }
    for (int c = 0; c < this->blob_top_1_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_->width(); ++w) {
          EXPECT_EQ(this->blob_bottom_->data_at(n, c + kSlicePoint0, h, w),
              this->blob_top_1_->data_at(n, c, h, w));
          err_sum += std::abs(this->blob_bottom_->data_at(n, c + kSlicePoint0, h, w) -
              this->blob_top_1_->data_at(n, c, h, w));
          sum += std::abs(this->blob_top_1_->data_at(n, c, h, w));
        }
      }
    }
    for (int c = 0; c < this->blob_top_2_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_->width(); ++w) {
          EXPECT_EQ(this->blob_bottom_->data_at(n, c + kSlicePoint1, h, w),
              this->blob_top_2_->data_at(n, c, h, w));
          err_sum += std::abs(this->blob_bottom_->data_at(n, c + kSlicePoint1, h, w) -
              this->blob_top_2_->data_at(n, c, h, w));
          sum += std::abs(this->blob_top_2_->data_at(n, c, h, w));
        }
      }
    }
  }
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  ERR_RATE(err_sum/sum);
  EVENT_TIME(layer.get_event_time());
}

template <typename TypeParam>
class MFUSSliceLayerTest : public MFUSDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MFUSSliceLayerTest()
      : blob_bottom_(new Blob<Dtype>(6, 12, 2, 3)),
        blob_top_0_(new Blob<Dtype>()),
        blob_top_1_(new Blob<Dtype>()),
        blob_top_2_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_top_vec_0_.push_back(blob_top_0_);
    blob_top_vec_0_.push_back(blob_top_1_);
    blob_top_vec_1_.push_back(blob_top_0_);
    blob_top_vec_1_.push_back(blob_top_1_);
    blob_top_vec_1_.push_back(blob_top_2_);
    blob_bottom_vec_.push_back(blob_bottom_);
  }

  virtual ~MFUSSliceLayerTest() {
    delete blob_top_0_;
    delete blob_top_1_;
    delete blob_top_2_;
    delete blob_bottom_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_0_;
  Blob<Dtype>* const blob_top_1_;
  Blob<Dtype>* const blob_top_2_;
  vector<Blob<Dtype>*> blob_top_vec_0_, blob_top_vec_1_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
};

TYPED_TEST_CASE(MFUSSliceLayerTest, TestMFUSDevices);

TYPED_TEST(MFUSSliceLayerTest, TestSetupNum) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_slice_param()->set_axis(0);
  MLUSliceLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_1_);
  EXPECT_EQ(this->blob_bottom_->num(), 3 * this->blob_top_0_->num());
  EXPECT_EQ(this->blob_top_0_->num(), this->blob_top_1_->num());
  EXPECT_EQ(this->blob_top_0_->num(), this->blob_top_2_->num());
  EXPECT_EQ(this->blob_bottom_->channels(), this->blob_top_0_->channels());
  EXPECT_EQ(this->blob_bottom_->height(), this->blob_top_0_->height());
  EXPECT_EQ(this->blob_bottom_->width(), this->blob_top_0_->width());
  OUTPUT("bottom1", this->blob_bottom_->shape_string().c_str());
}

TYPED_TEST(MFUSSliceLayerTest, TestSetupChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_slice_param()->add_slice_point(3);
  MLUSliceLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_0_);
  EXPECT_EQ(this->blob_top_0_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_0_->channels(), 3);
  EXPECT_EQ(this->blob_top_1_->channels(), 9);
  EXPECT_EQ(this->blob_bottom_->channels(),
    this->blob_top_0_->channels() + this->blob_top_1_->channels());
  EXPECT_EQ(this->blob_bottom_->height(), this->blob_top_0_->height());
  EXPECT_EQ(this->blob_bottom_->width(), this->blob_top_0_->width());
  OUTPUT("bottom1", this->blob_bottom_->shape_string().c_str());
}

TYPED_TEST(MFUSSliceLayerTest, TestSliceAcrossNum) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_slice_param()->set_axis(0);
  MLUSliceLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_0_);
  const int top_num = this->blob_bottom_->num() / 2;
  ASSERT_EQ(top_num, this->blob_top_0_->num());
  ASSERT_EQ(top_num, this->blob_top_1_->num());
  ASSERT_TRUE(layer.mfus_supported());

  MFusion<Dtype> fuser;
  fuser.reset();
  fuser.addInputs(this->blob_bottom_vec_);
  fuser.addOutputs(this->blob_top_vec_0_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_0_);
  layer.fuse(&fuser);
  fuser.compile();
  fuser.forward();
  float err_sum = 0, sum = 0;
  for (int n = 0; n < top_num; ++n) {
    for (int c = 0; c < this->blob_top_0_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_->width(); ++w) {
          EXPECT_EQ(this->blob_bottom_->data_at(n, c, h, w),
                    this->blob_top_0_->data_at(n, c, h, w));
          err_sum += std::abs(this->blob_bottom_->data_at(n, c, h, w) -
                                  this->blob_top_0_->data_at(n, c, h, w));
          sum += std::abs(this->blob_top_0_->data_at(n, c, h, w));
        }
      }
    }
    for (int c = 0; c < this->blob_top_1_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_->width(); ++w) {
          EXPECT_EQ(this->blob_bottom_->data_at(n + 3, c, h, w),
                    this->blob_top_1_->data_at(n, c, h, w));
          err_sum += std::abs(this->blob_top_1_->data_at(n, c, h, w) -
                    this->blob_bottom_->data_at(n + 3, c, h, w));
          sum += std::abs(this->blob_bottom_->data_at(n + 3, c, h, w));
        }
      }
    }
  }
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  ERR_RATE(err_sum/sum);
  EVENT_TIME(fuser.get_event_time());
}

TYPED_TEST(MFUSSliceLayerTest, TestSliceAcrossChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  // Slice at 2, 8: should produce output blobs with #channels 2, 6, 4.
  const int kSlicePoint0 = 2;
  const int kSlicePoint1 = 8;
  layer_param.mutable_slice_param()->add_slice_point(kSlicePoint0);
  layer_param.mutable_slice_param()->add_slice_point(kSlicePoint1);
  MLUSliceLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_1_);
  ASSERT_EQ(kSlicePoint0, this->blob_top_0_->channels());
  ASSERT_EQ(kSlicePoint1 - kSlicePoint0, this->blob_top_1_->channels());
  ASSERT_EQ(this->blob_bottom_->channels() - kSlicePoint1,
            this->blob_top_2_->channels());
  ASSERT_TRUE(layer.mfus_supported());

  MFusion<Dtype> fuser;
  fuser.reset();
  fuser.addInputs(this->blob_bottom_vec_);
  fuser.addOutputs(this->blob_top_vec_1_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_1_);
  layer.fuse(&fuser);
  fuser.compile();
  fuser.forward();
  float err_sum = 0, sum = 0;
  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    for (int c = 0; c < this->blob_top_0_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_->width(); ++w) {
          EXPECT_EQ(this->blob_bottom_->data_at(n, c, h, w),
              this->blob_top_0_->data_at(n, c, h, w));
          err_sum += std::abs(this->blob_top_0_->data_at(n, c, h, w) -
              this->blob_bottom_->data_at(n, c, h, w));
          sum += std::abs(this->blob_bottom_->data_at(n, c, h, w));
        }
      }
    }
    for (int c = 0; c < this->blob_top_1_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_->width(); ++w) {
          EXPECT_EQ(this->blob_bottom_->data_at(n, c + kSlicePoint0, h, w),
              this->blob_top_1_->data_at(n, c, h, w));
          err_sum += std::abs(this->blob_top_1_->data_at(n, c, h, w) -
                            this->blob_bottom_->data_at(n, c + kSlicePoint0, h, w));
          sum += std::abs(this->blob_bottom_->data_at(n, c + kSlicePoint0, h, w));
        }
      }
    }
    for (int c = 0; c < this->blob_top_2_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_->width(); ++w) {
          EXPECT_EQ(this->blob_bottom_->data_at(n, c + kSlicePoint1, h, w),
              this->blob_top_2_->data_at(n, c, h, w));
          err_sum += std::abs(this->blob_top_2_->data_at(n, c, h, w) -
              this->blob_bottom_->data_at(n, c + kSlicePoint1, h, w));
          sum += std::abs(this->blob_bottom_->data_at(n, c + kSlicePoint1, h, w));
        }
      }
    }
  }
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  ERR_RATE(err_sum/sum);
  EVENT_TIME(fuser.get_event_time());
}

#endif

}  // namespace caffe
