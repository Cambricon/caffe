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
#include "caffe/layers/concat_layer.hpp"
#include "caffe/layers/mlu_concat_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class ConcatLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  ConcatLayerTest()
      : blob_bottom_0_(new Blob<Dtype>(2, 3, 6, 5)),
        blob_bottom_1_(new Blob<Dtype>(2, 5, 6, 5)),
        blob_bottom_2_(new Blob<Dtype>(5, 3, 6, 5)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_0_);
    filler.Fill(this->blob_bottom_1_);
    filler.Fill(this->blob_bottom_2_);
    blob_bottom_vec_0_.push_back(blob_bottom_0_);
    blob_bottom_vec_0_.push_back(blob_bottom_1_);
    blob_bottom_vec_1_.push_back(blob_bottom_0_);
    blob_bottom_vec_1_.push_back(blob_bottom_2_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~ConcatLayerTest() {
    delete blob_bottom_0_; delete blob_bottom_1_;
    delete blob_bottom_2_; delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_0_;
  Blob<Dtype>* const blob_bottom_1_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_0_, blob_bottom_vec_1_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ConcatLayerTest, TestDtypesAndDevices);

TYPED_TEST(ConcatLayerTest, TestSetupNum) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_concat_param()->set_axis(0);
  ConcatLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_1_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(),
      this->blob_bottom_0_->num() + this->blob_bottom_2_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_0_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_0_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_0_->width());
}

TYPED_TEST(ConcatLayerTest, TestSetupChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConcatLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_0_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_0_->num());
  EXPECT_EQ(this->blob_top_->channels(),
      this->blob_bottom_0_->channels() + this->blob_bottom_1_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_0_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_0_->width());
}

TYPED_TEST(ConcatLayerTest, TestSetupChannelsNegativeIndexing) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConcatLayer<Dtype> layer(layer_param);
  // "channels" index is the third one from the end -- test negative indexing
  // by setting axis to -3 and checking that we get the same results as above in
  // TestSetupChannels.
  layer_param.mutable_concat_param()->set_axis(-3);
  layer.SetUp(this->blob_bottom_vec_0_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_0_->num());
  EXPECT_EQ(this->blob_top_->channels(),
      this->blob_bottom_0_->channels() + this->blob_bottom_1_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_0_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_0_->width());
}

TYPED_TEST(ConcatLayerTest, TestForwardTrivial) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConcatLayer<Dtype> layer(layer_param);
  this->blob_bottom_vec_0_.resize(1);
  layer.SetUp(this->blob_bottom_vec_0_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_0_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_bottom_0_->count(); ++i) {
    EXPECT_EQ(this->blob_bottom_0_->cpu_data()[i],
              this->blob_top_->cpu_data()[i]);
  }
}

TYPED_TEST(ConcatLayerTest, TestForwardNum) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_concat_param()->set_axis(0);
  ConcatLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_1_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_1_, this->blob_top_vec_);
  for (int n = 0; n < this->blob_bottom_vec_1_[0]->num(); ++n) {
    for (int c = 0; c < this->blob_top_->channels(); ++c) {
      for (int h = 0; h < this->blob_top_->height(); ++h) {
        for (int w = 0; w < this->blob_top_->width(); ++w) {
          EXPECT_EQ(this->blob_top_->data_at(n, c, h, w),
              this->blob_bottom_vec_1_[0]->data_at(n, c, h, w));
        }
      }
    }
  }
  for (int n = 0; n < this->blob_bottom_vec_1_[1]->num(); ++n) {
    for (int c = 0; c < this->blob_top_->channels(); ++c) {
      for (int h = 0; h < this->blob_top_->height(); ++h) {
        for (int w = 0; w < this->blob_top_->width(); ++w) {
          EXPECT_EQ(this->blob_top_->data_at(n + 2, c, h, w),
              this->blob_bottom_vec_1_[1]->data_at(n, c, h, w));
        }
      }
    }
  }
}

TYPED_TEST(ConcatLayerTest, TestForwardChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConcatLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_0_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_0_, this->blob_top_vec_);
  for (int n = 0; n < this->blob_top_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_0_->channels(); ++c) {
      for (int h = 0; h < this->blob_top_->height(); ++h) {
        for (int w = 0; w < this->blob_top_->width(); ++w) {
          EXPECT_EQ(this->blob_top_->data_at(n, c, h, w),
              this->blob_bottom_vec_0_[0]->data_at(n, c, h, w));
        }
      }
    }
    for (int c = 0; c < this->blob_bottom_1_->channels(); ++c) {
      for (int h = 0; h < this->blob_top_->height(); ++h) {
        for (int w = 0; w < this->blob_top_->width(); ++w) {
          EXPECT_EQ(this->blob_top_->data_at(n, c + 3, h, w),
              this->blob_bottom_vec_0_[1]->data_at(n, c, h, w));
        }
      }
    }
  }
}

TYPED_TEST(ConcatLayerTest, TestGradientTrivial) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConcatLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  this->blob_bottom_vec_0_.resize(1);
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_0_,
      this->blob_top_vec_);
}

TYPED_TEST(ConcatLayerTest, TestGradientNum) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_concat_param()->set_axis(0);
  ConcatLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradient(&layer, this->blob_bottom_vec_1_,
    this->blob_top_vec_);
}

TYPED_TEST(ConcatLayerTest, TestGradientChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConcatLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradient(&layer, this->blob_bottom_vec_0_,
    this->blob_top_vec_);
}

TYPED_TEST(ConcatLayerTest, TestGradientChannelsBottomOneOnly) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConcatLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradient(&layer, this->blob_bottom_vec_0_,
    this->blob_top_vec_, 1);
}

#ifdef USE_MLU

template <typename TypeParam>
class MLUConcatLayerTest : public MLUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MLUConcatLayerTest()
      : blob_bottom_0_(new Blob<Dtype>(2, 3, 6, 5)),
        blob_bottom_1_(new Blob<Dtype>(2, 5, 6, 5)),
        blob_bottom_2_(new Blob<Dtype>(5, 3, 6, 5)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_0_);
    filler.Fill(this->blob_bottom_1_);
    filler.Fill(this->blob_bottom_2_);
    blob_bottom_vec_0_.push_back(blob_bottom_0_);
    blob_bottom_vec_0_.push_back(blob_bottom_1_);
    blob_bottom_vec_1_.push_back(blob_bottom_0_);
    blob_bottom_vec_1_.push_back(blob_bottom_2_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~MLUConcatLayerTest() {
    delete blob_bottom_0_; delete blob_bottom_1_;
    delete blob_bottom_2_; delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_0_;
  Blob<Dtype>* const blob_bottom_1_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_0_, blob_bottom_vec_1_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MLUConcatLayerTest, TestMLUDevices);

TYPED_TEST(MLUConcatLayerTest, TestSetupNum) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_concat_param()->set_axis(0);
  MLUConcatLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_1_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(),
      this->blob_bottom_0_->num() + this->blob_bottom_2_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_0_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_0_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_0_->width());
}

TYPED_TEST(MLUConcatLayerTest, TestSetupChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MLUConcatLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_0_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_0_->num());
  EXPECT_EQ(this->blob_top_->channels(),
      this->blob_bottom_0_->channels() + this->blob_bottom_1_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_0_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_0_->width());
}

TYPED_TEST(MLUConcatLayerTest, TestSetupChannelsNegativeIndexing) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MLUConcatLayer<Dtype> layer(layer_param);
  // "channels" index is the third one from the end -- test negative indexing
  // by setting axis to -3 and checking that we get the same results as above in
  // TestSetupChannels.
  layer_param.mutable_concat_param()->set_axis(-3);
  layer.SetUp(this->blob_bottom_vec_0_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_0_->num());
  EXPECT_EQ(this->blob_top_->channels(),
      this->blob_bottom_0_->channels() + this->blob_bottom_1_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_0_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_0_->width());
}

TYPED_TEST(MLUConcatLayerTest, TestForwardTrivial) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MLUConcatLayer<Dtype> layer(layer_param);
  this->blob_bottom_vec_0_.resize(1);
  layer.SetUp(this->blob_bottom_vec_0_, this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_0_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_0_, this->blob_top_vec_);
  float err_sum = 0, sum = 0;
  for (int i = 0; i < this->blob_bottom_0_->count(); ++i) {
    EXPECT_EQ(this->blob_bottom_0_->cpu_data()[i],
              this->blob_top_->cpu_data()[i]);
    err_sum += std::abs(this->blob_top_->cpu_data()[i]-
       this->blob_bottom_0_->cpu_data()[i]);
    sum += std::abs(this->blob_bottom_0_->cpu_data()[i]);
  }
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_0_->shape_string().c_str() << "\t"
    << "bottom2:" << this->blob_bottom_1_->shape_string().c_str();
  BOTTOM(stream);
  ERR_RATE(err_sum/sum);
  EVENT_TIME(layer.get_event_time());
}

TYPED_TEST(MLUConcatLayerTest, TestForwardNum) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_concat_param()->set_axis(0);
  MLUConcatLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_1_, this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_1_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_1_, this->blob_top_vec_);
  float err_sum = 0, sum = 0;
  for (int n = 0; n < this->blob_bottom_vec_1_[0]->num(); ++n) {
    for (int c = 0; c < this->blob_top_->channels(); ++c) {
      for (int h = 0; h < this->blob_top_->height(); ++h) {
        for (int w = 0; w < this->blob_top_->width(); ++w) {
          EXPECT_EQ(this->blob_top_->data_at(n, c, h, w),
              this->blob_bottom_vec_1_[0]->data_at(n, c, h, w));
          err_sum += std::abs(this->blob_top_->data_at(n, c, h, w) -
              this->blob_bottom_vec_1_[0]->data_at(n, c, h, w));
          sum += std::abs(this->blob_bottom_vec_1_[0]->data_at(n, c, h, w));
        }
      }
    }
  }
  for (int n = 0; n < this->blob_bottom_vec_1_[1]->num(); ++n) {
    for (int c = 0; c < this->blob_top_->channels(); ++c) {
      for (int h = 0; h < this->blob_top_->height(); ++h) {
        for (int w = 0; w < this->blob_top_->width(); ++w) {
          EXPECT_EQ(this->blob_top_->data_at(n + 2, c, h, w),
              this->blob_bottom_vec_1_[1]->data_at(n, c, h, w));
          err_sum += std::abs(this->blob_top_->data_at(n + 2, c, h, w) -
              this->blob_bottom_vec_1_[1]->data_at(n, c, h, w));
        }
      }
    }
  }
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_0_->shape_string().c_str()<< "\t"
    << "bottom2:" << this->blob_bottom_2_->shape_string().c_str();
  BOTTOM(stream);
  ERR_RATE(err_sum/sum);
  EVENT_TIME(layer.get_event_time());
}

template <typename TypeParam>
class MFUSConcatLayerTest : public MFUSDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MFUSConcatLayerTest()
      : blob_bottom_0_(new Blob<Dtype>(2, 3, 6, 5)),
        blob_bottom_1_(new Blob<Dtype>(2, 5, 6, 5)),
        blob_bottom_2_(new Blob<Dtype>(5, 3, 6, 5)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_0_);
    filler.Fill(this->blob_bottom_1_);
    filler.Fill(this->blob_bottom_2_);
    blob_bottom_vec_0_.push_back(blob_bottom_0_);
    blob_bottom_vec_0_.push_back(blob_bottom_1_);
    blob_bottom_vec_1_.push_back(blob_bottom_0_);
    blob_bottom_vec_1_.push_back(blob_bottom_2_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~MFUSConcatLayerTest() {
    delete blob_bottom_0_; delete blob_bottom_1_;
    delete blob_bottom_2_; delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_0_;
  Blob<Dtype>* const blob_bottom_1_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_0_, blob_bottom_vec_1_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MFUSConcatLayerTest, TestMFUSDevices);

TYPED_TEST(MFUSConcatLayerTest, TestForwardTrivial) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MLUConcatLayer<Dtype> layer(layer_param);
  this->blob_bottom_vec_0_.resize(1);
  layer.SetUp(this->blob_bottom_vec_0_, this->blob_top_vec_);
  MFusion<Dtype> fuser;
  fuser.reset();
  fuser.addInputs(this->blob_bottom_vec_0_);
  fuser.addOutputs(this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_0_, this->blob_top_vec_);
  layer.fuse(&fuser);
  fuser.compile();
  fuser.forward();
  float err_sum = 0, sum = 0;
  for (int i = 0; i < this->blob_bottom_0_->count(); ++i) {
    EXPECT_EQ(this->blob_bottom_0_->cpu_data()[i],
              this->blob_top_->cpu_data()[i]);
    err_sum += std::abs(this->blob_top_->cpu_data()[i] -
        this->blob_bottom_0_->cpu_data()[i]);
    sum += std::abs(this->blob_bottom_0_->cpu_data()[i]);
  }
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_0_->shape_string().c_str() << "\t"
    << "bottom2:" << this->blob_bottom_1_->shape_string().c_str();
  BOTTOM(stream);
  ERR_RATE(err_sum/sum);
  EVENT_TIME(fuser.get_event_time());
}

TYPED_TEST(MFUSConcatLayerTest, TestForwardNum) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_concat_param()->set_axis(0);
  MLUConcatLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_1_, this->blob_top_vec_);
  MFusion<Dtype> fuser;
  fuser.reset();
  fuser.addInputs(this->blob_bottom_vec_1_);
  fuser.addOutputs(this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_1_, this->blob_top_vec_);
  layer.fuse(&fuser);
  fuser.compile();
  fuser.forward();

  float err_sum = 0, sum = 0;
  for (int n = 0; n < this->blob_bottom_vec_1_[0]->num(); ++n) {
    for (int c = 0; c < this->blob_top_->channels(); ++c) {
      for (int h = 0; h < this->blob_top_->height(); ++h) {
        for (int w = 0; w < this->blob_top_->width(); ++w) {
          EXPECT_EQ(this->blob_top_->data_at(n, c, h, w),
              this->blob_bottom_vec_1_[0]->data_at(n, c, h, w));
          err_sum += this->blob_top_->data_at(n, c, h, w) -
              this->blob_bottom_vec_1_[0]->data_at(n, c, h, w);
          sum += std::abs(this->blob_bottom_vec_1_[0]->data_at(n, c, h, w));
        }
      }
    }
  }
  for (int n = 0; n < this->blob_bottom_vec_1_[1]->num(); ++n) {
    for (int c = 0; c < this->blob_top_->channels(); ++c) {
      for (int h = 0; h < this->blob_top_->height(); ++h) {
        for (int w = 0; w < this->blob_top_->width(); ++w) {
          EXPECT_EQ(this->blob_top_->data_at(n + 2, c, h, w),
              this->blob_bottom_vec_1_[1]->data_at(n, c, h, w));
          err_sum += std::abs(this->blob_top_->data_at(n + 2, c, h, w) -
              this->blob_bottom_vec_1_[1]->data_at(n, c, h, w));
          sum += std::abs(this->blob_bottom_vec_1_[1]->data_at(n, c, h, w));
        }
      }
    }
  }
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_0_->shape_string().c_str() << "\t"
    << "bottom2:" << this->blob_bottom_2_->shape_string().c_str();
  BOTTOM(stream);
  ERR_RATE(err_sum/sum);
  EVENT_TIME(fuser.get_event_time());
}
#endif

}  // namespace caffe
