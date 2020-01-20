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
#include "caffe/layers/crop_layer.hpp"
#include "caffe/layers/mlu_crop_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class CropLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  CropLayerTest()
      : blob_bottom_0_(new Blob<Dtype>(2, 4, 5, 4)),
        blob_bottom_1_(new Blob<Dtype>(2, 3, 4, 2)),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_0_);
    filler.Fill(this->blob_bottom_1_);

    blob_bottom_vec_.push_back(blob_bottom_0_);
    blob_bottom_vec_.push_back(blob_bottom_1_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~CropLayerTest() {
    delete blob_bottom_0_; delete blob_bottom_1_;
    delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_0_;
  Blob<Dtype>* const blob_bottom_1_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};


TYPED_TEST_CASE(CropLayerTest, TestDtypesAndDevices);

TYPED_TEST(CropLayerTest, TestSetupShapeAll) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  // Crop all dimensions
  layer_param.mutable_crop_param()->set_axis(0);
  CropLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_top_->num_axes(); ++i) {
    EXPECT_EQ(this->blob_bottom_1_->shape(i), this->blob_top_->shape(i));
  }
}

TYPED_TEST(CropLayerTest, TestSetupShapeDefault) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  // Crop last two dimensions, axis is 2 by default
  CropLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_top_->num_axes(); ++i) {
    if (i < 2) {
      EXPECT_EQ(this->blob_bottom_0_->shape(i), this->blob_top_->shape(i));
    } else {
      EXPECT_EQ(this->blob_bottom_1_->shape(i), this->blob_top_->shape(i));
    }
  }
}

TYPED_TEST(CropLayerTest, TestSetupShapeNegativeIndexing) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  // Crop last dimension by negative indexing
  layer_param.mutable_crop_param()->set_axis(-1);
  CropLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_top_->num_axes(); ++i) {
    if (i < 3) {
      EXPECT_EQ(this->blob_bottom_0_->shape(i), this->blob_top_->shape(i));
    } else {
      EXPECT_EQ(this->blob_bottom_1_->shape(i), this->blob_top_->shape(i));
    }
  }
}

TYPED_TEST(CropLayerTest, TestDimensionsCheck) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  // Reshape size blob to have incompatible sizes for uncropped dimensions:
  // the size blob has more channels than the data blob, but this is fine
  // since the channels dimension is not cropped in this configuration.
  this->blob_bottom_1_->Reshape(2, 5, 4, 2);
  CropLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_top_->num_axes(); ++i) {
    if (i < 2) {
      EXPECT_EQ(this->blob_bottom_0_->shape(i), this->blob_top_->shape(i));
    } else {
      EXPECT_EQ(this->blob_bottom_1_->shape(i), this->blob_top_->shape(i));
    }
  }
}

TYPED_TEST(CropLayerTest, TestCropAll) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_crop_param()->set_axis(0);
  CropLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int n = 0; n < this->blob_bottom_0_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_0_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_0_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_0_->width(); ++w) {
          if ( n < this->blob_top_->shape(0) &&
              c < this->blob_top_->shape(1) &&
              h < this->blob_top_->shape(2) &&
              w < this->blob_top_->shape(3) ) {
            EXPECT_EQ(this->blob_top_->data_at(n, c, h, w),
                this->blob_bottom_0_->data_at(n, c, h, w));
          }
        }
      }
    }
  }
}

TYPED_TEST(CropLayerTest, TestCropAllOffset) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_crop_param()->set_axis(0);
  layer_param.mutable_crop_param()->add_offset(0);
  layer_param.mutable_crop_param()->add_offset(1);
  layer_param.mutable_crop_param()->add_offset(1);
  layer_param.mutable_crop_param()->add_offset(2);
  CropLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int n = 0; n < this->blob_bottom_0_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_0_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_0_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_0_->width(); ++w) {
          if ( n < this->blob_top_->shape(0) &&
              c < this->blob_top_->shape(1) &&
              h < this->blob_top_->shape(2) &&
              w < this->blob_top_->shape(3) ) {
            EXPECT_EQ(this->blob_top_->data_at(n, c, h, w),
                this->blob_bottom_0_->data_at(n, c+1, h+1, w+2));
          }
        }
      }
    }
  }
}

TYPED_TEST(CropLayerTest, TestCropHW) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_crop_param()->set_axis(2);
  layer_param.mutable_crop_param()->add_offset(1);
  layer_param.mutable_crop_param()->add_offset(2);
  CropLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int n = 0; n < this->blob_bottom_0_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_0_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_0_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_0_->width(); ++w) {
          if (n < this->blob_top_->shape(0) &&
              c < this->blob_top_->shape(1) &&
              h < this->blob_top_->shape(2) &&
              w < this->blob_top_->shape(3)) {
            EXPECT_EQ(this->blob_top_->data_at(n, c, h, w),
                this->blob_bottom_0_->data_at(n, c, h+1, w+2));
          }
        }
      }
    }
  }
}

TYPED_TEST(CropLayerTest, TestCropAllGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_crop_param()->set_axis(0);
  CropLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(CropLayerTest, TestCropHWGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_crop_param()->set_axis(2);
  layer_param.mutable_crop_param()->add_offset(1);
  layer_param.mutable_crop_param()->add_offset(2);
  CropLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

#ifdef USE_MLU
template <typename TypeParam>
class MLUCropLayerTest : public MLUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MLUCropLayerTest()
      : blob_bottom_0_(new Blob<Dtype>(2, 4, 5, 4)),
        blob_bottom_1_(new Blob<Dtype>(2, 3, 4, 2)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_0_);
    filler.Fill(this->blob_bottom_1_);

    blob_bottom_vec_.push_back(blob_bottom_0_);
    blob_bottom_vec_.push_back(blob_bottom_1_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~MLUCropLayerTest() {
    delete blob_bottom_0_; delete blob_bottom_1_;
    delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_0_;
  Blob<Dtype>* const blob_bottom_1_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MLUCropLayerTest, TestMLUDevices);

TYPED_TEST(MLUCropLayerTest, TestSetupShapeAll) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  // Crop all dimensions
  layer_param.mutable_crop_param()->set_axis(0);
  MLUCropLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_top_->num_axes(); ++i) {
    EXPECT_EQ(this->blob_bottom_1_->shape(i), this->blob_top_->shape(i));
  }
}

TYPED_TEST(MLUCropLayerTest, TestSetupShapeDefault) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  // Crop last two dimensions, axis is 2 by default
  MLUCropLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_top_->num_axes(); ++i) {
    if (i < 2) {
      EXPECT_EQ(this->blob_bottom_0_->shape(i), this->blob_top_->shape(i));
    } else {
      EXPECT_EQ(this->blob_bottom_1_->shape(i), this->blob_top_->shape(i));
    }
  }
}

TYPED_TEST(MLUCropLayerTest, TestSetupShapeNegativeIndexing) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  // Crop last dimension by negative indexing
  layer_param.mutable_crop_param()->set_axis(-1);
  MLUCropLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_top_->num_axes(); ++i) {
    if (i < 3) {
      EXPECT_EQ(this->blob_bottom_0_->shape(i), this->blob_top_->shape(i));
    } else {
      EXPECT_EQ(this->blob_bottom_1_->shape(i), this->blob_top_->shape(i));
    }
  }
}

TYPED_TEST(MLUCropLayerTest, TestDimensionsCheck) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  // Reshape size blob to have incompatible sizes for uncropped dimensions:
  // the size blob has more channels than the data blob, but this is fine
  // since the channels dimension is not cropped in this configuration.
  this->blob_bottom_1_->Reshape(2, 5, 4, 2);
  MLUCropLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_top_->num_axes(); ++i) {
    if (i < 2) {
      EXPECT_EQ(this->blob_bottom_0_->shape(i), this->blob_top_->shape(i));
    } else {
      EXPECT_EQ(this->blob_bottom_1_->shape(i), this->blob_top_->shape(i));
    }
  }
}

TYPED_TEST(MLUCropLayerTest, TestCropAll) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_crop_param()->set_axis(0);
  MLUCropLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  float err_sum = 0, sum = 0;
  for (int n = 0; n < this->blob_bottom_0_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_0_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_0_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_0_->width(); ++w) {
          if ( n < this->blob_top_->shape(0) &&
              c < this->blob_top_->shape(1) &&
              h < this->blob_top_->shape(2) &&
              w < this->blob_top_->shape(3) ) {
            EXPECT_EQ(this->blob_top_->data_at(n, c, h, w),
                this->blob_bottom_0_->data_at(n, c, h, w));
            err_sum += std::abs(this->blob_top_->data_at(n, c, h, w) -
                this->blob_bottom_0_->data_at(n, c, h, w));
            sum += std::abs(this->blob_bottom_0_->data_at(n, c, h, w));
          }
        }
      }
    }
  }
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_0_->shape_string().c_str() << "\t"
    << "bottom2:" << this->blob_bottom_1_->shape_string().c_str();
  BOTTOM(stream);
  ERR_RATE(err_sum/sum);
  EVENT_TIME(layer.get_event_time());
}

TYPED_TEST(MLUCropLayerTest, TestCropAllOffset) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_crop_param()->set_axis(0);
  layer_param.mutable_crop_param()->add_offset(0);
  layer_param.mutable_crop_param()->add_offset(1);
  layer_param.mutable_crop_param()->add_offset(1);
  layer_param.mutable_crop_param()->add_offset(2);
  MLUCropLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  float err_sum = 0, sum = 0;
  for (int n = 0; n < this->blob_bottom_0_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_0_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_0_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_0_->width(); ++w) {
          if ( n < this->blob_top_->shape(0) &&
              c < this->blob_top_->shape(1) &&
              h < this->blob_top_->shape(2) &&
              w < this->blob_top_->shape(3) ) {
            EXPECT_EQ(this->blob_top_->data_at(n, c, h, w),
                this->blob_bottom_0_->data_at(n, c+1, h+1, w+2));
            err_sum += std::abs(this->blob_top_->data_at(n, c, h, w) -
                this->blob_bottom_0_->data_at(n, c+1, h+1, w+2));
            sum += std::abs(this->blob_bottom_0_->data_at(n, c+1, h+1, w+2));
          }
        }
      }
    }
  }
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_0_->shape_string().c_str() << "\t"
    << "bottom2:" << this->blob_bottom_1_->shape_string().c_str();
  BOTTOM(stream);
  ERR_RATE(err_sum/sum);
  EVENT_TIME(layer.get_event_time());
}

TYPED_TEST(MLUCropLayerTest, TestCropHW) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_crop_param()->set_axis(2);
  layer_param.mutable_crop_param()->add_offset(1);
  layer_param.mutable_crop_param()->add_offset(2);
  MLUCropLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  float err_sum = 0, sum = 0;
  for (int n = 0; n < this->blob_bottom_0_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_0_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_0_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_0_->width(); ++w) {
          if (n < this->blob_top_->shape(0) &&
              c < this->blob_top_->shape(1) &&
              h < this->blob_top_->shape(2) &&
              w < this->blob_top_->shape(3)) {
            EXPECT_EQ(this->blob_top_->data_at(n, c, h, w),
                this->blob_bottom_0_->data_at(n, c, h+1, w+2));
            err_sum += std::abs(this->blob_top_->data_at(n, c, h, w) -
                this->blob_bottom_0_->data_at(n, c, h+1, w+2));
            sum += std::abs(this->blob_bottom_0_->data_at(n, c, h+1, w+2));
          }
        }
      }
    }
  }
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_0_->shape_string().c_str() << "\t"
    << "bottom2:" << this->blob_bottom_0_->shape_string().c_str();
  BOTTOM(stream);
  ERR_RATE(err_sum/sum);
  EVENT_TIME(layer.get_event_time());
}

template <typename TypeParam>
class MFUSCropLayerTest : public MFUSDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MFUSCropLayerTest()
      : blob_bottom_0_(new Blob<Dtype>(2, 4, 5, 4)),
        blob_bottom_1_(new Blob<Dtype>(2, 3, 4, 2)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_0_);
    filler.Fill(this->blob_bottom_1_);

    blob_bottom_vec_.push_back(blob_bottom_0_);
    blob_bottom_vec_.push_back(blob_bottom_1_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~MFUSCropLayerTest() {
    delete blob_bottom_0_; delete blob_bottom_1_;
    delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_0_;
  Blob<Dtype>* const blob_bottom_1_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MFUSCropLayerTest, TestMFUSDevices);

TYPED_TEST(MFUSCropLayerTest, TestSetupShapeAll) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  // Crop all dimensions
  layer_param.mutable_crop_param()->set_axis(0);
  MLUCropLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_top_->num_axes(); ++i) {
    EXPECT_EQ(this->blob_bottom_1_->shape(i), this->blob_top_->shape(i));
  }
}

TYPED_TEST(MFUSCropLayerTest, TestSetupShapeDefault) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  // Crop last two dimensions, axis is 2 by default
  MLUCropLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_top_->num_axes(); ++i) {
    if (i < 2) {
      EXPECT_EQ(this->blob_bottom_0_->shape(i), this->blob_top_->shape(i));
    } else {
      EXPECT_EQ(this->blob_bottom_1_->shape(i), this->blob_top_->shape(i));
    }
  }
}

TYPED_TEST(MFUSCropLayerTest, TestSetupShapeNegativeIndexing) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  // Crop last dimension by negative indexing
  layer_param.mutable_crop_param()->set_axis(-1);
  MLUCropLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_top_->num_axes(); ++i) {
    if (i < 3) {
      EXPECT_EQ(this->blob_bottom_0_->shape(i), this->blob_top_->shape(i));
    } else {
      EXPECT_EQ(this->blob_bottom_1_->shape(i), this->blob_top_->shape(i));
    }
  }
}

TYPED_TEST(MFUSCropLayerTest, TestDimensionsCheck) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  // Reshape size blob to have incompatible sizes for uncropped dimensions:
  // the size blob has more channels than the data blob, but this is fine
  // since the channels dimension is not cropped in this configuration.
  this->blob_bottom_1_->Reshape(2, 5, 4, 2);
  MLUCropLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_top_->num_axes(); ++i) {
    if (i < 2) {
      EXPECT_EQ(this->blob_bottom_0_->shape(i), this->blob_top_->shape(i));
    } else {
      EXPECT_EQ(this->blob_bottom_1_->shape(i), this->blob_top_->shape(i));
    }
  }
}

TYPED_TEST(MFUSCropLayerTest, TestCropAll) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_crop_param()->set_axis(0);
  MLUCropLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_TRUE(layer.mfus_supported());
  this->blob_bottom_vec_.pop_back();

  MFusion<Dtype> fuser;
  fuser.reset();
  fuser.addInputs(this->blob_bottom_vec_);
  fuser.addOutputs(this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.fuse(&fuser);
  fuser.compile();
  fuser.forward();
  float err_sum = 0, sum = 0;
  for (int n = 0; n < this->blob_bottom_0_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_0_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_0_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_0_->width(); ++w) {
          if ( n < this->blob_top_->shape(0) &&
              c < this->blob_top_->shape(1) &&
              h < this->blob_top_->shape(2) &&
              w < this->blob_top_->shape(3) ) {
            EXPECT_EQ(this->blob_top_->data_at(n, c, h, w),
                this->blob_bottom_0_->data_at(n, c, h, w));
            err_sum += std::abs(this->blob_top_->data_at(n, c, h, w) -
                this->blob_bottom_0_->data_at(n, c, h, w));
            sum += std::abs(this->blob_bottom_0_->data_at(n, c, h, w));
          }
        }
      }
    }
  }
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_0_->shape_string().c_str() << "\t"
    << "bottom2:" << this->blob_bottom_1_->shape_string().c_str();
  BOTTOM(stream);
  ERR_RATE(err_sum/sum);
  EVENT_TIME(fuser.get_event_time());
}

TYPED_TEST(MFUSCropLayerTest, TestCropAllOffset) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_crop_param()->set_axis(0);
  layer_param.mutable_crop_param()->add_offset(0);
  layer_param.mutable_crop_param()->add_offset(1);
  layer_param.mutable_crop_param()->add_offset(1);
  layer_param.mutable_crop_param()->add_offset(2);
  MLUCropLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_TRUE(layer.mfus_supported());
  this->blob_bottom_vec_.pop_back();

  MFusion<Dtype> fuser;
  fuser.reset();
  fuser.addInputs(this->blob_bottom_vec_);
  fuser.addOutputs(this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.fuse(&fuser);
  fuser.compile();
  fuser.forward();
  float err_sum = 0, sum = 0;
  for (int n = 0; n < this->blob_bottom_0_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_0_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_0_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_0_->width(); ++w) {
          if ( n < this->blob_top_->shape(0) &&
              c < this->blob_top_->shape(1) &&
              h < this->blob_top_->shape(2) &&
              w < this->blob_top_->shape(3) ) {
            EXPECT_EQ(this->blob_top_->data_at(n, c, h, w),
                this->blob_bottom_0_->data_at(n, c+1, h+1, w+2));
            err_sum += std::abs(this->blob_top_->data_at(n, c, h, w) -
                this->blob_bottom_0_->data_at(n, c+1, h+1, w+2));
            sum += std::abs(this->blob_bottom_0_->data_at(n, c+1, h+1, w+2));
          }
        }
      }
    }
  }
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_0_->shape_string().c_str() << "\t"
    << "bottom2:" << this->blob_bottom_1_->shape_string().c_str();
  BOTTOM(stream);
  ERR_RATE(err_sum/sum);
  EVENT_TIME(fuser.get_event_time());
}

TYPED_TEST(MFUSCropLayerTest, TestCropHW) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_crop_param()->set_axis(2);
  layer_param.mutable_crop_param()->add_offset(1);
  layer_param.mutable_crop_param()->add_offset(2);
  MLUCropLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_TRUE(layer.mfus_supported());
  this->blob_bottom_vec_.pop_back();

  MFusion<Dtype> fuser;
  fuser.reset();
  fuser.addInputs(this->blob_bottom_vec_);
  fuser.addOutputs(this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.fuse(&fuser);
  fuser.compile();
  fuser.forward();
  float err_sum = 0, sum = 0;
  for (int n = 0; n < this->blob_bottom_0_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_0_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_0_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_0_->width(); ++w) {
          if (n < this->blob_top_->shape(0) &&
              c < this->blob_top_->shape(1) &&
              h < this->blob_top_->shape(2) &&
              w < this->blob_top_->shape(3)) {
            EXPECT_EQ(this->blob_top_->data_at(n, c, h, w),
                this->blob_bottom_0_->data_at(n, c, h+1, w+2));
            err_sum += std::abs(this->blob_top_->data_at(n, c, h, w) -
                this->blob_bottom_0_->data_at(n, c, h+1, w+2));
            sum += std::abs(this->blob_bottom_0_->data_at(n, c, h+1, w+2));
          }
        }
      }
    }
  }
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_0_->shape_string().c_str() << "\t"
    << "bottom2:" << this->blob_bottom_1_->shape_string().c_str();
  BOTTOM(stream);
  ERR_RATE(err_sum/sum);
  EVENT_TIME(fuser.get_event_time());
}
#endif

}  // namespace caffe
