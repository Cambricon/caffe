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

#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/lrn_layer.hpp"
#include "caffe/layers/mlu_lrn_layer.hpp"

#ifdef USE_CUDNN
#include "caffe/layers/cudnn_lcn_layer.hpp"
#include "caffe/layers/cudnn_lrn_layer.hpp"
#endif

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using std::min;
using std::max;

namespace caffe {

template <typename Dtype>
void referenceLRNForward(const Blob<Dtype>& blob_bottom,
                         const LayerParameter& layer_param,
                         Blob<Dtype>* blob_top) {
  blob_top->Reshape(blob_bottom.num(), blob_bottom.channels(),
                    blob_bottom.height(), blob_bottom.width());
  Dtype* top_data = blob_top->mutable_cpu_data();
  LRNParameter lrn_param = layer_param.lrn_param();
  Dtype alpha = lrn_param.alpha();
  Dtype beta = lrn_param.beta();
  int size = lrn_param.local_size();
  switch (lrn_param.norm_region()) {
    case LRNParameter_NormRegion_ACROSS_CHANNELS:
      for (int n = 0; n < blob_bottom.num(); ++n) {
        for (int c = 0; c < blob_bottom.channels(); ++c) {
          for (int h = 0; h < blob_bottom.height(); ++h) {
            for (int w = 0; w < blob_bottom.width(); ++w) {
              int c_start = c - (size - 1) / 2;
              int c_end = min(c_start + size, blob_bottom.channels());
              c_start = max(c_start, 0);
              Dtype scale = 1.;
              for (int i = c_start; i < c_end; ++i) {
                Dtype value = blob_bottom.data_at(n, i, h, w);
                scale += value * value * alpha / size;
              }
              *(top_data + blob_top->offset(n, c, h, w)) =
                  blob_bottom.data_at(n, c, h, w) / pow(scale, beta);
            }
          }
        }
      }
      break;
    case LRNParameter_NormRegion_WITHIN_CHANNEL:
      for (int n = 0; n < blob_bottom.num(); ++n) {
        for (int c = 0; c < blob_bottom.channels(); ++c) {
          for (int h = 0; h < blob_bottom.height(); ++h) {
            int h_start = h - (size - 1) / 2;
            int h_end = min(h_start + size, blob_bottom.height());
            h_start = max(h_start, 0);
            for (int w = 0; w < blob_bottom.width(); ++w) {
              Dtype scale = 1.;
              int w_start = w - (size - 1) / 2;
              int w_end = min(w_start + size, blob_bottom.width());
              w_start = max(w_start, 0);
              for (int nh = h_start; nh < h_end; ++nh) {
                for (int nw = w_start; nw < w_end; ++nw) {
                  Dtype value = blob_bottom.data_at(n, c, nh, nw);
                  scale += value * value * alpha / (size * size);
                }
              }
              *(top_data + blob_top->offset(n, c, h, w)) =
                  blob_bottom.data_at(n, c, h, w) / pow(scale, beta);
            }
          }
        }
      }
      break;
    default:
      LOG(FATAL) << "Unknown normalization region.";
  }
}

template <typename TypeParam>
class LRNLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  LRNLayerTest()
      : epsilon_(Dtype(1e-5)),
        blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    blob_bottom_->Reshape(2, 7, 3, 3);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~LRNLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  Dtype epsilon_;
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(LRNLayerTest, TestDtypesAndDevices);

TYPED_TEST(LRNLayerTest, TestSetupAcrossChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  LRNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 7);
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 3);
}

TYPED_TEST(LRNLayerTest, TestForwardAcrossChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  LRNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Blob<Dtype> top_reference;
  referenceLRNForward(*(this->blob_bottom_), layer_param, &top_reference);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i], top_reference.cpu_data()[i],
                this->epsilon_);
  }
}

TYPED_TEST(LRNLayerTest, TestForwardAcrossChannelsParam) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  LRNParameter* lrn_param = layer_param.mutable_lrn_param();
  lrn_param->set_alpha(0.001);
  lrn_param->set_beta(0.75);
  LRNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Blob<Dtype> top_reference;
  referenceLRNForward(*(this->blob_bottom_), layer_param, &top_reference);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i], top_reference.cpu_data()[i],
                this->epsilon_);
  }
}

TYPED_TEST(LRNLayerTest, TestForwardAcrossChannelsLargeRegion) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_lrn_param()->set_local_size(15);
  LRNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Blob<Dtype> top_reference;
  referenceLRNForward(*(this->blob_bottom_), layer_param, &top_reference);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i], top_reference.cpu_data()[i],
                this->epsilon_);
  }
}

TYPED_TEST(LRNLayerTest, TestGradientAcrossChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  LRNLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    this->blob_top_->mutable_cpu_diff()[i] = 1.;
  }
  vector<bool> propagate_down(this->blob_bottom_vec_.size(), true);
  layer.Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);
  // for (int i = 0; i < this->blob_bottom_->count(); ++i) {
  //   LOG(INFO) << "CPU diff " << this->blob_bottom_->cpu_diff()[i]
  //       << std::endl;
  // }
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                  this->blob_top_vec_);
}

TYPED_TEST(LRNLayerTest, TestGradientAcrossChannelsLargeRegion) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_lrn_param()->set_local_size(15);
  LRNLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    this->blob_top_->mutable_cpu_diff()[i] = 1.;
  }
  vector<bool> propagate_down(this->blob_bottom_vec_.size(), true);
  layer.Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);
  // for (int i = 0; i < this->blob_bottom_->count(); ++i) {
  //   LOG(INFO) << "CPU diff " << this->blob_bottom_->cpu_diff()[i]
  //       << std::endl;
  // }
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                  this->blob_top_vec_);
}

TYPED_TEST(LRNLayerTest, TestSetupWithinChannel) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_lrn_param()->set_norm_region(
      LRNParameter_NormRegion_WITHIN_CHANNEL);
  layer_param.mutable_lrn_param()->set_local_size(3);
  LRNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 7);
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 3);
}

TYPED_TEST(LRNLayerTest, TestForwardWithinChannel) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_lrn_param()->set_norm_region(
      LRNParameter_NormRegion_WITHIN_CHANNEL);
  layer_param.mutable_lrn_param()->set_local_size(3);
  LRNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Blob<Dtype> top_reference;
  referenceLRNForward(*(this->blob_bottom_), layer_param, &top_reference);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i], top_reference.cpu_data()[i],
                this->epsilon_);
  }
}

TYPED_TEST(LRNLayerTest, TestForwardWithinChannelParam) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_lrn_param()->set_norm_region(
      LRNParameter_NormRegion_WITHIN_CHANNEL);
  layer_param.mutable_lrn_param()->set_alpha(0.001);
  layer_param.mutable_lrn_param()->set_beta(0.75);
  layer_param.mutable_lrn_param()->set_local_size(3);
  LRNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Blob<Dtype> top_reference;
  referenceLRNForward(*(this->blob_bottom_), layer_param, &top_reference);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i], top_reference.cpu_data()[i],
                this->epsilon_);
  }
}

TYPED_TEST(LRNLayerTest, TestGradientWithinChannel) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_lrn_param()->set_norm_region(
      LRNParameter_NormRegion_WITHIN_CHANNEL);
  layer_param.mutable_lrn_param()->set_local_size(3);
  LRNLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    this->blob_top_->mutable_cpu_diff()[i] = 1.;
  }
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                  this->blob_top_vec_);
}

#ifdef USE_CUDNN
template <typename Dtype>
class CuDNNLRNLayerTest : public GPUDeviceTest<Dtype> {
  protected:
  CuDNNLRNLayerTest()
      : epsilon_(Dtype(1e-5)),
        blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    blob_bottom_->Reshape(2, 7, 3, 3);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~CuDNNLRNLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  Dtype epsilon_;
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(CuDNNLRNLayerTest, TestDtypes);

TYPED_TEST(CuDNNLRNLayerTest, TestForwardAcrossChannelsCuDNN) {
  // typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CuDNNLRNLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Blob<TypeParam> top_reference;
  referenceLRNForward(*(this->blob_bottom_), layer_param, &top_reference);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i], top_reference.cpu_data()[i],
                this->epsilon_);
  }
}

TYPED_TEST(CuDNNLRNLayerTest, TestForwardAcrossChannelsLargeRegionCuDNN) {
  typedef TypeParam Dtype;
  LayerParameter layer_param;
  layer_param.mutable_lrn_param()->set_local_size(15);
  CuDNNLRNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Blob<Dtype> top_reference;
  referenceLRNForward(*(this->blob_bottom_), layer_param, &top_reference);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i], top_reference.cpu_data()[i],
                this->epsilon_);
  }
}

TYPED_TEST(CuDNNLRNLayerTest, TestGradientAcrossChannelsCuDNN) {
  typedef TypeParam Dtype;
  LayerParameter layer_param;
  CuDNNLRNLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    this->blob_top_->mutable_cpu_diff()[i] = 1.;
  }
  vector<bool> propagate_down(this->blob_bottom_vec_.size(), true);
  layer.Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                  this->blob_top_vec_);
}

TYPED_TEST(CuDNNLRNLayerTest, TestForwardWithinChannel) {
  typedef TypeParam Dtype;
  LayerParameter layer_param;
  layer_param.mutable_lrn_param()->set_norm_region(
      LRNParameter_NormRegion_WITHIN_CHANNEL);
  layer_param.mutable_lrn_param()->set_local_size(3);
  CuDNNLCNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Blob<Dtype> top_reference;
  this->ReferenceLRNForward(*(this->blob_bottom_), layer_param, &top_reference);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i], top_reference.cpu_data()[i],
                this->epsilon_);
  }
}

TYPED_TEST(CuDNNLRNLayerTest, TestGradientWithinChannel) {
  typedef TypeParam Dtype;
  LayerParameter layer_param;
  layer_param.mutable_lrn_param()->set_norm_region(
      LRNParameter_NormRegion_WITHIN_CHANNEL);
  layer_param.mutable_lrn_param()->set_local_size(3);
  CuDNNLCNLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    this->blob_top_->mutable_cpu_diff()[i] = 1.;
  }
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                  this->blob_top_vec_);
}

TYPED_TEST(CuDNNLRNLayerTest, TestGradientAcrossChannelsLargeRegionCuDNN) {
  typedef TypeParam Dtype;
  LayerParameter layer_param;
  layer_param.mutable_lrn_param()->set_local_size(15);
  CuDNNLRNLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    this->blob_top_->mutable_cpu_diff()[i] = 1.;
  }
  vector<bool> propagate_down(this->blob_bottom_vec_.size(), true);
  layer.Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                  this->blob_top_vec_);
}

#endif

#ifdef USE_MLU

template <typename TypeParam>
class MLULRNLayerTest : public MLUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MLULRNLayerTest()
      : epsilon_(Dtype(0.1)),
        blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {
    Caffe::set_random_seed(1701);
    blob_bottom_->Reshape(2, 7, 3, 3);
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(0.1);
    ConstantFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MLULRNLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  Dtype epsilon_;
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MLULRNLayerTest, TestMLUDevices);

TYPED_TEST(MLULRNLayerTest, TestSetupAcrossChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MLULRNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 7);
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 3);
}


TYPED_TEST(MLULRNLayerTest, TestForwardAcrossChannelsInt8) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BlobDataType blob_dtype;  // set position
  blob_dtype = get_quantized_info(*this->blob_bottom_,
                                layer_param, "common", DT_INT8);
  layer_param.add_bottom_mlu_dtype()->CopyFrom(blob_dtype);
  MLULRNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Blob<Dtype> top_reference;
  referenceLRNForward(*(this->blob_bottom_), layer_param, &top_reference);
  float err_sum = 0, sum = 0;
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i], top_reference.cpu_data()[i],
                5e-3);
    err_sum += std::abs(this->blob_top_->cpu_data()[i] - top_reference.cpu_data()[i]);
    sum += std::abs(top_reference.cpu_data()[i]);
  }
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  ERR_RATE(err_sum/sum);
  EVENT_TIME(layer.get_event_time());
}
TYPED_TEST(MLULRNLayerTest, TestForwardAcrossChannelsParamIn8) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  LRNParameter* lrn_param = layer_param.mutable_lrn_param();
  lrn_param->set_alpha(0.001);
  lrn_param->set_beta(0.75);
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
  MLULRNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Blob<Dtype> top_reference;
  referenceLRNForward(*(this->blob_bottom_), layer_param, &top_reference);
  float err_sum = 0, sum = 0;
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i], top_reference.cpu_data()[i],
                this->epsilon_);
    err_sum += std::abs(this->blob_top_->cpu_data()[i] - top_reference.cpu_data()[i]);
    sum += std::abs(top_reference.cpu_data()[i]);
  }
  std::ostringstream stream, param;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  param << "alpha:" << lrn_param->alpha() << "\t"
        << "beta:" << lrn_param->beta();
  BOTTOM(stream);
  PARAM(param);
  ERR_RATE(err_sum / sum);
  EVENT_TIME(layer.get_event_time());
}

TYPED_TEST(MLULRNLayerTest, TestForwardAcrossChannelsLargeRegionInt8) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_lrn_param()->set_local_size(15);
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
  MLULRNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Blob<Dtype> top_reference;
  referenceLRNForward(*(this->blob_bottom_), layer_param, &top_reference);
  Dtype err_sum = 0, sum = 0;
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i], top_reference.cpu_data()[i],
                1);
    err_sum +=
        std::abs(this->blob_top_->cpu_data()[i] - top_reference.cpu_data()[i]);
    sum += std::abs(top_reference.cpu_data()[i]);
  }
  EXPECT_LE(err_sum / sum, 1);
  std::ostringstream stream, param;
  param << "local_size:" << layer_param.mutable_lrn_param()->local_size();
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  PARAM(param);
  ERR_RATE(err_sum/sum);
  EVENT_TIME(layer.get_event_time());
}

TYPED_TEST(MLULRNLayerTest, TestSetupWithinChannel) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_lrn_param()->set_norm_region(
      LRNParameter_NormRegion_WITHIN_CHANNEL);
  layer_param.mutable_lrn_param()->set_local_size(3);
  MLULRNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 7);
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 3);
}

TYPED_TEST(MLULRNLayerTest, TestForwardWithinChannelParamInt8) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_lrn_param()->set_norm_region(
      LRNParameter_NormRegion_WITHIN_CHANNEL);
  layer_param.mutable_lrn_param()->set_alpha(0.001);
  layer_param.mutable_lrn_param()->set_beta(0.75);
  layer_param.mutable_lrn_param()->set_local_size(3);
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
  MLULRNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Blob<Dtype> top_reference;
  referenceLRNForward(*(this->blob_bottom_), layer_param, &top_reference);
  float err_sum = 0, sum = 0;
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i], top_reference.cpu_data()[i],
                5e-3);
    err_sum +=
        std::abs(this->blob_top_->cpu_data()[i] - top_reference.cpu_data()[i]);
    sum += std::abs(top_reference.cpu_data()[i]);
  }
  std::ostringstream stream, param;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  param << "alpha:" << layer_param.mutable_lrn_param()->alpha() << "\t"
        << "beta:" << layer_param.mutable_lrn_param()->beta() << "\t"
        << "local_size:" << layer_param.mutable_lrn_param()->local_size();
  BOTTOM(stream);
  PARAM(param);
  ERR_RATE(err_sum / sum);
  EVENT_TIME(layer.get_event_time());
}

TYPED_TEST(MLULRNLayerTest, TestForwardWithinChannel) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_lrn_param()->set_norm_region(
      LRNParameter_NormRegion_WITHIN_CHANNEL);
  layer_param.mutable_lrn_param()->set_local_size(3);
  MLULRNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Blob<Dtype> top_reference;
  referenceLRNForward(*(this->blob_bottom_), layer_param, &top_reference);
  float err_sum = 0, sum = 0;
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i], top_reference.cpu_data()[i],
                this->epsilon_);
    err_sum += std::abs(this->blob_top_->cpu_data()[i] -
        top_reference.cpu_data()[i]);
    sum += std::abs(top_reference.cpu_data()[i]);
  }
  std::ostringstream stream, param;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  param << "local_size:" << layer_param.mutable_lrn_param()->local_size();
  BOTTOM(stream);
  PARAM(param);
  ERR_RATE(err_sum / sum);
  EVENT_TIME(layer.get_event_time());
}

template <typename TypeParam>
class MFUSLRNLayerTest : public MFUSDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MFUSLRNLayerTest()
      : epsilon_(Dtype(6e-3)),
        blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {
    Caffe::set_random_seed(1701);
    blob_bottom_->Reshape(2, 7, 3, 3);
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(0.1);
    ConstantFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MFUSLRNLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  Dtype epsilon_;
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MFUSLRNLayerTest, TestMFUSDevices);

TYPED_TEST(MFUSLRNLayerTest, TestForwardAcrossChannelsInt8) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
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
  MLULRNLayer<Dtype> layer(layer_param);
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
  Blob<Dtype> top_reference;
  referenceLRNForward(*(this->blob_bottom_), layer_param, &top_reference);
  float err_sum = 0, sum = 0;
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i], top_reference.cpu_data()[i],
                5e-3);
    err_sum += std::abs(this->blob_top_->cpu_data()[i] -
                top_reference.cpu_data()[i]);
    sum += top_reference.cpu_data()[i];
  }
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  ERR_RATE(err_sum/sum);
  EVENT_TIME(layer.get_event_time());
}

TYPED_TEST(MFUSLRNLayerTest, TestForwardAcrossChannelsParamInt8) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  LRNParameter* lrn_param = layer_param.mutable_lrn_param();
  lrn_param->set_alpha(0.001);
  lrn_param->set_beta(0.75);
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
  MLULRNLayer<Dtype> layer(layer_param);
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
  Blob<Dtype> top_reference;
  referenceLRNForward(*(this->blob_bottom_), layer_param, &top_reference);
  float err_sum = 0, sum = 0;
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i], top_reference.cpu_data()[i],
                this->epsilon_);
    err_sum += std::abs(this->blob_top_->cpu_data()[i] - top_reference.cpu_data()[i]);
    sum += std::abs(top_reference.cpu_data()[i]);
  }
  std::ostringstream stream, param;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  param << "alpha:" << lrn_param->alpha() << "\t"
    << "beta:" << lrn_param->beta();
  BOTTOM(stream);
  PARAM(param);
  ERR_RATE(err_sum / sum);
  EVENT_TIME(layer.get_event_time());
}

TYPED_TEST(MFUSLRNLayerTest, TestForwardAcrossChannelsLargeRegionInt8) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_lrn_param()->set_local_size(15);
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
  MLULRNLayer<Dtype> layer(layer_param);
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
  Blob<Dtype> top_reference;
  referenceLRNForward(*(this->blob_bottom_), layer_param, &top_reference);
  Dtype err_sum = 0, sum = 0;
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i], top_reference.cpu_data()[i],
                5e-3);
    err_sum +=
        std::abs(this->blob_top_->cpu_data()[i] - top_reference.cpu_data()[i]);
    sum += std::abs(top_reference.cpu_data()[i]);
  }
  EXPECT_LE(err_sum / sum, 2e-1);
  std::ostringstream stream, param;
  param << "local_size:" << layer_param.mutable_lrn_param()->local_size();
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  PARAM(param);
  ERR_RATE(err_sum/sum);
  EVENT_TIME(layer.get_event_time());
}

TYPED_TEST(MFUSLRNLayerTest, TestForwardWithinChannelParam) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_lrn_param()->set_norm_region(
      LRNParameter_NormRegion_WITHIN_CHANNEL);
  layer_param.mutable_lrn_param()->set_alpha(0.001);
  layer_param.mutable_lrn_param()->set_beta(0.75);
  layer_param.mutable_lrn_param()->set_local_size(3);
  MLULRNLayer<Dtype> layer(layer_param);
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

  Blob<Dtype> top_reference;
  referenceLRNForward(*(this->blob_bottom_), layer_param, &top_reference);
  float err_sum = 0, sum = 0;
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i], top_reference.cpu_data()[i],
                5e-3);
    err_sum +=
        std::abs(this->blob_top_->cpu_data()[i] - top_reference.cpu_data()[i]);
    sum += std::abs(top_reference.cpu_data()[i]);
  }
  EXPECT_LE(err_sum / sum, 5e-2);
  std::ostringstream stream, param;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  param << "alpha:" << layer_param.mutable_lrn_param()->alpha() << "\t"
    << "beta:" << layer_param.mutable_lrn_param()->beta() << "\t"
    << "local_size:" << layer_param.mutable_lrn_param()->local_size();
  BOTTOM(stream);
  PARAM(param);
  ERR_RATE(err_sum / sum);
  EVENT_TIME(layer.get_event_time());
}

TYPED_TEST(MFUSLRNLayerTest, TestForwardWithinChannel) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_lrn_param()->set_norm_region(
      LRNParameter_NormRegion_WITHIN_CHANNEL);
  layer_param.mutable_lrn_param()->set_local_size(3);
  MLULRNLayer<Dtype> layer(layer_param);
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

  Blob<Dtype> top_reference;
  referenceLRNForward(*(this->blob_bottom_), layer_param, &top_reference);
  float err_sum = 0, sum = 0;
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i], top_reference.cpu_data()[i],
                this->epsilon_);
    err_sum += std::abs(this->blob_top_->cpu_data()[i] -
        top_reference.cpu_data()[i]);
    sum += std::abs(top_reference.cpu_data()[i]);
  }
  std::ostringstream stream, param;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  param << "local_size:" << layer_param.mutable_lrn_param()->local_size();
  BOTTOM(stream);
  PARAM(param);
  ERR_RATE(err_sum / sum);
  EVENT_TIME(layer.get_event_time());
}


#endif

}  // namespace caffe
