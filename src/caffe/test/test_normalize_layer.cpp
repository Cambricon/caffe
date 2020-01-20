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
#include <memory>
#include <vector>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/mlu_normalize_layer.hpp"
#include "caffe/layers/normalize_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include "gtest/gtest.h"

namespace caffe {

template <typename TypeParam>
class NormalizeLayerTest : public CPUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
    NormalizeLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype>()),
        blob_ca_top_(new Blob<Dtype>()) {}
  void SetUp() {
    Caffe::set_random_seed(1701);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.clear();
    blob_top_vec_.clear();
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~NormalizeLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
    delete blob_ca_top_;
  }
  void TestSetUp(bool channel_shared, bool across_spatial,
                 Dtype eps) {
    this->SetUp();
    LayerParameter layer_param;
    NormalizeParameter* normalize_param =
      layer_param.mutable_norm_param();
    FillerParameter* normalize_filler_param = new FillerParameter();
    normalize_filler_param->set_type("constant");
    normalize_filler_param->set_value(1.0);
    normalize_param->set_allocated_scale_filler(normalize_filler_param);
    normalize_param->set_channel_shared(channel_shared);
    normalize_param->set_across_spatial(across_spatial);
    normalize_param->set_eps(eps);
    NormalizeLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    // top shape equal bottom shape
    EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
    EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
    EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height());
    EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width());
    EXPECT_GE(this->blob_bottom_->num_axes(), 2);
  }
  void TestForward(bool channel_shared, bool across_spatial,
                   Dtype eps) {
    this->SetUp();
    LayerParameter layer_param;
    NormalizeParameter* normalize_param =
      layer_param.mutable_norm_param();
    FillerParameter* normalize_filler_param = new FillerParameter();
    normalize_filler_param->set_type("constant");
    normalize_filler_param->set_value(1.0);
    normalize_param->set_allocated_scale_filler(normalize_filler_param);
    normalize_param->set_channel_shared(channel_shared);
    normalize_param->set_across_spatial(across_spatial);
    NormalizeLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    // prepare needed values
    this->blob_ca_top_->ReshapeLike(*blob_bottom_);
    this->buffer_.Reshape(1, blob_bottom_->channels(), blob_bottom_->height(),
                          blob_bottom_->width());
    if (across_spatial) {
      norm_.Reshape(blob_bottom_->num(), 1, 1, 1);
    } else {
      norm_.Reshape(blob_bottom_->num(), 1,
          blob_bottom_->height(), blob_bottom_->width());
    }
    int channels = blob_bottom_->channels();
    int spatial_dim = blob_bottom_->height() * blob_bottom_->width();
    this->sum_channel_multiplier_.Reshape(1, channels, 1, 1);
    caffe_set(channels, Dtype(1),
              sum_channel_multiplier_.mutable_cpu_data());
    this->sum_spatial_multiplier_.Reshape(
        1, 1, blob_bottom_->height(), blob_bottom_->width());
    caffe_set(spatial_dim, Dtype(1),
              sum_spatial_multiplier_.mutable_cpu_data());
    if (channel_shared) {
      this->blob_.reset(new Blob<Dtype>(vector<int>(0)));
    } else {
      this->blob_.reset(new Blob<Dtype>(vector<int>(1, channels)));
    }
    shared_ptr<Filler<Dtype> > scale_filler;
    scale_filler.reset(GetFiller<Dtype>(*normalize_filler_param));
    scale_filler->Fill(this->blob_.get());
    if (channel_shared) {
      EXPECT_EQ(this->blob_->count(), 1);
    } else {
      EXPECT_EQ(this->blob_->count(), channels);
    }
    // Now, check values
    const Dtype* bottom_data = blob_bottom_vec_[0]->cpu_data();
    const Dtype* top_data = blob_top_vec_[0]->cpu_data();
    Dtype* temp_top_data = blob_ca_top_->mutable_cpu_data();
    const Dtype* scale = blob_->cpu_data();
    Dtype* buffer_data = buffer_.mutable_cpu_data();
    Dtype* norm_data = norm_.mutable_cpu_data();
    // add eps to avoid overflow
    caffe_set<Dtype>(norm_.count(), Dtype(eps), norm_data);
    const Dtype* sum_channel_multiplier = sum_channel_multiplier_.cpu_data();
    const Dtype* sum_spatial_multiplier = sum_spatial_multiplier_.cpu_data();
    int num = blob_bottom_->num();
    int dim = blob_bottom_->count() / num;
    for (int n = 0; n < num; ++n) {
      // sqr to make bottom_data no-negative
      caffe_sqr<Dtype>(dim, bottom_data, buffer_data);
      // across_spatial means normalize in batch_size
      // else will normalize in channels
      if (across_spatial) {
        // caffe_cpu_asum to get the sum of buffer_data,
        // pow 0.5 means square root.
        // norm_data[n] is the denominator,
        // add a small eps to avoid denominator becomes 0
        norm_data[n] = pow(caffe_cpu_asum<Dtype>(dim, buffer_data)+eps,
                           Dtype(0.5));
        // make bottom_data scale norm_data[n], complete normalization
        caffe_cpu_scale<Dtype>(dim, Dtype(1.0 / norm_data[n]), bottom_data,
                               temp_top_data);
      } else {
        // math calculate from matrix to vector
        caffe_cpu_gemv<Dtype>(CblasTrans, channels, spatial_dim, Dtype(1),
                              buffer_data, sum_channel_multiplier, Dtype(1),
                              norm_data);
        // compute norm with spatial_dim
        caffe_powx<Dtype>(spatial_dim, norm_data, Dtype(0.5), norm_data);
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, spatial_dim,
                          1, Dtype(1), sum_channel_multiplier, norm_data,
                          Dtype(0), buffer_data);
        caffe_div<Dtype>(dim, bottom_data, buffer_data, temp_top_data);
        norm_data += spatial_dim;
      }
      // scale the output
      if (channel_shared) {
        caffe_scal<Dtype>(dim, scale[0], temp_top_data);
      } else {
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, spatial_dim,
                              1, Dtype(1), scale, sum_spatial_multiplier,
                              Dtype(0),
                              buffer_data);
        caffe_mul<Dtype>(dim, temp_top_data, buffer_data, temp_top_data);
      }
      bottom_data += dim;
      temp_top_data += dim;
    }
    temp_top_data = blob_ca_top_->mutable_cpu_data();
    for (int i = 0; i < blob_bottom_->count(); ++i) {
       EXPECT_NEAR(top_data[i], temp_top_data[i], 1e-5);
    }
  }
  shared_ptr<Blob<Dtype> > blob_;
  Blob<Dtype> norm_;
  Blob<Dtype> sum_channel_multiplier_, sum_spatial_multiplier_;
  Blob<Dtype> buffer_;
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_ca_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(NormalizeLayerTest, TestDtypesAndDevices);

TYPED_TEST(NormalizeLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  bool channel_shared = false;
  bool across_spatial = false;
  Dtype eps = 1e-10;
  this->TestSetUp(channel_shared, across_spatial, eps);
}

TYPED_TEST(NormalizeLayerTest, TestChannelShared) {
  typedef typename TypeParam::Dtype Dtype;
  bool channel_shared = true;
  bool across_spatial = false;
  Dtype eps = 1e-10;
  this->TestForward(channel_shared, across_spatial, eps);
}
TYPED_TEST(NormalizeLayerTest, TestAcrossSpatial) {
  typedef typename TypeParam::Dtype Dtype;
  bool channel_shared = false;
  bool across_spatial = true;
  Dtype eps = 1e-10;
  this->TestForward(channel_shared, across_spatial, eps);
}

TYPED_TEST(NormalizeLayerTest, TestEps) {
  typedef typename TypeParam::Dtype Dtype;
  bool channel_shared = false;
  bool across_spatial = false;
  Dtype eps = 5e-11;
  this->TestForward(channel_shared, across_spatial, eps);
}

#ifdef USE_MLU

template <typename TypeParam>
class MLUNormalizeLayerTest : public MLUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
    MLUNormalizeLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype>()),
        blob_ca_top_(new Blob<Dtype>()) {}
  void SetUp() {
    Caffe::set_random_seed(1701);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.clear();
    blob_top_vec_.clear();
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MLUNormalizeLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
    delete blob_ca_top_;
  }
  void TestSetUp(bool channel_shared, bool across_spatial,
                 Dtype eps) {
    this->SetUp();
    LayerParameter layer_param;
    NormalizeParameter* normalize_param =
      layer_param.mutable_norm_param();
    FillerParameter* normalize_filler_param = new FillerParameter();
    normalize_filler_param->set_type("constant");
    normalize_filler_param->set_value(1.0);
    normalize_param->set_allocated_scale_filler(normalize_filler_param);
    normalize_param->set_channel_shared(channel_shared);
    normalize_param->set_across_spatial(across_spatial);
    normalize_param->set_eps(eps);
    MLUNormalizeLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    // top shape equal bottom shape
    EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
    EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
    EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height());
    EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width());
    EXPECT_GE(this->blob_bottom_->num_axes(), 2);
  }
  void TestForward(bool channel_shared, bool across_spatial,
                   Dtype eps, bool int8) {
    this->SetUp();
    LayerParameter layer_param;
    NormalizeParameter* normalize_param =
      layer_param.mutable_norm_param();
    FillerParameter* normalize_filler_param = new FillerParameter();
    normalize_filler_param->set_type("constant");
    normalize_filler_param->set_value(1.0);
    normalize_param->set_allocated_scale_filler(normalize_filler_param);
    normalize_param->set_channel_shared(channel_shared);
    normalize_param->set_across_spatial(across_spatial);
    if (int8) {  // int8
      BlobDataType blob_dtype;
      blob_dtype = get_quantized_info(*this->blob_bottom_,
                                    layer_param, "common", DT_INT8);
      layer_param.add_bottom_mlu_dtype()->CopyFrom(blob_dtype);
      BlobDataType blobs_dtype;
      blobs_dtype.set_type(DT_INT8);
      blobs_dtype.add_position(-3);
      blobs_dtype.add_scale(1.5875);
      layer_param.add_bottom_mlu_dtype()->CopyFrom(blobs_dtype);
      layer_param.add_blobs_dtype()->CopyFrom(blobs_dtype);
    }
    MLUNormalizeLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    // prepare needed values
    this->blob_ca_top_->ReshapeLike(*blob_bottom_);
    this->buffer_.Reshape(1, blob_bottom_->channels(), blob_bottom_->height(),
                          blob_bottom_->width());
    if (across_spatial) {
      norm_.Reshape(blob_bottom_->num(), 1, 1, 1);
    } else {
      norm_.Reshape(blob_bottom_->num(), 1,
          blob_bottom_->height(), blob_bottom_->width());
    }
    int channels = blob_bottom_->channels();
    int spatial_dim = blob_bottom_->width() * blob_bottom_->height();
    this->sum_channel_multiplier_.Reshape(1, channels, 1, 1);
    caffe_set(channels, Dtype(1),
              sum_channel_multiplier_.mutable_cpu_data());
    this->sum_spatial_multiplier_.Reshape(
        1, 1, blob_bottom_->height(), blob_bottom_->width());
    caffe_set(spatial_dim, Dtype(1),
              sum_spatial_multiplier_.mutable_cpu_data());
    if (channel_shared) {
      this->blob_.reset(new Blob<Dtype>(vector<int>(0)));
    } else {
      this->blob_.reset(new Blob<Dtype>(vector<int>(1, channels)));
    }
    shared_ptr<Filler<Dtype> > scale_filler;
    scale_filler.reset(GetFiller<Dtype>(*normalize_filler_param));
    scale_filler->Fill(this->blob_.get());
    if (channel_shared) {
      EXPECT_EQ(this->blob_->count(), 1);
    } else {
      EXPECT_EQ(this->blob_->count(), channels);
    }
    // Now, check values
    const Dtype* bottom_data = blob_bottom_vec_[0]->cpu_data();
    Dtype* top_data = blob_top_->mutable_cpu_data();
    Dtype* temp_top_data = blob_ca_top_->mutable_cpu_data();
    const Dtype* scale = this->blob_->cpu_data();
    Dtype* buffer_data = buffer_.mutable_cpu_data();
    Dtype* norm_data = norm_.mutable_cpu_data();
    // add eps to avoid overflow
    caffe_set<Dtype>(norm_.count(), Dtype(eps), norm_data);
    const Dtype* sum_channel_multiplier = sum_channel_multiplier_.cpu_data();
    const Dtype* sum_spatial_multiplier = sum_spatial_multiplier_.cpu_data();
    int num = blob_bottom_->num();
    int dim = blob_bottom_->count() / num;
    for (int n = 0; n < num; ++n) {
      caffe_sqr<Dtype>(dim, bottom_data, buffer_data);
      // across_spatial means normalize in batch_size
      // else will normalize in channels
      if (across_spatial) {
        norm_data[n] = pow(caffe_cpu_asum<Dtype>(dim, buffer_data) + eps,
                           Dtype(0.5));
        caffe_cpu_scale<Dtype>(dim, Dtype(1.0 / norm_data[n]), bottom_data,
                               temp_top_data);
      } else {
        caffe_cpu_gemv<Dtype>(CblasTrans, channels, spatial_dim, Dtype(1),
                              buffer_data, sum_channel_multiplier, Dtype(1),
                              norm_data);
        caffe_powx<Dtype>(spatial_dim, norm_data, Dtype(0.5), norm_data);
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, spatial_dim,
                          1, Dtype(1), sum_channel_multiplier, norm_data,
                          Dtype(0), buffer_data);
        caffe_div<Dtype>(dim, bottom_data, buffer_data, temp_top_data);
        norm_data += spatial_dim;
      }
      // scale the output
      if (channel_shared) {
        caffe_scal<Dtype>(dim, scale[0], temp_top_data);
      } else {
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, spatial_dim,
                              1, Dtype(1), scale, sum_spatial_multiplier,
                              Dtype(0), buffer_data);
        caffe_mul<Dtype>(dim, temp_top_data, buffer_data, temp_top_data);
      }
      bottom_data += dim;
      temp_top_data += dim;
    }
    temp_top_data = blob_ca_top_->mutable_cpu_data();
    float err_sum = 0, sum = 0;
    for (int i = 0; i < blob_bottom_->count(); ++i) {
      Dtype err = (top_data[i] - temp_top_data[i]) / top_data[i];
      EXPECT_LT(err,  0.16);
      err_sum += std::abs(top_data[i] - temp_top_data[i]);
      sum += std::abs(temp_top_data[i]);
    }
    ERR_RATE(err_sum/sum);
    EVENT_TIME(layer.get_event_time());
    std::ostringstream stream, param;
    stream << "bottom1:" << blob_bottom_->shape_string().c_str();
    param << "channel_shared:"<< channel_shared << "\t"
          << "across_spatial:" << across_spatial << "\t"
          << "eps:" << eps;
    BOTTOM(stream);
    PARAM(param);
  }
  shared_ptr<Blob<Dtype> > blob_;
  Blob<Dtype> norm_;
  Blob<Dtype> sum_channel_multiplier_, sum_spatial_multiplier_;
  Blob<Dtype> buffer_;
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_ca_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MLUNormalizeLayerTest, TestMLUDevices);

TYPED_TEST(MLUNormalizeLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  bool channel_shared = false;
  bool across_spatial = false;
  Dtype eps = 1e-10;
  this->TestSetUp(channel_shared, across_spatial, eps);
}

TYPED_TEST(MLUNormalizeLayerTest, TestChannelSharedInt8) {
  typedef typename TypeParam::Dtype Dtype;
  bool channel_shared = true;
  bool across_spatial = false;
  Dtype eps = 1e-10;
  this->TestForward(channel_shared, across_spatial, eps, true);
}

TYPED_TEST(MLUNormalizeLayerTest, TestAcrossSpatialInt8) {
  typedef typename TypeParam::Dtype Dtype;
  bool channel_shared = false;
  bool across_spatial = true;
  Dtype eps = 1e-10;
  this->TestForward(channel_shared, across_spatial, eps, true);
}

TYPED_TEST(MLUNormalizeLayerTest, TestEpsInt8) {
  typedef typename TypeParam::Dtype Dtype;
  bool channel_shared = false;
  bool across_spatial = false;
  Dtype eps = 1e-11;
  this->TestForward(channel_shared, across_spatial, eps, true);
}

template <typename TypeParam>
class MFUSNormalizeLayerTest : public MFUSDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
    MFUSNormalizeLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype>()),
        blob_ca_top_(new Blob<Dtype>()) {}
  void SetUp() {
    Caffe::set_random_seed(1701);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.clear();
    blob_top_vec_.clear();
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MFUSNormalizeLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
    delete blob_ca_top_;
  }
  void TestForward(bool channel_shared, bool across_spatial,
                   Dtype eps, bool int8) {
    this->SetUp();
    LayerParameter layer_param;
    NormalizeParameter* normalize_param =
      layer_param.mutable_norm_param();
    FillerParameter* normalize_filler_param = new FillerParameter();
    normalize_filler_param->set_type("constant");
    normalize_filler_param->set_value(1.0);
    normalize_param->set_allocated_scale_filler(normalize_filler_param);
    normalize_param->set_channel_shared(channel_shared);
    normalize_param->set_across_spatial(across_spatial);
    if (int8) {  // int8
      BlobDataType blob_dtype;
      blob_dtype = get_quantized_info(*this->blob_bottom_,
                                    layer_param, "common", DT_INT8);
      layer_param.add_bottom_mlu_dtype()->CopyFrom(blob_dtype);
      BlobDataType blobs_dtype;
      blobs_dtype.set_type(DT_INT8);
      blobs_dtype.add_position(-3);
      blobs_dtype.add_scale(1.5875);
      layer_param.add_bottom_mlu_dtype()->CopyFrom(blobs_dtype);
      layer_param.add_blobs_dtype()->CopyFrom(blobs_dtype);
    }
    MLUNormalizeLayer<Dtype> layer(layer_param);
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
    // prepare needed values
    this->blob_ca_top_->ReshapeLike(*blob_bottom_);
    this->buffer_.Reshape(1, blob_bottom_->channels(), blob_bottom_->height(),
                          blob_bottom_->width());
    if (across_spatial) {
      norm_.Reshape(blob_bottom_->num(), 1, 1, 1);
    } else {
      norm_.Reshape(blob_bottom_->num(), 1,
          blob_bottom_->height(), blob_bottom_->width());
    }
    int channels = blob_bottom_->channels();
    int spatial_dim = blob_bottom_->width() * blob_bottom_->height();
    this->sum_channel_multiplier_.Reshape(1, channels, 1, 1);
    caffe_set(channels, Dtype(1),
              sum_channel_multiplier_.mutable_cpu_data());
    this->sum_spatial_multiplier_.Reshape(
        1, 1, blob_bottom_->height(), blob_bottom_->width());
    caffe_set(spatial_dim, Dtype(1),
              sum_spatial_multiplier_.mutable_cpu_data());
    if (channel_shared) {
      this->blob_.reset(new Blob<Dtype>(vector<int>(0)));
    } else {
      this->blob_.reset(new Blob<Dtype>(vector<int>(1, channels)));
    }
    shared_ptr<Filler<Dtype> > scale_filler;
    scale_filler.reset(GetFiller<Dtype>(*normalize_filler_param));
    scale_filler->Fill(this->blob_.get());
    if (channel_shared) {
      EXPECT_EQ(this->blob_->count(), 1);
    } else {
      EXPECT_EQ(this->blob_->count(), channels);
    }
    // Now, check values
    const Dtype* bottom_data = blob_bottom_vec_[0]->cpu_data();
    Dtype* top_data = blob_top_->mutable_cpu_data();
    Dtype* temp_top_data = blob_ca_top_->mutable_cpu_data();
    const Dtype* scale = this->blob_->cpu_data();
    Dtype* buffer_data = buffer_.mutable_cpu_data();
    Dtype* norm_data = norm_.mutable_cpu_data();
    // add eps to avoid overflow
    caffe_set<Dtype>(norm_.count(), Dtype(eps), norm_data);
    const Dtype* sum_channel_multiplier = sum_channel_multiplier_.cpu_data();
    const Dtype* sum_spatial_multiplier = sum_spatial_multiplier_.cpu_data();
    int num = blob_bottom_->num();
    int dim = blob_bottom_->count() / num;
    for (int n = 0; n < num; ++n) {
      caffe_sqr<Dtype>(dim, bottom_data, buffer_data);
      // across_spatial means normalize in batch_size
      // else will normalize in channels
      if (across_spatial) {
        norm_data[n] = pow(caffe_cpu_asum<Dtype>(dim, buffer_data) + eps,
                           Dtype(0.5));
        caffe_cpu_scale<Dtype>(dim, Dtype(1.0 / norm_data[n]), bottom_data,
                               temp_top_data);
      } else {
        caffe_cpu_gemv<Dtype>(CblasTrans, channels, spatial_dim, Dtype(1),
                              buffer_data, sum_channel_multiplier, Dtype(1),
                              norm_data);
        caffe_powx<Dtype>(spatial_dim, norm_data, Dtype(0.5), norm_data);
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, spatial_dim,
                          1, Dtype(1), sum_channel_multiplier, norm_data,
                          Dtype(0), buffer_data);
        caffe_div<Dtype>(dim, bottom_data, buffer_data, temp_top_data);
        norm_data += spatial_dim;
      }
      // scale the output
      if (channel_shared) {
        caffe_scal<Dtype>(dim, scale[0], temp_top_data);
      } else {
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, spatial_dim,
                              1, Dtype(1), scale, sum_spatial_multiplier,
                              Dtype(0), buffer_data);
        caffe_mul<Dtype>(dim, temp_top_data, buffer_data, temp_top_data);
      }
      bottom_data += dim;
      temp_top_data += dim;
    }
    temp_top_data = blob_ca_top_->mutable_cpu_data();
    float err_sum = 0, sum = 0;
    for (int i = 0; i < blob_bottom_->count(); ++i) {
      Dtype err = (top_data[i] - temp_top_data[i]) / top_data[i];
      EXPECT_LT(err,  0.16);
      err_sum += std::abs(top_data[i] - temp_top_data[i]);
      sum += std::abs(temp_top_data[i]);
    }
    ERR_RATE(err_sum/sum);
    EVENT_TIME(fuser.get_event_time());
    std::ostringstream stream, param;
    stream << "bottom1:" << blob_bottom_->shape_string().c_str();
    param << "channel_shared:"<< channel_shared << "\t"
      << "across_spatial:" << across_spatial << "\t"
      << "eps:" << eps;
    PARAM(param);
    BOTTOM(stream);
  }
  shared_ptr<Blob<Dtype> > blob_;
  Blob<Dtype> norm_;
  Blob<Dtype> sum_channel_multiplier_, sum_spatial_multiplier_;
  Blob<Dtype> buffer_;
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_ca_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MFUSNormalizeLayerTest, TestMFUSDevices);

TYPED_TEST(MFUSNormalizeLayerTest, TestChannelSharedInt8) {
  typedef typename TypeParam::Dtype Dtype;
  bool channel_shared = true;
  bool across_spatial = false;
  Dtype eps = 1e-10;
  this->TestForward(channel_shared, across_spatial, eps, true);
}

TYPED_TEST(MFUSNormalizeLayerTest, TestAcrossSpatialInt8) {
  typedef typename TypeParam::Dtype Dtype;
  bool channel_shared = false;
  bool across_spatial = true;
  Dtype eps = 1e-10;
  this->TestForward(channel_shared, across_spatial, eps, true);
}

TYPED_TEST(MFUSNormalizeLayerTest, TestEpsInt8) {
  typedef typename TypeParam::Dtype Dtype;
  bool channel_shared = false;
  bool across_spatial = false;
  Dtype eps = 1e-11;
  this->TestForward(channel_shared, across_spatial, eps, true);
}

#endif

}  // namespace caffe
