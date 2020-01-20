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

#include <assert.h>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/mlu_roi_pooling_layer.hpp"
#include "caffe/layers/roi_pooling_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/util/math_functions.hpp"
#include "gtest/gtest.h"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

template <typename Dtype>
void caffe_roipool(const Blob<Dtype>* in, const Blob<Dtype>* in2,
                   ROIPoolingParameter* roi_pool_param,
                   Blob<Dtype>* out, int mode_option) {
  int channels_ = in->channels();
  int height_ = in->height();
  int width_ = in->width();

  CHECK_GT(roi_pool_param->pooled_h(), 0) << "pooled_h must be > 0";
  CHECK_GT(roi_pool_param->pooled_w(), 0) << "pooled_w must be > 0";
  int pooled_height_ = roi_pool_param->pooled_h();
  int pooled_width_ = roi_pool_param->pooled_w();
  float spatial_scale_ = roi_pool_param->spatial_scale();

  const Dtype* bottom_data = in->cpu_data();
  const Dtype* bottom_rois = in2->cpu_data();
  // Number of ROIs
  int num_rois = in2->num();
  int batch_size = in->num();
  int top_count = out->count();
  Dtype* top_data = out->mutable_cpu_data();
  caffe_set(top_count, Dtype(0), top_data);

  // For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
  int roi_batch_ind;
  int roi_start_w;
  int roi_start_h;
  int roi_end_w;
  int roi_end_h;
  for (int n = 0; n < num_rois; ++n) {
    if (mode_option == 0) {
      roi_batch_ind = bottom_rois[0];
      roi_start_w = round(bottom_rois[1] * spatial_scale_);
      roi_start_h = round(bottom_rois[2] * spatial_scale_);
      roi_end_w = round(bottom_rois[3] * spatial_scale_);
      roi_end_h = round(bottom_rois[4] * spatial_scale_);
    } else {
      roi_batch_ind = bottom_rois[4];
      roi_start_w = round(bottom_rois[0] * spatial_scale_);
      roi_start_h = round(bottom_rois[1] * spatial_scale_);
      roi_end_w = round(bottom_rois[2] * spatial_scale_);
      roi_end_h = round(bottom_rois[3] * spatial_scale_);
    }
    CHECK_GE(roi_batch_ind, 0);
    CHECK_LT(roi_batch_ind, batch_size);

    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    const Dtype bin_size_h =
        static_cast<Dtype>(roi_height) / static_cast<Dtype>(pooled_height_);
    const Dtype bin_size_w =
        static_cast<Dtype>(roi_width) / static_cast<Dtype>(pooled_width_);

    const Dtype* batch_data = bottom_data + in->offset(roi_batch_ind);

    for (int c = 0; c < channels_; ++c) {
      for (int ph = 0; ph < pooled_height_; ++ph) {
        for (int pw = 0; pw < pooled_width_; ++pw) {
          // Compute pooling region for this output unit:
          //  start (included) = floor(ph * roi_height / pooled_height_)
          //  end (excluded) = ceil((ph + 1) * roi_height / pooled_height_)
          int hstart =
              static_cast<int>(floor(static_cast<Dtype>(ph) * bin_size_h));
          int wstart =
              static_cast<int>(floor(static_cast<Dtype>(pw) * bin_size_w));
          int hend =
              static_cast<int>(ceil(static_cast<Dtype>(ph + 1) * bin_size_h));
          int wend =
              static_cast<int>(ceil(static_cast<Dtype>(pw + 1) * bin_size_w));

          hstart = min(max(hstart + roi_start_h, 0), height_);
          hend = min(max(hend + roi_start_h, 0), height_);
          wstart = min(max(wstart + roi_start_w, 0), width_);
          wend = min(max(wend + roi_start_w, 0), width_);

          bool is_empty = (hend <= hstart) || (wend <= wstart);

          const int pool_index = ph * pooled_width_ + pw;
          if (is_empty) {
            top_data[pool_index] = 0;
          }

          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              const int index = h * width_ + w;
              if (batch_data[index] > top_data[pool_index]) {
                top_data[pool_index] = batch_data[index];
              }
            }
          }
        }
      }
      // Increment all data pointers by one channel
      batch_data += in->offset(0, 1);
      top_data += out->offset(0, 1);
    }
    // Increment ROI data pointer
    bottom_rois += 5;
  }
}
void pad_uniform_distribution_data_roi_pooling(float* data, int count,
                                               int seed, float min, float max) {
  assert(data);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> uniform(min, max);
  for (int i = 0; i < count; i++) {
    data[i] = uniform(gen);
  }
}

template <typename Dtype>
void setInputData(Blob<Dtype>* bottom0, Blob<Dtype>* bottom1, int mode_option) {
  int num_rois = bottom1->count() / 5;
  int raw_count = bottom0->count();
  float* data_raw = static_cast<float*>(malloc(raw_count * sizeof(float)));
  for (int i = 0; i < raw_count; i++) {
    data_raw[i] = i * 0.01;
  }
  float* cx = static_cast<float*>(malloc(num_rois * sizeof(float)));
  float* cy = static_cast<float*>(malloc(num_rois * sizeof(float)));
  float* w = static_cast<float*>(malloc(num_rois * sizeof(float)));
  float* h = static_cast<float*>(malloc(num_rois * sizeof(float)));
  float* idx = static_cast<float*>(malloc(num_rois * sizeof(float)));
  memset(idx, 0, num_rois * sizeof(float));  // NOLINT
  pad_uniform_distribution_data_roi_pooling(cx, num_rois, 1000, 5, 32);
  pad_uniform_distribution_data_roi_pooling(cy, num_rois, 1000, 5, 32);
  pad_uniform_distribution_data_roi_pooling(w, num_rois, 1000, 0, 10);
  pad_uniform_distribution_data_roi_pooling(h, num_rois, 1000, 0, 10);

  float* x1 = static_cast<float*>(malloc(num_rois * sizeof(float)));
  float* y1 = static_cast<float*>(malloc(num_rois * sizeof(float)));
  float* x2 = static_cast<float*>(malloc(num_rois * sizeof(float)));
  float* y2 = static_cast<float*>(malloc(num_rois * sizeof(float)));
  for (int i = 0; i < num_rois; i++) {
    x1[i] = cx[i] - w[i] / 2;
    x1[i] = std::min(x1[i], static_cast<float>(32));
    y1[i] = cy[i] - h[i] / 2;
    y1[i] = std::min(y1[i], static_cast<float>(32));
    x2[i] = cx[i] + w[i] / 2;
    x2[i] = std::min(x2[i], static_cast<float>(32));
    y2[i] = cy[i] + h[i] / 2;
    y2[i] = std::min(y2[i], static_cast<float>(32));
  }
  free(cx);
  free(cy);
  free(w);
  free(h);

  int unit_num = 5;
  if (mode_option == 0) {
    float* concat_data =
        static_cast<float*>(malloc(num_rois * unit_num * sizeof(float)));
    for (int i = 0; i < num_rois; i++) {
      concat_data[i * unit_num] = idx[i];
      concat_data[i * unit_num + 1] = x1[i];
      concat_data[i * unit_num + 2] = y1[i];
      concat_data[i * unit_num + 3] = x2[i];
      concat_data[i * unit_num + 4] = y2[i];
    }
    for (int i = 0; i < bottom0->count(); i++) {
      bottom0->mutable_cpu_data()[i] = data_raw[i];
    }
    for (int i = 0; i < bottom1->count(); i++) {
      bottom1->mutable_cpu_data()[i] = concat_data[i];
    }
  } else {
    float* rois_conc_data =
        static_cast<float*>(malloc(num_rois * unit_num * sizeof(float)));
    for (int i = 0; i < num_rois; i++) {
      rois_conc_data[i * unit_num] = x1[i];
      rois_conc_data[i * unit_num + 1] = y1[i];
      rois_conc_data[i * unit_num + 2] = x2[i];
      rois_conc_data[i * unit_num + 3] = y2[i];
      rois_conc_data[i * unit_num + 4] = idx[i];
    }
    for (int i = 0; i < bottom0->count(); i++) {
      bottom0->mutable_cpu_data()[i] = data_raw[i];
    }
    for (int i = 0; i < bottom1->count(); i++) {
      bottom1->mutable_cpu_data()[i] = rois_conc_data[i];
    }
  }
  free(x1);
  free(y1);
  free(x2);
  free(y2);
  free(idx);
}

template <typename TypeParam>
class RoiPoolingLayerTest : public CPUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  RoiPoolingLayerTest()
      : blob_bottom_(new Blob<Dtype>(1, 38, 20, 12)),
        blob_bottom2_(new Blob<Dtype>(1, 1, 1, 5)),
        blob_top_(new Blob<Dtype>()) {}

  virtual ~RoiPoolingLayerTest() {
    delete blob_bottom_;
    delete blob_bottom2_;
    delete blob_top_;
  }

  virtual void SetUp() {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_bottom2_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom2_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
    this->ref_blob_top_.reset(new Blob<Dtype>());
    this->ref_blob_top_->ReshapeLike(*top);
    return this->ref_blob_top_.get();
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom2_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  shared_ptr<Blob<Dtype>> ref_blob_top_;
};

TYPED_TEST_CASE(RoiPoolingLayerTest, TestDtypesAndDevices);

TYPED_TEST(RoiPoolingLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ROIPoolingParameter* roi_pool_param = layer_param.mutable_roi_pooling_param();
  roi_pool_param->set_pooled_h(3);
  roi_pool_param->set_pooled_w(3);
  roi_pool_param->set_spatial_scale(3);
  shared_ptr<ROIPoolingLayer<Dtype>> layer(
      new ROIPoolingLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
}

TYPED_TEST(RoiPoolingLayerTest, Forward) {
  typedef typename TypeParam::Dtype Dtype;
  setInputData(this->blob_bottom_, this->blob_bottom2_, 0);
  LayerParameter layer_param;
  ROIPoolingParameter* roi_pool_param = layer_param.mutable_roi_pooling_param();
  roi_pool_param->set_pooled_h(3);
  roi_pool_param->set_pooled_w(3);
  roi_pool_param->set_spatial_scale(3);
  ROIPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* top_data = this->blob_top_->cpu_data();
  caffe_roipool(this->blob_bottom_, this->blob_bottom2_, roi_pool_param,
                this->MakeReferenceTop(this->blob_top_), 0);
  const Dtype* ref_top_data = this->ref_blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  for (int i = 0; i < count; i++) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 5e-3);
  }
}

#ifdef USE_MLU

template <typename TypeParam>
class MLUBangRoiPoolingLayerTest : public MLUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MLUBangRoiPoolingLayerTest()
      : blob_bottom_(new Blob<Dtype>(1, 32, 80, 90)),
        blob_bottom2_(new Blob<Dtype>(1, 1, 1, 5)),
        blob_top_(new Blob<Dtype>()) {}

  virtual ~MLUBangRoiPoolingLayerTest() {
    delete blob_bottom_;
    delete blob_bottom2_;
    delete blob_top_;
  }

  virtual void SetUp() {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_bottom2_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom2_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
    this->ref_blob_top_.reset(new Blob<Dtype>());
    this->ref_blob_top_->ReshapeLike(*top);
    return this->ref_blob_top_.get();
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom2_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  shared_ptr<Blob<Dtype>> ref_blob_top_;
};

TYPED_TEST_CASE(MLUBangRoiPoolingLayerTest, TestMLUDevices);

TYPED_TEST(MLUBangRoiPoolingLayerTest, TestSetUpBangOp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ROIPoolingParameter* roi_pool_param = layer_param.mutable_roi_pooling_param();
  roi_pool_param->set_pooled_h(3);
  roi_pool_param->set_pooled_w(3);
  roi_pool_param->set_spatial_scale(3);
  shared_ptr<MLUROIPoolingLayer<Dtype>> layer(
      new MLUROIPoolingLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  OUTPUT("bottom1", this->blob_bottom_->shape_string().c_str());
}

TYPED_TEST(MLUBangRoiPoolingLayerTest, ForwardBangOp) {
  typedef typename TypeParam::Dtype Dtype;
  setInputData(this->blob_bottom_, this->blob_bottom2_, 1);
  LayerParameter layer_param;
  ROIPoolingParameter* roi_pool_param = layer_param.mutable_roi_pooling_param();
  roi_pool_param->set_pooled_h(3);
  roi_pool_param->set_pooled_w(3);
  roi_pool_param->set_spatial_scale(3);
  MLUROIPoolingLayer<Dtype> layer(layer_param);
  Caffe::setDetectOpMode(1);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* top_data = this->blob_top_->cpu_data();
  caffe_roipool(this->blob_bottom_, this->blob_bottom2_, roi_pool_param,
                this->MakeReferenceTop(this->blob_top_), 1);
  const Dtype* ref_top_data = this->ref_blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  float err_sum = 0, sum = 0;
  for (int i = 0; i < count; i++) {
    err_sum += std::abs(top_data[i] - ref_top_data[i]);
    sum += std::abs(ref_top_data[i]);
  }
  EXPECT_LT(err_sum/sum, 0.1);
}

template <typename TypeParam>
class MFUSBangRoiPoolingLayerTest : public MFUSDeviceTest<TypeParam> {
typedef typename TypeParam :: Dtype Dtype;

  protected:
    MFUSBangRoiPoolingLayerTest()
      : blob_bottom_(new Blob<Dtype>(1, 32, 80, 90)),
        blob_bottom2_(new Blob<Dtype>(1, 1, 1, 5)),
        blob_top_(new Blob<Dtype>()) {}

    virtual ~MFUSBangRoiPoolingLayerTest() {
      delete blob_bottom_;
      delete blob_bottom2_;
      delete blob_top_;
    }

  virtual void SetUp() {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_bottom2_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom2_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
    this->ref_blob_top_.reset(new Blob<Dtype>());
    this->ref_blob_top_->ReshapeLike(*top);
    return this->ref_blob_top_.get();
  }

    Blob<Dtype>* const blob_bottom_;
    Blob<Dtype>* const blob_bottom2_;
    Blob<Dtype>* const blob_top_;
    vector<Blob<Dtype>*> blob_bottom_vec_;
    vector<Blob<Dtype>*> blob_top_vec_;
    shared_ptr<Blob<Dtype> > ref_blob_top_;
};

TYPED_TEST_CASE(MFUSBangRoiPoolingLayerTest, TestMFUSDevices);

TYPED_TEST(MFUSBangRoiPoolingLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ROIPoolingParameter* roi_pool_param = layer_param.mutable_roi_pooling_param();
  roi_pool_param->set_pooled_h(3);
  roi_pool_param->set_pooled_w(3);
  roi_pool_param->set_spatial_scale(3);
  shared_ptr<MLUROIPoolingLayer<Dtype>> layer(
      new MLUROIPoolingLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
}

TYPED_TEST(MFUSBangRoiPoolingLayerTest, ForwardBang) {
  typedef typename TypeParam::Dtype Dtype;
  setInputData(this->blob_bottom_, this->blob_bottom2_, 1);
  LayerParameter layer_param;
  ROIPoolingParameter* roi_pool_param = layer_param.mutable_roi_pooling_param();
  roi_pool_param->set_pooled_h(3);
  roi_pool_param->set_pooled_w(3);
  roi_pool_param->set_spatial_scale(3);
  MLUROIPoolingLayer<Dtype> layer(layer_param);
  Caffe::setDetectOpMode(1);
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
  const Dtype* top_data = this->blob_top_->cpu_data();
  caffe_roipool(this->blob_bottom_, this->blob_bottom2_, roi_pool_param,
                this->MakeReferenceTop(this->blob_top_), 1);
  const Dtype* ref_top_data = this->ref_blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  float err_sum = 0, sum = 0;
  for (int i = 0; i < count; i++) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 5e-3);
    err_sum += std::abs(top_data[i] - ref_top_data[i]);
    sum += std::abs(ref_top_data[i]);
  }
  EXPECT_LT(err_sum/sum, 0.1);
}
#endif
}  // namespace caffe
