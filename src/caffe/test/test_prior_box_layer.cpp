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
#include "caffe/layers/mlu_prior_box_layer.hpp"
#include "caffe/layers/mlu_relu_layer.hpp"
#include "caffe/layers/prior_box_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "gtest/gtest.h"

namespace caffe {

template <typename Dtype>
void caffe_prior(const Blob<Dtype>* in, const Blob<Dtype>* in2,
                 PriorBoxParameter* prior_box_param,
                 const vector<shared_ptr<Blob<Dtype> > >& weights,
                 Blob<Dtype>* out) {
  vector<float> min_sizes_;
  vector<float> max_sizes_;
  vector<float> aspect_ratios_;
  bool flip_;
  int num_priors_;
  bool clip_;
  vector<float> variance_;

  enum PriorType { CLASSICAL_PRIOR = 0, DENSE_PRIOR = 1 };
  PriorType p_type;
  int inner_scale;
  int img_w_;
  int img_h_;
  float step_w_;
  float step_h_;

  float offset_;
  if (prior_box_param->p_type() == prior_box_param->DENSE) {
    p_type = DENSE_PRIOR;
    inner_scale = prior_box_param->inner_scale();
  } else {
    p_type = CLASSICAL_PRIOR;
  }
  CHECK_GT(prior_box_param->min_size_size(), 0) << "must provide min_size.";
  for (int i = 0; i < prior_box_param->min_size_size(); ++i) {
    min_sizes_.push_back(prior_box_param->min_size(i));
    CHECK_GT(min_sizes_.back(), 0) << "min_size must be positive.";
  }
  aspect_ratios_.clear();
  aspect_ratios_.push_back(1.);
  flip_ = prior_box_param->flip();
  for (int i = 0; i < prior_box_param->aspect_ratio_size(); ++i) {
    float ar = prior_box_param->aspect_ratio(i);
    bool already_exist = false;
    for (int j = 0; j < aspect_ratios_.size(); ++j) {
      if (fabs(ar - aspect_ratios_[j]) < 1e-6) {
        already_exist = true;
        break;
      }
    }
    if (!already_exist) {
      aspect_ratios_.push_back(ar);
      if (flip_) {
        aspect_ratios_.push_back(1. / ar);
      }
    }
  }
  num_priors_ = aspect_ratios_.size() * min_sizes_.size();
  if (prior_box_param->max_size_size() > 0) {
    CHECK_EQ(prior_box_param->min_size_size(),
             prior_box_param->max_size_size());
    for (int i = 0; i < prior_box_param->max_size_size(); ++i) {
      max_sizes_.push_back(prior_box_param->max_size(i));
      CHECK_GT(max_sizes_[i], min_sizes_[i])
          << "max_size must be greater than min_size.";
      if (p_type == CLASSICAL_PRIOR) {
        num_priors_ += 1;
      } else if (p_type == DENSE_PRIOR) {
        num_priors_ *= inner_scale;
      }
    }
  }
  clip_ = prior_box_param->clip();
  if (prior_box_param->variance_size() > 1) {
    // Must and only provide 4 variance.
    CHECK_EQ(prior_box_param->variance_size(), 4);
    for (int i = 0; i < prior_box_param->variance_size(); ++i) {
      CHECK_GT(prior_box_param->variance(i), 0);
      variance_.push_back(prior_box_param->variance(i));
    }
  } else if (prior_box_param->variance_size() == 1) {
    CHECK_GT(prior_box_param->variance(0), 0);
    variance_.push_back(prior_box_param->variance(0));
  } else {
    // Set default to 0.1.
    variance_.push_back(0.1);
  }

  if (prior_box_param->has_img_h() || prior_box_param->has_img_w()) {
    CHECK(!prior_box_param->has_img_size())
        << "Either img_size or img_h/img_w should be specified; not both.";
    img_h_ = prior_box_param->img_h();
    CHECK_GT(img_h_, 0) << "img_h should be larger than 0.";
    img_w_ = prior_box_param->img_w();
    CHECK_GT(img_w_, 0) << "img_w should be larger than 0.";
  } else if (prior_box_param->has_img_size()) {
    const int img_size = prior_box_param->img_size();
    CHECK_GT(img_size, 0) << "img_size should be larger than 0.";
    img_h_ = img_size;
    img_w_ = img_size;
  } else {
    img_h_ = 0;
    img_w_ = 0;
  }

  if (prior_box_param->has_step_h() || prior_box_param->has_step_w()) {
    CHECK(!prior_box_param->has_step())
        << "Either step or step_h/step_w should be specified; not both.";
    step_h_ = prior_box_param->step_h();
    CHECK_GT(step_h_, 0.) << "step_h should be larger than 0.";
    step_w_ = prior_box_param->step_w();
    CHECK_GT(step_w_, 0.) << "step_w should be larger than 0.";
  } else if (prior_box_param->has_step()) {
    const float step = prior_box_param->step();
    CHECK_GT(step, 0) << "step should be larger than 0.";
    step_h_ = step;
    step_w_ = step;
  } else {
    step_h_ = 0;
    step_w_ = 0;
  }

  offset_ = prior_box_param->offset();

  const int layer_width = in->width();
  const int layer_height = in->height();
  int img_width, img_height;
  if (img_h_ == 0 || img_w_ == 0) {
    img_width = in2->width();
    img_height = in2->height();
  } else {
    img_width = img_w_;
    img_height = img_h_;
  }
  float step_w, step_h;
  if (step_w_ == 0 || step_h_ == 0) {
    step_w = static_cast<float>(img_width) / layer_width;
    step_h = static_cast<float>(img_height) / layer_height;
  } else {
    step_w = step_w_;
    step_h = step_h_;
  }
  Dtype* top_data = out->mutable_cpu_data();
  int dim = layer_height * layer_width * num_priors_ * 4;
  int idx = 0;
  for (int h = 0; h < layer_height; ++h) {
    for (int w = 0; w < layer_width; ++w) {
      float center_x = (w + offset_) * step_w;
      float center_y = (h + offset_) * step_h;
      float box_width, box_height;
      for (int s = 0; s < min_sizes_.size(); ++s) {
        int min_size_ = min_sizes_[s];
        // first prior: aspect_ratio = 1, size = min_size
        box_width = box_height = min_size_;
        // xmin
        top_data[idx++] = (center_x - box_width / 2.) / img_width;
        // ymin
        top_data[idx++] = (center_y - box_height / 2.) / img_height;
        // xmax
        top_data[idx++] = (center_x + box_width / 2.) / img_width;
        // ymax
        top_data[idx++] = (center_y + box_height / 2.) / img_height;

        if (max_sizes_.size() > 0) {
          CHECK_EQ(min_sizes_.size(), max_sizes_.size());
          int max_size_ = max_sizes_[s];
          if (p_type == CLASSICAL_PRIOR) {
            // second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
            box_width = box_height = sqrt(min_size_ * max_size_);
            // xmin
            top_data[idx++] = (center_x - box_width / 2.) / img_width;
            // ymin
            top_data[idx++] = (center_y - box_height / 2.) / img_height;
            // xmax
            top_data[idx++] = (center_x + box_width / 2.) / img_width;
            // ymax
            top_data[idx++] = (center_y + box_height / 2.) / img_height;
          } else if (p_type == DENSE_PRIOR) {
            // intermediate size
            for (int scale_i = 1; scale_i < inner_scale; scale_i++) {
              float temp_size = min_size_ *
                  pow((max_size_ * max_size_) / (min_size_ * min_size_),
                      static_cast<float>(scale_i) / (inner_scale * 2));
              // add together normal ratio euqals 1 to what has been set by users
              bool has_normal_ratio_ = false;
              for (int r = -1; r < aspect_ratios_.size(); ++r) {
                float ar = (r == -1 ? 1 : aspect_ratios_[r]);
                if (fabs(ar - 1.) < 1e-6 && has_normal_ratio_) {
                  continue;
                } else {
                  has_normal_ratio_ = true;
                }
                box_width = temp_size * sqrt(ar);
                box_height = temp_size / sqrt(ar);
                // xmin
                top_data[idx++] = (center_x - box_width / 2.) / img_width;
                // ymin
                top_data[idx++] = (center_y - box_height / 2.) / img_height;
                // xmax
                top_data[idx++] = (center_x + box_width / 2.) / img_width;
                // ymax
                top_data[idx++] = (center_y + box_height / 2.) / img_height;
              }
            }
          }
        }
        // rest of priors
        for (int r = 0; r < aspect_ratios_.size(); ++r) {
          float ar = aspect_ratios_[r];
          if (fabs(ar - 1.) < 1e-6) {
            continue;
          }
          box_width = min_size_ * sqrt(ar);
          box_height = min_size_ / sqrt(ar);
          // xmin
          top_data[idx++] = (center_x - box_width / 2.) / img_width;
          // ymin
          top_data[idx++] = (center_y - box_height / 2.) / img_height;
          // xmax
          top_data[idx++] = (center_x + box_width / 2.) / img_width;
          // ymax
          top_data[idx++] = (center_y + box_height / 2.) / img_height;
        }
      }
    }
  }
  // clip the prior's coordidate such that it is within [0, 1]
  if (clip_) {
    for (int d = 0; d < dim; ++d) {
      top_data[d] = std::min<Dtype>(std::max<Dtype>(top_data[d], 0.), 1.);
    }
  }
  // set the variance.
  top_data += out->offset(0, 1);
  if (variance_.size() == 1) {
    caffe_set<Dtype>(dim, Dtype(variance_[0]), top_data);
  } else {
    int count = 0;
    for (int h = 0; h < layer_height; ++h) {
      for (int w = 0; w < layer_width; ++w) {
        for (int i = 0; i < num_priors_; ++i) {
          for (int j = 0; j < 4; ++j) {
            top_data[count] = variance_[j];
            ++count;
          }
        }
      }
    }
  }
}

template <typename TypeParam>
class PriorBoxLayerTest : public CPUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  PriorBoxLayerTest()
      : blob_bottom_(new Blob<Dtype>(1, 25, 38, 38)),
        blob_bottom1_(new Blob<Dtype>(1, 12, 14, 14)),
        blob_top_(new Blob<Dtype>()) {}

  virtual ~PriorBoxLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  virtual void SetUp() {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_bottom1_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom1_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
    this->ref_blob_top_.reset(new Blob<Dtype>());
    this->ref_blob_top_->ReshapeLike(*top);
    return this->ref_blob_top_.get();
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom1_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  shared_ptr<Blob<Dtype> > ref_blob_top_;
};

TYPED_TEST_CASE(PriorBoxLayerTest, TestDtypesAndDevices);

TYPED_TEST(PriorBoxLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PriorBoxParameter* prior_param = layer_param.mutable_prior_box_param();
  prior_param->add_min_size(4);
  prior_param->add_max_size(7);
  PriorBoxLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->shape(0), 1);
  EXPECT_EQ(this->blob_top_->shape(1), 2);
  EXPECT_EQ(this->blob_top_->shape(2), 11552);
}

TYPED_TEST(PriorBoxLayerTest, TestSetUpWithAspect_ratio) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PriorBoxParameter* prior_param = layer_param.mutable_prior_box_param();
  prior_param->add_min_size(4);
  prior_param->add_max_size(7);
  prior_param->add_aspect_ratio(2);
  prior_param->add_variance(0.1);
  PriorBoxLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->shape(0), 1);
  EXPECT_EQ(this->blob_top_->shape(1), 2);
  EXPECT_EQ(this->blob_top_->shape(2), 23104);
}

TYPED_TEST(PriorBoxLayerTest, TestSetUpWithmutiAspect_ratio) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PriorBoxParameter* prior_param = layer_param.mutable_prior_box_param();
  prior_param->add_min_size(4);
  prior_param->add_max_size(7);
  prior_param->add_aspect_ratio(2);
  prior_param->add_variance(0.1);
  prior_param->add_variance(0.1);
  prior_param->add_variance(0.1);
  prior_param->add_variance(0.1);
  PriorBoxLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->shape(0), 1);
  EXPECT_EQ(this->blob_top_->shape(1), 2);
  EXPECT_EQ(this->blob_top_->shape(2), 23104);
}

TYPED_TEST(PriorBoxLayerTest, ForwardDenseImg_hwStep_hw) {
  typedef typename TypeParam::Dtype Dtype;

  for (int i = 0; i < this->blob_bottom_->count(); i++) {
    this->blob_bottom_->mutable_cpu_data()[i] = i / 10.0 + 0.1;
  }

  for (int i = 0; i < this->blob_bottom1_->count(); i++) {
    this->blob_bottom1_->mutable_cpu_data()[i] = i / 10.0 + 0.1;
  }

  LayerParameter layer_param;
  PriorBoxParameter* prior_param = layer_param.mutable_prior_box_param();
  prior_param->add_min_size(4);
  prior_param->add_max_size(7);
  prior_param->set_p_type(prior_param->DENSE);
  prior_param->set_img_h(4);
  prior_param->set_img_w(4);
  prior_param->set_step_h(2);
  prior_param->set_step_w(2);
  PriorBoxLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  const Dtype* top_data = this->blob_top_->cpu_data();
  caffe_prior(this->blob_bottom_, this->blob_bottom1_, prior_param,
              layer.blobs(), this->MakeReferenceTop(this->blob_top_));
  const Dtype* ref_top_data = this->ref_blob_top_->cpu_data();
  const int count = this->blob_top_->count();

  for (int i = 0; i < count; i++) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 5e-3);
  }
}

TYPED_TEST(PriorBoxLayerTest, ForwardDenseImg_sizeStep_size) {
  typedef typename TypeParam::Dtype Dtype;

  for (int i = 0; i < this->blob_bottom_->count(); i++) {
    this->blob_bottom_->mutable_cpu_data()[i] = i / 10.0 + 0.1;
  }

  for (int i = 0; i < this->blob_bottom1_->count(); i++) {
    this->blob_bottom1_->mutable_cpu_data()[i] = i / 10.0 + 0.1;
  }

  LayerParameter layer_param;
  PriorBoxParameter* prior_param = layer_param.mutable_prior_box_param();
  prior_param->set_p_type(prior_param->DENSE);
  prior_param->add_min_size(4);
  prior_param->add_max_size(7);
  prior_param->set_img_size(4);
  prior_param->set_step(4);
  PriorBoxLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  const Dtype* top_data = this->blob_top_->cpu_data();
  caffe_prior(this->blob_bottom_, this->blob_bottom1_, prior_param,
              layer.blobs(), this->MakeReferenceTop(this->blob_top_));
  const Dtype* ref_top_data = this->ref_blob_top_->cpu_data();
  const int count = this->blob_top_->count();

  for (int i = 0; i < count; i++) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 5e-3);
  }
}

// CLASSIC
TYPED_TEST(PriorBoxLayerTest, ForwardClassicImg_hwStep_hw) {
  typedef typename TypeParam::Dtype Dtype;

  for (int i = 0; i < this->blob_bottom_->count(); i++) {
    this->blob_bottom_->mutable_cpu_data()[i] = i / 10.0 + 0.1;
  }

  for (int i = 0; i < this->blob_bottom1_->count(); i++) {
    this->blob_bottom1_->mutable_cpu_data()[i] = i / 10.0 + 0.1;
  }

  LayerParameter layer_param;
  PriorBoxParameter* prior_param = layer_param.mutable_prior_box_param();
  prior_param->add_min_size(4);
  prior_param->add_max_size(7);
  prior_param->set_p_type(PriorBoxParameter_PriorType_CLASSIC);
  prior_param->set_img_h(4);
  prior_param->set_img_w(4);
  prior_param->set_step_h(2);
  prior_param->set_step_w(2);
  PriorBoxLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  const Dtype* top_data = this->blob_top_->cpu_data();
  caffe_prior(this->blob_bottom_, this->blob_bottom1_, prior_param,
              layer.blobs(), this->MakeReferenceTop(this->blob_top_));
  const Dtype* ref_top_data = this->ref_blob_top_->cpu_data();
  const int count = this->blob_top_->count();

  for (int i = 0; i < count; i++) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 5e-3);
  }
}

TYPED_TEST(PriorBoxLayerTest, ForwardClassicImg_sizeStep_size) {
  typedef typename TypeParam::Dtype Dtype;

  for (int i = 0; i < this->blob_bottom_->count(); i++) {
    this->blob_bottom_->mutable_cpu_data()[i] = i / 10.0 + 0.1;
  }

  for (int i = 0; i < this->blob_bottom1_->count(); i++) {
    this->blob_bottom1_->mutable_cpu_data()[i] = i / 10.0 + 0.1;
  }

  LayerParameter layer_param;
  PriorBoxParameter* prior_param = layer_param.mutable_prior_box_param();
  prior_param->set_p_type(prior_param->CLASSIC);
  prior_param->add_min_size(4);
  prior_param->add_max_size(7);
  prior_param->set_img_size(4);
  prior_param->set_step(4);
  PriorBoxLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  const Dtype* top_data = this->blob_top_->cpu_data();
  caffe_prior(this->blob_bottom_, this->blob_bottom1_, prior_param,
              layer.blobs(), this->MakeReferenceTop(this->blob_top_));
  const Dtype* ref_top_data = this->ref_blob_top_->cpu_data();
  const int count = this->blob_top_->count();

  for (int i = 0; i < count; i++) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 5e-3);
  }
}

#ifdef USE_MLU

template <typename TypeParam>
class MLUPriorBoxLayerTest : public MLUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MLUPriorBoxLayerTest()
      : blob_bottom_(new Blob<Dtype>(1, 25, 38, 38)),
        blob_bottom1_(new Blob<Dtype>(1, 12, 14, 14)),
        blob_top_(new Blob<Dtype>()) {}

  virtual ~MLUPriorBoxLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  virtual void SetUp() {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_bottom1_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom1_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
    this->ref_blob_top_.reset(new Blob<Dtype>());
    this->ref_blob_top_->ReshapeLike(*top);
    return this->ref_blob_top_.get();
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom1_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  shared_ptr<Blob<Dtype> > ref_blob_top_;
};

TYPED_TEST_CASE(MLUPriorBoxLayerTest, TestMLUDevices);

TYPED_TEST(MLUPriorBoxLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PriorBoxParameter* prior_param = layer_param.mutable_prior_box_param();
  prior_param->add_min_size(4);
  prior_param->add_max_size(7);
  MLUPriorBoxLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->shape(0), 1);
  EXPECT_EQ(this->blob_top_->shape(1), 2);
  EXPECT_EQ(this->blob_top_->shape(2), 11552);
}

TYPED_TEST(MLUPriorBoxLayerTest, TestSetUpWithAspect_ratio) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PriorBoxParameter* prior_param = layer_param.mutable_prior_box_param();
  prior_param->add_min_size(4);
  prior_param->add_max_size(7);
  prior_param->add_aspect_ratio(2);
  prior_param->add_variance(0.1);
  MLUPriorBoxLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->shape(0), 1);
  EXPECT_EQ(this->blob_top_->shape(1), 2);
  EXPECT_EQ(this->blob_top_->shape(2), 23104);
}
TYPED_TEST(MLUPriorBoxLayerTest, TestSetUpWithmutiAspect_ratio) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PriorBoxParameter* prior_param = layer_param.mutable_prior_box_param();
  prior_param->add_min_size(4);
  prior_param->add_max_size(7);
  prior_param->add_aspect_ratio(2);
  prior_param->add_variance(0.1);
  prior_param->add_variance(0.1);
  prior_param->add_variance(0.1);
  prior_param->add_variance(0.1);
  MLUPriorBoxLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->shape(0), 1);
  EXPECT_EQ(this->blob_top_->shape(1), 2);
  EXPECT_EQ(this->blob_top_->shape(2), 23104);
}

TYPED_TEST(MLUPriorBoxLayerTest, ForwardDenseImg_hwStep_hw) {
  typedef typename TypeParam::Dtype Dtype;

  for (int i = 0; i < this->blob_bottom_->count(); i++) {
    this->blob_bottom_->mutable_cpu_data()[i] = i / 10.0 + 0.1;
  }

  for (int i = 0; i < this->blob_bottom1_->count(); i++) {
    this->blob_bottom1_->mutable_cpu_data()[i] = i / 10.0 + 0.1;
  }

  LayerParameter layer_param;
  PriorBoxParameter* prior_param = layer_param.mutable_prior_box_param();
  prior_param->add_min_size(4);
  prior_param->add_max_size(7);
  prior_param->set_p_type(prior_param->DENSE);
  prior_param->set_img_h(4);
  prior_param->set_img_w(4);
  prior_param->set_step_h(2);
  prior_param->set_step_w(2);
  MLUPriorBoxLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  const Dtype* top_data = this->blob_top_->cpu_data();
  caffe_prior(this->blob_bottom_, this->blob_bottom1_, prior_param,
              layer.blobs(), this->MakeReferenceTop(this->blob_top_));
  const Dtype* ref_top_data = this->ref_blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  float err_sum = 0, sum = 0;
  for (int i = 0; i < count; i++) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 5e-3);
    err_sum += std::abs(top_data[i]-ref_top_data[i]);
    sum+=std::abs(ref_top_data[i]);
  }
  ERR_RATE(err_sum/sum);
  std::ostringstream stream, param;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  EVENT_TIME(layer.get_event_time());
}

TYPED_TEST(MLUPriorBoxLayerTest, ForwardDenseImg_sizeStep_size) {
  typedef typename TypeParam::Dtype Dtype;

  for (int i = 0; i < this->blob_bottom_->count(); i++) {
    this->blob_bottom_->mutable_cpu_data()[i] = i / 10.0 + 0.1;
  }

  for (int i = 0; i < this->blob_bottom1_->count(); i++) {
    this->blob_bottom1_->mutable_cpu_data()[i] = i / 10.0 + 0.1;
  }

  LayerParameter layer_param;
  PriorBoxParameter* prior_param = layer_param.mutable_prior_box_param();
  prior_param->set_p_type(prior_param->DENSE);
  prior_param->add_min_size(4);
  prior_param->add_max_size(7);
  prior_param->set_img_size(4);
  prior_param->set_step(4);
  MLUPriorBoxLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  const Dtype* top_data = this->blob_top_->cpu_data();
  caffe_prior(this->blob_bottom_, this->blob_bottom1_, prior_param,
              layer.blobs(), this->MakeReferenceTop(this->blob_top_));
  const Dtype* ref_top_data = this->ref_blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  float err_sum = 0, sum = 0;
  for (int i = 0; i < count; i++) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 5e-3);
    err_sum += std::abs(top_data[i] - ref_top_data[i]);
    sum+=std::abs(ref_top_data[i]);
  }
  ERR_RATE(err_sum/sum);
  std::ostringstream stream, param;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  EVENT_TIME(layer.get_event_time());
}

// CLASSIC
TYPED_TEST(MLUPriorBoxLayerTest, ForwardClassicImg_hwStep_hw) {
  typedef typename TypeParam::Dtype Dtype;

  for (int i = 0; i < this->blob_bottom_->count(); i++) {
    this->blob_bottom_->mutable_cpu_data()[i] = i / 10.0 + 0.1;
  }

  for (int i = 0; i < this->blob_bottom1_->count(); i++) {
    this->blob_bottom1_->mutable_cpu_data()[i] = i / 10.0 + 0.1;
  }

  LayerParameter layer_param;
  PriorBoxParameter* prior_param = layer_param.mutable_prior_box_param();
  prior_param->add_min_size(4);
  prior_param->add_max_size(7);
  prior_param->set_p_type(prior_param->CLASSIC);
  prior_param->set_img_h(4);
  prior_param->set_img_w(4);
  prior_param->set_step_h(2);
  prior_param->set_step_w(2);
  MLUPriorBoxLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  const Dtype* top_data = this->blob_top_->cpu_data();
  caffe_prior(this->blob_bottom_, this->blob_bottom1_, prior_param,
              layer.blobs(), this->MakeReferenceTop(this->blob_top_));
  const Dtype* ref_top_data = this->ref_blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  float err_sum = 0, sum = 0;
  for (int i = 0; i < count; i++) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 5e-3);
    err_sum += std::abs(top_data[i] - ref_top_data[i]);
    sum+=std::abs(ref_top_data[i]);
  }
  ERR_RATE(err_sum/sum);
  std::ostringstream stream, param;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  EVENT_TIME(layer.get_event_time());
}

TYPED_TEST(MLUPriorBoxLayerTest, ForwardClassicImg_sizeStep_size) {
  typedef typename TypeParam::Dtype Dtype;

  for (int i = 0; i < this->blob_bottom_->count(); i++) {
    this->blob_bottom_->mutable_cpu_data()[i] = i / 10.0 + 0.1;
  }

  for (int i = 0; i < this->blob_bottom1_->count(); i++) {
    this->blob_bottom1_->mutable_cpu_data()[i] = i / 10.0 + 0.1;
  }

  LayerParameter layer_param;
  PriorBoxParameter* prior_param = layer_param.mutable_prior_box_param();
  prior_param->set_p_type(prior_param->CLASSIC);
  prior_param->add_min_size(4);
  prior_param->add_max_size(7);
  prior_param->set_img_size(4);
  prior_param->set_step(4);
  MLUPriorBoxLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  const Dtype* top_data = this->blob_top_->cpu_data();
  caffe_prior(this->blob_bottom_, this->blob_bottom1_, prior_param,
              layer.blobs(), this->MakeReferenceTop(this->blob_top_));
  const Dtype* ref_top_data = this->ref_blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  float err_sum = 0, sum = 0;
  for (int i = 0; i < count; i++) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 5e-3);
    err_sum += std::abs(top_data[i] - ref_top_data[i]);
    sum+=std::abs(ref_top_data[i]);
  }
  ERR_RATE(err_sum/sum);
  std::ostringstream stream, param;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  EVENT_TIME(layer.get_event_time());
}
template <typename TypeParam>
class MFUSPriorBoxLayerTest : public MFUSDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MFUSPriorBoxLayerTest()
      : relu_bottom_(new Blob<Dtype>(1, 25, 38, 38)),
        blob_bottom_(new Blob<Dtype>(1, 25, 38, 38)),
        blob_bottom1_(new Blob<Dtype>(1, 12, 14, 14)),
        blob_top_(new Blob<Dtype>()) {}

  virtual ~MFUSPriorBoxLayerTest() {
    delete blob_bottom_;
    delete blob_bottom1_;
    delete blob_top_;
    delete relu_bottom_;
  }

  virtual void SetUp() {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->relu_bottom_);
    filler.Fill(this->blob_bottom1_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
    relu_bottom_vec_.push_back(relu_bottom_);
  }

  virtual Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
    this->ref_blob_top_.reset(new Blob<Dtype>());
    this->ref_blob_top_->ReshapeLike(*top);
    return this->ref_blob_top_.get();
  }

  Blob<Dtype>* const relu_bottom_;
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom1_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> relu_bottom_vec_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  shared_ptr<Blob<Dtype> > ref_blob_top_;
};

TYPED_TEST_CASE(MFUSPriorBoxLayerTest, TestMFUSDevices);

TYPED_TEST(MFUSPriorBoxLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom1_);
  LayerParameter layer_param;
  PriorBoxParameter* prior_param = layer_param.mutable_prior_box_param();
  prior_param->add_min_size(4);
  prior_param->add_max_size(7);
  MLUPriorBoxLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->shape(0), 1);
  EXPECT_EQ(this->blob_top_->shape(1), 2);
  EXPECT_EQ(this->blob_top_->shape(2), 11552);
}

TYPED_TEST(MFUSPriorBoxLayerTest, TestSetUpWithAspect_ratio) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom1_);
  LayerParameter layer_param;
  PriorBoxParameter* prior_param = layer_param.mutable_prior_box_param();
  prior_param->add_min_size(4);
  prior_param->add_max_size(7);
  prior_param->add_aspect_ratio(2);
  prior_param->add_variance(0.1);
  MLUPriorBoxLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->shape(0), 1);
  EXPECT_EQ(this->blob_top_->shape(1), 2);
  EXPECT_EQ(this->blob_top_->shape(2), 23104);
}
TYPED_TEST(MFUSPriorBoxLayerTest, TestSetUpWithmutiAspect_ratio) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom1_);
  LayerParameter layer_param;
  PriorBoxParameter* prior_param = layer_param.mutable_prior_box_param();
  prior_param->add_min_size(4);
  prior_param->add_max_size(7);
  prior_param->add_aspect_ratio(2);
  prior_param->add_variance(0.1);
  prior_param->add_variance(0.1);
  prior_param->add_variance(0.1);
  prior_param->add_variance(0.1);
  MLUPriorBoxLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->shape(0), 1);
  EXPECT_EQ(this->blob_top_->shape(1), 2);
  EXPECT_EQ(this->blob_top_->shape(2), 23104);
}

TYPED_TEST(MFUSPriorBoxLayerTest, ForwardDenseImg_hwStep_hw) {
  typedef typename TypeParam::Dtype Dtype;

  for (int i = 0; i < this->blob_bottom_->count(); i++) {
    this->blob_bottom_->mutable_cpu_data()[i] = i / 10.0 + 0.1;
  }

  for (int i = 0; i < this->blob_bottom1_->count(); i++) {
    this->blob_bottom1_->mutable_cpu_data()[i] = i / 10.0 + 0.1;
  }

  LayerParameter layer_param;
  PriorBoxParameter* prior_param = layer_param.mutable_prior_box_param();
  prior_param->add_min_size(4);
  prior_param->add_max_size(7);
  prior_param->set_p_type(prior_param->DENSE);
  prior_param->set_img_h(4);
  prior_param->set_img_w(4);
  prior_param->set_step_h(2);
  prior_param->set_step_w(2);
  MLUReLULayer<Dtype> relu(layer_param);
  relu.SetUp(this->relu_bottom_vec_, this->blob_bottom_vec_);
  relu.Reshape_dispatch(this->relu_bottom_vec_, this->blob_bottom_vec_);
  ASSERT_TRUE(relu.mfus_supported());
  this->blob_bottom_vec_.push_back(this->blob_bottom1_);
  MLUPriorBoxLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_TRUE(layer.mfus_supported());

  MFusion<Dtype> fuser;
  fuser.reset();
  fuser.addInputs(this->relu_bottom_vec_);
  fuser.addOutputs(this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.fuse(&fuser);
  relu.fuse(&fuser);
  fuser.compile();
  fuser.forward();

  const Dtype* top_data = this->blob_top_->cpu_data();
  caffe_prior(this->blob_bottom_, this->blob_bottom1_, prior_param,
              layer.blobs(), this->MakeReferenceTop(this->blob_top_));
  const Dtype* ref_top_data = this->ref_blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  float err_sum = 0, sum = 0;
  for (int i = 0; i < count; i++) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 5e-3);
    err_sum += std::abs(top_data[i] - ref_top_data[i]);
    sum+=std::abs(ref_top_data[i]);
  }
  ERR_RATE(err_sum/sum);
  std::ostringstream stream, param;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  EVENT_TIME(fuser.get_event_time());
}

TYPED_TEST(MFUSPriorBoxLayerTest, ForwardDenseImg_sizeStep_size) {
  typedef typename TypeParam::Dtype Dtype;
  for (int i = 0; i < this->blob_bottom_->count(); i++) {
    this->blob_bottom_->mutable_cpu_data()[i] = i / 10.0 + 0.1;
  }

  for (int i = 0; i < this->blob_bottom1_->count(); i++) {
    this->blob_bottom1_->mutable_cpu_data()[i] = i / 10.0 + 0.1;
  }

  LayerParameter layer_param;
  PriorBoxParameter* prior_param = layer_param.mutable_prior_box_param();
  prior_param->set_p_type(prior_param->DENSE);
  prior_param->add_min_size(4);
  prior_param->add_max_size(7);
  prior_param->set_img_size(4);
  prior_param->set_step(4);
  MLUReLULayer<Dtype> relu(layer_param);
  relu.SetUp(this->relu_bottom_vec_, this->blob_bottom_vec_);
  relu.Reshape_dispatch(this->relu_bottom_vec_, this->blob_bottom_vec_);
  ASSERT_TRUE(relu.mfus_supported());
  this->blob_bottom_vec_.push_back(this->blob_bottom1_);
  MLUPriorBoxLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_TRUE(layer.mfus_supported());

  MFusion<Dtype> fuser;
  fuser.reset();
  fuser.addInputs(this->relu_bottom_vec_);
  fuser.addOutputs(this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);

  layer.fuse(&fuser);
  relu.fuse(&fuser);
  fuser.compile();
  fuser.forward();

  const Dtype* top_data = this->blob_top_->cpu_data();
  caffe_prior(this->blob_bottom_, this->blob_bottom1_, prior_param,
              layer.blobs(), this->MakeReferenceTop(this->blob_top_));
  const Dtype* ref_top_data = this->ref_blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  float err_sum = 0, sum = 0;
  for (int i = 0; i < count; i++) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 5e-3);
    err_sum += std::abs(top_data[i] - ref_top_data[i]);
    sum+=std::abs(ref_top_data[i]);
  }
  ERR_RATE(err_sum/sum);
  std::ostringstream stream, param;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  EVENT_TIME(fuser.get_event_time());
}

#endif

}  // namespace caffe
