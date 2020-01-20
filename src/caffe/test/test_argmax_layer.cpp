/*
All modification made by Cambricon Corporation: © 2018--2019 Cambricon Corporation
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
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/argmax_layer.hpp"
#include "caffe/layers/mlu_argmax_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class ArgMaxLayerTest : public CPUDeviceTest<Dtype> {
  protected:
  ArgMaxLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 4, 5, 6)),
        blob_top_(new Blob<Dtype>()),
        top_k_(3) {
    Caffe::set_random_seed(1701);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~ArgMaxLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  size_t top_k_;
};

TYPED_TEST_CASE(ArgMaxLayerTest, TestDtypes);

TYPED_TEST(ArgMaxLayerTest, TestSetup) {
  LayerParameter layer_param;
  ArgMaxLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), 1);
}

TYPED_TEST(ArgMaxLayerTest, TestSetupMaxVal) {
  LayerParameter layer_param;
  ArgMaxParameter* argmax_param = layer_param.mutable_argmax_param();
  argmax_param->set_out_max_val(true);
  ArgMaxLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), 2);
}

TYPED_TEST(ArgMaxLayerTest, TestSetupAxis) {
  LayerParameter layer_param;
  ArgMaxParameter* argmax_param = layer_param.mutable_argmax_param();
  argmax_param->set_axis(0);
  ArgMaxLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->shape(0), argmax_param->top_k());
  EXPECT_EQ(this->blob_top_->shape(1), this->blob_bottom_->shape(1));
  EXPECT_EQ(this->blob_top_->shape(2), this->blob_bottom_->shape(2));
  EXPECT_EQ(this->blob_top_->shape(3), this->blob_bottom_->shape(3));
}

TYPED_TEST(ArgMaxLayerTest, TestSetupAxisNegativeIndexing) {
  LayerParameter layer_param;
  ArgMaxParameter* argmax_param = layer_param.mutable_argmax_param();
  argmax_param->set_axis(-2);
  ArgMaxLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->shape(0), this->blob_bottom_->shape(0));
  EXPECT_EQ(this->blob_top_->shape(1), this->blob_bottom_->shape(1));
  EXPECT_EQ(this->blob_top_->shape(2), argmax_param->top_k());
  EXPECT_EQ(this->blob_top_->shape(3), this->blob_bottom_->shape(3));
}

TYPED_TEST(ArgMaxLayerTest, TestSetupAxisMaxVal) {
  LayerParameter layer_param;
  ArgMaxParameter* argmax_param = layer_param.mutable_argmax_param();
  argmax_param->set_axis(2);
  argmax_param->set_out_max_val(true);
  ArgMaxLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->shape(0), this->blob_bottom_->shape(0));
  EXPECT_EQ(this->blob_top_->shape(1), this->blob_bottom_->shape(1));
  EXPECT_EQ(this->blob_top_->shape(2), argmax_param->top_k());
  EXPECT_EQ(this->blob_top_->shape(3), this->blob_bottom_->shape(3));
}

TYPED_TEST(ArgMaxLayerTest, TestCPU) {
  LayerParameter layer_param;
  ArgMaxLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const TypeParam* bottom_data = this->blob_bottom_->cpu_data();
  const TypeParam* top_data = this->blob_top_->cpu_data();
  int max_ind;
  TypeParam max_val;
  int num = this->blob_bottom_->num();
  int dim = this->blob_bottom_->count() / num;
  for (int i = 0; i < num; ++i) {
    EXPECT_GE(top_data[i], 0);
    EXPECT_LE(top_data[i], dim);
    max_ind = top_data[i];
    max_val = bottom_data[i * dim + max_ind];
    for (int j = 0; j < dim; ++j) {
      EXPECT_LE(bottom_data[i * dim + j], max_val);
    }
  }
}

TYPED_TEST(ArgMaxLayerTest, TestCPUMaxVal) {
  LayerParameter layer_param;
  ArgMaxParameter* argmax_param = layer_param.mutable_argmax_param();
  argmax_param->set_out_max_val(true);
  ArgMaxLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const TypeParam* bottom_data = this->blob_bottom_->cpu_data();
  const TypeParam* top_data = this->blob_top_->cpu_data();
  int max_ind;
  TypeParam max_val;
  int num = this->blob_bottom_->num();
  int dim = this->blob_bottom_->count() / num;
  for (int i = 0; i < num; ++i) {
    EXPECT_GE(top_data[i], 0);
    EXPECT_LE(top_data[i], dim);
    max_ind = top_data[i * 2];
    max_val = top_data[i * 2 + 1];
    EXPECT_EQ(bottom_data[i * dim + max_ind], max_val);
    for (int j = 0; j < dim; ++j) {
      EXPECT_LE(bottom_data[i * dim + j], max_val);
    }
  }
}

TYPED_TEST(ArgMaxLayerTest, TestCPUTopK) {
  LayerParameter layer_param;
  ArgMaxParameter* argmax_param = layer_param.mutable_argmax_param();
  argmax_param->set_top_k(this->top_k_);
  ArgMaxLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const TypeParam* bottom_data = this->blob_bottom_->cpu_data();
  int max_ind;
  TypeParam max_val;
  int num = this->blob_bottom_->num();
  int dim = this->blob_bottom_->count() / num;
  for (int i = 0; i < num; ++i) {
    EXPECT_GE(this->blob_top_->data_at(i, 0, 0, 0), 0);
    EXPECT_LE(this->blob_top_->data_at(i, 0, 0, 0), dim);
    for (int j = 0; j < this->top_k_; ++j) {
      max_ind = this->blob_top_->data_at(i, 0, j, 0);
      max_val = bottom_data[i * dim + max_ind];
      int count = 0;
      for (int k = 0; k < dim; ++k) {
        if (bottom_data[i * dim + k] > max_val) {
          ++count;
        }
      }
      EXPECT_EQ(j, count);
    }
  }
}

TYPED_TEST(ArgMaxLayerTest, TestCPUMaxValTopK) {
  LayerParameter layer_param;
  ArgMaxParameter* argmax_param = layer_param.mutable_argmax_param();
  argmax_param->set_out_max_val(true);
  argmax_param->set_top_k(this->top_k_);
  ArgMaxLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const TypeParam* bottom_data = this->blob_bottom_->cpu_data();
  int max_ind;
  TypeParam max_val;
  int num = this->blob_bottom_->num();
  int dim = this->blob_bottom_->count() / num;
  for (int i = 0; i < num; ++i) {
    EXPECT_GE(this->blob_top_->data_at(i, 0, 0, 0), 0);
    EXPECT_LE(this->blob_top_->data_at(i, 0, 0, 0), dim);
    for (int j = 0; j < this->top_k_; ++j) {
      max_ind = this->blob_top_->data_at(i, 0, j, 0);
      max_val = this->blob_top_->data_at(i, 1, j, 0);
      EXPECT_EQ(bottom_data[i * dim + max_ind], max_val);
      int count = 0;
      for (int k = 0; k < dim; ++k) {
        if (bottom_data[i * dim + k] > max_val) {
          ++count;
        }
      }
      EXPECT_EQ(j, count);
    }
  }
}

TYPED_TEST(ArgMaxLayerTest, TestCPUAxis) {
  LayerParameter layer_param;
  ArgMaxParameter* argmax_param = layer_param.mutable_argmax_param();
  argmax_param->set_axis(0);
  ArgMaxLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  int max_ind;
  TypeParam max_val;
  std::vector<int> shape = this->blob_bottom_->shape();
  for (int i = 0; i < shape[1]; ++i) {
    for (int j = 0; j < shape[2]; ++j) {
      for (int k = 0; k < shape[3]; ++k) {
        max_ind = this->blob_top_->data_at(0, i, j, k);
        max_val = this->blob_bottom_->data_at(max_ind, i, j, k);
        EXPECT_GE(max_ind, 0);
        EXPECT_LE(max_ind, shape[0]);
        for (int l = 0; l < shape[0]; ++l) {
          EXPECT_LE(this->blob_bottom_->data_at(l, i, j, k), max_val);
        }
      }
    }
  }
}

TYPED_TEST(ArgMaxLayerTest, TestCPUAxisTopK) {
  LayerParameter layer_param;
  ArgMaxParameter* argmax_param = layer_param.mutable_argmax_param();
  argmax_param->set_axis(2);
  argmax_param->set_top_k(this->top_k_);
  ArgMaxLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  int max_ind;
  TypeParam max_val;
  std::vector<int> shape = this->blob_bottom_->shape();
  for (int i = 0; i < shape[0]; ++i) {
    for (int j = 0; j < shape[1]; ++j) {
      for (int k = 0; k < shape[3]; ++k) {
        for (int m = 0; m < this->top_k_; ++m) {
          max_ind = this->blob_top_->data_at(i, j, m, k);
          max_val = this->blob_bottom_->data_at(i, j, max_ind, k);
          EXPECT_GE(max_ind, 0);
          EXPECT_LE(max_ind, shape[2]);
          int count = 0;
          for (int l = 0; l < shape[2]; ++l) {
            if (this->blob_bottom_->data_at(i, j, l, k) > max_val) {
              ++count;
            }
          }
          EXPECT_EQ(m, count);
        }
      }
    }
  }
}

TYPED_TEST(ArgMaxLayerTest, TestCPUAxisMaxValTopK) {
  LayerParameter layer_param;
  ArgMaxParameter* argmax_param = layer_param.mutable_argmax_param();
  argmax_param->set_axis(-1);
  argmax_param->set_top_k(this->top_k_);
  argmax_param->set_out_max_val(true);
  ArgMaxLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  TypeParam max_val;
  std::vector<int> shape = this->blob_bottom_->shape();
  for (int i = 0; i < shape[0]; ++i) {
    for (int j = 0; j < shape[1]; ++j) {
      for (int k = 0; k < shape[2]; ++k) {
        for (int m = 0; m < this->top_k_; ++m) {
          max_val = this->blob_top_->data_at(i, j, k, m);
          int count = 0;
          for (int l = 0; l < shape[3]; ++l) {
            if (this->blob_bottom_->data_at(i, j, k, l) > max_val) {
              ++count;
            }
          }
          EXPECT_EQ(m, count);
        }
      }
    }
  }
}

#ifdef USE_MLU

template <typename TypeParam>
class MLUArgMaxLayerTest : public MLUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MLUArgMaxLayerTest()
      : blob_bottom_(new Blob<Dtype>(3, 5, 8, 10)),
        blob_top_(new Blob<Dtype>()),
        top_k_(1) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MLUArgMaxLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  size_t top_k_;
};

TYPED_TEST_CASE(MLUArgMaxLayerTest, TestMLUDevices);

TYPED_TEST(MLUArgMaxLayerTest, TestSetupAxis) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ArgMaxParameter* argmax_param = layer_param.mutable_argmax_param();
  argmax_param->set_axis(1);
  MLUArgMaxLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->shape(0), this->blob_bottom_->shape(0));
  EXPECT_EQ(this->blob_top_->shape(1), argmax_param->top_k());
  EXPECT_EQ(this->blob_top_->shape(2), this->blob_bottom_->shape(2));
  EXPECT_EQ(this->blob_top_->shape(3), this->blob_bottom_->shape(3));
}

TYPED_TEST(MLUArgMaxLayerTest, TestSetupAxisNegativeIndexing) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ArgMaxParameter* argmax_param = layer_param.mutable_argmax_param();
  argmax_param->set_axis(-2);
  MLUArgMaxLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->shape(0), this->blob_bottom_->shape(0));
  EXPECT_EQ(this->blob_top_->shape(1), this->blob_bottom_->shape(1));
  EXPECT_EQ(this->blob_top_->shape(2), argmax_param->top_k());
  EXPECT_EQ(this->blob_top_->shape(3), this->blob_bottom_->shape(3));
}

TYPED_TEST(MLUArgMaxLayerTest, TestSetupAxisMaxVal) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ArgMaxParameter* argmax_param = layer_param.mutable_argmax_param();
  argmax_param->set_axis(2);
  argmax_param->set_out_max_val(true);
  MLUArgMaxLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->shape(0), this->blob_bottom_->shape(0));
  EXPECT_EQ(this->blob_top_->shape(1), this->blob_bottom_->shape(1));
  EXPECT_EQ(this->blob_top_->shape(2), argmax_param->top_k());
  EXPECT_EQ(this->blob_top_->shape(3), this->blob_bottom_->shape(3));
}

TYPED_TEST(MLUArgMaxLayerTest, TestMLUAxis) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ArgMaxParameter* argmax_param = layer_param.mutable_argmax_param();
  argmax_param->set_axis(0);
  MLUArgMaxLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  vector<Dtype> data(this->blob_bottom_->count());
  for (int i = 0; i < this->blob_bottom_->count(); i++) {
    data[i] = i;
  }
  std::random_shuffle(data.begin(), data.end());
  for (int i = 0; i < this->blob_bottom_->count(); i++)
    this->blob_bottom_->mutable_cpu_data()[i] = data[i];
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  int max_ind;
  Dtype max_val;
  std::vector<int> shape = this->blob_bottom_->shape();
  for (int i = 0; i < shape[1]; ++i) {
    for (int j = 0; j < shape[2]; ++j) {
      for (int k = 0; k < shape[3]; ++k) {
        max_ind = this->blob_top_->data_at(0, i, j, k);
        max_val = this->blob_bottom_->data_at(max_ind, i, j, k);
        EXPECT_GE(max_ind, 0);
        EXPECT_LE(max_ind, shape[0]);
        for (int l = 0; l < shape[0]; ++l) {
          EXPECT_LE(this->blob_bottom_->data_at(l, i, j, k), max_val);
        }
      }
    }
  }
  std::ostringstream stream, param;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  param << "axis:" << argmax_param->axis();
  PARAM(param);
  EVENT_TIME(layer.get_event_time());
}

TYPED_TEST(MLUArgMaxLayerTest, TestMLUAxisTopK) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ArgMaxParameter* argmax_param = layer_param.mutable_argmax_param();
  argmax_param->set_axis(2);
  argmax_param->set_top_k(this->top_k_);
  MLUArgMaxLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  vector<Dtype> data(this->blob_bottom_->count());
  for (int i = 0; i < this->blob_bottom_->count(); i++) {
    data[i] = i;
  }
  std::random_shuffle(data.begin(), data.end());
  for (int i = 0; i < this->blob_bottom_->count(); i++)
    this->blob_bottom_->mutable_cpu_data()[i] = data[i];
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  int max_ind;
  Dtype max_val;
  std::vector<int> shape = this->blob_bottom_->shape();
  for (int i = 0; i < shape[0]; ++i) {
    for (int j = 0; j < shape[1]; ++j) {
      for (int k = 0; k < shape[3]; ++k) {
        for (int m = 0; m < this->top_k_; ++m) {
          max_ind = this->blob_top_->data_at(i, j, m, k);
          max_val = this->blob_bottom_->data_at(i, j, max_ind, k);
          EXPECT_GE(max_ind, 0);
          EXPECT_LE(max_ind, shape[2]);
          int count = 0;
          for (int l = 0; l < shape[2]; ++l) {
            if (this->blob_bottom_->data_at(i, j, l, k) > max_val) {
              ++count;
            }
          }
          EXPECT_EQ(m, count);
        }
      }
    }
  }
  std::ostringstream stream, param;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  param << "axis:" << argmax_param->axis()
        << "\ttop_k:" << argmax_param->top_k();
  PARAM(param);
  EVENT_TIME(layer.get_event_time());
}

TYPED_TEST(MLUArgMaxLayerTest, TestMLUTopK) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ArgMaxParameter* argmax_param = layer_param.mutable_argmax_param();
  argmax_param->set_top_k(this->top_k_);
  MLUArgMaxLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  int max_ind;
  Dtype max_val;
  int num = this->blob_bottom_->num();
  int dim = this->blob_bottom_->count() / num;
  for (int i = 0; i < num; ++i) {
    EXPECT_GE(this->blob_top_->data_at(i, 0, 0, 0), 0);
    EXPECT_LE(this->blob_top_->data_at(i, 0, 0, 0), dim);
    for (int j = 0; j < this->top_k_; ++j) {
      max_ind = this->blob_top_->data_at(i, 0, j, 0);
      max_val = bottom_data[i * dim + max_ind];
      int count = 0;
      for (int k = 0; k < dim; ++k) {
        if (bottom_data[i * dim + k] > max_val) {
          ++count;
        }
      }
      EXPECT_EQ(j, count);
    }
  }
  std::ostringstream stream, param;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  param << "top_k:" << argmax_param->top_k();
  PARAM(param);
  EVENT_TIME(layer.get_event_time());
}

TYPED_TEST(MLUArgMaxLayerTest, TestMLUMaxValue) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  cnmlSetFusionThreadContext(false);

  ArgMaxParameter* argmax_param = layer_param.mutable_argmax_param();
  argmax_param->set_out_max_val(true);
  MLUArgMaxLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const Dtype* top_data = this->blob_top_->cpu_data();
  int max_ind;
  Dtype max_val;
  int num = this->blob_bottom_->num();
  int dim = this->blob_bottom_->count() / num;
  for (int i = 0; i < num; ++i) {
    EXPECT_GE(top_data[i], 0);
    EXPECT_LE(top_data[i], dim);
    max_ind = top_data[i * 2];
    max_val = top_data[i * 2 + 1];
    EXPECT_EQ(bottom_data[i * dim + max_ind], max_val);
    for (int j = 0; j < dim; ++j) {
      EXPECT_LE(bottom_data[i * dim + j], max_val);
    }
  }
  std::ostringstream stream, param;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  param << "out_max_val:" << argmax_param->out_max_val();
  PARAM(param);
  EVENT_TIME(layer.get_event_time());
}

TYPED_TEST(MLUArgMaxLayerTest, TestMLUMaxValueTopK) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ArgMaxParameter* argmax_param = layer_param.mutable_argmax_param();
  argmax_param->set_out_max_val(true);
  argmax_param->set_top_k(this->top_k_);
  MLUArgMaxLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const Dtype* top_data = this->blob_top_->cpu_data();
  int max_ind;
  Dtype max_val;
  int num = this->blob_bottom_->num();
  int dim = this->blob_bottom_->count() / num;
  for (int i = 0; i < num; ++i) {
    EXPECT_GE(top_data[i], 0);
    EXPECT_LE(top_data[i], dim);
    max_ind = top_data[i * 2];
    max_val = top_data[i * 2 + 1];
    EXPECT_EQ(bottom_data[i * dim + max_ind], max_val);
    for (int j = 0; j < dim; ++j) {
      EXPECT_LE(bottom_data[i * dim + j], max_val);
    }
  }
  std::ostringstream stream, param;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  param << "top_k:" << argmax_param->top_k();
  PARAM(param);
  EVENT_TIME(layer.get_event_time());
}

TYPED_TEST(MLUArgMaxLayerTest, TestMLUAxisMaxValueTopK) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ArgMaxParameter* argmax_param = layer_param.mutable_argmax_param();
  argmax_param->set_axis(-1);
  argmax_param->set_top_k(this->top_k_);
  argmax_param->set_out_max_val(true);
  MLUArgMaxLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  Dtype max_val;
  std::vector<int> shape = this->blob_bottom_->shape();
  for (int i = 0; i < shape[0]; ++i) {
    for (int j = 0; j < shape[1]; ++j) {
      for (int k = 0; k < shape[2]; ++k) {
        for (int m = 0; m < this->top_k_; ++m) {
          max_val = this->blob_top_->data_at(i, j, k, m);
          int count = 0;
          for (int l = 0; l < shape[3]; ++l) {
            if (this->blob_bottom_->data_at(i, j, k, l) > max_val) {
              ++count;
            }
          }
          EXPECT_EQ(m, count);
        }
      }
    }
  }
  std::ostringstream stream, param;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  param << "axis:" << argmax_param->axis()
        << "top_k:" << argmax_param->top_k()
        << "out_max_val:" << argmax_param->out_max_val();
  PARAM(param);
  EVENT_TIME(layer.get_event_time());
}


// ------------------------------  TEST MFUS --------------------------------

template <typename TypeParam>
class MFUSArgMaxLayerTest : public MFUSDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MFUSArgMaxLayerTest()
      : blob_bottom_(new Blob<Dtype>(3, 5, 8, 10)),
        blob_top_(new Blob<Dtype>()),
        top_k_(1) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MFUSArgMaxLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  size_t top_k_;
};

TYPED_TEST_CASE(MFUSArgMaxLayerTest, TestMFUSDevices);

TYPED_TEST(MFUSArgMaxLayerTest, TestSetupAxis) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ArgMaxParameter* argmax_param = layer_param.mutable_argmax_param();
  argmax_param->set_axis(1);
  MLUArgMaxLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->shape(0), this->blob_bottom_->shape(0));
  EXPECT_EQ(this->blob_top_->shape(1), argmax_param->top_k());
  EXPECT_EQ(this->blob_top_->shape(2), this->blob_bottom_->shape(2));
  EXPECT_EQ(this->blob_top_->shape(3), this->blob_bottom_->shape(3));
}

TYPED_TEST(MFUSArgMaxLayerTest, TestSetupAxisNegativeIndexing) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ArgMaxParameter* argmax_param = layer_param.mutable_argmax_param();
  argmax_param->set_axis(-2);
  MLUArgMaxLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->shape(0), this->blob_bottom_->shape(0));
  EXPECT_EQ(this->blob_top_->shape(1), this->blob_bottom_->shape(1));
  EXPECT_EQ(this->blob_top_->shape(2), argmax_param->top_k());
  EXPECT_EQ(this->blob_top_->shape(3), this->blob_bottom_->shape(3));
}

TYPED_TEST(MFUSArgMaxLayerTest, TestSetupAxisMaxVal) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ArgMaxParameter* argmax_param = layer_param.mutable_argmax_param();
  argmax_param->set_axis(2);
  argmax_param->set_out_max_val(true);
  MLUArgMaxLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->shape(0), this->blob_bottom_->shape(0));
  EXPECT_EQ(this->blob_top_->shape(1), this->blob_bottom_->shape(1));
  EXPECT_EQ(this->blob_top_->shape(2), argmax_param->top_k());
  EXPECT_EQ(this->blob_top_->shape(3), this->blob_bottom_->shape(3));
}

TYPED_TEST(MFUSArgMaxLayerTest, TestMFUSAxis) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ArgMaxParameter* argmax_param = layer_param.mutable_argmax_param();
  argmax_param->set_axis(0);
  MLUArgMaxLayer<Dtype> layer(layer_param);
  vector<Dtype> data(this->blob_bottom_->count());
  for (int i = 0; i < this->blob_bottom_->count(); i++) {
    data[i] = i;
  }
  std::random_shuffle(data.begin(), data.end());
  for (int i = 0; i < this->blob_bottom_->count(); i++)
    this->blob_bottom_->mutable_cpu_data()[i] = data[i];
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

  // Now, check values
  int max_ind;
  Dtype max_val;
  std::vector<int> shape = this->blob_bottom_->shape();
  for (int i = 0; i < shape[1]; ++i) {
    for (int j = 0; j < shape[2]; ++j) {
      for (int k = 0; k < shape[3]; ++k) {
        max_ind = this->blob_top_->data_at(0, i, j, k);
        max_val = this->blob_bottom_->data_at(max_ind, i, j, k);
        EXPECT_GE(max_ind, 0);
        EXPECT_LE(max_ind, shape[0]);
        for (int l = 0; l < shape[0]; ++l) {
          EXPECT_LE(this->blob_bottom_->data_at(l, i, j, k), max_val);
        }
      }
    }
  }
  std::ostringstream stream, param;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  param << "axis:" << argmax_param->axis()
    << "top_k:" <<argmax_param->top_k()
    << "out_max_val:" << argmax_param->out_max_val();
  PARAM(param);
  EVENT_TIME(fuser.get_event_time());
}

TYPED_TEST(MFUSArgMaxLayerTest, TestMFUSAxisTopK) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ArgMaxParameter* argmax_param = layer_param.mutable_argmax_param();
  argmax_param->set_axis(2);
  argmax_param->set_top_k(this->top_k_);
  MLUArgMaxLayer<Dtype> layer(layer_param);
  vector<Dtype> data(this->blob_bottom_->count());
  for (int i = 0; i < this->blob_bottom_->count(); i++) {
    data[i] = i;
  }
  std::random_shuffle(data.begin(), data.end());
  for (int i = 0; i < this->blob_bottom_->count(); i++)
    this->blob_bottom_->mutable_cpu_data()[i] = data[i];
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
  // Now, check values
  int max_ind;
  Dtype max_val;
  std::vector<int> shape = this->blob_bottom_->shape();
  for (int i = 0; i < shape[0]; ++i) {
    for (int j = 0; j < shape[1]; ++j) {
      for (int k = 0; k < shape[3]; ++k) {
        for (int m = 0; m < this->top_k_; ++m) {
          max_ind = this->blob_top_->data_at(i, j, m, k);
          max_val = this->blob_bottom_->data_at(i, j, max_ind, k);
          EXPECT_GE(max_ind, 0);
          EXPECT_LE(max_ind, shape[2]);
          int count = 0;
          for (int l = 0; l < shape[2]; ++l) {
            if (this->blob_bottom_->data_at(i, j, l, k) > max_val) {
              ++count;
            }
          }
          EXPECT_EQ(m, count);
        }
      }
    }
  }
  std::ostringstream stream, param;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  param << "axis:" << argmax_param->axis()
    << "top_k:" <<argmax_param->top_k();
  PARAM(param);
  EVENT_TIME(fuser.get_event_time());
}

TYPED_TEST(MFUSArgMaxLayerTest, TestMFUSTopK) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ArgMaxParameter* argmax_param = layer_param.mutable_argmax_param();
  argmax_param->set_top_k(this->top_k_);
  MLUArgMaxLayer<Dtype> layer(layer_param);
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
  // Now, check values
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  int max_ind;
  Dtype max_val;
  int num = this->blob_bottom_->num();
  int dim = this->blob_bottom_->count() / num;
  for (int i = 0; i < num; ++i) {
    EXPECT_GE(this->blob_top_->data_at(i, 0, 0, 0), 0);
    EXPECT_LE(this->blob_top_->data_at(i, 0, 0, 0), dim);
    for (int j = 0; j < this->top_k_; ++j) {
      max_ind = this->blob_top_->data_at(i, 0, j, 0);
      max_val = bottom_data[i * dim + max_ind];
      int count = 0;
      for (int k = 0; k < dim; ++k) {
        if (bottom_data[i * dim + k] > max_val) {
          ++count;
        }
      }
      EXPECT_EQ(j, count);
    }
  }
  std::ostringstream stream, param;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  param << "top_k:" <<argmax_param->top_k();
  PARAM(param);
  EVENT_TIME(layer.get_event_time());
}

TYPED_TEST(MFUSArgMaxLayerTest, TestMFUSMaxValue) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ArgMaxParameter* argmax_param = layer_param.mutable_argmax_param();
  argmax_param->set_out_max_val(true);
  MLUArgMaxLayer<Dtype> layer(layer_param);
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
  // Now, check values
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const Dtype* top_data = this->blob_top_->cpu_data();
  int max_ind;
  Dtype max_val;
  int num = this->blob_bottom_->num();
  int dim = this->blob_bottom_->count() / num;
  for (int i = 0; i < num; ++i) {
    EXPECT_GE(top_data[i], 0);
    EXPECT_LE(top_data[i], dim);
    max_ind = top_data[i * 2];
    max_val = top_data[i * 2 + 1];
    EXPECT_EQ(bottom_data[i * dim + max_ind], max_val);
    for (int j = 0; j < dim; ++j) {
      EXPECT_LE(bottom_data[i * dim + j], max_val);
    }
  }
  std::ostringstream stream, param;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  param << "out_max_val:" << argmax_param->out_max_val();
  PARAM(param);
  EVENT_TIME(layer.get_event_time());
}

TYPED_TEST(MFUSArgMaxLayerTest, TestMFUSMaxValueTopK) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ArgMaxParameter* argmax_param = layer_param.mutable_argmax_param();
  argmax_param->set_out_max_val(true);
  argmax_param->set_top_k(this->top_k_);
  MLUArgMaxLayer<Dtype> layer(layer_param);
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
  // Now, check values
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const Dtype* top_data = this->blob_top_->cpu_data();
  int max_ind;
  Dtype max_val;
  int num = this->blob_bottom_->num();
  int dim = this->blob_bottom_->count() / num;
  for (int i = 0; i < num; ++i) {
    EXPECT_GE(top_data[i], 0);
    EXPECT_LE(top_data[i], dim);
    max_ind = top_data[i * 2];
    max_val = top_data[i * 2 + 1];
    EXPECT_EQ(bottom_data[i * dim + max_ind], max_val);
    for (int j = 0; j < dim; ++j) {
      EXPECT_LE(bottom_data[i * dim + j], max_val);
    }
  }
  std::ostringstream stream, param;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  param << "top_k:" <<argmax_param->top_k()
    << "out_max_val: " << argmax_param->out_max_val();
  PARAM(param);
  EVENT_TIME(layer.get_event_time());
}

TYPED_TEST(MFUSArgMaxLayerTest, TestMFUSAxisMaxValueTopK) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ArgMaxParameter* argmax_param = layer_param.mutable_argmax_param();
  argmax_param->set_axis(-1);
  argmax_param->set_top_k(this->top_k_);
  argmax_param->set_out_max_val(true);
  MLUArgMaxLayer<Dtype> layer(layer_param);
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
  // Now, check values
  Dtype max_val;
  std::vector<int> shape = this->blob_bottom_->shape();
  for (int i = 0; i < shape[0]; ++i) {
    for (int j = 0; j < shape[1]; ++j) {
      for (int k = 0; k < shape[2]; ++k) {
        for (int m = 0; m < this->top_k_; ++m) {
          max_val = this->blob_top_->data_at(i, j, k, m);
          int count = 0;
          for (int l = 0; l < shape[3]; ++l) {
            if (this->blob_bottom_->data_at(i, j, k, l) > max_val) {
              ++count;
            }
          }
          EXPECT_EQ(m, count);
        }
      }
    }
  }
  std::ostringstream stream, param;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  param << "top_k:" <<argmax_param->top_k()
    << "out_max_val: " << argmax_param->out_max_val()
    << "axis:" << argmax_param->axis();
  PARAM(param);
  EVENT_TIME(layer.get_event_time());
}


#endif

}  // namespace caffe
