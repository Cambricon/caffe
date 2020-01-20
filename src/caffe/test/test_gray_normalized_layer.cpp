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

#include <time.h>
#include <cstdlib>
#include <memory>
#include <vector>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "gtest/gtest.h"

#ifdef USE_MLU
#include "caffe/layers/mlu_gray_normalized_layer.hpp"
#endif

#include "caffe/filler.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#ifdef USE_MLU

template <typename TypeParam>
class MLUGrayNormalizedLayerTest : public MLUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  int num, height, width;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  Blob<Dtype>* blob_bottom_;
  Blob<Dtype>* blob_top_;
  float* rgb_data;
  float* yuv_data;
  bool have_data;

  MLUGrayNormalizedLayerTest()
    : num(1), height(4), width(4),
    blob_bottom_(new Blob<Dtype>()),
    blob_top_(new Blob<Dtype>()),
    rgb_data(nullptr), yuv_data(nullptr),
    have_data(false) {}
  virtual void SetUp() {
    blob_bottom_->Reshape(num, 1, height, width);
    blob_top_->Reshape(num, 1, height, width);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MLUGrayNormalizedLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
    if (rgb_data) delete rgb_data;
    if (yuv_data) delete yuv_data;
  }
  void GenerateData() {
    if (have_data) return;
    have_data = true;
    srand(height * width %256 +1);
    /* generate rgb data */
    rgb_data = new float[height * width * 3];
    for (int c = 0; c < 3; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < height; w++) {
            rgb_data[(c * height + h) * width + w] =
               static_cast<float>(rand() % 256);  // NOLINT
        }
      }
    }
    /* convert rgb data to yuv data, and need Y data to compare err  */
    int coordinate;
    yuv_data = new float[height * width / 2 * 3];
    for (int h = 0; h < height / 2; h++) {
      for (int w = 0; w < width / 2; w++) {
        float Cr, Cb, avg_r = 0, avg_g = 0, avg_b = 0;
        for (int hh = 0; hh < 2; hh++) {
          for (int ww = 0; ww < 2; ww++) {
            float r, g, b, Y;
            coordinate = (h * 2 + hh) * width + (w * 2 + ww);
            r = rgb_data[coordinate + 0 * width * height];
            g = rgb_data[coordinate + 1 * width * height];
            b = rgb_data[coordinate + 2 * width * height];
            avg_r += r;
            avg_g += g;
            avg_b += b;
            Y = 0.257 * r + 0.504 * g + 0.098 * b + 16;
            yuv_data[coordinate] = Y;
          }
        }
        avg_r /= 4;
        avg_g /= 4;
        avg_b /= 4;
        Cb = -0.148 * avg_r - 0.291 * avg_g + 0.439 * avg_b + 128;
        Cr = 0.439 * avg_r - 0.368 * avg_g - 0.071 * avg_b + 128;
        Cb = static_cast<int>(Cb);
        Cr = static_cast<int>(Cr);
        coordinate = (h + height) * width + w * 2;
        yuv_data[coordinate] = static_cast<float>(Cr);
        yuv_data[coordinate + 1] = static_cast<float>(Cb);
      }
    }
  }

  void BottomSetValue() {
    for (int i = 0; i < num; i++) {
      for (int cood = 0; cood < height * width; cood++) {
        reinterpret_cast<uint8_t*>(
            blob_bottom_->mutable_cpu_data())[i * height * width  + cood]
            = yuv_data[cood];
      }
    }
  }
  void LayerSizeCheck() {
    EXPECT_EQ(blob_top_->num(), num);
    EXPECT_EQ(blob_top_->channels(), 1);
    EXPECT_EQ(blob_top_->height(), height);
    EXPECT_EQ(blob_top_->width(), width);
  }

  float OutputCompare() {
    const Dtype* data = blob_top_->cpu_data();
    float err_sum = 0, sum = 0;
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < height; w++) {
        float Y = yuv_data[h * width + w] / 255 - 0.5;  // normalized_Y;
        EXPECT_NEAR(Y, data[h * width + w], 5e-2);
        err_sum += std::abs(data[h * width + w] - Y);
        sum += std::abs(Y);
      }
    }
    return err_sum/sum;
  }

  void TestForwardToGray() {
    LayerParameter layer_param;
    int channel = 1;
    GenerateData();
    blob_bottom_->Reshape(num, channel, height, width);
    BottomSetValue();
    MLUGrayNormalizedLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    LayerSizeCheck();
    float rate = OutputCompare();
    std::ostringstream stream;
    stream << "bottom1:" << blob_bottom_->shape_string().c_str();
    BOTTOM(stream);
    ERR_RATE(rate);
    EVENT_TIME(layer.get_event_time());
  }
};

TYPED_TEST_CASE(MLUGrayNormalizedLayerTest, TestMLUDevices);

TYPED_TEST(MLUGrayNormalizedLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  shared_ptr<MLUGrayNormalizedLayer<Dtype> > layer(
      new MLUGrayNormalizedLayer<Dtype>(layer_param));

  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_vec_[0]->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width());
  OUTPUT("bottom", this->blob_bottom_->shape_string().c_str());
}

TYPED_TEST(MLUGrayNormalizedLayerTest, TestForward) {
  this->TestForwardToGray();
}


template <typename TypeParam>
class MFUSGrayNormalizedLayerTest : public MFUSDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  int num, height, width;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  Blob<Dtype>* blob_bottom_;
  Blob<Dtype>* blob_top_;
  float* rgb_data;
  float* yuv_data;
  bool have_data;

  MFUSGrayNormalizedLayerTest()
    : num(1), height(4), width(4),
    blob_bottom_(new Blob<Dtype>()),
    blob_top_(new Blob<Dtype>()),
    rgb_data(nullptr), yuv_data(nullptr),
    have_data(false) {}
  virtual void SetUp() {
    blob_bottom_->Reshape(num, 1, height, width);
    blob_top_->Reshape(num, 1, height, width);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MFUSGrayNormalizedLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
    if (rgb_data) delete rgb_data;
    if (yuv_data) delete yuv_data;
  }
  void GenerateData() {
    if (have_data) return;
    have_data = true;
    srand(height * width % 256 +1);
    /* generate rgb data */
    rgb_data = new float[height * width * 3];
    for (int c = 0; c < 3; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < height; w++) {
            rgb_data[(c * height + h) * width + w] =
               static_cast<float>(rand() % 256);  // NOLINT
        }
      }
    }
    /* convert rgb data to yuv data, and need Y data to compare err  */
    int coordinate;
    yuv_data = new float[height * width / 2 * 3];
    for (int h = 0; h < height / 2; h++) {
      for (int w = 0; w < width / 2; w++) {
        float Cr, Cb, avg_r = 0, avg_g = 0, avg_b = 0;
        for (int hh = 0; hh < 2; hh++) {
          for (int ww = 0; ww < 2; ww++) {
            float r, g, b, Y;
            coordinate = (h * 2 + hh) * width + (w * 2 + ww);
            r = rgb_data[coordinate + 0 * width * height];
            g = rgb_data[coordinate + 1 * width * height];
            b = rgb_data[coordinate + 2 * width * height];
            avg_r += r;
            avg_g += g;
            avg_b += b;
            Y = 0.257 * r + 0.504 * g + 0.098 * b + 16;
            yuv_data[coordinate] = Y;
          }
        }
        avg_r /= 4;
        avg_g /= 4;
        avg_b /= 4;
        Cb = -0.148 * avg_r - 0.291 * avg_g + 0.439 * avg_b + 128;
        Cr = 0.439 * avg_r - 0.368 * avg_g - 0.071 * avg_b + 128;
        Cb = static_cast<int>(Cb);
        Cr = static_cast<int>(Cr);
        coordinate = (h + height) * width + w * 2;
        yuv_data[coordinate] = static_cast<float>(Cr);
        yuv_data[coordinate + 1] = static_cast<float>(Cb);
      }
    }
  }

  void BottomSetValue() {
    for (int i = 0; i < num; i++) {
      for (int cood = 0; cood < height * width; cood++) {
        reinterpret_cast<uint8_t*>(
            blob_bottom_->mutable_cpu_data())[i * height * width + cood]
            = yuv_data[cood];
      }
    }
  }
  void LayerSizeCheck() {
    EXPECT_EQ(blob_top_->num(), num);
    EXPECT_EQ(blob_top_->channels(), 1);
    EXPECT_EQ(blob_top_->height(), height);
    EXPECT_EQ(blob_top_->width(), width);
  }

  float OutputCompare() {
    const Dtype* data = blob_top_->cpu_data();
    float err_sum = 0, sum = 0;
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < height; w++) {
        float Y = yuv_data[h * width + w] / 255 - 0.5;  // normalized_Y;
        EXPECT_NEAR(Y, data[h * width + w], 5e-2);
        err_sum += std::abs(data[h * width + w] - Y);
        sum += std::abs(Y);
      }
    }
    return err_sum/sum;
  }

  void TestForwardToGray() {
    LayerParameter layer_param;
    int channel = 1;
    GenerateData();
    blob_bottom_->Reshape(num, channel, height, width);
    BottomSetValue();
    MLUGrayNormalizedLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    ASSERT_TRUE(layer.mfus_supported());

    MFusion<Dtype> fuser;
    fuser.reset();
    fuser.addInputs(this->blob_bottom_vec_);
    fuser.addOutputs(this->blob_top_vec_);
    layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.fuse(&fuser);
    fuser.compile();
    fuser.forward();
    LayerSizeCheck();
    float rate = OutputCompare();
    std::ostringstream stream;
    stream << "bottom1:" << blob_bottom_->shape_string().c_str();
    BOTTOM(stream);
    ERR_RATE(rate);
    EVENT_TIME(fuser.get_event_time());
  }
};

TYPED_TEST_CASE(MFUSGrayNormalizedLayerTest, TestMFUSDevices);

TYPED_TEST(MFUSGrayNormalizedLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  shared_ptr<MLUGrayNormalizedLayer<Dtype> > layer(
      new MLUGrayNormalizedLayer<Dtype>(layer_param));

  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width());
  OUTPUT("bottom", this->blob_bottom_->shape_string().c_str());
}

TYPED_TEST(MFUSGrayNormalizedLayerTest, TestForward) {
  this->TestForwardToGray();
}

#endif
}  // namespace caffe
