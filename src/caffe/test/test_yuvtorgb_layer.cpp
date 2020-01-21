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

#include <cstdlib>
#include <memory>
#include <vector>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "gtest/gtest.h"

#ifdef USE_MLU
#include "caffe/layers/mlu_yuvtorgb_layer.hpp"
#endif

#include "caffe/filler.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {
#ifdef USE_MLU
template <typename TypeParam>
class MLUYUVtoRGBLayerTest : public MLUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  int Height, Width, Nums;  // Height and Width have to be in the first line
  typedef unsigned char TType;
  // char for Uint8 , float for Normal Float TestType
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_1_;
  Blob<Dtype>* const blob_top_;
  // blob set and compare.
  bool data_generated_;
  TType* yuv420spNV21_;
  TType* yuv420spNV12_;
  float* random_jpg_;
  TType* result_jpg_;  // Calc back from YUV, use this to compare result.

  enum intype { NV12 = 0, NV21 };
  enum outtype { RGB0 = 0, BGR0, ARGB };

  MLUYUVtoRGBLayerTest()
      : Height(4), Width(4), Nums(1),  // Set size here
                      // H & W is JPG size, must be even, n > 0
        blob_bottom_(new Blob<Dtype>(this->Nums, 1,
                                     this->Height, this->Width)),
        blob_bottom_1_(new Blob<Dtype>(this->Nums, 1,
                                       this->Height/2, this->Width)),
        blob_top_(new Blob<Dtype>()), data_generated_(false),
        yuv420spNV21_(nullptr), yuv420spNV12_(nullptr), random_jpg_(nullptr),
        result_jpg_(nullptr) {}

  virtual ~MLUYUVtoRGBLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_1_;
    delete blob_top_;
    clear();
  }
  void clear() {  // clear all ptrs
    if (yuv420spNV21_) delete yuv420spNV21_;
    if (yuv420spNV12_) delete yuv420spNV12_;
    if (random_jpg_) delete random_jpg_;
    if (result_jpg_) delete result_jpg_;
  }
/*
  dataGenerator only generate the same size image once.
*/
  void dataGenerator() {
    if (data_generated_) return;
    data_generated_ = true;
    srand(Height * Width % 256 + 1);

    int coordinate;
    clear();
    random_jpg_ = new float[Height * Width * 3];
    result_jpg_ = new TType[Height * Width * 3];
    for (int c_ = 0; c_ < 3; c_++)
      for (int h_ = 0; h_ < Height; h_++)
        for (int w_ = 0; w_ < Width; w_++)
          random_jpg_[(c_ * Height + h_) * Width + w_] =
              static_cast<float>(rand() % 256);  //  NOLINT thread safe OK

    yuv420spNV21_ = new TType[Height * Width / 2 * 3];
    yuv420spNV12_ = new TType[Height * Width / 2 * 3];
    for (int h_ = 0; h_ < Height / 2; h_++)
      for (int w_ = 0; w_ < Width / 2; w_++) {
        float Cr, Cb, avg_r = 0, avg_g = 0, avg_b = 0;
        for (int hh = 0; hh < 2; hh++)
          for (int ww = 0; ww < 2; ww++) {
            float r, g, b, Y;
            coordinate = (h_ * 2 + hh) * Width + (w_ * 2 + ww);
            r = random_jpg_[coordinate + 0 * Width * Height];
            g = random_jpg_[coordinate + 1 * Width * Height];
            b = random_jpg_[coordinate + 2 * Width * Height];
            avg_r += r;
            avg_g += g;
            avg_b += b;
            Y = 0.257 * r + 0.504 * g + 0.098 * b + 16;
            yuv420spNV21_[coordinate] = static_cast<TType>(static_cast<int>(Y));
            yuv420spNV12_[coordinate] = static_cast<TType>(static_cast<int>(Y));
          }

        avg_r /= 4;
        avg_g /= 4;
        avg_b /= 4;
        Cb = -0.148 * avg_r - 0.291 * avg_g + 0.439 * avg_b + 128;
        Cr = 0.439 * avg_r - 0.368 * avg_g - 0.071 * avg_b + 128;
        Cb = static_cast<int>(Cb);
        Cr = static_cast<int>(Cr);
        coordinate = (h_ + Height) * Width + w_ * 2;
        /*
           YYYY
           YYYY
           YYYY
           YYYY  Height * Width
           UVUV  U at: (Height + h_) + w_ *2;
           UVUV
           in NV12 U=Cb V=Cr
           in NV21 U=Cr V=Cb
        */
        yuv420spNV12_[coordinate] = static_cast<TType>(Cr);
        yuv420spNV12_[coordinate + 1] = static_cast<TType>(Cb);
        yuv420spNV21_[coordinate] = static_cast<TType>(Cb);
        yuv420spNV21_[coordinate + 1] = static_cast<TType>(Cr);
      }
    // End 4 for
      // Calc rgb back from YUV.
      for (int h_ = 0; h_ < Height; h_ ++)
        for (int w_ = 0; w_ < Width; w_ ++) {
          float Y, Cr, Cb, r, g, b;
          Y = yuv420spNV12_[h_ * Width + w_];
          Cb = yuv420spNV12_[(Height + h_ / 2) * Width + w_ / 2 * 2];
          Cr = yuv420spNV12_[(Height + h_ / 2) * Width + w_ / 2 * 2 + 1];
          coordinate = h_ * Width + w_;
          r = Y * 1.164 + Cr * 1.596 - 222.912;
          g = Y * 1.164 - Cb * 0.392 - Cr * 0.813 + 135.616;
          b = Y * 1.164 + Cb * 2.017 - 276.8;
          r = r < 0 ? 0 : (r > 255 ? 255 : r);
          g = g < 0 ? 0 : (g > 255 ? 255 : g);
          b = b < 0 ? 0 : (b > 255 ? 255 : b);
          result_jpg_[coordinate + 0 * Height * Width] = static_cast<TType>(
              static_cast<int>(r));
          result_jpg_[coordinate + 1 * Height * Width] = static_cast<TType>(
              static_cast<int>(g));
          result_jpg_[coordinate + 2 * Height * Width] = static_cast<TType>(
              static_cast<int>(b));
        }
    }
  //  End of datagenerator
  void LayerBottomSet(intype shape) {
    for (int n_ = 0; n_ < Nums; n_++)
      for (int cood = 0; cood < Height * Width/ 2 * 3; cood++)
        if (cood < Height * Width) {
            reinterpret_cast<TType*>(
                blob_bottom_->mutable_cpu_data())[n_ * Height * Width + cood]
                =(shape == NV12) ? yuv420spNV12_[cood] : yuv420spNV21_[cood];
        } else {
            reinterpret_cast<TType*>
                (blob_bottom_1_->mutable_cpu_data())[n_ * Height * Width / 2
                + cood - (Height * Width)]
                = (shape == NV12) ? yuv420spNV12_[cood] : yuv420spNV21_[cood];
        }
  }

  int ResultAt(outtype shape, int coodinate, int channel) {
    if (shape == RGB0) {
      if (channel == 3)
        return 0;
      else
        return result_jpg_[channel * Height * Width + coodinate];
    } else if (shape == BGR0) {
      if (channel == 3)
        return 0;
      else
        return result_jpg_[(2 - channel) * Height * Width + coodinate];
    } else if (shape == ARGB) {
      if (channel == 0)
        return 0;
      else
        return result_jpg_[(channel - 1) * Height * Width + coodinate];
    }
    return -1;
  }
  void LayerTopCmp(outtype shape) {
    float deviation = 2;
    for (int n_ = 0; n_ < Nums; n_++) {
      for (int c_ = 0; c_ < 4; c_++) {
        for (int cood = 0; cood < Height * Width; cood++) {
          float Output = static_cast<float>(
              reinterpret_cast<const TType*>(blob_top_vec_[0]->cpu_data())[(
              n_ * 4)  * Height * Width + 4 * cood+ c_]);
          unsigned char RightAns = ResultAt(shape, cood, c_);
          EXPECT_NEAR(Output, RightAns, deviation) << "  Where  N=" << n_
            << " C=" << c_ << " H=" << cood / Width << " W=" << cood % Width;
         }
      }
    }
  }

  void LayerSizeCmp() {
    EXPECT_EQ(blob_top_->num(), Nums);
    EXPECT_EQ(blob_top_->channels(), 4);
    EXPECT_EQ(blob_top_->height(), Height);
    EXPECT_EQ(blob_top_->width(), Width);
  }
  virtual void SetUp() {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_bottom_1_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_1_);
    blob_top_vec_.push_back(blob_top_);
  }

void TestForwardTtoT(int a, int b) {
  intype inshape; outtype outshape;
  if (a == 0)
    inshape = NV12;
  else
    inshape = NV21;
  if (b == 0)
    outshape = RGB0;
  else if (b == 1)
    outshape = BGR0;
  else
    outshape = ARGB;
  LayerParameter layer_param;
  YuvToRgbParameter* yuvtorgb_param = layer_param.mutable_yuvtorgb_param();
  int channels = 1;
  dataGenerator();
  blob_bottom_->Reshape(Nums, channels, Height, Width);
  blob_bottom_1_->Reshape(Nums, channels, Height / 2, Width);
  blob_top_->Reshape(Nums, 4, Height, Width);
  LayerBottomSet(inshape);
  if (inshape == NV12)
    yuvtorgb_param->set_input_format(YuvToRgbParameter_InputFormat_YUV420SP_NV12);
  else
    yuvtorgb_param->set_input_format(YuvToRgbParameter_InputFormat_YUV420SP_NV21);
  if (outshape == RGB0)
    yuvtorgb_param->set_output_format(YuvToRgbParameter_OutputFormat_RGB0);
  else if (outshape == BGR0)
    yuvtorgb_param->set_output_format(YuvToRgbParameter_OutputFormat_BGR0);
  else
    yuvtorgb_param->set_output_format(YuvToRgbParameter_OutputFormat_ARGB);
  MLUYUVtoRGBLayer<Dtype> layer(layer_param);
  layer.SetUp(blob_bottom_vec_, blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(blob_bottom_vec_, blob_top_vec_);
  LayerSizeCmp();
  LayerTopCmp(outshape);
  OUTPUT("bottom", blob_bottom_->shape_string().c_str());
  EVENT_TIME(layer.get_event_time());
}
};

TYPED_TEST_CASE(MLUYUVtoRGBLayerTest, TestMLUDevices);

TYPED_TEST(MLUYUVtoRGBLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  shared_ptr<MLUYUVtoRGBLayer<Dtype> > layer(
      new MLUYUVtoRGBLayer<Dtype>(layer_param));

  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), 4);
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width());
}

TYPED_TEST(MLUYUVtoRGBLayerTest, TestForwardNV12RGB0) {
  this->TestForwardTtoT(0, 0);
}

TYPED_TEST(MLUYUVtoRGBLayerTest, TestForwardNV12BGR0) {
  this->TestForwardTtoT(0, 1);
}

TYPED_TEST(MLUYUVtoRGBLayerTest, TestForwardNV12ARGB) {
  this->TestForwardTtoT(0, 2);
}

TYPED_TEST(MLUYUVtoRGBLayerTest, TestForwardNV21RGB0) {
  this->TestForwardTtoT(1, 0);
}

TYPED_TEST(MLUYUVtoRGBLayerTest, TestForwardNV21BGR0) {
  this->TestForwardTtoT(1, 1);
}

TYPED_TEST(MLUYUVtoRGBLayerTest, TestForwardNV21ARGB) {
  this->TestForwardTtoT(1, 2);
}

template <typename TypeParam>
class MFUSYUVtoRGBLayerTest : public MFUSDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  int Height, Width, Nums;  // Height and Width have to be in the first line
  typedef unsigned char TType;
  // char for Uint8 , float for Normal Float TestType
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_1_;
  Blob<Dtype>* const blob_top_;
  // blob set and compare.
  bool data_generated_;
  TType* yuv420spNV21_;
  TType* yuv420spNV12_;
  float* random_jpg_;
  TType* result_jpg_;  // Calc back from YUV, use this to compare result.

  enum intype { NV12 = 0, NV21 };
  enum outtype { RGB0 = 0, BGR0, ARGB };

    MFUSYUVtoRGBLayerTest()
        : Height(4), Width(4), Nums(1),  // Set size here
                        // H & W is JPG size, must be even, n > 0
          blob_bottom_(new Blob<Dtype>(this->Nums, 1,
                                       this->Height, this->Width)),
          blob_bottom_1_(new Blob<Dtype>(this->Nums, 1, this->Height/2, this->Width)),
          blob_top_(new Blob<Dtype>()), data_generated_(false),
          yuv420spNV21_(nullptr), yuv420spNV12_(nullptr), random_jpg_(nullptr),
          result_jpg_(nullptr) {}

    virtual ~MFUSYUVtoRGBLayerTest() {
      delete blob_bottom_;
      delete blob_bottom_1_;
      delete blob_top_;
      clear();
    }
    void clear() {  // clear all ptrs
      if (yuv420spNV21_) delete yuv420spNV21_;
      if (yuv420spNV12_) delete yuv420spNV12_;
      if (random_jpg_) delete random_jpg_;
      if (result_jpg_) delete result_jpg_;
    }
/*
  dataGenerator only generate the same size image once.
*/
    void dataGenerator() {
      if (data_generated_) return;
      data_generated_ = true;
      srand(Height * Width % 256 + 1);

    int coordinate;
    clear();
    random_jpg_ = new float[Height * Width * 3];
    result_jpg_ = new TType[Height * Width * 3];
    for (int c_ = 0; c_ < 3; c_++)
      for (int h_ = 0; h_ < Height; h_++)
        for (int w_ = 0; w_ < Width; w_++)
          random_jpg_[(c_ * Height + h_) * Width + w_] =
              static_cast<float>(rand() % 256);  //  NOLINT thread safe OK

    yuv420spNV21_ = new TType[Height * Width / 2 * 3];
    yuv420spNV12_ = new TType[Height * Width / 2 * 3];
    for (int h_ = 0; h_ < Height / 2; h_++)
      for (int w_ = 0; w_ < Width / 2; w_++) {
        float Cr, Cb, avg_r = 0, avg_g = 0, avg_b = 0;
        for (int hh = 0; hh < 2; hh++)
          for (int ww = 0; ww < 2; ww++) {
            float r, g, b, Y;
            coordinate = (h_ * 2 + hh) * Width + (w_ * 2 + ww);
            r = random_jpg_[coordinate + 0 * Width * Height];
            g = random_jpg_[coordinate + 1 * Width * Height];
            b = random_jpg_[coordinate + 2 * Width * Height];
            avg_r += r;
            avg_g += g;
            avg_b += b;
            Y = 0.257 * r + 0.504 * g + 0.098 * b + 16;
            yuv420spNV21_[coordinate] = static_cast<TType>(static_cast<int>(Y));
            yuv420spNV12_[coordinate] = static_cast<TType>(static_cast<int>(Y));
          }

        avg_r /= 4;
        avg_g /= 4;
        avg_b /= 4;
        Cb = -0.148 * avg_r - 0.291 * avg_g + 0.439 * avg_b + 128;
        Cr = 0.439 * avg_r - 0.368 * avg_g - 0.071 * avg_b + 128;
        Cb = static_cast<int>(Cb);
        Cr = static_cast<int>(Cr);
        coordinate = (h_ + Height) * Width + w_ * 2;
        /*
           YYYY
           YYYY
           YYYY
           YYYY  Height * Width
           UVUV  U at: (Height + h_) + w_ *2;
           UVUV
           in NV12 U=Cb V=Cr
           in NV21 U=Cr V=Cb
        */
        yuv420spNV12_[coordinate] = static_cast<TType>(Cr);
        yuv420spNV12_[coordinate + 1] = static_cast<TType>(Cb);
        yuv420spNV21_[coordinate] = static_cast<TType>(Cb);
        yuv420spNV21_[coordinate + 1] = static_cast<TType>(Cr);
      }
    // End 4 for

      // Calc rgb back from YUV.
      for (int h_ = 0; h_ < Height; h_ ++)
        for (int w_ = 0; w_ < Width; w_ ++) {
          float Y, Cr, Cb, r, g, b;
          Y = yuv420spNV12_[h_ * Width + w_];
          Cb = yuv420spNV12_[(Height + h_ / 2) * Width + w_ / 2 * 2];
          Cr = yuv420spNV12_[(Height + h_ / 2) * Width + w_ / 2 * 2 + 1];
          coordinate = h_ * Width + w_;
          r = Y * 1.164 + Cr * 1.596 - 222.912;
          g = Y * 1.164 - Cb * 0.392 - Cr * 0.813 + 135.616;
          b = Y * 1.164 + Cb * 2.017 - 276.8;
          r = r < 0 ? 0 : (r > 255 ? 255 : r);
          g = g < 0 ? 0 : (g > 255 ? 255 : g);
          b = b < 0 ? 0 : (b > 255 ? 255 : b);
          result_jpg_[coordinate + 0 * Height * Width] = static_cast<TType>(
              static_cast<int>(r));
          result_jpg_[coordinate + 1 * Height * Width] = static_cast<TType>(
              static_cast<int>(g));
          result_jpg_[coordinate + 2 * Height * Width] = static_cast<TType>(
              static_cast<int>(b));
        }
    }
  //  End of datagenerator
  void LayerBottomSet(intype shape) {
    for (int n_ = 0; n_ < Nums; n_++)
      for (int cood = 0; cood < Height * Width / 2 * 3; cood++)
        if (cood < Height * Width) {
            reinterpret_cast<TType*>(
            blob_bottom_->mutable_cpu_data())[n_ * Height * Width + cood]
            = (shape == NV12) ? yuv420spNV12_[cood] : yuv420spNV21_[cood];
        } else {
            reinterpret_cast<TType*>(
            blob_bottom_1_->mutable_cpu_data())[n_ * Height * Width / 2
            + cood - (Height * Width)]
            = (shape == NV12) ? yuv420spNV12_[cood] : yuv420spNV21_[cood];
       }
  }

  int ResultAt(outtype shape, int coodinate, int channel) {
    if (shape == RGB0) {
      if (channel == 3)
        return 0;
      else
        return result_jpg_[channel * Height * Width + coodinate];
    } else if (shape == BGR0) {
      if (channel == 3)
        return 0;
      else
        return result_jpg_[(2 - channel) * Height * Width + coodinate];
    } else if (shape == ARGB) {
      if (channel == 0)
        return 0;
      else
        return result_jpg_[(channel - 1) * Height * Width + coodinate];
    }
    return -1;
  }
  void LayerTopCmp(outtype shape) {
    float deviation = 2;
    for (int n_ = 0; n_ < Nums; n_++) {
      for (int c_ = 0; c_ < 4; c_++) {
        for (int cood = 0; cood < Height * Width; cood++) {
            float Output = static_cast<float>(reinterpret_cast<const TType*>(
            blob_top_->cpu_data())[(n_ * 4) * Height * Width + cood * 4 + c_]);
          float RightAns = ResultAt(shape, cood, c_);
          EXPECT_NEAR(Output, RightAns, deviation) << "  Where  N=" << n_
            << " C=" << c_ << " H=" << cood / Width << " W=" << cood % Width;
         }
      }
    }
  }

  void LayerSizeCmp() {
    EXPECT_EQ(blob_top_->num(), Nums);
    EXPECT_EQ(blob_top_->channels(), 4);
    EXPECT_EQ(blob_top_->height(), Height);
    EXPECT_EQ(blob_top_->width(), Width);
  }
  virtual void SetUp() {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_bottom_1_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_1_);
    blob_top_vec_.push_back(blob_top_);
  }

void TestForwardTtoT(int a, int b) {
  intype inshape; outtype outshape;
  if (a == 0)
    inshape = NV12;
  else
    inshape = NV21;
  if (b == 0)
    outshape = RGB0;
  else if (b == 1)
    outshape = BGR0;
  else
    outshape = ARGB;
  LayerParameter layer_param;
  YuvToRgbParameter* yuvtorgb_param = layer_param.mutable_yuvtorgb_param();
  int channels = 1;
  dataGenerator();
  blob_bottom_->Reshape(Nums, channels, Height, Width);
  blob_bottom_1_->Reshape(Nums, channels, Height / 2, Width);
  blob_top_->Reshape(Nums, 4, Height, Width);
  LayerBottomSet(inshape);
  if (inshape == NV12)
    yuvtorgb_param->set_input_format(YuvToRgbParameter_InputFormat_YUV420SP_NV12);
  else
    yuvtorgb_param->set_input_format(YuvToRgbParameter_InputFormat_YUV420SP_NV21);
  if (outshape == RGB0)
    yuvtorgb_param->set_output_format(YuvToRgbParameter_OutputFormat_RGB0);
  else if (outshape == BGR0)
    yuvtorgb_param->set_output_format(YuvToRgbParameter_OutputFormat_BGR0);
  else
    yuvtorgb_param->set_output_format(YuvToRgbParameter_OutputFormat_ARGB);
  MLUYUVtoRGBLayer<Dtype> layer(layer_param);
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
  LayerSizeCmp();
  LayerTopCmp(outshape);
  OUTPUT("bottom", blob_bottom_->shape_string().c_str());
  EVENT_TIME(fuser.get_event_time());
}
};

TYPED_TEST_CASE(MFUSYUVtoRGBLayerTest, TestMFUSDevices);

TYPED_TEST(MFUSYUVtoRGBLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  shared_ptr<MLUYUVtoRGBLayer<Dtype> > layer(
      new MLUYUVtoRGBLayer<Dtype>(layer_param));

  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), 4);
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width());
}

TYPED_TEST(MFUSYUVtoRGBLayerTest, TestForwardNV12RGB0) {
  this->TestForwardTtoT(0, 0);
}

TYPED_TEST(MFUSYUVtoRGBLayerTest, TestForwardNV12BGR0) {
  this->TestForwardTtoT(0, 1);
}

TYPED_TEST(MFUSYUVtoRGBLayerTest, TestForwardNV12ARGB) {
  this->TestForwardTtoT(0, 2);
}

TYPED_TEST(MFUSYUVtoRGBLayerTest, TestForwardNV21RGB0) {
  this->TestForwardTtoT(1, 0);
}

TYPED_TEST(MFUSYUVtoRGBLayerTest, TestForwardNV21BGR0) {
  this->TestForwardTtoT(1, 1);
}

TYPED_TEST(MFUSYUVtoRGBLayerTest, TestForwardNV21ARGB) {
  this->TestForwardTtoT(1, 2);
}
#endif
}  // namespace caffe
