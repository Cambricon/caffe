/*
All modification made by Cambricon Corporation: © 2018--2019 Cambricon Corporation
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

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <sys/time.h>
#include <time.h>
#include <cstdlib>
#include <memory>
#include <vector>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "gtest/gtest.h"
#ifdef USE_MLU
#include "caffe/layers/mlu_resizeconvert_layer.hpp"
#endif

#include "caffe/filler.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

const int PAD_SIZE = 64;
const int CI  = 64;
const int MULTCI = 64;
const int CO = 256;
#define PAD_UP(x, y) (x / y + static_cast<int>((x) % y > 0)) * y
#define PAD_DN(x, y) (x / y) * y


namespace caffe {

#ifdef USE_MLU
template <typename TypeParam>
class MLUResizeconvertLayerTest : public MLUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MLUResizeconvertLayerTest()
    : blob_Y_ptrs_(new Blob<Dtype>(16, 2, 1, 1, DT_INT32, DT_INT32, CNML_TENSOR)),
      blob_UV_ptrs_(new Blob<Dtype>(16, 2, 1, 1, DT_INT32, DT_INT32, CNML_TENSOR)),
      blob_size_(new Blob<Dtype>(1, 4, 1, 1, DT_INT32, DT_INT32, CNML_TENSOR)),
      blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    caffe::Caffe::setBatchsize(16);
    caffe::Caffe::setCoreNumber(16);
    caffe::Caffe::setSimpleFlag(true);
    blob_bottom_vec_.clear();
    blob_top_vec_.clear();
    blob_bottom_vec_.push_back(blob_Y_ptrs_);
    blob_bottom_vec_.push_back(blob_UV_ptrs_);
    blob_bottom_vec_.push_back(blob_size_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MLUResizeconvertLayerTest() {
    delete blob_Y_ptrs_;
    delete blob_UV_ptrs_;
    delete blob_size_;
    delete blob_top_;
  }
  void TestForward(int resize_h, int resize_w, int s_col, int s_row) {
    LayerParameter layer_param;
    ResizeConvertParameter* resize_param =
                         layer_param.mutable_resize_convert_param();
    resize_param->set_resize_h(resize_h);
    resize_param->set_resize_w(resize_w);
    MLUResizeConvertLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(this->blob_top_->num(), 16);
    EXPECT_EQ(this->blob_top_->channels(), 4);
    EXPECT_EQ(this->blob_top_->height(), 224);
    EXPECT_EQ(this->blob_top_->width(), 224);
    layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  }
  Blob<Dtype>* const blob_Y_ptrs_;
  Blob<Dtype>* const blob_UV_ptrs_;
  Blob<Dtype>* const blob_size_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MLUResizeconvertLayerTest, TestMLUDevices);

TYPED_TEST(MLUResizeconvertLayerTest, TestForward) {
  int resize_h = 224;
  int resize_w = 224;
  int channel = 3;
  // read and process real image data
  // Modify this part
  int batchNum = 16;
  int s_col = 1920;
  int s_row = 1080;
  int d_col = 224;
  int d_row = 224;
  int roi_x = 0;
  int roi_y = 0;
  int roi_w = s_col - roi_x;
  int roi_h = s_row - roi_y;
  int color_mode = YUV_TO_RGBA_NV12;

  int layerIn = 3;
  int d_col_pad = PAD_UP((d_col * 4), PAD_SIZE) / 4;
  LOG(INFO) << "(" << roi_w << ", " << roi_h << ") ==> (" << d_col << ", "
        << d_row << ")\n";
  LOG(INFO) << "The real output d_col is: " << d_col_pad << "\n";
  int d_len_cpu = d_row * d_col * 3;

  cv::Rect roi;
  roi.x = roi_x;
  roi.y = roi_y;
  roi.width = roi_w;
  roi.height = roi_h;

  /*------------------------------ PREPARE DATA -------------------------------*/
  // [ture image or random values] in BGR
  std::string frameName = string(TEST_SOURCE_DIR())+"demo.jpg";
  LOG(INFO) << "framePath is " << frameName << "\n";
  cv::Mat img = cv::imread(frameName, CV_LOAD_IMAGE_COLOR);
  unsigned char* srcYs = reinterpret_cast<unsigned char*>(malloc(
                           batchNum * s_col * s_row));
  unsigned char* srcUVs = reinterpret_cast<unsigned char*>(malloc(
                           batchNum * s_col * s_row / 2));
  unsigned char* srcYUVs = reinterpret_cast<unsigned char*>(malloc(
                           batchNum * s_col * s_row * 3 / 2));
  unsigned char* cvResults = reinterpret_cast<unsigned char*>(malloc(
                           batchNum * d_col * d_row * 4));
  char* srcYUV = reinterpret_cast<char*>(malloc(
                           s_col * s_row * 3 / 2* sizeof(char)));
  unsigned char *dstf = reinterpret_cast<unsigned char*>(malloc(
                           d_len_cpu * sizeof(char)));

  for (int batchId = 0; batchId < batchNum; batchId++) {
    // src img
    cv::Mat srcImg =
      cv::Mat(cv::Size(s_col, s_row), img.type(), cv::Scalar::all(0));
    cv::resize(img, srcImg, srcImg.size());

    // RGB data
    cv::Mat imgRGB;
    imgRGB.convertTo(imgRGB, CV_8UC3, 1.f);
    cvtColor(srcImg, imgRGB, CV_BGR2RGB);

    // YUV420_I420 data
    cv::Mat imgYUV;
    cv::Mat imgYUV420sp2RGB;
    imgYUV.convertTo(imgYUV, CV_8UC3, 1.f);
    cvtColor(imgRGB, imgYUV, CV_RGB2YUV_I420);

    // YUV_I420 -> YUV420sp_NV12
    char* srcU = reinterpret_cast<char *>(imgYUV.data) + s_col * s_row;
    char* srcV = srcU + s_col * s_row / 4;
    char* srcUV = srcYUV + s_col * s_row;
    memcpy(srcYUV, reinterpret_cast<char *>(imgYUV.data), s_col * s_row);
    int CV_MODE = CV_YUV2RGB_NV12;
    for (int i = 0; i < s_col * s_row / 4; i++) {
      if (color_mode % 2) {
        (*srcUV++) = (*srcU++);
        (*srcUV++) = (*srcV++);
      } else {
        (*srcUV++) = (*srcV++);
        (*srcUV++) = (*srcU++);
        CV_MODE = CV_YUV2RGB_NV21;
      }
    }
    imgYUV.data = (unsigned char*)srcYUV;

    // stack input datai
    memcpy(srcYs + batchId * s_col * s_row,
       srcYUV,
       s_col * s_row);
    memcpy(srcUVs + batchId * s_col * s_row / 2,
       srcYUV + s_col * s_row,
       s_col * s_row / 2);
    memcpy(srcYUVs + batchId * s_col * s_row * 3 / 2,
       srcYUV,
       s_col * s_row * 3 / 2);

    cvtColor(imgYUV, imgYUV420sp2RGB, CV_MODE);

    // opencv result
    srcImg = srcImg(roi);
    srcImg.convertTo(srcImg, CV_32FC3, 1);
    cv::Mat imgd =
      cv::Mat(cv::Size(d_col, d_row), srcImg.type(), cv::Scalar::all(0));
    cv::Mat imgcv =
      cv::Mat(cv::Size(d_col, d_row), imgRGB.type(), cv::Scalar::all(0));
    if (layerIn == 1) {
      imgRGB = imgRGB(roi);
      cv::resize(imgRGB, imgcv, imgcv.size());
    } else {
      imgYUV420sp2RGB = imgYUV420sp2RGB(roi);
      cv::resize(imgYUV420sp2RGB, imgcv, imgcv.size());
    }
    memcpy(cvResults + batchId * d_row * d_col * channel,
       reinterpret_cast<char*>(imgcv.data), d_col * d_row * channel);
  }
  /*----------------------------- ResizeAndConvert ------------------------------*/
    int* size = reinterpret_cast<int *>(this->blob_size_->mutable_cpu_data());
    size[0] = 1080;
    size[1] = 1920;
    size[2] = 1080;
    size[3] = 1920;
    int height = size[0];
    int width  = size[1];

    void* mlutensor_input_ptrs[2];
    mlutensor_input_ptrs[0] = reinterpret_cast<void *>(
                              this->blob_Y_ptrs_->mutable_mlu_data());
    mlutensor_input_ptrs[1] = reinterpret_cast<void *>(
                              this->blob_UV_ptrs_->mutable_mlu_data());

    void **inputY_addrs_cpu = reinterpret_cast<void **>(malloc(
                                     sizeof(void *) * 16));
    void **inputUV_addrs_cpu = reinterpret_cast<void **>(malloc(
                                     sizeof(void *) * 16));
    for (int i = 0; i < 16; i++) {
      CNRT_CHECK(cnrtMalloc(&inputY_addrs_cpu[i], height * width));
      CNRT_CHECK(cnrtMalloc(&inputUV_addrs_cpu[i], height * width / 2));
    }
    for (int i = 0; i < 16; i++) {
      cnrtMemcpy(inputY_addrs_cpu[i], srcYs + i * s_col * s_row,
                 height * width, CNRT_MEM_TRANS_DIR_HOST2DEV);
      cnrtMemcpy(inputUV_addrs_cpu[i], srcUVs + i * s_row * s_col / 2 - 1,
                 height * width / 2, CNRT_MEM_TRANS_DIR_HOST2DEV);
    }

    cnrtMemcpy(mlutensor_input_ptrs[0], inputY_addrs_cpu, 16 * sizeof(void*),
                    CNRT_MEM_TRANS_DIR_HOST2DEV);
    cnrtMemcpy(mlutensor_input_ptrs[1], inputUV_addrs_cpu, 16 * sizeof(void*),
                    CNRT_MEM_TRANS_DIR_HOST2DEV);

    this->TestForward(resize_h, resize_w, s_col, s_row);

    uint8_t* dst = reinterpret_cast<uint8_t*>(this->blob_top_->mutable_cpu_data());
    /*---------------------------------- MSE -----------------------------------*/
    for (int batchId = 0; batchId < batchNum; batchId++) {
      LOG(INFO) << "\n===== Check result for batch " << batchId <<" \n";
      // compare and compute diff
      bool flag = true;
      float thres = 0.02;
      float diff = 0.0;
      float diffSum = 0.0;
      float max  = 0.0;
      float mae  = 0.0;
      float ma   = 0.0;
      unsigned char *mlu_ptr = (unsigned char *)dst + batchId * d_col_pad * d_row * 4;
      unsigned char *cpu_ptr = cvResults + batchId * d_col * d_row * 3;
      int channelOut = 4;

      for (int i = 0; i < d_row; i++) {
        for (int j = 0; j < d_col; j++) {
          for (int k = 0; k < channel; k++) {
            int rgbIdx = (((color_mode - 1) / 2) == 0) * (k) +
                         (((color_mode - 1) / 2) == 1) * (2 - k) +
                         (((color_mode - 1) / 2) == 2) * (k + 1) +
                         (((color_mode - 1) / 2) == 3) * (3 - k);
            int cpu_idx = i * d_col * channel + j * channel + k;
            int mlu_idx = i * d_col_pad * channelOut + j * channelOut + rgbIdx;
            diff = static_cast<float>(mlu_ptr[mlu_idx])
                 - static_cast<float>(cpu_ptr[cpu_idx]) + 0;
            ma += static_cast<float>(cpu_ptr[cpu_idx]);

            if (std::abs(diff) > max) max = std::abs(diff);
            mae += std::abs(diff);
            diffSum += diff;
            dstf[i * d_col * channel + j * channel + k] =
                mlu_ptr[mlu_idx]
              - cpu_ptr[cpu_idx] * 0;
          }
        }
      }
      mae /= ma;
      LOG(INFO) << "max diff: " << max << "\n";
      LOG(INFO) << "sum diff: " << diffSum << "\n";
      if ( mae > thres ) {
        flag = false;
      }
      LOG(INFO) << flag ? "PASSED!\n" : "FAILED!\n";
      LOG(INFO) << "diff1: " << mae*100 << "%" << "\n";
      EXPECT_LE(mae, 2e-2);
    }
    free(srcYs);
    free(srcUVs);
    free(srcYUVs);
    free(cvResults);
}
template <typename TypeParam>
class MFUSResizeconvertLayerTest : public MFUSDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MFUSResizeconvertLayerTest()
    : blob_Y_ptrs_(new Blob<Dtype>(16, 2, 1, 1, DT_INT32, DT_INT32, CNML_TENSOR)),
      blob_UV_ptrs_(new Blob<Dtype>(16, 2, 1, 1, DT_INT32, DT_INT32, CNML_TENSOR)),
      blob_size_(new Blob<Dtype>(1, 2, 1, 1, DT_INT32, DT_INT32, CNML_TENSOR)),
      blob_top_(new Blob<Dtype>()) {}
  void SetUp() {
    blob_bottom_vec_.clear();
    blob_top_vec_.clear();
    blob_bottom_vec_.push_back(blob_Y_ptrs_);
    blob_bottom_vec_.push_back(blob_UV_ptrs_);
    blob_bottom_vec_.push_back(blob_size_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MFUSResizeconvertLayerTest() {
    delete blob_Y_ptrs_;
    delete blob_UV_ptrs_;
    delete blob_size_;
    delete blob_top_;
  }
  void TestForward(int resize_h, int resize_w, int s_col, int s_row) {
    caffe::Caffe::setBatchsize(16);
    caffe::Caffe::setCoreNumber(16);
    caffe::Caffe::setSimpleFlag(true);
    LayerParameter layer_param;
    ResizeConvertParameter* resize_param =
                         layer_param.mutable_resize_convert_param();
    resize_param->set_resize_h(resize_h);
    resize_param->set_resize_w(resize_w);
    MLUResizeConvertLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(this->blob_top_->num(), 16);
    EXPECT_EQ(this->blob_top_->channels(), 4);
    EXPECT_EQ(this->blob_top_->height(), 224);
    EXPECT_EQ(this->blob_top_->width(), 224);
    ASSERT_TRUE(layer.mfus_supported());
    MFusion<Dtype> fuser;
    fuser.reset();
    fuser.addInputs(this->blob_bottom_vec_);
    fuser.addOutputs(this->blob_top_vec_);
    layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.fuse(&fuser);
    fuser.compile();
    fuser.forward();
  }
  Blob<Dtype>* const blob_Y_ptrs_;
  Blob<Dtype>* const blob_UV_ptrs_;
  Blob<Dtype>* const blob_size_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MFUSResizeconvertLayerTest, TestMLUDevices);

TYPED_TEST(MFUSResizeconvertLayerTest, TestForward) {
  int resize_h = 224;
  int resize_w = 224;
  int channel = 3;
  // read and process real image data
  // Modify this part
  int batchNum = 16;
  int s_col = 1920;
  int s_row = 1080;
  int d_col = 224;
  int d_row = 224;
  int roi_x = 0;
  int roi_y = 0;
  int roi_w = s_col - roi_x;
  int roi_h = s_row - roi_y;
  int color_mode = YUV_TO_RGBA_NV12;

  int layerIn = 3;
  int channelOut = 4;
  int d_col_pad = PAD_UP((d_col * 4), PAD_SIZE) / 4;
  LOG(INFO) << "(" << roi_w << ", " << roi_h << ") ==> (" << d_col << ", "
        << d_row << ")\n";
  LOG(INFO) << "The real output d_col is: " << d_col_pad << "\n";
  int d_len_cpu = d_row * d_col * 3;

  cv::Rect roi;
  roi.x = roi_x;
  roi.y = roi_y;
  roi.width = roi_w;
  roi.height = roi_h;

  /*------------------------------ PREPARE DATA -------------------------------*/
  // [ture image or random values] in BGR
  std::string frameName = string(TEST_SOURCE_DIR())+"demo.jpg";
  LOG(INFO) << "framePath is " << frameName << "\n";
  cv::Mat img = cv::imread(frameName, CV_LOAD_IMAGE_COLOR);
  unsigned char* srcYs = reinterpret_cast<unsigned char*>(malloc(
                           batchNum * s_col * s_row));
  unsigned char* srcUVs = reinterpret_cast<unsigned char*>(malloc(
                           batchNum * s_col * s_row / 2));
  unsigned char* srcYUVs = reinterpret_cast<unsigned char*>(malloc(
                           batchNum * s_col * s_row * 3 / 2));
  unsigned char* cvResults = reinterpret_cast<unsigned char*>(malloc(
                           batchNum * d_col * d_row * 4));
  char* srcYUV = reinterpret_cast<char*>(malloc(
                           s_col * s_row * 3 / 2* sizeof(char)));
  unsigned char *dstf = reinterpret_cast<unsigned char*>(malloc(
                           d_len_cpu * sizeof(char)));
  channel = 3;

  for (int batchId = 0; batchId < batchNum; batchId++) {
    // src img
    cv::Mat srcImg =
      cv::Mat(cv::Size(s_col, s_row), img.type(), cv::Scalar::all(0));
    cv::resize(img, srcImg, srcImg.size());

    // RGB data
    cv::Mat imgRGB;
    imgRGB.convertTo(imgRGB, CV_8UC3, 1.f);
    cvtColor(srcImg, imgRGB, CV_BGR2RGB);

    // YUV420_I420 data
    cv::Mat imgYUV;
    cv::Mat imgYUV420sp2RGB;
    imgYUV.convertTo(imgYUV, CV_8UC3, 1.f);
    cvtColor(imgRGB, imgYUV, CV_RGB2YUV_I420);

    // YUV_I420 -> YUV420sp_NV12
    char* srcU = reinterpret_cast<char *>(imgYUV.data) + s_col * s_row;
    char* srcV = srcU + s_col * s_row / 4;
    char* srcUV = srcYUV + s_col * s_row;
    memcpy(srcYUV, reinterpret_cast<char *>(imgYUV.data), s_col * s_row);
    int CV_MODE = CV_YUV2RGB_NV12;
    for (int i = 0; i < s_col * s_row / 4; i++) {
      if (color_mode % 2) {
        (*srcUV++) = (*srcU++);
        (*srcUV++) = (*srcV++);
      } else {
        (*srcUV++) = (*srcV++);
        (*srcUV++) = (*srcU++);
        CV_MODE = CV_YUV2RGB_NV21;
      }
    }
    imgYUV.data = (unsigned char*)srcYUV;

    // stack input datai
    memcpy(srcYs + batchId * s_col * s_row,
       srcYUV,
       s_col * s_row);
    memcpy(srcUVs + batchId * s_col * s_row / 2,
       srcYUV + s_col * s_row,
       s_col * s_row / 2);
    memcpy(srcYUVs + batchId * s_col * s_row * 3 / 2,
       srcYUV,
       s_col * s_row * 3 / 2);

    cvtColor(imgYUV, imgYUV420sp2RGB, CV_MODE);

    // opencv result
    srcImg = srcImg(roi);
    srcImg.convertTo(srcImg, CV_32FC3, 1);
    cv::Mat imgd =
      cv::Mat(cv::Size(d_col, d_row), srcImg.type(), cv::Scalar::all(0));
    cv::Mat imgcv =
      cv::Mat(cv::Size(d_col, d_row), imgRGB.type(), cv::Scalar::all(0));
    if (layerIn == 1) {
      imgRGB = imgRGB(roi);
      cv::resize(imgRGB, imgcv, imgcv.size());
    } else {
      imgYUV420sp2RGB = imgYUV420sp2RGB(roi);
      cv::resize(imgYUV420sp2RGB, imgcv, imgcv.size());
    }
    memcpy(cvResults + batchId * d_row * d_col * channel,
       reinterpret_cast<char*>(imgcv.data), d_col * d_row * channel);
  }
  /*----------------------------- ResizeAndConvert ------------------------------*/
    int* size = reinterpret_cast<int *>(this->blob_size_->mutable_cpu_data());
    size[0] = 1080;
    size[1] = 1920;
    size[2] = 1080;
    size[3] = 1920;
    int height = size[0];
    int width  = size[1];

    void* mlutensor_input_ptrs[2];
    mlutensor_input_ptrs[0] = reinterpret_cast<void *>(
                              this->blob_Y_ptrs_->mutable_mlu_data());
    mlutensor_input_ptrs[1] = reinterpret_cast<void *>(
                              this->blob_UV_ptrs_->mutable_mlu_data());

    void **inputY_addrs_cpu = reinterpret_cast<void **>(malloc(
                                     sizeof(void *) * 16));
    void **inputUV_addrs_cpu = reinterpret_cast<void **>(malloc(
                                     sizeof(void *) * 16));
    for (int i = 0; i < 16; i++) {
      CNRT_CHECK(cnrtMalloc(&inputY_addrs_cpu[i], height * width));
      CNRT_CHECK(cnrtMalloc(&inputUV_addrs_cpu[i], height * width / 2));
    }
    for (int i = 0; i < 16; i++) {
      cnrtMemcpy(inputY_addrs_cpu[i], srcYs + i * s_col * s_row,
                 height * width, CNRT_MEM_TRANS_DIR_HOST2DEV);
      cnrtMemcpy(inputUV_addrs_cpu[i], srcUVs + i * s_row * s_col / 2 - 1,
                 height * width / 2, CNRT_MEM_TRANS_DIR_HOST2DEV);
    }

    cnrtMemcpy(mlutensor_input_ptrs[0], inputY_addrs_cpu, 16 * sizeof(void*),
                    CNRT_MEM_TRANS_DIR_HOST2DEV);
    cnrtMemcpy(mlutensor_input_ptrs[1], inputUV_addrs_cpu, 16 * sizeof(void*),
                    CNRT_MEM_TRANS_DIR_HOST2DEV);

    this->TestForward(resize_h, resize_w, s_col, s_row);

    uint8_t* dst = reinterpret_cast<uint8_t*>(this->blob_top_->mutable_cpu_data());
    /*---------------------------------- MSE -----------------------------------*/
    for (int batchId = 0; batchId < batchNum; batchId++) {
      LOG(INFO) << "\n===== Check result for batch " << batchId <<" \n";
      // compare and compute diff
      bool flag = true;
      float thres = 0.02;
      float diff = 0.0;
      float diffSum = 0.0;
      float max  = 0.0;
      float mae  = 0.0;
      float ma   = 0.0;
      unsigned char *mlu_ptr = (unsigned char *)dst + batchId * d_col_pad * d_row * 4;
      unsigned char *cpu_ptr = cvResults + batchId * d_col * d_row * 3;

      for (int i = 0; i < d_row; i++) {
        for (int j = 0; j < d_col; j++) {
          for (int k = 0; k < channel; k++) {
            int rgbIdx = (((color_mode - 1) / 2) == 0) * (k) +
                         (((color_mode - 1) / 2) == 1) * (2 - k) +
                         (((color_mode - 1) / 2) == 2) * (k + 1) +
                         (((color_mode - 1) / 2) == 3) * (3 - k);
            int cpu_idx = i * d_col * channel + j * channel + k;
            int mlu_idx = i * d_col_pad * channelOut + j * channelOut + rgbIdx;
            diff = static_cast<float>(mlu_ptr[mlu_idx])
                 - static_cast<float>(cpu_ptr[cpu_idx]) + 0;
            ma += static_cast<float>(cpu_ptr[cpu_idx]);

            if (std::abs(diff) > max) max = std::abs(diff);
            mae += std::abs(diff);
            diffSum += diff;
            dstf[i * d_col * channel + j * channel + k] =
                mlu_ptr[mlu_idx]
              - cpu_ptr[cpu_idx] * 0;
          }
        }
      }
      mae /= ma;
      LOG(INFO) << "max diff: " << max << "\n";
      LOG(INFO) << "sum diff: " << diffSum << "\n";
      if ( mae > thres ) {
        flag = false;
      }
      LOG(INFO) << flag ? "PASSED!\n" : "FAILED!\n";
      LOG(INFO) << "diff1: " << mae*100 << "%" << "\n";
      EXPECT_LE(mae, 2e-2);
    }

  free(srcYs);
  free(srcUVs);
  free(srcYUVs);
  free(cvResults);
}
#endif
}  // namespace caffe
