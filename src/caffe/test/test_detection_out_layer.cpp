

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

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/detection_out_layer.hpp"
#include "caffe/layers/mlu_detection_out_layer.hpp"
#include "caffe/test/test_detection_out_data.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "gtest/gtest.h"


namespace caffe {

template <typename TypeParam>
class DetectionOutLayerTest : public CPUDeviceTest<TypeParam> {
typedef typename TypeParam::Dtype Dtype;

  protected:
  DetectionOutLayerTest()
      : blob_bottom_(new Blob<Dtype>(1, 125, 13, 13)),
         blob_top_(new Blob<Dtype>()) {}
     void SetUp() {
       Dtype* input_data = this->blob_bottom_->mutable_cpu_data();
       for (int i = 0; i < this->blob_bottom_->count(); ++i) {
         input_data[i] = detection_out_data::input_data[i];
       }
       blob_bottom_vec_.push_back(blob_bottom_);
       blob_top_vec_.push_back(blob_top_);
     }
     virtual ~DetectionOutLayerTest() {
       delete blob_bottom_;
       delete blob_top_;
     }

    Blob<Dtype>* const blob_bottom_;
    Blob<Dtype>* const blob_top_;
    vector<Blob<Dtype>*> blob_bottom_vec_;
    vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(DetectionOutLayerTest, TestDtypesAndDevices);

TYPED_TEST(DetectionOutLayerTest, TestForwardDetection) {
  typedef typename TypeParam::Dtype Dtype;

  LayerParameter layer_param;
  DetectionOutParameter* param = layer_param.mutable_detection_out_param();
  param->set_side(13);
  param->set_num_classes(20);
  param->set_num_box(5);
  param->set_coords(4);
  param->set_confidence_threshold(0.5);
  param->set_nms_threshold(0.45);
  param->add_biases(1.3221);
  param->add_biases(1.73145);
  param->add_biases(3.19275);
  param->add_biases(4.00944);
  param->add_biases(5.05587);
  param->add_biases(8.09892);
  param->add_biases(9.47112);
  param->add_biases(4.84053);
  param->add_biases(11.2364);
  param->add_biases(10.0071);
  DetectionOutLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 7);

  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_top_->count(); i++) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i],
        detection_out_data::output_data[i], 1e-5);
  }
}

#ifdef USE_MLU
// MLU
template <typename TypeParam>
class MLUDetectionOutLayerTest : public MLUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MLUDetectionOutLayerTest()
      : blob_bottom_(new Blob<Dtype>(1, 125, 13, 13)),
          blob_top_(new Blob<Dtype>()) {}

     void SetUp() {
       Dtype* input_data = this->blob_bottom_->mutable_cpu_data();
       for (int i = 0; i < this->blob_bottom_->count(); ++i) {
         input_data[i] = detection_out_data::input_data[i];
       }
       blob_bottom_vec_.push_back(blob_bottom_);
       blob_top_vec_.push_back(blob_top_);
     }
     virtual ~MLUDetectionOutLayerTest() {
       delete blob_bottom_;
       delete blob_top_;
     }

    Blob<Dtype>* const blob_bottom_;
    Blob<Dtype>* const blob_top_;
    vector<Blob<Dtype>*> blob_bottom_vec_;
    vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MLUDetectionOutLayerTest, TestMLUDevices);

TYPED_TEST(MLUDetectionOutLayerTest, TestForwardDetection) {
  typedef typename TypeParam::Dtype Dtype;

  LayerParameter layer_param;
  DetectionOutParameter* param = layer_param.mutable_detection_out_param();
  param->set_side(13);
  param->set_num_classes(20);
  param->set_num_box(5);
  param->set_coords(4);
  param->set_confidence_threshold(0.5);
  param->set_nms_threshold(0.45);
  param->add_biases(1.3221);
  param->add_biases(1.73145);
  param->add_biases(3.19275);
  param->add_biases(4.00944);
  param->add_biases(5.05587);
  param->add_biases(8.09892);
  param->add_biases(9.47112);
  param->add_biases(4.84053);
  param->add_biases(11.2364);
  param->add_biases(10.0071);

  MLUDetectionOutLayer<Dtype> layer(layer_param);
  layer.set_int8_context(0);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 256);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 7);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  // sort out the output and prepare for the final result
  int iOutputCount = 0;
  for (int i = 0; i < this->blob_top_->count(); i++) {
    if (this->blob_top_vec_[0]->mutable_cpu_data()[i] == -1.0) break;
    iOutputCount++;
  }
  // sort
  for (int i = 0; i < iOutputCount; i++)
    for (int j = 0; j < iOutputCount; j++) {
      float fSwap = 0.0;
      float temp1 = this->blob_top_vec_[0]->mutable_cpu_data()[i];
      float temp2 = detection_out_data::output_data[j];
      float fThreshold = 0.0;
      if (i%7 == 0 || i%7 == 1) fThreshold = 1e-6;  // check for index, hard matching
      else
        fThreshold = 1e-3;  // check for value, soft matching

      if (std::abs(temp1 - temp2) < fThreshold) {
        fSwap = this->blob_top_vec_[0]->mutable_cpu_data()[i];
        this->blob_top_vec_[0]->mutable_cpu_data()[i] =
                     this->blob_top_vec_[0]->mutable_cpu_data()[j];
        this->blob_top_vec_[0]->mutable_cpu_data()[j] = fSwap;
      }
    }

  for (int i = 0; i < iOutputCount; i++)
    EXPECT_NEAR(this->blob_top_->cpu_data()[i],
      detection_out_data::output_data[i], 5e-1);
}

// MFUS
template <typename TypeParam>
class MFUSDetectionOutLayerTest : public MFUSDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MFUSDetectionOutLayerTest()
      : blob_bottom_(new Blob<Dtype>(1, 125, 13, 13)),
          blob_top_(new Blob<Dtype>()) {}

     void SetUp() {
       Dtype* input_data = this->blob_bottom_->mutable_cpu_data();
       for (int i = 0; i < this->blob_bottom_->count(); ++i) {
         input_data[i] = detection_out_data::input_data[i];
       }
       blob_bottom_vec_.push_back(blob_bottom_);
       blob_top_vec_.push_back(blob_top_);
     }
     virtual ~MFUSDetectionOutLayerTest() {
       delete blob_bottom_;
       delete blob_top_;
     }

    Blob<Dtype>* const blob_bottom_;
    Blob<Dtype>* const blob_top_;
    vector<Blob<Dtype>*> blob_bottom_vec_;
    vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MFUSDetectionOutLayerTest, TestMFUSDevices);

TYPED_TEST(MFUSDetectionOutLayerTest, TestForwardDetection) {
  typedef typename TypeParam::Dtype Dtype;

  LayerParameter layer_param;
  DetectionOutParameter* param = layer_param.mutable_detection_out_param();
  param->set_side(13);
  param->set_num_classes(20);
  param->set_num_box(5);
  param->set_coords(4);
  param->set_confidence_threshold(0.5);
  param->set_nms_threshold(0.45);
  param->add_biases(1.3221);
  param->add_biases(1.73145);
  param->add_biases(3.19275);
  param->add_biases(4.00944);
  param->add_biases(5.05587);
  param->add_biases(8.09892);
  param->add_biases(9.47112);
  param->add_biases(4.84053);
  param->add_biases(11.2364);
  param->add_biases(10.0071);

  MLUDetectionOutLayer<Dtype> layer(layer_param);
  layer.set_int8_context(0);
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

  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 256);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 7);

  // sort out the output and prepare for the final result
  int iOutputCount = 0;
  for (int i = 0; i < this->blob_top_->count(); i++) {
    if (this->blob_top_vec_[0]->mutable_cpu_data()[i] == -1.0) break;
    iOutputCount++;
  }
  // sort
  for (int i = 0; i < iOutputCount; i++)
    for (int j = 0; j < iOutputCount; j++) {
      float fSwap = 0.0;
      float temp1 = this->blob_top_vec_[0]->mutable_cpu_data()[i];
      float temp2 = detection_out_data::output_data[j];
      float fThreshold = 0.0;
      if (i%7 == 0 || i%7 == 1) fThreshold = 1e-6;  // check for index, hard matching
      else
        fThreshold = 1e-3;  // check for value, soft matching

      if (std::abs(temp1 - temp2) < fThreshold) {
        fSwap = this->blob_top_vec_[0]->mutable_cpu_data()[i];
        this->blob_top_vec_[0]->mutable_cpu_data()[i] =
                           this->blob_top_vec_[0]->mutable_cpu_data()[j];
        this->blob_top_vec_[0]->mutable_cpu_data()[j] = fSwap;
      }
    }

  // check
  for (int i = 0; i < iOutputCount; i++)
    EXPECT_NEAR(this->blob_top_->cpu_data()[i],
     detection_out_data::output_data[i], 5e-1);
}
#endif

}  // namespace caffe
