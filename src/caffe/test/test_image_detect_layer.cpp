/*
All modification made by Cambricon Corporation: © 2019 Cambricon Corporation
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
#include <iostream>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/image_detect_layer.hpp"
#include "caffe/layers/mlu_image_detect_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "gtest/gtest.h"
#include "test_image_detect_input_data.hpp"

namespace caffe {

template <typename Dtype>
void get_valid_box(Dtype* in_data, std::vector<Dtype>& bboxes, int length) {  // NOLINT
  int box_count = length / 6;

  for (int i = 0; i < box_count; i++) {
    if (in_data[i * 6 + 4] > 0.0) {
      bboxes.push_back(in_data[i * 6]);
      bboxes.push_back(in_data[i * 6 + 1]);
      bboxes.push_back(in_data[i * 6 + 2]);
      bboxes.push_back(in_data[i * 6 + 3]);
      bboxes.push_back(in_data[i * 6 + 4]);
      bboxes.push_back(in_data[i * 6 + 5]);
    }
  }
}

// Reference add for checking results:
template <typename Dtype>
float caffe_image_detect(std::vector<Blob<Dtype>*> bottom_blob_vec_,
                         std::vector<Blob<Dtype>*> top_blob_vec_,
                         bool cpu_mode, float errRate) {
  Dtype* top_data = top_blob_vec_[0]->mutable_cpu_data();

  int num_class_ = 21;
  int im_h_ = 720;
  int im_w_ = 1280;
  float scale_ = 1;
  float nms_thresh_ = 0.3;
  float score_thresh_ = 0.5;

  std::vector<Dtype> rois(bottom_blob_vec_[2]->count());
  if (!cpu_mode) {
    for (int i = 0; i < bottom_blob_vec_[2]->count() / 5; i++) {
      rois[i * 5] = 0;
      for (int j = 0; j < 4; j++) {
        rois[i * 5 + j + 1] = bottom_blob_vec_[2]->cpu_data()[i * 5 + j] / scale_;
      }
    }
  } else {
    for (int i = 0; i < bottom_blob_vec_[2]->count(); i++)
      rois[i] = bottom_blob_vec_[2]->cpu_data()[i] / scale_;
  }

  std::vector<std::vector<Dtype> > boxes;
  std::vector<std::vector<Dtype> > size;
  std::vector<std::vector<Dtype> > scoreCpu;
  std::vector<std::vector<int> > use;

  int batch = bottom_blob_vec_[0]->num();
  for (int i = 0; i < batch; i++) {
    std::vector<Dtype> score_vec;
    std::vector<int> use_vec;
    for (int j = 0; j < num_class_; j++) {
      score_vec.push_back(bottom_blob_vec_[1]->cpu_data()[i * num_class_ + j]);
      use_vec.push_back(1);
    }
    scoreCpu.push_back(score_vec);
    use.push_back(use_vec);
  }

  for (int i = 0; i < batch; i++) {
    Dtype width = rois[i * 5 + 3] - rois[i * 5 + 1] + 1;
    Dtype height = rois[i * 5 + 4] - rois[i * 5 + 2] + 1;
    Dtype c_x = rois[i * 5 + 1] + static_cast<Dtype>(width) / 2;
    Dtype c_y = rois[i * 5 + 2] + static_cast<Dtype>(height) / 2;

    std::vector<Dtype> size_vec;
    std::vector<Dtype> boxes_vec;
    for (int j = 0; j < num_class_; j++) {
      Dtype pc_x = c_x + width *
          bottom_blob_vec_[0]->cpu_data()[i * num_class_ * 4 + j * 4];
      Dtype pc_y = c_y + height *
          bottom_blob_vec_[0]->cpu_data()[i * num_class_ * 4 + j * 4 + 1];
      Dtype pc_w = exp(bottom_blob_vec_[0]->cpu_data()[i *
          num_class_ * 4 + j * 4 + 2]) * width;
      Dtype pc_h = exp(bottom_blob_vec_[0]->cpu_data()[i * num_class_ * 4 +
          j * 4 + 3]) * height;
      Dtype box_x1 = ((pc_x - 0.5 * pc_w) > 0 ? (pc_x - 0.5 * pc_w) : 0);
      Dtype box_y1 = ((pc_y - 0.5 * pc_h) > 0 ? (pc_y - 0.5 * pc_h) : 0);
      Dtype box_x2 = ((pc_x + 0.5 * pc_w) < im_w_ ? (pc_x + 0.5 * pc_w)
                                    : im_w_ - 1);
      Dtype box_y2 = ((pc_y + 0.5 * pc_h) < im_h_ ? (pc_y + 0.5 * pc_h)
                                    : im_h_ - 1);
      boxes_vec.push_back(box_x1);
      boxes_vec.push_back(box_y1);
      boxes_vec.push_back(box_x2);
      boxes_vec.push_back(box_y2);
      size_vec.push_back((box_y2 - box_y1) * (box_x2 - box_x1));
    }
    size.push_back(size_vec);
    boxes.push_back(boxes_vec);
  }

  std::vector<Dtype> resultInfo(top_blob_vec_[0]->count(), 0.0);
  int bbox_count = 0;
  for (int t = 0; t < num_class_; t++) {
    for (int i = 0; i < batch; i++) {
      if (use[i][t]) {
        for (int j = i + 1; j < batch; j++) {
          int overlap_x = std::min(boxes[i][4 * t + 2], boxes[j][4 * t + 2]) -
                          std::max(boxes[i][4 * t + 0], boxes[j][4 * t + 0]);
          int overlap_y = std::min(boxes[i][4 * t + 3], boxes[j][4 * t + 3]) -
                          std::max(boxes[i][4 * t + 1], boxes[j][4 * t + 1]);
          int overlap =
              (overlap_x > 0 ? overlap_x : 0) * (overlap_y > 0 ? overlap_y : 0);
          Dtype nms = static_cast<Dtype>(overlap) /
                      static_cast<Dtype>((size[i][t] + size[j][t] - overlap));
          if (nms < nms_thresh_) continue;
          if (scoreCpu[j][t] > scoreCpu[i][t]) {
            use[i][t] = 0;
            break;
          } else if (scoreCpu[j][t] <= scoreCpu[i][t]) {
            use[j][t] = 0;
          }
        }
        if (use[i][t] && scoreCpu[i][t] > score_thresh_) {
          resultInfo[bbox_count++] = boxes[i][4 * t + 0];
          resultInfo[bbox_count++] = boxes[i][4 * t + 1];
          resultInfo[bbox_count++] = boxes[i][4 * t + 2];
          resultInfo[bbox_count++] = boxes[i][4 * t + 3];
          resultInfo[bbox_count++] = t;
          resultInfo[bbox_count++] = scoreCpu[i][t];
        }
      }
    }
  }

  std::vector<Dtype> sort_refer;
  std::vector<Dtype> sort_output;
  get_valid_box(resultInfo.data(), sort_refer, top_blob_vec_[0]->count());
  get_valid_box(top_data, sort_output, top_blob_vec_[0]->count());
  EXPECT_EQ(sort_refer.size(), sort_output.size());

  float err_sum = 0, sum = 0;
  for (int i = 0; i < sort_refer.size(); i++) {
    EXPECT_NEAR(sort_output[i], sort_refer[i], errRate);
    err_sum += std::abs(sort_output[i] - sort_refer[i]);
    sum += std::abs(sort_refer[i]);
  }
  EXPECT_LE(err_sum/sum, 1e0);
  return err_sum / sum;
}

template <typename TypeParam>
class ImageDetectLayerTest : public CPUDeviceTest<TypeParam> {
typedef typename TypeParam::Dtype Dtype;

  protected:
  ImageDetectLayerTest()
      : blob_bottom_box_(new Blob<Dtype>(304, 84, 1, 1)),
        blob_bottom_score_(new Blob<Dtype>(304, 21, 1, 1)),
        blob_bottom_rois_(new Blob<Dtype>(1, 304, 1, 5)),
        blob_top_(new Blob<Dtype>()) {}
     virtual void SetUp() {
       FillerParameter filler_param;
       GaussianFiller<Dtype> filler(filler_param);
       filler.Fill(this->blob_bottom_box_);
       filler.Fill(this->blob_bottom_score_);
       filler.Fill(this->blob_bottom_rois_);
       blob_bottom_vec_.push_back(blob_bottom_box_);
       blob_bottom_vec_.push_back(blob_bottom_score_);
       blob_bottom_vec_.push_back(blob_bottom_rois_);
       blob_top_vec_.push_back(blob_top_);
     }

     virtual ~ImageDetectLayerTest() {
       delete blob_bottom_box_;
       delete blob_bottom_score_;
       delete blob_bottom_rois_;
       delete blob_top_;
     }

    Blob<Dtype>* const blob_bottom_box_;
    Blob<Dtype>* const blob_bottom_score_;
    Blob<Dtype>* const blob_bottom_rois_;
    Blob<Dtype>* const blob_top_;
    vector<Blob<Dtype>*> blob_bottom_vec_;
    vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ImageDetectLayerTest, TestDtypesAndDevices);

TYPED_TEST(ImageDetectLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;

  LayerParameter layer_param;
  ImageDetectParameter* param = layer_param.mutable_image_detect_param();
  param->set_num_class(21);
  param->set_im_h(720);
  param->set_im_w(1280);
  param->set_scale(1.0);
  param->set_nms_thresh(0.3);
  param->set_score_thresh(0.5);
  ImageDetectLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 304 * (84 / 4));
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 6);
}

TYPED_TEST(ImageDetectLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;

  float scale_ = 1.0;
  this->blob_bottom_vec_.clear();
  this->blob_top_vec_.clear();

  Dtype* input_bbox_data;
  input_bbox_data = this->blob_bottom_box_->mutable_cpu_data();
  for (int i = 0; i < this->blob_bottom_box_->count(); i++) {
    input_bbox_data[i] = image_detect_input_data::input_bbox_data[i];
  }

  Dtype* input_score_data;
  input_score_data = this->blob_bottom_score_->mutable_cpu_data();
  for (int i = 0; i < this->blob_bottom_score_->count(); i++) {
    input_score_data[i] = image_detect_input_data::input_score_data[i];
  }

  Dtype* input_rois_data;
  input_rois_data = this->blob_bottom_rois_->mutable_cpu_data();
  for (int i = 0; i < this->blob_bottom_rois_->count() / 5; i++) {
    input_rois_data[i * 5] = 0;
    for (int j = 0; j < 4; j++) {
      input_rois_data[i * 5 + j + 1] =
           image_detect_input_data::input_rois_data[i * 5 + j] / scale_;
    }
  }

  this->blob_bottom_vec_.push_back(this->blob_bottom_box_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_score_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_rois_);
  this->blob_top_vec_.push_back(this->blob_top_);

  LayerParameter layer_param;
  ImageDetectParameter* param = layer_param.mutable_image_detect_param();
  param->set_num_class(21);
  param->set_im_h(720);
  param->set_im_w(1280);
  param->set_scale(scale_);
  param->set_nms_thresh(0.3);
  param->set_score_thresh(0.5);
  ImageDetectLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  caffe_image_detect(this->blob_bottom_vec_, this->blob_top_vec_, true, 1e-5);
}

#ifdef USE_MLU
template <typename TypeParam>
class MLUImageDetectLayerTest : public MLUDeviceTest<TypeParam> {
typedef typename TypeParam::Dtype Dtype;

  protected:
  MLUImageDetectLayerTest()
      : blob_bottom_box_(new Blob<Dtype>(304, 84, 1, 1)),
        blob_bottom_score_(new Blob<Dtype>(304, 21, 1, 1)),
        blob_bottom_rois_(new Blob<Dtype>(1, 304, 1, 5)),
        blob_top_(new Blob<Dtype>()) {}
     virtual void SetUp() {
       FillerParameter filler_param;
       GaussianFiller<Dtype> filler(filler_param);
       filler.Fill(this->blob_bottom_box_);
       filler.Fill(this->blob_bottom_score_);
       filler.Fill(this->blob_bottom_rois_);
       blob_bottom_vec_.push_back(blob_bottom_box_);
       blob_bottom_vec_.push_back(blob_bottom_score_);
       blob_bottom_vec_.push_back(blob_bottom_rois_);
       blob_top_vec_.push_back(blob_top_);
     }

     virtual ~MLUImageDetectLayerTest() {
       delete blob_bottom_box_;
       delete blob_bottom_score_;
       delete blob_bottom_rois_;
       delete blob_top_;
     }

    Blob<Dtype>* const blob_bottom_box_;
    Blob<Dtype>* const blob_bottom_score_;
    Blob<Dtype>* const blob_bottom_rois_;
    Blob<Dtype>* const blob_top_;
    vector<Blob<Dtype>*> blob_bottom_vec_;
    vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MLUImageDetectLayerTest, TestMLUDevices);

TYPED_TEST(MLUImageDetectLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;

  LayerParameter layer_param;
  ImageDetectParameter* param = layer_param.mutable_image_detect_param();
  param->set_num_class(21);
  param->set_im_h(720);
  param->set_im_w(1280);
  param->set_scale(1.0);
  param->set_nms_thresh(0.3);
  param->set_score_thresh(0.5);
  MLUImageDetectLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 304 * (84 / 4));
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 6);
}

TYPED_TEST(MLUImageDetectLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.clear();
  this->blob_top_vec_.clear();

  Dtype* input_bbox_data;
  input_bbox_data = this->blob_bottom_box_->mutable_cpu_data();
  for (int i = 0; i < this->blob_bottom_box_->count(); i++) {
    input_bbox_data[i] = image_detect_input_data::input_bbox_data[i];
  }

  Dtype* input_score_data;
  input_score_data = this->blob_bottom_score_->mutable_cpu_data();
  for (int i = 0; i < this->blob_bottom_score_->count(); i++) {
    input_score_data[i] = image_detect_input_data::input_score_data[i];
  }

  Dtype* input_rois_data;
  input_rois_data = this->blob_bottom_rois_->mutable_cpu_data();
  for (int i = 0; i < this->blob_bottom_rois_->count(); i++) {
    input_rois_data[i] = image_detect_input_data::input_rois_data[i];
  }
  this->blob_bottom_vec_.push_back(this->blob_bottom_box_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_score_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_rois_);
  this->blob_top_vec_.push_back(this->blob_top_);

  LayerParameter layer_param;
  ImageDetectParameter* param = layer_param.mutable_image_detect_param();
  param->set_num_class(21);
  param->set_im_h(720);
  param->set_im_w(1280);
  param->set_scale(1.0);
  param->set_nms_thresh(0.3);
  param->set_score_thresh(0.5);
  MLUImageDetectLayer<Dtype> layer(layer_param);
  Caffe::setDetectOpMode(1);
  layer.set_int8_context(0);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  caffe_image_detect(this->blob_bottom_vec_, this->blob_top_vec_, false, 2.5e0);
}

template <typename TypeParam>
class MFUSImageDetectLayerTest : public MFUSDeviceTest<TypeParam> {
typedef typename TypeParam::Dtype Dtype;

  protected:
  MFUSImageDetectLayerTest()
      : blob_bottom_box_(new Blob<Dtype>(304, 84, 1, 1)),
        blob_bottom_score_(new Blob<Dtype>(304, 21, 1, 1)),
        blob_bottom_rois_(new Blob<Dtype>(1, 304, 1, 5)),
        blob_top_(new Blob<Dtype>()) {}
     virtual void SetUp() {
       FillerParameter filler_param;
       GaussianFiller<Dtype> filler(filler_param);
       filler.Fill(this->blob_bottom_box_);
       filler.Fill(this->blob_bottom_score_);
       filler.Fill(this->blob_bottom_rois_);
       blob_bottom_vec_.push_back(blob_bottom_box_);
       blob_bottom_vec_.push_back(blob_bottom_score_);
       blob_bottom_vec_.push_back(blob_bottom_rois_);
       blob_top_vec_.push_back(blob_top_);
     }

     virtual ~MFUSImageDetectLayerTest() {
       delete blob_bottom_box_;
       delete blob_bottom_score_;
       delete blob_bottom_rois_;
       delete blob_top_;
     }

    Blob<Dtype>* const blob_bottom_box_;
    Blob<Dtype>* const blob_bottom_score_;
    Blob<Dtype>* const blob_bottom_rois_;
    Blob<Dtype>* const blob_top_;
    vector<Blob<Dtype>*> blob_bottom_vec_;
    vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MFUSImageDetectLayerTest, TestMFUSDevices);

TYPED_TEST(MFUSImageDetectLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;

  LayerParameter layer_param;
  ImageDetectParameter* param = layer_param.mutable_image_detect_param();
  param->set_num_class(21);
  param->set_im_h(720);
  param->set_im_w(1280);
  param->set_scale(1.0);
  param->set_nms_thresh(0.3);
  param->set_score_thresh(0.5);
  MLUImageDetectLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 304 * (84 / 4));
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 6);
}

TYPED_TEST(MFUSImageDetectLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.clear();
  this->blob_top_vec_.clear();

  Dtype* input_bbox_data;
  input_bbox_data = this->blob_bottom_box_->mutable_cpu_data();
  for (int i = 0; i < this->blob_bottom_box_->count(); i++) {
    input_bbox_data[i] = image_detect_input_data::input_bbox_data[i];
  }

  Dtype* input_score_data;
  input_score_data = this->blob_bottom_score_->mutable_cpu_data();
  for (int i = 0; i < this->blob_bottom_score_->count(); i++) {
    input_score_data[i] = image_detect_input_data::input_score_data[i];
  }

  Dtype* input_rois_data;
  input_rois_data = this->blob_bottom_rois_->mutable_cpu_data();
  for (int i = 0; i < this->blob_bottom_rois_->count(); i++) {
    input_rois_data[i] = image_detect_input_data::input_rois_data[i];
  }
  this->blob_bottom_vec_.push_back(this->blob_bottom_box_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_score_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_rois_);
  this->blob_top_vec_.push_back(this->blob_top_);

  LayerParameter layer_param;
  ImageDetectParameter* param = layer_param.mutable_image_detect_param();
  param->set_num_class(21);
  param->set_im_h(720);
  param->set_im_w(1280);
  param->set_scale(1.0);
  param->set_nms_thresh(0.3);
  param->set_score_thresh(0.5);
  MLUImageDetectLayer<Dtype> layer(layer_param);
  Caffe::setDetectOpMode(1);
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

  caffe_image_detect(this->blob_bottom_vec_, this->blob_top_vec_, false, 2.5e0);
}
#endif

}  // namespace caffe
