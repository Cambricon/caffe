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
#include <cstring>
#include <vector>


#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/proposal_layer.hpp"
#ifdef USE_MLU
#include "caffe/layers/mlu_proposal_layer.hpp"
#endif

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include "rpn_cls_score_input.hpp"
#include "rpn_bbox_pred_input.hpp"

#include "gtest/gtest.h"
namespace caffe {
#ifdef USE_MLU
static const float iou_thres = 0.5;
static const float expect_iou_qualified_ratio = 0.75;
template <typename TypeParam>
class MLUProposalLayerTest : public MLUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MLUProposalLayerTest()
      : rpn_cls_score_(new Blob<Dtype>(1, 18, 23, 40)),
        rpn_bbox_pred_(new Blob<Dtype>(1, 36, 23, 40)),
        im_info_(new Blob<Dtype>(1, 3, 1, 1)),
        mlu_blob_top_(new Blob<Dtype>()),
        cpu_blob_top_(new Blob<Dtype>()) {
    for (int i = 0; i < rpn_cls_score_->count(); i++) {
      rpn_cls_score_->mutable_cpu_data()[i] =
        rpn_cls_score::rpn_cls_score_data[i];
    }
    for (int i = 0; i < rpn_bbox_pred_->count(); i++) {
      rpn_bbox_pred_->mutable_cpu_data()[i] =
        rpn_bbox_pred::rpn_bbox_pred_data[i];
    }
    auto im_info_data = im_info_->mutable_cpu_data();
    im_info_data[0] = 720;
    im_info_data[1] = 1280;
    im_info_data[2] = 1;
    blob_bottom_vec_.push_back(rpn_cls_score_);
    blob_bottom_vec_.push_back(rpn_bbox_pred_);
    blob_bottom_vec_.push_back(im_info_);
    mlu_blob_top_vec_.push_back(mlu_blob_top_);
    cpu_blob_top_vec_.push_back(cpu_blob_top_);
  }

  void reconstruct_box(vector<Dtype>* box) {  //NOLINT
    Dtype x1 = box->at(0);
    Dtype y1 = box->at(1);
    Dtype x2 = box->at(2);
    Dtype y2 = box->at(3);
    box->at(0) = min(x1, x2);
    box->at(1) = min(y1, y2);
    box->at(2) = max(x1, x2);
    box->at(3) = max(y1, y2);
  }

  bool in_range(const Dtype x, const Dtype a, const Dtype b) {
    return x >= a && x <= b;
  }

  bool intersect(const vector<Dtype>* box_a, const vector<Dtype>* box_b) {
    Dtype xa1 = box_a->at(0);
    Dtype ya1 = box_a->at(1);
    Dtype xa2 = box_a->at(2);
    Dtype ya2 = box_a->at(3);

    Dtype xb1 = box_b->at(0);
    Dtype yb1 = box_b->at(1);
    Dtype xb2 = box_b->at(2);
    Dtype yb2 = box_b->at(3);
    return (in_range(xb1, xa1, xa2) || in_range(xb2, xa1, xa2))
      && (in_range(yb1, ya1, ya2) || in_range(yb2, ya1, ya2));
  }

  Dtype iou(vector<Dtype>* const box_a, vector<Dtype>* const box_b) {
    reconstruct_box(box_a);
    reconstruct_box(box_b);
    if (!intersect(box_a, box_b)) {
      return 0;
    }
    Dtype xa1 = box_a->at(0);
    Dtype ya1 = box_a->at(1);
    Dtype xa2 = box_a->at(2);
    Dtype ya2 = box_a->at(3);

    Dtype xb1 = box_b->at(0);
    Dtype yb1 = box_b->at(1);
    Dtype xb2 = box_b->at(2);
    Dtype yb2 = box_b->at(3);

    Dtype xA = max(xa1, xb1);
    Dtype yA = max(ya1, yb1);
    Dtype xB = min(xa2, xb2);
    Dtype yB = min(ya2, yb2);
    Dtype intersect = abs(xB-xA) * abs(yB-yA);
    Dtype S_box_a = abs(xa1 - xa2) * abs(ya1 - ya2);
    Dtype S_box_b = abs(xb1 - xb2) * abs(yb1 - yb2);
    if (S_box_a == 0 || S_box_b == 0) {
      return 0;
    }
    return intersect * 1.0/ (S_box_a + S_box_b - intersect);
  }

  virtual ~MLUProposalLayerTest() {
    delete rpn_cls_score_;
    delete rpn_bbox_pred_;
    delete im_info_;
    delete mlu_blob_top_;
    delete cpu_blob_top_;
  }

  Blob<Dtype>* const rpn_cls_score_;
  Blob<Dtype>* const rpn_bbox_pred_;
  Blob<Dtype>* const im_info_;
  Blob<Dtype>* const mlu_blob_top_;
  Blob<Dtype>* const cpu_blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> mlu_blob_top_vec_;
  vector<Blob<Dtype>*> cpu_blob_top_vec_;
};

TYPED_TEST_CASE(MLUProposalLayerTest, TestMLUDevices);

TYPED_TEST(MLUProposalLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.set_phase(TEST);
  ProposalParameter* proposal_param =
    layer_param.mutable_proposal_param();
  proposal_param->set_stride(32);
  proposal_param->set_im_min_w(16);
  proposal_param->set_im_min_h(16);
  proposal_param->set_top_num(6000);
  proposal_param->set_nms_thresh(0.7);
  proposal_param->set_nms_num(304);
  proposal_param->set_anchor_num(9);
  ProposalLayer<Dtype> cpu_layer(layer_param);
  Caffe::setDetectOpMode(1);
  cpu_layer.SetUp(this->blob_bottom_vec_, this->cpu_blob_top_vec_);
  cpu_layer.Forward(this->blob_bottom_vec_, this->cpu_blob_top_vec_);

  MLUProposalLayer<Dtype> mlu_layer(layer_param);
  mlu_layer.SetUp(this->blob_bottom_vec_, this->mlu_blob_top_vec_);
  mlu_layer.Reshape_dispatch(this->blob_bottom_vec_, this->mlu_blob_top_vec_);
  mlu_layer.Forward(this->blob_bottom_vec_, this->mlu_blob_top_vec_);
  // CPU 304 5 1 1
  // MLU 1 304 1 5
  CHECK_EQ(this->cpu_blob_top_->height(), 1);
  CHECK_EQ(this->cpu_blob_top_->width(), 1);
  CHECK_EQ(this->mlu_blob_top_->num(), 1);
  CHECK_EQ(this->mlu_blob_top_->height(), 1);
  CHECK_EQ(this->cpu_blob_top_->num(), this->mlu_blob_top_->channels());
  CHECK_EQ(this->cpu_blob_top_->channels(), this->mlu_blob_top_->width());
  CHECK_EQ(this->cpu_blob_top_->count() % 5, 0);
  int length = this->cpu_blob_top_->count() / 5;
  int iou_qualified_cnt = 0;
  auto cpu_data = this->cpu_blob_top_->cpu_data();
  auto mlu_data = this->mlu_blob_top_->cpu_data();
  for (int i = 0; i < length; i++) {
    vector<Dtype> data_a{cpu_data[i*5+1], cpu_data[i*5+2], cpu_data[i*5+3],
                         cpu_data[i*5+4]};
    float max_iou = 0.;
    for (int j = 0; j < length; j++) {
      vector<Dtype> data_b{mlu_data[j*5], mlu_data[j*5+1], mlu_data[j*5+2],
                           mlu_data[j*5+3]};
      float iou = this->iou(&data_a, &data_b);
      if (iou > max_iou) {
        max_iou = iou;
      }
    }
    if (max_iou > iou_thres) {
      iou_qualified_cnt++;
    }
  }
  CHECK_GE(iou_qualified_cnt * 1.0 / length, expect_iou_qualified_ratio);
  std::ostringstream stream;
  stream << "bottom1:" << this->rpn_cls_score_->shape_string().c_str() << "\t"
    << "bottom2:" << this->rpn_bbox_pred_->shape_string().c_str() << "\t"
    << "bottom3:" << this->im_info_->shape_string().c_str();
  ERR_RATE(iou_qualified_cnt * 1.0/length - expect_iou_qualified_ratio);
  BOTTOM(stream);
  EVENT_TIME(mlu_layer.get_event_time());
}

template <typename TypeParam>
class MFUSProposalLayerTest : public MFUSDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MFUSProposalLayerTest()
      : rpn_cls_score_(new Blob<Dtype>(1, 18, 23, 40)),
        rpn_bbox_pred_(new Blob<Dtype>(1, 36, 23, 40)),
        im_info_(new Blob<Dtype>(1, 3, 1, 1)),
        mlu_blob_top_(new Blob<Dtype>()),
        cpu_blob_top_(new Blob<Dtype>()) {
    for (int i = 0; i < rpn_cls_score_->count(); i++) {
      rpn_cls_score_->mutable_cpu_data()[i] =
        rpn_cls_score::rpn_cls_score_data[i];
    }
    for (int i = 0; i < rpn_bbox_pred_->count(); i++) {
      rpn_bbox_pred_->mutable_cpu_data()[i] =
        rpn_bbox_pred::rpn_bbox_pred_data[i];
    }
    auto im_info_data = im_info_->mutable_cpu_data();
    im_info_data[0] = 720;
    im_info_data[1] = 1280;
    im_info_data[2] = 1;
    blob_bottom_vec_.push_back(rpn_cls_score_);
    blob_bottom_vec_.push_back(rpn_bbox_pred_);
    blob_bottom_vec_.push_back(im_info_);
    mlu_blob_top_vec_.push_back(mlu_blob_top_);
    cpu_blob_top_vec_.push_back(cpu_blob_top_);
  }

  void reconstruct_box(vector<Dtype>* box) {
    Dtype x1 = box->at(0);
    Dtype y1 = box->at(1);
    Dtype x2 = box->at(2);
    Dtype y2 = box->at(3);
    box->at(0) = min(x1, x2);
    box->at(1) = min(y1, y2);
    box->at(2) = max(x1, x2);
    box->at(3) = max(y1, y2);
  }

  bool intersect(const vector<Dtype>* box_a, const vector<Dtype>* box_b) {
    if (max(box_a->at(0), box_a->at(2)) <= min(box_b->at(0), box_b->at(2)) ||
          max(box_b->at(0), box_b->at(2)) <= min(box_a->at(0), box_a->at(2)) ||
          max(box_a->at(1), box_a->at(3)) <= min(box_b->at(1), box_b->at(3)) ||
          max(box_b->at(1), box_b->at(3)) <= min(box_b->at(1), box_b->at(3))) {
          return false;
        } else {
          return true;
        }
  }

  Dtype iou(vector<Dtype>* const box_a, vector<Dtype>* const box_b) {
    reconstruct_box(box_a);
    reconstruct_box(box_b);
    if (!intersect(box_a, box_b)) {
      return 0;
    }
    Dtype xa1 = box_a->at(0);
    Dtype ya1 = box_a->at(1);
    Dtype xa2 = box_a->at(2);
    Dtype ya2 = box_a->at(3);

    Dtype xb1 = box_b->at(0);
    Dtype yb1 = box_b->at(1);
    Dtype xb2 = box_b->at(2);
    Dtype yb2 = box_b->at(3);

    Dtype xA = max(xa1, xb1);
    Dtype yA = max(ya1, yb1);
    Dtype xB = min(xa2, xb2);
    Dtype yB = min(ya2, yb2);
    Dtype intersect = abs(xB-xA) * abs(yB-yA);
    Dtype S_box_a = abs(xa1 - xa2) * abs(ya1 - ya2);
    Dtype S_box_b = abs(xb1 - xb2) * abs(yb1 - yb2);
    if (S_box_a == 0 || S_box_b == 0) {
      return 0;
    }
    return intersect * 1.0/ (S_box_a + S_box_b - intersect);
  }

  virtual ~MFUSProposalLayerTest() {
    delete rpn_cls_score_;
    delete rpn_bbox_pred_;
    delete im_info_;
    delete mlu_blob_top_;
    delete cpu_blob_top_;
  }

  Blob<Dtype>* const rpn_cls_score_;
  Blob<Dtype>* const rpn_bbox_pred_;
  Blob<Dtype>* const im_info_;
  Blob<Dtype>* const mlu_blob_top_;
  Blob<Dtype>* const cpu_blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> mlu_blob_top_vec_;
  vector<Blob<Dtype>*> cpu_blob_top_vec_;
};

TYPED_TEST_CASE(MFUSProposalLayerTest, TestMLUDevices);

TYPED_TEST(MFUSProposalLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.set_phase(TEST);
  ProposalParameter* proposal_param =
    layer_param.mutable_proposal_param();
  proposal_param->set_stride(32);
  proposal_param->set_im_min_w(16);
  proposal_param->set_im_min_h(16);
  proposal_param->set_top_num(6000);
  proposal_param->set_nms_thresh(0.7);
  proposal_param->set_nms_num(304);
  proposal_param->set_anchor_num(9);
  ProposalLayer<Dtype> cpu_layer(layer_param);
  cpu_layer.SetUp(this->blob_bottom_vec_, this->cpu_blob_top_vec_);
  cpu_layer.Forward(this->blob_bottom_vec_, this->cpu_blob_top_vec_);

  MLUProposalLayer<Dtype> mfus_layer(layer_param);
  Caffe::setDetectOpMode(1);
  mfus_layer.SetUp(this->blob_bottom_vec_, this->mlu_blob_top_vec_);
  MFusion<Dtype> fuser;
  fuser.reset();
  fuser.addInputs(this->blob_bottom_vec_);
  fuser.addOutputs(this->mlu_blob_top_vec_);
  mfus_layer.Reshape_dispatch(this->blob_bottom_vec_, this->mlu_blob_top_vec_);
  mfus_layer.fuse(&fuser);
  fuser.compile();
  fuser.forward();
  // CPU 304 5 1 1
  // MLU 1 304 1 5
  CHECK_EQ(this->cpu_blob_top_->height(), 1);
  CHECK_EQ(this->cpu_blob_top_->width(), 1);
  CHECK_EQ(this->mlu_blob_top_->num(), 1);
  CHECK_EQ(this->mlu_blob_top_->height(), 1);
  CHECK_EQ(this->cpu_blob_top_->num(), this->mlu_blob_top_->channels());
  CHECK_EQ(this->cpu_blob_top_->channels(), this->mlu_blob_top_->width());
  CHECK_EQ(this->cpu_blob_top_->count() % 5, 0);
  int length = this->cpu_blob_top_->count() / 5;
  int iou_qualified_cnt = 0;
  auto cpu_data = this->cpu_blob_top_->cpu_data();
  auto mlu_data = this->mlu_blob_top_->cpu_data();
  for (int i = 0; i < length; i++) {
    vector<Dtype> data_a{cpu_data[i*5+1], cpu_data[i*5+2], cpu_data[i*5+3],
                         cpu_data[i*5+4]};
    float max_iou = 0.;
    for (int j = 0; j < length; j++) {
      vector<Dtype> data_b{mlu_data[j*5], mlu_data[j*5+1], mlu_data[j*5+2],
                           mlu_data[j*5+3]};
      float iou = this->iou(&data_a, &data_b);
      if (iou > max_iou) {
        max_iou = iou;
      }
    }
    if (max_iou > iou_thres) {
      iou_qualified_cnt++;
    }
  }
  CHECK_GE(iou_qualified_cnt * 1.0 / length, expect_iou_qualified_ratio);
  std::ostringstream stream;
  stream << "bottom1:" << this->rpn_cls_score_->shape_string().c_str() << "\t"
    << "bottom2:" << this->rpn_bbox_pred_->shape_string().c_str() << "\t"
    << "bottom3:" << this->im_info_->shape_string().c_str();
  BOTTOM(stream);
  ERR_RATE(iou_qualified_cnt * 1.0/length - expect_iou_qualified_ratio);
  EVENT_TIME(mfus_layer.get_event_time());
}
#endif

}  // namespace caffe
