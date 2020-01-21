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

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "caffe/layers/proposal_layer.hpp"
#define SIZE 1000

#define ROUND(x) ((int)((x) + (Dtype)0.5))  // NOLINT

using namespace std;  // NOLINT

namespace caffe {

template <typename Dtype>
void ProposalLayer<Dtype>::Sort(vector<Dtype>* scores, vector<int>* id,
                                int start, int length) {
  if (length < 2) return;
  Dtype swap;
  int swap_id;
  if (length > 2) {
    int i = 0;
    int j = length - 1;
    while (j > i) {
      for (; j > i; j--)
        if ((*scores)[i + start] < (*scores)[j + start]) {
          swap = (*scores)[i + start];
          swap_id = (*id)[i + start];
          (*scores)[i + start] = (*scores)[j + start];
          (*id)[i + start] = (*id)[j + start];
          (*scores)[j + start] = swap;
          (*id)[j + start] = swap_id;
          break;
        }
      for (; i < j; i++)
        if ((*scores)[i + start] < (*scores)[j + start]) {
          swap = (*scores)[i + start];
          swap_id = (*id)[i + start];
          (*scores)[i + start] = (*scores)[j + start];
          (*id)[i + start] = (*id)[j + start];
          (*scores)[j + start] = swap;
          (*id)[j + start] = swap_id;
          break;
        }
    }
    Sort(scores, id, start, i + 1);
    Sort(scores, id, start + i + 1, length - i - 1);
    return;
  } else {
    if ((*scores)[0 + start] < (*scores)[1 + start]) {
      swap = (*scores)[0 + start];
      swap_id = (*id)[0 + start];
      (*scores)[0 + start] = (*scores)[1 + start];
      (*id)[0 + start] = (*id)[1 + start];
      (*scores)[1 + start] = swap;
      (*id)[1 + start] = swap_id;
    }
    return;
  }
}
template <typename Dtype>
void ProposalLayer<Dtype>::nSort(vector<Dtype>* list_cpu,
                                 int start, int end, int num_top) {
  const Dtype score = (*list_cpu)[start * 5 + 4];
  int left = start + 1, right = end;
  Dtype temp[5];
  while (left <= right) {
    while (left <= end && (*list_cpu)[left * 5 + 4] >= score) ++left;
    while (right > start && (*list_cpu)[right * 5 + 4] <= score) --right;
    if (left <= right) {
      for (int i = 0; i < 5; ++i) {
        temp[i] = (*list_cpu)[left * 5 + i];
      }
      for (int i = 0; i < 5; ++i) {
        (*list_cpu)[left * 5 + i] = (*list_cpu)[right * 5 + i];
      }
      for (int i = 0; i < 5; ++i) {
        (*list_cpu)[right * 5 + i] = temp[i];
      }
      ++left;
      --right;
    }
  }
  if (right > start) {
    for (int i = 0; i < 5; ++i) {
      temp[i] = (*list_cpu)[start * 5 + i];
    }
    for (int i = 0; i < 5; ++i) {
      (*list_cpu)[start * 5 + i] = (*list_cpu)[right * 5 + i];
    }
    for (int i = 0; i < 5; ++i) {
      (*list_cpu)[right * 5 + i] = temp[i];
    }
  }
  if (start < right - 1) {
    nSort(list_cpu, start, right - 1, num_top);
  }
  if (right + 1 < num_top && right + 1 < end) {
    nSort(list_cpu, right + 1, end, num_top);
  }
}
template <typename Dtype>
void ProposalLayer<Dtype>::GetTopScores(vector<Dtype>* scores, vector<Dtype>* box,
                                        vector<int>* id, int* size, int THRESH) {
  vector<Dtype> list_cpu(5*scores->size(), 0);
  if (!nproposal_mode_) {
    Sort(scores, id, 0, *size);
    *size = min(*size, THRESH);
  } else {
    for (int i =0; i< scores->size(); i++) {
      list_cpu[i*5 +0] = (*box)[i*4 +0];
      list_cpu[i*5 +1] = (*box)[i*4 +1];
      list_cpu[i*5 +2] = (*box)[i*4 +2];
      list_cpu[i*5 +3] = (*box)[i*4 +3];
      list_cpu[i*5 +4] = (*scores)[i];
    }
    nSort(&list_cpu, 0, *size -1, THRESH);
    for (int i =0; i< scores->size(); i++) {
      (*box)[i*4 +0] = list_cpu[i*5 + 0];
      (*box)[i*4 +1] = list_cpu[i*5 + 1];
      (*box)[i*4 +2] = list_cpu[i*5 + 2];
      (*box)[i*4 +3] = list_cpu[i*5 + 3];
      (*scores)[i] =   list_cpu[i*5 + 4];
    }
  }
}

template <typename Dtype>
void ProposalLayer<Dtype>::CreateAnchor(vector<Dtype>* anchor,
                                        int A, int W, int H,
                                        Dtype stride) {
  anchor->resize(A * W * H * 4);
  for (int i = 0; i < H; i++) {
    for (int j = 0; j < W; j++) {
      for (int k = 0; k < A; k++) {
        (*anchor)[i * W * A * 4 + j * A * 4 + k * 4 + 0] =
            j * stride + init_anchor_[k * 4 + 0];
        (*anchor)[i * W * A * 4 + j * A * 4 + k * 4 + 1] =
            i * stride + init_anchor_[k * 4 + 1];
        (*anchor)[i * W * A * 4 + j * A * 4 + k * 4 + 2] =
            j * stride + init_anchor_[k * 4 + 2];
        (*anchor)[i * W * A * 4 + j * A * 4 + k * 4 + 3] =
            i * stride + init_anchor_[k * 4 + 3];
      }
    }
  }
  if (this->layer_param().proposal_param().anchor_ratio_size())
    free(init_anchor_);
}

template <typename Dtype>
void ProposalLayer<Dtype>::CreateBox(vector<Dtype>* box, Dtype* scores,
                                     vector<Dtype>* newscores, vector<Dtype> anchor,
                                     const Dtype* delt, int A, int W, int H,
                                     Dtype im_w, Dtype im_h) {
  box->resize(A * W * H * 4);
  int l = 0;
  for (int i = 0; i < H; i++)
    for (int j = 0; j < W; j++)
      for (int k = 0; k < A; k++) {
        int anchor_loc = i * W * A * 4 + j * A * 4 + k * 4;
        Dtype x0 = anchor[anchor_loc + 0];
        Dtype y0 = anchor[anchor_loc + 1];
        Dtype x1 = anchor[anchor_loc + 2];
        Dtype y1 = anchor[anchor_loc + 3];
        int delt_loc = i * W + j + k * H * W * 4;
        Dtype dx = delt[delt_loc + 0 * H * W];
        Dtype dy = delt[delt_loc + 1 * H * W];
        Dtype dw = delt[delt_loc + 2 * H * W];
        Dtype dh = delt[delt_loc + 3 * H * W];
        Dtype cx = (x0 + x1 + 1) / 2;
        Dtype cy = (y0 + y1 + 1) / 2;
        Dtype w = x1 - x0 + 1;
        Dtype h = y1 - y0 + 1;
        Dtype ncx = cx + dx * w;
        Dtype ncy = cy + dy * h;
        Dtype nw = exp(dw) * w;
        Dtype nh = exp(dh) * h;
        // here choice the order of box as anchor
        // or as scores
        int box_loc = i * W * 4 + j * 4 + k * H * W * 4;
        if (nproposal_mode_) box_loc = anchor_loc;
        (*box)[box_loc + 0] = max(ncx - nw / 2, (Dtype)0);
        (*box)[box_loc + 1] = max(ncy - nh / 2, (Dtype)0);
        (*box)[box_loc + 2] = min(ncx + nw / 2, im_w - 1);
        (*box)[box_loc + 3] = min(ncy + nh / 2, im_h - 1);
        // change the order of scores
        if (nproposal_mode_) {
          int score_loc = i * W + j + k * H * W;
          (*newscores)[l++] = scores[score_loc];
        }
      }
}

template <typename Dtype>
void ProposalLayer<Dtype>::RemoveSmallBox(vector<Dtype>* box, vector<int> id,
                                          vector<int>* keep, int* keep_num,
                                          int total, Dtype w_min_size,
                                          Dtype h_min_size, vector<Dtype>* newscores) {
  keep->resize(total);
  int j = 0;
  for (int i = 0; i < total; i++) {
    if (((*box)[i * 4 + 2] - (*box)[i * 4 + 0] + 1) >= w_min_size &&
        ((*box)[i * 4 + 3] - (*box)[i * 4 + 1] + 1) >= h_min_size) {
      (*keep)[j] = id[i];
      j++;
    } else if (nproposal_mode_) {
      (*newscores)[i] = 0;
      (*keep)[j] = id[i];
      j++;
    }
  }
  *keep_num = j;
}

template <typename Dtype>
void ProposalLayer<Dtype>::GetNewScoresByKeep(const Dtype* scores,
                                              vector<Dtype>* new_scores,
                                              vector<int> keep, int keep_num) {
  new_scores->resize(keep_num);
  for (int i = 0; i < keep_num; i++) {
    (*new_scores)[i] = scores[keep[i]];
  }
}

template <typename Dtype>
void ProposalLayer<Dtype>::NMS(const vector<Dtype>& box, vector<int>* id,
                               int* id_size, Dtype THRESH, int MAX_NUM) {
  vector<bool> is_used(*id_size, true);
  vector<Dtype> area(*id_size);
  for (int i = 0; i < *id_size; i++) {
    area[i] = (box[(*id)[i] * 4 + 2] - box[(*id)[i] * 4 + 0] + 1) *
              (box[(*id)[i] * 4 + 3] - box[(*id)[i] * 4 + 1] + 1);
  }
  int j = 0;
  for (int i = 0; i < *id_size; i++) {
    if (is_used[i]) {
      (*id)[j] = (*id)[i];
      for (int k = i + 1; k < *id_size; k++) {
        if (is_used[k]) {
          Dtype inter_x1 = max(box[(*id)[i] * 4 + 0], box[(*id)[k] * 4 + 0]);
          Dtype inter_y1 = max(box[(*id)[i] * 4 + 1], box[(*id)[k] * 4 + 1]);
          Dtype inter_x2 = min(box[(*id)[i] * 4 + 2], box[(*id)[k] * 4 + 2]);
          Dtype inter_y2 = min(box[(*id)[i] * 4 + 3], box[(*id)[k] * 4 + 3]);
          Dtype inter_area = max((Dtype)0, inter_x2 - inter_x1 + 1) *
                             max((Dtype)0, inter_y2 - inter_y1 + 1);
          Dtype over = inter_area / (area[i] + area[k] - inter_area);
          if (over > THRESH) is_used[k] = false;
        }
      }
      j = j + 1;
      if (j >= MAX_NUM) break;
    }
  }
  *id_size = j;
}

template <typename Dtype>
void ProposalLayer<Dtype>::GetNewBox(const vector<Dtype>& box, Dtype* new_box,
                                     vector<int> id, int id_size, vector<Dtype>* scores) {
  for (int i = 0; i < id_size; i++) {
    if (nproposal_mode_) {
      new_box[i * 5 + 0] = box[id[i] * 4 + 0];
      new_box[i * 5 + 1] = box[id[i] * 4 + 1];
      new_box[i * 5 + 2] = box[id[i] * 4 + 2];
      new_box[i * 5 + 3] = box[id[i] * 4 + 3];
      new_box[i * 5 + 4] = (*scores)[id[i]];
    } else {
      new_box[i * 5 + 0] = 0;
      new_box[i * 5 + 1] = box[id[i] * 4 + 0];
      new_box[i * 5 + 2] = box[id[i] * 4 + 1];
      new_box[i * 5 + 3] = box[id[i] * 4 + 2];
      new_box[i * 5 + 4] = box[id[i] * 4 + 3];
    }
  }
}

template <typename Dtype>
int ProposalLayer<Dtype>::Proposal(const Dtype* bbox_pred, Dtype* scores,
                                   int H, int W, int A, Dtype stride, Dtype im_w,
                                   Dtype im_h, Dtype im_min_w, Dtype im_min_h,
                                   Dtype top_thresh, Dtype nms_thresh,
                                   int nms_num, Dtype* new_box) {
  int total = A * W * H;
  int keep_num = 0;
  vector<Dtype> anchor;
  vector<int> keep(total, 0);
  vector<Dtype> box(4*total, 0);
  vector<Dtype> new_scores(total, 0);
  vector<int> id(total);
  for (int i = 0; i < total; i++) id[i] = i;
  CreateAnchor(&anchor, A, W, H, stride);
  CreateBox(&box, scores, &new_scores, anchor, bbox_pred, A, W, H, im_w, im_h);
  RemoveSmallBox(&box, id, &keep, &keep_num, total, im_min_w, im_min_h, &new_scores);
  if (!nproposal_mode_) GetNewScoresByKeep(scores, &new_scores, keep, keep_num);
  GetTopScores(&new_scores, &box, &keep, &keep_num, top_thresh);
  NMS(box, &keep, &keep_num, nms_thresh, nms_num);
  GetNewBox(box, new_box, keep, keep_num, &new_scores);
  return keep_num;
}

template <typename Dtype>
void ProposalLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
  ProposalParameter proposal_param = this->layer_param().proposal_param();
  stride_ = proposal_param.stride();
  im_min_w_ = proposal_param.im_min_w();
  im_min_h_ = proposal_param.im_min_h();
  top_num_ = proposal_param.top_num();
  nms_thresh_ = proposal_param.nms_thresh();
  nms_num_ = proposal_param.nms_num();
  A_ = proposal_param.anchor_num();
  nproposal_mode_ = proposal_param.nproposal_mode();
  shuffle_channel_ = proposal_param.shuffle_channel();
}

template <typename Dtype>
void ProposalLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top) {
  std::vector<int> top_shape;
  top_shape.push_back(nms_num_);
  top_shape.push_back(5);
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void ProposalLayer<Dtype>::get_anchor() {
  ProposalParameter param = this->layer_param().proposal_param();
  vector<float> scales;
  for (int i = 0; i < param.anchor_scale_size(); i++) {
    scales.push_back(param.anchor_scale(i));
  }
  vector<float> ratios;
  for (int i = 0; i < param.anchor_ratio_size(); i++) {
    ratios.push_back(param.anchor_ratio(i));
  }
  float base_size = param.base_size();
  this->init_anchor_ =
      reinterpret_cast<Dtype*>(malloc(sizeof(Dtype) * 4 * param.anchor_ratio_size() *
              param.anchor_scale_size()));
  if (param.pvanet_mode() || param.nproposal_mode()) {
    float center = (base_size - 1) / 2;
    float size = base_size;
    for (int j = 0; j < param.anchor_ratio_size(); j++) {
      float sqt = sqrt(param.anchor_ratio(j));
      for (int i = 0; i < param.anchor_scale_size(); i++) {
          const float ratio_w = ROUND(size / sqt);
          const float ratio_h = ROUND(ratio_w * sqt * sqt);
          init_anchor_[(j * param.anchor_scale_size() + i) * 4 + 0]
              = center - 0.5 * (ratio_w * param.anchor_scale(i) - (Dtype)1);
          init_anchor_[(j * param.anchor_scale_size() + i) * 4 + 1]
              = center - 0.5 * (ratio_h * param.anchor_scale(i) - (Dtype)1);
          init_anchor_[(j * param.anchor_scale_size() + i) * 4 + 2]
              = center + 0.5 * (ratio_w * param.anchor_scale(i) - (Dtype)1);
          init_anchor_[(j * param.anchor_scale_size() + i) * 4 + 3]
              = center + 0.5 * (ratio_h * param.anchor_scale(i) - (Dtype)1);
      }
    }
  } else {
    float center = (base_size - 1) / 2;
    float size = base_size / 2;
    for (int i = 0; i < param.anchor_scale_size(); i++) {
      for (int j = 0; j < param.anchor_ratio_size(); j++) {
          float sqt = sqrt(param.anchor_ratio(j));
          init_anchor_[(i * param.anchor_ratio_size() + j) * 4 + 0]
              = center - size / sqt * param.anchor_scale(i);
          init_anchor_[(i * param.anchor_ratio_size() + j) * 4 + 1]
              = center - size * sqt * param.anchor_scale(i);
          init_anchor_[(i * param.anchor_ratio_size() + j) * 4 + 2]
              = center + size / sqt * param.anchor_scale(i);
          init_anchor_[(i * param.anchor_ratio_size() + j) * 4 + 3]
              = center + size * sqt * param.anchor_scale(i);
      }
    }
  }
}


template <typename Dtype>
void ProposalLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
  Dtype H = bottom[0]->height();
  Dtype W = bottom[0]->width();
  Dtype* scores = bottom[0]->mutable_cpu_data() + bottom[0]->offset(0, A_);
  const Dtype* bbox_pred = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype init_anchor[] = {-84, -40, 99, 55, -176, -88, 191, 103, -360,
                         -184, 375, 199, -56, -56, 71, 71, -120, -120,
                         135, 135, -248, -248, 263, 263, -36, -80, 51,
                         95, -80, -168, 95,  183, -168, -344, 183, 359};
  Dtype im_h = this->layer_param_.proposal_param().im_h();
  Dtype im_w = this->layer_param_.proposal_param().im_w();
  Dtype scale = this->layer_param_.proposal_param().scale();
  // If the value of im_info is passed through the third input,
  // the value of im_info will be overwritten.
  const Dtype* im_info = bottom[2]->cpu_data();
  if (bottom.size() == 3 && im_info[0] != 0) {
    im_h = im_info[0];
    im_w = im_info[1];
    scale = im_info[2];
  }
  ProposalParameter param = this->layer_param().proposal_param();
  if (!param.anchor_ratio_size()) {
    init_anchor_ = init_anchor;
  } else {
    get_anchor();
  }
  int box_num = Proposal(bbox_pred, scores, H, W, A_, stride_, im_w, im_h,
                         im_min_w_ * scale, im_min_h_ * scale, top_num_,
                         nms_thresh_, nms_num_, top_data);
  for (int i = box_num * 5; i < nms_num_ * 5; i++)
    top_data[i] = 0;
}

INSTANTIATE_CLASS(ProposalLayer);

}  // namespace caffe
