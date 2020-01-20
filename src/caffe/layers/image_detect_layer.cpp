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

#include <vector>
#include <algorithm>
#include "caffe/layers/image_detect_layer.hpp"

namespace caffe {

template <typename Dtype>
void ImageDetectLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                                 const vector<Blob<Dtype>*>& top) {
  auto detection_param = this->layer_param().image_detect_param();
  num_class_ = detection_param.num_class();
  im_h_ = detection_param.im_h();
  im_w_ = detection_param.im_w();
  nms_thresh_ = detection_param.nms_thresh();
  score_thresh_ = detection_param.score_thresh();
  scale_ = detection_param.scale();
}

template <typename Dtype>
void ImageDetectLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                              const vector<Blob<Dtype>*>& top) {
  auto bbox_N = bottom[0]->num();
  auto bbox_C = bottom[0]->channels();
  /* each box has 6 elements: x1, y1, x2, y2, label, score*/
  top[0]->Reshape(1, bbox_N * (bbox_C / 4), 1, 6);
}

template <typename Dtype>
void ImageDetectLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                                  const vector<Blob<Dtype>*>& top) {
  std::vector<Dtype> rois(bottom[2]->count());
  for (int i = 0; i < bottom[2]->count(); i++)
    rois[i] = bottom[2]->cpu_data()[i] / scale_;

  std::vector<std::vector<Dtype> > boxes;
  std::vector<std::vector<Dtype> > size;
  std::vector<std::vector<Dtype> > scoreCpu;
  std::vector<std::vector<int> > use;

  int batch = bottom[0]->num();
  for (int i = 0; i < batch; i++) {
    std::vector<Dtype> score_vec;
    std::vector<int> use_vec;
    for (int j = 0; j < num_class_; j++) {
      score_vec.push_back(bottom[1]->cpu_data()[i * num_class_ + j]);
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
      Dtype pc_x = c_x + width * bottom[0]->cpu_data()[i * num_class_ * 4 + j * 4];
      Dtype pc_y = c_y + height * bottom[0]->cpu_data()[i * num_class_ * 4 + j * 4 + 1];
      Dtype pc_w = exp(bottom[0]->cpu_data()[i * num_class_ * 4 + j * 4 + 2]) * width;
      Dtype pc_h = exp(bottom[0]->cpu_data()[i * num_class_ * 4 + j * 4 + 3]) * height;
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

  std::vector<std::vector<Dtype> > resultInfo;
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
          vector<Dtype> tmp(6);
          tmp[0] = boxes[i][4 * t + 0];
          tmp[1] = boxes[i][4 * t + 1];
          tmp[2] = boxes[i][4 * t + 2];
          tmp[3] = boxes[i][4 * t + 3];
          tmp[4] = t;
          tmp[5] = scoreCpu[i][t];
          resultInfo.push_back(tmp);
        }
      }
    }
  }

  int box_count = 0;
  /* Write detection box to top[0] blob */
  for (int j = 0; j < resultInfo.size(); j++)
    for (int k = 0; k < resultInfo[0].size(); k++)
      top[0]->mutable_cpu_data()[box_count++] = resultInfo[j][k];

  for (int i = box_count; i < top[0]->count(); i++) {
    top[0]->mutable_cpu_data()[i] = 0;
  }
}

INSTANTIATE_CLASS(ImageDetectLayer);

}  // namespace caffe
