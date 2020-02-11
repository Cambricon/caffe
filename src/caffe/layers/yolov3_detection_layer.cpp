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

#include <cmath>
#include <vector>

#include "caffe/layers/yolov3_detection_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// transpose data order to 255*169->169*255
template <typename Dtype>
void transpose(Blob<Dtype>* bot, int pos0, int pos1) {
  int batch = bot->num();
  // count is single batch size
  int count = bot->count(1);
  int channels = bot->shape(pos0);
  int hw = bot->shape(pos1);
  CHECK_EQ(count, hw * channels) << "count should equal to channels* hw";
  Dtype* tmp_data = new Dtype[count]();
  Dtype* bot_data = bot->mutable_cpu_data();
  for (int n = 0; n < batch; n++) {
    // transpose the bot data every batch
    Dtype* bot_buffer = bot_data + n * count;
    caffe_copy(count, bot_buffer, tmp_data);
    int nr = 0;
    for (int i = 0; i < hw; i++) {
      for (int j = 0; j < channels; j++) {
        bot_buffer[nr++] = tmp_data[j * hw + i];
      }
    }
  }
  free(tmp_data);
}

template <typename Dtype>
void xy_add_mul(Blob<Dtype>* bot, const vector<Dtype>& xy_offsets, int stride) {
  int step = bot->width();
  int batch = bot->num();
  int count = bot->count(1);
  Dtype* bot_data = bot->mutable_cpu_data();
  for (int n = 0; n < batch; n++) {
    Dtype* bot_buffer = bot_data + n * count;
    int temp_index = 0;
    for (int i = 0; i < count; i += step) {
      bot_buffer[i] = stride * (bot_buffer[i] + xy_offsets[temp_index++]);
      bot_buffer[i + 1] =
          stride * (bot_buffer[i + 1] + xy_offsets[temp_index++]);
    }
  }
}

// sigmoid vectors
template <typename Dtype>
void sigmoid(Blob<Dtype>* bot, int colsLeft, int colsRight) {
  Dtype* bot_data = bot->mutable_cpu_data();
  int step = bot->width();
  int batch = bot->num();
  int count = bot->count() / batch;
  for (int n = 0; n < batch; n++) {
    Dtype* bot_buffer = bot_data + n * count;
    for (int i = 0; i < count; i += step) {
      for (int j = colsLeft; j < colsRight; j++) {
        bot_buffer[i + j] = 1 / (1 + std::exp(-bot_buffer[i + j]));
      }
    }
  }
}

template <typename Dtype>
void anchor_mul(Blob<Dtype>* bot, const Dtype* anchor, int im_h, int im_w) {
  Dtype* bot_data = bot->mutable_cpu_data();
  int step = bot->width();
  int batch = bot->num();
  int count = bot->count(1);
  for (int n = 0; n < batch; n++) {
    Dtype* bot_buffer = bot_data + n * count;
    int temp_index = 0;
    for (int i = 0; i < count; i += step) {
      bot_buffer[i + 2] =
          std::exp(bot_buffer[i + 2]) * anchor[(temp_index++) % 6];
      bot_buffer[i + 3] =
          std::exp(bot_buffer[i + 3]) * anchor[(temp_index++) % 6];
    }
  }
}

template <typename Dtype>
void concatenate(Blob<Dtype>* concat_bot, const vector<Blob<Dtype>*>& bottom) {
  int batch = bottom[0]->num();
  int top_batch_count = 0;
  for (int i = 0; i < bottom.size(); i++) {
    top_batch_count += bottom[i]->count(1);
  }
  Dtype* concat_data = concat_bot->mutable_cpu_data();
  for (int n = 0; n < batch; n++) {
    int top_step_count = 0;
    for (int i = 0; i < bottom.size(); i++) {
      const Dtype* bot_data = bottom[i]->cpu_data();
      int bot_batch_count = bottom[i]->count();
      int bot_channel_count = bottom[i]->count(1);
      caffe_copy(bot_channel_count, bot_data + n * bot_batch_count,
                 concat_data + n * top_batch_count + top_step_count);
      top_step_count += bot_channel_count;
    }
  }
}

template <typename Dtype>
void fill_zeros(Blob<Dtype>* bot, int cols, float confidence) {
  Dtype* bot_data = bot->mutable_cpu_data();
  int step = bot->width();
  int count = bot->count();
  for (int i = 0; i < count; i += step) {
    if (bot_data[i + cols] > confidence)
      continue;
    else
      bot_data[i + cols] = 0;
  }
}

template <typename Dtype>
void conf_filter(Blob<Dtype>* bot, float confidence, int num_classes,
                 vector<vector<vector<Dtype>>>* detections) {
  const Dtype* bot_data = bot->cpu_data();
  int batch = bot->num();
  int step = bot->width();
  int count = bot->count(1);

  for (int n = 0; n < batch; ++n) {
    int num_boxes = 0;
    const Dtype* batch_data = bot_data + n * count;
    for (int i = 0; i < count; i += step) {
      if (batch_data[i + 4] > confidence) {
        ++num_boxes;
      }
    }
    // add current class
    detections->push_back(vector<vector<Dtype>>(
        num_boxes, vector<Dtype>(5 + num_classes + 1, 0)));
  }
  for (int n = 0; n < batch; ++n) {
    int box_count = 0;
    const Dtype* batch_data = bot_data + n * count;
    for (int i = 0; i < count; i += step) {
      if (batch_data[i + 4] <= confidence) {
        continue;
      } else {
        (*detections)[n][box_count][0] =
            batch_data[i + 0] - batch_data[i + 2] / 2;
        (*detections)[n][box_count][2] =
            batch_data[i + 0] + batch_data[i + 2] / 2;
        (*detections)[n][box_count][1] =
            batch_data[i + 1] - batch_data[i + 3] / 2;
        (*detections)[n][box_count][3] =
            batch_data[i + 1] + batch_data[i + 3] / 2;

        (*detections)[n][box_count][4] = batch_data[i + 4];
        Dtype objectness = bot_data[i + 4];
        for (int j = 0; j < num_classes; ++j) {
          Dtype prob = objectness * batch_data[i + 5 + j];
          (*detections)[n][box_count][5 + j] = (prob > confidence) ? prob : 0;
        }
        // current class
        (*detections)[n][box_count][num_classes + 5] = -1;
        ++box_count;
      }
    }
  }
}

template <typename Dtype>
Dtype box_iou(int box1_id, int box2_id,
              const vector<vector<Dtype>>& detections) {
  Dtype inter_x1 = std::max(detections[box1_id][0], detections[box2_id][0]);
  Dtype inter_y1 = std::max(detections[box1_id][1], detections[box2_id][1]);
  Dtype inter_x2 = std::min(detections[box1_id][2], detections[box2_id][2]);
  Dtype inter_y2 = std::min(detections[box1_id][3], detections[box2_id][3]);
  Dtype inter_area, first_area, next_area, union_area, iou;
  if ((inter_x2 - inter_x1 + 1 > 0) && (inter_y2 - inter_y1 + 1 > 0))
    inter_area = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1);
  else
    inter_area = 0;
  first_area = (detections[box1_id][2] - detections[box1_id][0] + 1) *
               (detections[box1_id][3] - detections[box1_id][1] + 1);
  next_area = (detections[box2_id][2] - detections[box2_id][0] + 1) *
              (detections[box2_id][3] - detections[box2_id][1] + 1);
  union_area = first_area + next_area - inter_area;
  iou = inter_area / union_area;
  return iou;
}

template <typename Dtype>
void do_nms_sort(vector<Blob<Dtype>*> top, Blob<Dtype>* bot, int num_classes,
    int im_h, int im_w, float confidence, float nms_thresh, int num_box) {
  int batch = bot->num();
  Dtype* top_buffer = top[0]->mutable_cpu_data();
  caffe_set(top[0]->count(), Dtype(-1), top_buffer);
  int box_size = 0;

  vector<vector<vector<Dtype>>> detections;
  conf_filter(bot, confidence, num_classes, &detections);
  for (int n = 0; n < batch; ++n) {
    for (int k = 0; k < num_classes; ++k) {
      for (int idx = 0; idx < detections[n].size(); ++idx) {
        detections[n][idx][num_classes + 5] = k;
      }
      std::sort(
          detections[n].begin(), detections[n].end(),
          [](const vector<Dtype>& pa, const vector<Dtype>& pb) {
            return (pa[pa[pa.size() - 1] + 5] > pb[pb[pb.size() - 1] + 5]);
          });
      for (int i = 0; i < detections[n].size(); ++i) {
        if (detections[n][i][5 + k] == 0) {
          continue;
        }
        int box1_id = i;
        for (int j = i + 1; j < detections[n].size(); ++j) {
          int box2_id = j;
          if (box_iou(box1_id, box2_id, detections[n]) > nms_thresh) {
            detections[n][j][5 + k] = 0;
          }
        }
      }
    }
    for (int i = 0; i < detections[n].size(); ++i) {
      for (int j = 0; j < num_classes; ++j) {
        if (detections[n][i][j + 5] > confidence) {
          top_buffer[64 + box_size * 7 + 0] = n;
          top_buffer[64 + box_size * 7 + 1] = j;
          top_buffer[64 + box_size * 7 + 2] = detections[n][i][j + 5];
          top_buffer[64 + box_size * 7 + 3] = detections[n][i][0] / im_w;
          top_buffer[64 + box_size * 7 + 4] = detections[n][i][1] / im_h;
          top_buffer[64 + box_size * 7 + 5] = detections[n][i][2] / im_w;
          top_buffer[64 + box_size * 7 + 6] = detections[n][i][3] / im_h;
          ++box_size;
        }
      }
    }
  }
  top_buffer[0] = box_size;
}

template <typename Dtype>
vector<vector<Dtype>> filter_boxes(vector<vector<Dtype>>* all_boxes,
                                   vector<Dtype>* max_class_score,
                                   vector<Dtype>* max_class_idx) {
  vector<vector<Dtype>> temp(all_boxes->size(), vector<Dtype>(5 + 2, 0));
  for (int i = 0; i < all_boxes->size(); i++) {
    for (int j = 0; j < 7; j++) {
      if (j < 5)
        temp[i][j] = (*all_boxes)[i][j];
      else if (j == 5)
        temp[i][j] = (*max_class_score)[i];
      else if (j == 6)
        temp[i][j] = (*max_class_idx)[i];
    }
  }
  vector<vector<Dtype>> vec;
  for (int m = 0; m < temp.size(); m++) {
    if (temp[m][4] == 0)
      continue;
    else
      vec.push_back(temp[m]);
  }
  return vec;
}

template <typename Dtype>
void unique_vector(vector<vector<Dtype>>* input_vector,
                   vector<Dtype>* output_vector) {
  // (*input_vector)[i][6] means label index
  for (int i = 0; i < input_vector->size(); i++) {
    (*output_vector).push_back((*input_vector)[i][6]);
  }
  sort((*output_vector).begin(), (*output_vector).end());
  auto new_end = unique((*output_vector).begin(), (*output_vector).end());
  (*output_vector).erase(new_end, (*output_vector).end());
}

template <typename Dtype>
Dtype findMax(vector<Dtype> vec) {
  Dtype max = -999;
  for (auto v : vec) {
    if (max < v) max = v;
  }
  return max;
}

template <typename Dtype>
int getPositionOfMax(vector<Dtype> vec, float max) {
  auto distance = find(vec.begin(), vec.end(), max);
  return distance - vec.begin();
}

template <typename Dtype>
void nms_by_classes(vector<vector<Dtype>> sort_boxes, vector<Dtype>* ious,
                    int start) {
  for (int i = start + 1; i < sort_boxes.size(); i++) {
    float first_x1 = sort_boxes[start][0];
    float first_y1 = sort_boxes[start][1];
    float first_x2 = sort_boxes[start][2];
    float first_y2 = sort_boxes[start][3];

    float next_x1 = sort_boxes[i][0];
    float next_y1 = sort_boxes[i][1];
    float next_x2 = sort_boxes[i][2];
    float next_y2 = sort_boxes[i][3];

    float inter_x1 = std::max(first_x1, next_x1);
    float inter_y1 = std::max(first_y1, next_y1);
    float inter_x2 = std::min(first_x2, next_x2);
    float inter_y2 = std::min(first_y2, next_y2);
    float inter_area, first_area, next_area, union_area, iou;
    if ((inter_x2 - inter_x1 + 1 > 0) && (inter_y2 - inter_y1 + 1 > 0))
      inter_area = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1);
    else
      inter_area = 0;
    first_area = (first_x2 - first_x1 + 1) * (first_y2 - first_y1 + 1);
    next_area = (next_x2 - next_x1 + 1) * (next_y2 - next_y1 + 1);
    union_area = first_area + next_area - inter_area;
    iou = inter_area / union_area;
    (*ious).push_back(iou);
  }
}

template <typename Dtype>
void Yolov3DetectionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
  Yolov3DetectionParameter yolo_param = this->layer_param_.yolov3_param();
  num_classes_ = yolo_param.num_classes();
  anchor_num_ = yolo_param.anchor_num();
  int mask_size = anchor_num_ * bottom.size();
  vector<int> const_shape = {1, 64, 1, 1};
  c_arr_blob_.Reshape(const_shape);
  h_arr_blob_.Reshape(const_shape);
  w_arr_blob_.Reshape(const_shape);
  // const_shape[1] = 2 * mask_size;
  const_shape[1] = 64;
  biases_blob_.Reshape(const_shape);
  int* c_arr_data = reinterpret_cast<int*>(c_arr_blob_.mutable_cpu_data());
  int* h_arr_data = reinterpret_cast<int*>(h_arr_blob_.mutable_cpu_data());
  int* w_arr_data = reinterpret_cast<int*>(w_arr_blob_.mutable_cpu_data());
  for (int i = 0; i < bottom.size(); i++) {
    c_arr_data[i] = bottom[i]->channels();
    w_arr_data[i] = bottom[i]->width();
    h_arr_data[i] = bottom[i]->height();
  }
  Dtype* biases_data = biases_blob_.mutable_cpu_data();
  CHECK_EQ(2 * mask_size, yolo_param.biases_size());
  for (int i = 0; i < 2 * mask_size; i++) {
    biases_data[i] = yolo_param.biases(i);
  }
  im_w_ = yolo_param.im_w();
  im_h_ = yolo_param.im_h();
  num_box_ = yolo_param.num_box();
  confidence_threshold_ = yolo_param.confidence_threshold();
  nms_threshold_ = yolo_param.nms_threshold();
}

template <typename Dtype>
void Yolov3DetectionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape(4, 1);
  top_shape[0] = 1;
  top_shape[1] = bottom[0]->num() * (7 * this->num_box_ + 64);
  top[0]->Reshape(top_shape);
  for (int i = 0; i < bottom.size(); i++) {
    vector<int> bottom_shape(4, 1);
    bottom_shape[0] = bottom[i]->num();
    bottom_shape[1] = bottom[i]->channels();
    bottom_shape[3] = bottom[i]->height() * bottom[i]->width();
    bottom[i]->Reshape(bottom_shape);
  }
}

template <typename Dtype>
void Yolov3DetectionLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int* w_arr_ = reinterpret_cast<const int*>(w_arr_blob_.cpu_data());
  // transpose the bottom data and cacculate boxes coordanates
  const Dtype* biases_ = biases_blob_.cpu_data();
  int bbox_attrs = 5 + num_classes_;
  for (int i = 0; i < bottom.size(); i++) {
    // hw looks like grid count in the input data
    int hw = bottom[i]->width();
    // use im_w as input dim because im_w == im_h
    int stride = im_w_ / std::sqrt(hw);
    x_y_offsets_.clear();
    // 255*169->169*255
    transpose(bottom[i], 1, 3);
    // bottom[i]->shape(3) == hw, now shape looks like 507*85
    bottom[i]->Reshape(bottom[i]->num(), hw * anchor_num_, 1, bbox_attrs);
    for (int k = 0; k < bottom[i]->channels() / anchor_num_; k++) {
      for (int j = 0; j < anchor_num_; j++) {
        x_y_offsets_.push_back(k % w_arr_[i]);
        x_y_offsets_.push_back(k / w_arr_[i]);
      }
    }
    // sigmoid 0,1 means only sigmoid position 0
    sigmoid(bottom[i], 0, 1);
    sigmoid(bottom[i], 1, 2);
    xy_add_mul(bottom[i], x_y_offsets_, stride);
    anchor_mul(bottom[i], &biases_[6 * i], im_h_, im_w_);
    sigmoid(bottom[i], 4, 5);
    sigmoid(bottom[i], 5, bbox_attrs);
  }
  vector<vector<float>> all_boxes, tmp_boxes;
  vector<vector<vector<float>>> final_boxes;
  Blob<Dtype> concat_bot;
  vector<int> concat_shape(4, 0);
  concat_shape[0] = bottom[0]->num();
  for (int i = 0; i < bottom.size(); i++) {
    concat_shape[1] += bottom[i]->channels();
  }
  concat_shape[2] = 1;
  concat_shape[3] = bbox_attrs;
  concat_bot.Reshape(concat_shape);
  // 10647*85
  concatenate(&concat_bot, bottom);
  do_nms_sort(top, &concat_bot, num_classes_, im_h_, im_w_,
              confidence_threshold_, nms_threshold_, this->num_box_);
}
#ifndef USE_CUDA
STUB_GPU(Yolov3DetectionLayer);
#endif

INSTANTIATE_CLASS(Yolov3DetectionLayer);

}  // namespace caffe
