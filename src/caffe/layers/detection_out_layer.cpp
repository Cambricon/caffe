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

#include <algorithm>
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "boost/filesystem.hpp"
#include "boost/foreach.hpp"

#include "caffe/layers/region_loss_layer.hpp"
#include "caffe/layers/detection_out_layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/bbox.hpp"

namespace caffe {
template <typename Dtype>
void DetectionOutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const DetectionOutParameter& detection_out_param =
      this->layer_param_.detection_out_param();
  CHECK(detection_out_param.has_num_classes()) << "Must specify num_classes";
  side_ = detection_out_param.side();
  num_classes_ = detection_out_param.num_classes();
  num_box_ = detection_out_param.num_box();
  coords_ = detection_out_param.coords();
  confidence_threshold_ = detection_out_param.confidence_threshold();
  nms_threshold_ = detection_out_param.nms_threshold();

  for (int c = 0; c < detection_out_param.biases_size(); ++c) {
     biases_.push_back(detection_out_param.biases(c));
  }

  if (detection_out_param.has_label_map_file()) {
    string label_map_file = detection_out_param.label_map_file();
    if (label_map_file.empty()) {
      LOG(WARNING) << "Provide label_map_file if out results to files.";
    } else {
      LabelMap label_map;
      CHECK(ReadProtoFromTextFile(label_map_file, &label_map))
          << "Failed to read label map file: " << label_map_file;
      CHECK(MapLabelToName(label_map, true, &label_to_name_))
          << "Failed to convert label to name.";
    }
  }
}

template <typename Dtype>
void DetectionOutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape(2, 1);
  top_shape.push_back(1);
  top_shape.push_back(7);
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void DetectionOutLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int num = bottom[0]->num();
  Blob<Dtype> swap;
  swap.Reshape(bottom[0]->num(), bottom[0]->height()*bottom[0]->width(),
               num_box_, bottom[0]->channels() / num_box_);
  Dtype* swap_data = swap.mutable_cpu_data();
  int index = 0;
  for (int b = 0; b < bottom[0]->num(); ++b)
    for (int h = 0; h < bottom[0]->height(); ++h)
      for (int w = 0; w < bottom[0]->width(); ++w)
        for (int c = 0; c < bottom[0]->channels(); ++c) {
          swap_data[index++] = bottom[0]->data_at(b, c, h, w);
        }

  vector< PredictionResult<Dtype> > predicts;
  PredictionResult<Dtype> predict;
  predicts.clear();
  vector<vector<float>> results;
  map<int, int> valid_index;
  for (int b = 0; b < swap.num(); ++b) {
    for (int j = 0; j < side_; ++j)
      for (int i = 0; i < side_; ++i)
        for (int n = 0; n < num_box_; ++n) {
          int index = b * swap.channels() * swap.height() * swap.width() +
                      (j * side_ + i) * swap.height() * swap.width() +
                      n * swap.width();
          CHECK_EQ(swap_data[index], swap.data_at(b, j * side_ + i, n, 0));
          get_region_box(swap_data, &predict, biases_,
                         n, index, i, j, side_, side_);
          predict.objScore = sigmoid(swap_data[index + 4]);
          map<int, float> prob_index;
          class_index_and_score(swap_data + index + 5,
                                num_classes_, confidence_threshold_, &prob_index);
          for (int k = 0; k < num_classes_; k++) {
            predict.confidence = predict.objScore * prob_index[k];
            if (predict.confidence > confidence_threshold_) {
              predict.classType = k;
              predicts.push_back(predict);
            }
          }
        }

    vector<int> idxes;
    int num_kept = 0;
    if (predicts.size() > 0) {
      ApplyNms(&predicts, &idxes, nms_threshold_, &results, b, num_classes_);
    }
    num_kept = results.size();
    vector<int> top_shape{1, 1, num_kept, 7};
    if (num_kept == 0) {
      LOG(INFO) << "Couldn't find any detections, Generate fake results per image";
      top_shape[2] = swap.num();
      top[0]->Reshape(top_shape);
      Dtype* top_data = top[0]->mutable_cpu_data();
      caffe_set<Dtype>(top[0]->count(), -1, top_data);
      // Generate fake results per image.
      for (int i = 0; i < num; ++i) {
        top_data[0] = i;
        top_data += 7;
      }
    }
  }
  vector<int> top_shape{1, static_cast<int>(results.size()), 1, 7};
  top[0]->Reshape(top_shape);
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int i = 0; i < results.size(); i++) {
    *top_data++ = results[i][0];  //  Image_Id
    *top_data++ = results[i][1];  //  label
    *top_data++ = results[i][2];  //  confidence
    *top_data++ = results[i][3];
    *top_data++ = results[i][4];
    *top_data++ = results[i][5];
    *top_data++ = results[i][6];
  }
}

INSTANTIATE_CLASS(DetectionOutLayer);

}  // namespace caffe
