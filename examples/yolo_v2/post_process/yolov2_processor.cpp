
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

#include "glog/logging.h"
#ifdef USE_MLU
#include "cnrt.h" // NOLINT
#include "runner.hpp"
#include "yolov2_processor.hpp"
#include "command_option.hpp"
#include "common_functions.hpp"


template <typename Dtype, template <typename> class Qtype>
void YoloV2Processor<Dtype, Qtype>::IntersectBBox(const NormalizedBBox& bbox1,
                   const NormalizedBBox& bbox2,
                   NormalizedBBox* intersect_bbox) {
  if (bbox2.xmin() > bbox1.xmax() || bbox2.xmax() < bbox1.xmin() ||
      bbox2.ymin() > bbox1.ymax() || bbox2.ymax() < bbox1.ymin()) {
    // Return [0, 0, 0, 0] if there is no intersection.
    intersect_bbox->set_xmin(0);
    intersect_bbox->set_ymin(0);
    intersect_bbox->set_xmax(0);
    intersect_bbox->set_ymax(0);
  } else {
    intersect_bbox->set_xmin(std::max(bbox1.xmin(), bbox2.xmin()));
    intersect_bbox->set_ymin(std::max(bbox1.ymin(), bbox2.ymin()));
    intersect_bbox->set_xmax(std::min(bbox1.xmax(), bbox2.xmax()));
    intersect_bbox->set_ymax(std::min(bbox1.ymax(), bbox2.ymax()));
  }
}

template <typename Dtype, template <typename> class Qtype>
float YoloV2Processor<Dtype, Qtype>::BBoxSize(const NormalizedBBox& bbox,
               const bool normalized) {
  if (bbox.xmax() < bbox.xmin() || bbox.ymax() < bbox.ymin()) {
    // If bbox is invalid (e.g. xmax < xmin or ymax < ymin), return 0.
    return 0;
  } else {
    if (bbox.has_size()) {
      return bbox.size();
    } else {
      float width = bbox.xmax() - bbox.xmin();
      float height = bbox.ymax() - bbox.ymin();
      if (normalized) {
        return width * height;
      } else {
        // If bbox is not within range [0, 1].
        return (width + 1) * (height + 1);
      }
    }
  }
}

template <typename Dtype, template <typename> class Qtype>
float YoloV2Processor<Dtype, Qtype>::JaccardOverlap(
            const NormalizedBBox& bbox1, const NormalizedBBox& bbox2,
            const bool normalized) {
  NormalizedBBox intersect_bbox;
  IntersectBBox(bbox1, bbox2, &intersect_bbox);
  float intersect_width, intersect_height;
  if (normalized) {
    intersect_width = intersect_bbox.xmax() - intersect_bbox.xmin();
    intersect_height = intersect_bbox.ymax() - intersect_bbox.ymin();
  } else {
    intersect_width = intersect_bbox.xmax() - intersect_bbox.xmin() + 1;
    intersect_height = intersect_bbox.ymax() - intersect_bbox.ymin() + 1;
  }
  if (intersect_width > 0 && intersect_height > 0) {
    float intersect_size = intersect_width * intersect_height;
    float bbox1_size = BBoxSize(bbox1);
    float bbox2_size = BBoxSize(bbox2);
    return intersect_size / (bbox1_size + bbox2_size - intersect_size);
  } else {
    return 0.;
  }
}

template <typename Dtype, template <typename> class Qtype>
void YoloV2Processor<Dtype, Qtype>::setNormalizedBBox(NormalizedBBox* bbox,
                       float x,
                       float y,
                       float w,
                       float h) {
  float xmin = x - w / 2.0;
  float xmax = x + w / 2.0;
  float ymin = y - h / 2.0;
  float ymax = y + h / 2.0;

  if (xmin < 0.0) {
    xmin = 0.0;
  }
  if (xmax > 1.0) {
    xmax = 1.0;
  }
  if (ymin < 0.0) {
    ymin = 0.0;
  }
  if (ymax > 1.0) {
    ymax = 1.0;
  }
  bbox->set_xmin(xmin);
  bbox->set_ymin(ymin);
  bbox->set_xmax(xmax);
  bbox->set_ymax(ymax);
  float bbox_size = BBoxSize(*bbox, true);
  bbox->set_size(bbox_size);
}

template <typename Dtype, template <typename> class Qtype>
void YoloV2Processor<Dtype, Qtype>::class_index_and_score(float* input,
                           int classes,
                           float confidence_threshold_,
                           map<int, float> *prob_index) {
  float sum = 0;
  float large = input[0];
  for (int i = 0; i < classes; ++i) {
    if (input[i] > large)
      large = input[i];
  }
  for (int i = 0; i < classes; ++i) {
    float e = exp(input[i] - large);
    sum += e;
    input[i] = e;
  }

  for (int i = 0; i < classes; ++i) {
    input[i] = input[i] / sum;
  }
  large = input[0];

  for (int i = 0; i < classes; ++i) {
    if (input[i] > large) {
      large = input[i];
    }
  }
  for (int i = 0; i < classes; i++) {
    if (input[i] > confidence_threshold_) {
      (*prob_index)[i] = input[i];
    }
  }
}

template <typename Dtype, template <typename> class Qtype>
void YoloV2Processor<Dtype, Qtype>::get_region_box(float* x, PredictionResult* predict,
                    vector<float> biases, int n, int index,
                    int i, int j, int w, int h) {
  predict->x = (i + sigmoid(x[index + 0])) / w;
  predict->y = (j + sigmoid(x[index + 1])) / h;
  predict->w = exp(x[index + 2]) * biases[2 * n] / w;
  predict->h = exp(x[index + 3]) * biases[2 * n + 1] / h;
}

template <typename Dtype, template <typename> class Qtype>
void YoloV2Processor<Dtype, Qtype>::ApplyNms(vector<PredictionResult>* boxes,
                    vector<int>* idxes, float threshold,
                    vector< vector<float>>* result,
                    int b, int num_classes_) {
  for (int k = 0; k < num_classes_; k++) {
    vector<PredictionResult> cur_boxes;
    for (int i = 0; i < (*boxes).size(); i++) {
      if ((*boxes)[i].classType == k) {
        cur_boxes.push_back((*boxes)[i]);
      }
    }
    if (cur_boxes.empty()) continue;
    std::sort(cur_boxes.begin(), cur_boxes.end(),
        [](const PredictionResult& pa, const PredictionResult& pb){
          float diff = pa.confidence - pb.confidence;
          if (diff > 0) return 1;
          return 0;
        });

    map<int, int> idx_map;
    for (int i = 0; i < cur_boxes.size() - 1; ++i) {
      if (idx_map.find(i) != idx_map.end()) {
        continue;
      }
      for (int j = i + 1; j < cur_boxes.size(); ++j) {
        if (idx_map.find(j) != idx_map.end()) {
          continue;
        }
        NormalizedBBox Bbox1, Bbox2;
        setNormalizedBBox(&Bbox1, cur_boxes[i].x, cur_boxes[i].y,
                          cur_boxes[i].w, cur_boxes[i].h);
        setNormalizedBBox(&Bbox2, cur_boxes[j].x, cur_boxes[j].y,
                          cur_boxes[j].w, cur_boxes[j].h);

        float overlap = JaccardOverlap(Bbox1, Bbox2, true);

        if (overlap >= threshold) {
          if (cur_boxes[i].confidence > cur_boxes[j].confidence) idx_map[j] = 1;
          else
            idx_map[i] = 1;
        }
      }
    }
    for (int i = 0; i < cur_boxes.size(); ++i) {
      if (idx_map.find(i) == idx_map.end()) {
        std::vector<float> tmp;
        tmp.push_back(b);
        tmp.push_back(cur_boxes[i].classType);
        tmp.push_back(cur_boxes[i].confidence);
        tmp.push_back(cur_boxes[i].x);
        tmp.push_back(cur_boxes[i].y);
        tmp.push_back(cur_boxes[i].w);
        tmp.push_back(cur_boxes[i].h);
        (*result).push_back(tmp);
      }
    }
  }
}

template <typename Dtype, template <typename> class Qtype>
std::vector<std::vector<float> > YoloV2Processor<Dtype, Qtype>::detection_out(
                                  float* net_output,
                                  int out_n, int out_c, int out_h, int out_w) {
  vector< vector<float> > result;

  // MLU output
  if (FLAGS_Bangop != 0) {
    for (int b = 0; b < out_n; b++) {
      for (int i = 0+b*out_c*out_w; i < (1+b)*out_c*out_w; i = i+out_w) {
        if (net_output[i] !=  -1.0) {
          std::vector<float> temp;
          temp.push_back(b);  // Image_Id
          temp.push_back(net_output[i+1]);  // label
          temp.push_back(net_output[i+2]);  // confidence
          temp.push_back(net_output[i+3]);  // x
          temp.push_back(net_output[i+4]);  // y
          temp.push_back(net_output[i+5]);  // w
          temp.push_back(net_output[i+6]);  // h
          result.push_back(temp);
        }
      }
    }
  } else {
    float* swap_data = reinterpret_cast<float*>
                   (malloc(sizeof(float) * out_n * out_c * out_h * out_w));
    int index = 0;
    for (int b = 0; b < out_n; ++b)
      for (int h = 0; h < out_h; ++h)
        for (int w = 0; w < out_w; ++w)
          for (int c = 0; c < out_c; ++c) {
            swap_data[index++] =
             net_output[ b*out_c*out_h*out_w + c*out_h*out_w + h*out_w +w];
          }

    vector<PredictionResult> predicts;
    PredictionResult predict;
    predicts.clear();

    vector<float> biases_;
    biases_.push_back(1.3221);
    biases_.push_back(1.73145);
    biases_.push_back(3.19275);
    biases_.push_back(4.00944);
    biases_.push_back(5.05587);
    biases_.push_back(8.09892);
    biases_.push_back(9.47112);
    biases_.push_back(4.84053);
    biases_.push_back(11.2364);
    biases_.push_back(10.0071);

    int side_ = 13;
    int num_box_ = 5;
    int num_classes_ = 20;
    float nms_threshold_ = 0.45;
    float confidence_threshold_  = 0.005;
    if (FLAGS_confidencethreshold) {
      confidence_threshold_  = FLAGS_confidencethreshold;
    }
    for (int b = 0; b < out_n; ++b) {
      for (int j = 0; j < side_; ++j)
        for (int i = 0; i < side_; ++i)
          for (int n = 0; n < num_box_; ++n) {
            int index = b * out_c * out_h * out_w + (j * side_ + i) * out_c +
                        n * out_c / num_box_;
            get_region_box(swap_data, &predict, biases_, n,
                           index, i, j, side_, side_);
            predict.objScore = sigmoid(swap_data[index + 4]);
            map<int, float> prob_index;
            class_index_and_score(swap_data+index + 5, num_classes_,
                                  confidence_threshold_, &prob_index);
            for (int k = 0; k < num_classes_; k++) {
              predict.confidence = predict.objScore * prob_index[k];
              if (predict.confidence > confidence_threshold_) {
                predict.classType = k;
                predicts.push_back(predict);
              }
            }
          }
      vector<int> idxes;
      if (predicts.size() > 0) {
        ApplyNms(&predicts, &idxes, nms_threshold_, &result, b, num_classes_);
      }
      predicts.clear();
    }
    free(swap_data);
  }
  return result;
}
template <typename Dtype, template <typename> class Qtype>
void YoloV2Processor<Dtype, Qtype>::get_point_position(const vector<float> pos,
                        cv::Point* p1, cv::Point* p2, int h, int w) {
  int left = (pos[3] - pos[5] / 2) * w;
  int right = (pos[3] + pos[5] / 2) * w;
  int top = (pos[4] - pos[6] / 2) * h;
  int bottom = (pos[4] + pos[6] / 2) * h;
  if (left < 0) left = 0;
  if (top < 0) top = 0;
  if (right > w) right = w;
  if (bottom > h) bottom = h;
  p1->x = left;
  p1->y = top;
  p2->x = right;
  p2->y = bottom;
  return;
}

template <typename Dtype, template <typename> class Qtype>
void YoloV2Processor<Dtype, Qtype>::correct_region_boxes(vector<vector<float>>* boxes,
                        const cv::Mat image) {
  int new_w = 0;
  int new_h = 0;
  int netw = 416;
  int neth = 416;
  int w = image.cols;
  int h = image.rows;
  if ((static_cast<float>(netw)/w) < (static_cast<float>(neth)/h)) {
    new_w = netw;
    new_h = (h * netw)/w;
  }  else {
    new_h = neth;
    new_w = (w * neth)/h;
  }
  for (int i = 0; i < (*boxes).size(); i++) {
    (*boxes)[i][3] = ((*boxes)[i][3] - (netw - new_w)/2./netw) /
                      (static_cast<float>(new_w)/netw);
    (*boxes)[i][4] = ((*boxes)[i][4] - (neth - new_h)/2./neth) /
                      (static_cast<float>(new_h)/neth);
    (*boxes)[i][5] *= static_cast<float>(netw)/new_w;
    (*boxes)[i][6] *= static_cast<float>(neth)/new_h;
  }
}

template <typename Dtype, template <typename> class Qtype>
void YoloV2Processor<Dtype, Qtype>::WriteVisualizeBBox_offline(
                   const vector<cv::Mat>& images,
                   const vector<vector<float>>& detections,
                   const vector<string>& labels_,
                   const vector<string>& img_names,
                   const int from, const int to) {
  // Retrieve detections.
  for (int i = from; i < to; ++i) {
    cv::Mat tmp = images[i];
    cv::Mat image;
    if (FLAGS_yuv) {
      image = yuv420sp2Bgr24(tmp);
    } else {
      image = tmp;
    }

    vector<vector<float>> boxes = detections;
    correct_region_boxes(&boxes, image);

    std::string name = img_names[i];
    int pos_map = img_names[i].rfind("/");
    if (pos_map > 0 && pos_map < img_names[i].size()) {
      name = name.substr(pos_map + 1);
    }
    pos_map = name.rfind(".");
    if (pos_map > 0 && pos_map < name.size()) {
      name = name.substr(0, pos_map);
    }
    name = FLAGS_outputdir + "/" + name + ".txt";
    std::ofstream file_map(name);
    for (auto box : boxes) {
      if (box[0] != i) {
        continue;
      }
      cv::Point p1, p2;
      get_point_position(box, &p1, &p2,
                         image.rows, image.cols);
      cv::rectangle(image, p1, p2, cv::Scalar(0, 255, 0), 8, 8, 0);
      std::stringstream s0;
      s0 << box[2];
      string s1 = s0.str();
      cv::putText(image, labels_[box[1]], cv::Point(p1.x, p1.y - 10),
                  2, 0.5, cv::Scalar(255, 0, 0), 0, 8, 0);
      cv::putText(image, s1.c_str(), cv::Point(p1.x, (p1.y + p2.y) / 2 + 10),
                  2, 0.5, cv::Scalar(255, 0, 0), 0, 8, 0);
      LOG(INFO) << ">>> label: " << labels_[box[1]] << " score: "
                << box[2] << " , position : ";
      for (int idx = 0; idx < 4; ++idx) {
        LOG(INFO) << box[3 + idx] << " ";
      }
      LOG(INFO);
      LOG(INFO) << "detection result: " << box[1] << " " << box[2]
          << " " << box[3] << " " << box[4] << " " << box[5]
          << " " << box[6];
      // the order for mAP is: label score x_min y_min x_max y_max
      file_map << labels_[box[1]] << " "
          << static_cast<float>(box[2]) << " "
          << static_cast<float>(p1.x) / image.cols << " "
          << static_cast<float>(p1.y) / image.rows << " "
          << static_cast<float>(p2.x) / image.cols << " "
          << static_cast<float>(p2.y) / image.rows
          << std::endl;
    }
    file_map.close();

    stringstream ss;
    string outFile;
    int position = img_names[i].find_last_of('/');
    string fileName(img_names[i].substr(position + 1));
    ss << FLAGS_outputdir << "/yolov2_" << fileName;
    if (FLAGS_yuv) ss << ".jpg";
    ss >> outFile;

    cv::imwrite(outFile.c_str(), image);
  }
}

template <typename Dtype, template <typename> class Qtype>
void YoloV2Processor<Dtype, Qtype>::WriteVisualizeBBox_online(
                   const vector<cv::Mat>& images,
                   const vector<vector<vector<float>>> detections,
                   const vector<string>& labelToDisplayName,
                   const vector<string>& imageNames) {
  float scale = 1.0;
  for (int imageIdx = 0; imageIdx < detections.size(); ++imageIdx) {
    if (imageNames[imageIdx] == "null")
      continue;

    std::string name = imageNames[imageIdx];
    int positionMap = imageNames[imageIdx].rfind("/");
    if (positionMap > 0 && positionMap < imageNames[imageIdx].size()) {
      name = name.substr(positionMap + 1);
    }
    positionMap = name.rfind(".");
    if (positionMap > 0 && positionMap < name.size()) {
      name = name.substr(0, positionMap);
    }
    name = FLAGS_outputdir + "/" + name + ".txt";
    std::ofstream fileMap(name);
    vector<vector<float> > boxes = detections[imageIdx];
    cv::Mat tmp = images[imageIdx];
    cv::Mat image;
    if (FLAGS_yuv) {
      image = yuv420sp2Bgr24(tmp);
    } else {
      image = tmp;
    }
    correct_region_boxes(&boxes, image);
    for (int j = 0; j < boxes.size(); ++j) {
      if (boxes[j][1] == -1 && boxes[j][2] == -1 &&
          boxes[j][3] == -1 && boxes[j][4] == -1 &&
          boxes[j][5] == -1) {
        continue;
      }
      cv::Point p1, p2;
      get_point_position(boxes[j], &p1, &p2,
          image.size().height, image.size().width);
      p1.x = p1.x / scale;
      p1.y = p1.y / scale;
      p2.x = p2.x / scale;
      p2.y = p2.y / scale;
      cv::rectangle(image, p1, p2, cv::Scalar(0, 255, 0), 8, 8, 0);
      std::stringstream s0;
      s0 << boxes[j][2];
      string s1 = s0.str();
      cv::putText(image, labelToDisplayName[boxes[j][1]],
                  cv::Point(p1.x, p1.y - 10),
                  2, 0.5, cv::Scalar(255, 0, 0), 0, 8, 0);
      cv::putText(image, s1.c_str(),
                  cv::Point(p1.x, (p1.y + p2.y) / 2 + 10),
                  2, 0.5, cv::Scalar(255, 0, 0), 0, 8, 0);
      LOG(INFO) << ">>> label: " << labelToDisplayName[boxes[j][1]] << " score: "
                << boxes[j][2] << " , position : ";
      for (int idx = 0; idx < 4; ++idx) {
        LOG(INFO) << boxes[j][3 + idx] << " ";
      }
      LOG(INFO);
      LOG(INFO) << "detection result: " << boxes[j][1] << " " << boxes[j][2]
          << " " << boxes[j][3] << " " << boxes[j][4] << " " << boxes[j][5]
          << " " << boxes[j][6];
      // the order for mAP is: label score x_min y_min x_max y_max
      fileMap << labelToDisplayName[boxes[j][1]] << " "
          << static_cast<float>(boxes[j][2]) << " "
          << static_cast<float>(p1.x) / image.size().width << " "
          << static_cast<float>(p1.y) / image.size().height << " "
          << static_cast<float>(p2.x) / image.size().width << " "
          << static_cast<float>(p2.y) / image.size().height
          << std::endl;
    }
    fileMap.close();

    stringstream ss;
    string outFile;
    int position = imageNames[imageIdx].find_last_of('/');
    string fileName(imageNames[imageIdx].substr(position + 1));
    ss << FLAGS_outputdir << "/yolov2_" << fileName;
    if (FLAGS_yuv) ss << ".jpg";
    ss >> outFile;

    cv::imwrite(outFile.c_str(), images[imageIdx]);
  }
}

template <typename Dtype, template <typename> class Qtype>
void YoloV2Processor<Dtype, Qtype>::readLabels(vector<std::string>* labels) {
  if (!FLAGS_labels.empty()) {
    std::ifstream file(FLAGS_labels);
    std::string line;
    while (std::getline(file, line)) {
      labels->push_back(line);
    }
    file.close();
  }
}

INSTANTIATE_ALL_CLASS(YoloV2Processor);
#endif
