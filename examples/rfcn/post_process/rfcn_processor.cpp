
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
#include "rfcn_processor.hpp"
#include "command_option.hpp"
#include "common_functions.hpp"

using std::vector;

const std::string objects[21] = {"__background__",  // NOLINT
                             "aeroplane", "bicycle", "bird", "boat",
                             "bottle", "bus", "car", "cat", "chair",
                             "cow", "diningtable", "dog", "horse",
                             "motorbike", "person", "pottedplant",
                             "sheep", "sofa", "train", "tvmonitor"};

float min(float a, int b) {
  return a < static_cast<float>(b) ? a : static_cast<float>(b);
}
float max(float a, int b) {
  return a > b ? a : static_cast<float>(b);
}

template <typename Dtype, template <typename> class Qtype>
void RfcnProcessor<Dtype, Qtype>::WriteVisualizeBBox_offline(
                   const vector<cv::Mat>& images,
                   const vector<vector<float>>& detections,
                   const vector<string>& labels_,
                   const vector<string>& img_names,
                   const int inHeight, const int inWidth,
                   const int from, const int to) {
  // Retrieve detections.
  for (int i = from; i < to; ++i) {
    cv::Mat tmp = images[i];
    cv::Mat image;
    cv::resize(tmp, image, cv::Size(inWidth, inHeight));
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
    for (auto box : detections) {
      if (box[6] != i) continue;
      if (box[4] == 0) continue;
      cv::Point p1, p2;
      p1.x = box[0];
      p1.y = box[1];
      p2.x = box[2];
      p2.y = box[3];
      cv::rectangle(image, p1, p2, cv::Scalar(0, 255, 0), 8, 8, 0);
      std::stringstream s0;
      s0 << box[5];
      string s1 = s0.str();
      putText(image, objects[static_cast<int>(box[4])],
              cv::Point(p1.x, (p1.y + p2.y) / 2 - 10), 2, 0.5, cv::Scalar(255, 0, 0), 0,
              8, 0);
      putText(image, s1.c_str(), cv::Point(p1.x, (p1.y + p2.y) / 2 + 10), 2, 0.5,
            cv::Scalar(255, 0, 0), 0, 8, 0);
      LOG(INFO) << "detection result: "<<"obj is "<<objects[static_cast<int>(box[4])]
          << ";" << box[4] << " " << box[5] << " " << box[0] << " " << box[1]
          << " " << box[2] << " " << box[3] << " " << box[6];
      // the order for mAP is: label score x_min y_min x_max y_max
      file_map << objects[static_cast<int>(box[4])] << " "
          << static_cast<float>(box[5]) << " "
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
    ss << FLAGS_outputdir << "/rfcn_" << fileName;
    ss >> outFile;
    cv::imwrite(outFile.c_str(), image);
  }
}

template <typename Dtype, template <typename> class Qtype>
void RfcnProcessor<Dtype, Qtype>::readLabels(vector<std::string>* labels) {
  if (!FLAGS_labels.empty()) {
    std::ifstream file(FLAGS_labels);
    std::string line;
    while (std::getline(file, line)) {
      labels->push_back(line);
    }
    file.close();
  }
}

template <typename Dtype, template <typename> class Qtype>
vector<vector<float>> RfcnProcessor<Dtype, Qtype>::detection_out(
    float* roiDatatmp, float* scoreDatatmp, float* boxDatatmp,
    int roiDataCount, int scoreDataCount, int boxDataCount, int batchsize,
    int inputWidth, int inputHeight) {
  vector<vector<float>> resultInfo;
  for (int n = 0; n < batchsize; n++) {
    float* roiData = roiDatatmp + n * roiDataCount;
    float* scoreData = scoreDatatmp + n * scoreDataCount;
    float* boxData = boxDatatmp + n * boxDataCount;
    std::vector<float> rois(roiDataCount);
    for (int i = 0; i < roiDataCount / 5; i++) {
      rois[i * 5] = 0;
      for (int j = 0; j < 4; j++) {
        rois[i * 5 + j + 1] = roiData[i * 5 + j];
      }
    }
    int batch = 304;
    int **boxes = reinterpret_cast<int **>(malloc(sizeof(int *) * batch));
    float **result = reinterpret_cast<float **>(malloc(sizeof(float *) * batch));
    int *size = reinterpret_cast<int *>(malloc(sizeof(int) * batch));
    int **use = reinterpret_cast<int **>(malloc(sizeof(int *) * batch));
    float **scoreCpu = reinterpret_cast<float **>(malloc(sizeof(float *) * batch));
    for (int i = 0; i < batch; i++) {
      boxes[i] = reinterpret_cast<int *>(malloc((sizeof(int)) * 8));
      scoreCpu[i] = reinterpret_cast<float *>(malloc((sizeof(float)) * 21));
      result[i] = reinterpret_cast<float *>(malloc((sizeof(float)) * 6));
      use[i] = reinterpret_cast<int *>(malloc(sizeof(int) * 21));
      for (int j = 0; j < 21; j++) {
        if (FLAGS_Bangop)
          scoreCpu[i][j] = scoreData[i * 21 + j];
        else
          scoreCpu[i][j] = scoreData[i * 32 + j];
        use[i][j] = 1;
      }
    }

    int mlu_option;
    if (FLAGS_Bangop)
      mlu_option = 0;
    else
      mlu_option = 1;

    for (int i = 0; i < batch; i++) {
      int width = rois[i * 5 + 3] - rois[i * 5 + 1] + 1;
      int height = rois[i * 5 + 4] - rois[i * 5 + 2] + 1;
      float c_x = rois[i * 5 + 1] + static_cast<float>(width) / 2;
      float c_y = rois[i * 5 + 2] + static_cast<float>(height) / 2;
      for (int j = 0; j < 2; j++) {
        float pc_x = c_x + width * boxData[i * (mlu_option ? 16 : 8) + 4];
        float pc_y = c_y + height * boxData[i * (mlu_option ? 16 : 8) + 5];
        float pc_w = exp(boxData[i * (mlu_option ? 16 : 8) + 6]) * width;
        float pc_h = exp(boxData[i * (mlu_option ? 16 : 8) + 7]) * height;
        boxes[i][4] = ((pc_x - 0.5 * pc_w) > 0 ? (pc_x - 0.5 * pc_w) : 0);
        boxes[i][5] = ((pc_y - 0.5 * pc_h) > 0 ? (pc_y - 0.5 * pc_h) : 0);
        boxes[i][6] =
            ((pc_x + 0.5 * pc_w) < inputWidth ? (pc_x + 0.5 * pc_w) : inputWidth - 1);
        boxes[i][7] =
            ((pc_y + 0.5 * pc_h) < inputHeight ? (pc_y + 0.5 * pc_h) : inputHeight - 1);
        size[i] = (boxes[i][7] - boxes[i][5]) * (boxes[i][6] - boxes[i][4]);
      }
    }

    float confidenceThreshold =  0.1;
    for (int t = 1; t < 21; t++) {
      for (int i = 0; i < batch; i++) {
        if (use[i][t]) {
          for (int j = i + 1; j < batch; j++) {
            int overlap_x = min(boxes[i][6], boxes[j][6]) - max(boxes[i][4], boxes[j][4]);
            int overlap_y = min(boxes[i][7], boxes[j][7]) - max(boxes[i][5], boxes[j][5]);
            int overlap =
                (overlap_x > 0 ? overlap_x : 0) * (overlap_y > 0 ? overlap_y : 0);
            float nms = static_cast<float>(overlap) /
                        static_cast<float>(size[i] + size[j] - overlap);
            if (nms < 0.3) continue;
            if (scoreCpu[j][t] > scoreCpu[i][t]) {
              use[i][t] = 0;
              break;
            } else if (scoreCpu[j][t] <= scoreCpu[i][t]) {
              use[j][t] = 0;
            }
          }
          if (use[i][t] && scoreCpu[i][t] > confidenceThreshold) {
            vector<float> tmp(7);
            tmp[0] = boxes[i][4];
            tmp[1] = boxes[i][5];
            tmp[2] = boxes[i][6];
            tmp[3] = boxes[i][7];
            tmp[4] = t;
            tmp[5] = scoreCpu[i][t];
            tmp[6] = n;  // batch_idx
            resultInfo.push_back(tmp);
          }
        }
      }
    }

    for (int i = 0; i < batch; i++) {
      free(boxes[i]);
      free(scoreCpu[i]);
      free(result[i]);
      free(use[i]);
    }
    free(boxes);
    free(result);
    free(size);
    free(use);
    free(scoreCpu);
  }
  return resultInfo;
}
INSTANTIATE_ALL_CLASS(RfcnProcessor);
#endif
