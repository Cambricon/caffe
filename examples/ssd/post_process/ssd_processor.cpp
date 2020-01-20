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

#include "glog/logging.h"
#ifdef USE_MLU
#include "cnrt.h" // NOLINT
#include "ssd_processor.hpp"
#include "runner.hpp"
#include "command_option.hpp"
#include "common_functions.hpp"

using std::pair;

template <typename Dtype, template <typename> class Qtype>
cv::Scalar SsdProcessor<Dtype, Qtype>::HSV2RGB(
           const float h, const float s, const float v) {
  const int h_i = static_cast<int>(h * 6);
  const float f = h * 6 - h_i;
  const float p = v * (1 - s);
  const float q = v * (1 - f*s);
  const float t = v * (1 - (1 - f) * s);
  float r, g, b;
  switch (h_i) {
    case 0:
      r = v; g = t; b = p;
      break;
    case 1:
      r = q; g = v; b = p;
      break;
    case 2:
      r = p; g = v; b = t;
      break;
    case 3:
      r = p; g = q; b = v;
      break;
    case 4:
      r = t; g = p; b = v;
      break;
    case 5:
      r = v; g = p; b = q;
      break;
    default:
      r = 1; g = 1; b = 1;
      break;
  }
  return cv::Scalar(r * 255, g * 255, b * 255);
}
template <typename Dtype, template <typename> class Qtype>
vector<cv::Scalar> SsdProcessor<Dtype, Qtype>::getColors(const int n) {
  vector<cv::Scalar> colors;
  cv::RNG rng(12345);
  const float golden_ratio_conjugate = 0.618033988749895;
  const float s = 0.3;
  const float v = 0.99;
  for (int i = 0; i < n; ++i) {
    const float h = std::fmod(rng.uniform(0.f, 1.f) + golden_ratio_conjugate, 1.f);
    colors.push_back(HSV2RGB(h, s, v));
  }
  return colors;
}

template <typename Dtype, template <typename> class Qtype>
void SsdProcessor<Dtype, Qtype>::WriteVisualizeBBox(const vector<cv::Mat>& images,
                   const vector<vector<vector<float>>>& detections,
                   const float threshold, const vector<cv::Scalar>& colors,
                   const map<int, string>& labelNameMap,
                   const vector<string>& names, const int& from, const int& to) {
  // Retrieve detections.
  // const int num_img = images.size();
  int fontface = cv::FONT_HERSHEY_SIMPLEX;
  double scale = 1;
  int thickness = 2;
  int baseline = 0;
  char buffer[50];

  // for (int i = 0; i < num_img; ++i) {
  for (int i = from; i < to; ++i) {
    cv::Mat tmp = images[i];
    cv::Mat image;
    if (FLAGS_yuv) {
      image = yuv420sp2Bgr24(tmp);
    } else {
      image = tmp;
    }
    LabelBBox labelMap;
    for (int j = 0; j < detections[i].size(); j++) {
      const int label = detections[i][j][1];
      const float score = detections[i][j][2];
      if (score < threshold) {
        continue;
      }
      NormalizedBBox bbox;
      bbox.set_xmin(detections[i][j][3] *
                    image.cols);
      bbox.set_ymin(detections[i][j][4] *
                    image.rows);
      bbox.set_xmax(detections[i][j][5] *
                    image.cols);
      bbox.set_ymax(detections[i][j][6] *
                    image.rows);
      bbox.set_score(score);
      labelMap[label].push_back(bbox);
    }

    std::string name = names[i];
    int pos = names[i].rfind("/");
    if (pos > 0 && pos < names[i].size()) {
      name = name.substr(pos + 1);
    }
    pos = name.find(".");
    if (pos > 0 && pos < name.size()) {
      name = name.substr(0, pos);
    }
    name = FLAGS_outputdir + "/" + name + ".txt";
    std::ofstream file(name);
    // Draw bboxes.
    for (map<int, vector<NormalizedBBox> >::iterator it =
         labelMap.begin(); it != labelMap.end(); ++it) {
      int label = it->first;
      string label_name = "Unknown";
      if (labelNameMap.find(label) != labelNameMap.end()) {
        label_name = labelNameMap.find(label)->second;
      }

      CHECK_LT(label, colors.size());
      const cv::Scalar& color = colors[label];
      const vector<NormalizedBBox>& bboxes = it->second;
      for (int j = 0; j < bboxes.size(); ++j) {
        cv::Point top_left_pt(bboxes[j].xmin(), bboxes[j].ymin());
        cv::Point bottom_right_pt(bboxes[j].xmax(), bboxes[j].ymax());
        cv::rectangle(image, top_left_pt, bottom_right_pt, color, 4);
        cv::Point bottom_left_pt(bboxes[j].xmin(), bboxes[j].ymax());
        snprintf(buffer, sizeof(buffer), "%s: %.2f", label_name.c_str(),
                 bboxes[j].score());
        cv::Size text = cv::getTextSize(buffer, fontface, scale, thickness, &baseline);
        cv::rectangle(image, bottom_left_pt + cv::Point(0, 0),
            bottom_left_pt + cv::Point(text.width, -text.height - baseline),
            color, CV_FILLED);
        cv::putText(image, buffer, bottom_left_pt - cv::Point(0, baseline),
                    fontface, scale, CV_RGB(0, 0, 0), thickness, 8);
        LOG(INFO) << "detection result: " << label << " " << bboxes[j].score()
            << " " << bboxes[j].xmin() << " " << bboxes[j].ymin()
            << " " << bboxes[j].xmax() << " " << bboxes[j].ymax();
        file << label_name << " " << bboxes[j].score() << " "
             << bboxes[j].xmin() / image.cols << " "
             << bboxes[j].ymin() / image.rows << " "
             << bboxes[j].xmax() / image.cols
             << " " << bboxes[j].ymax() / image.rows << std::endl;
      }
    }
    file.close();
    LOG(INFO) << "close txt file ...";

    stringstream ss;
    string outFile;
    int position = names[i].find_last_of('/');
    string fileName(names[i].substr(position + 1));
    ss << FLAGS_outputdir << "/ssd_" << fileName;
    if (FLAGS_yuv)
      ss << ".jpg";
    ss >> outFile;

    cv::imwrite(outFile.c_str(), image);
  }
}

template <typename Dtype, template <typename> class Qtype>
void SsdProcessor<Dtype, Qtype>::readLabels(map<int, string>* label_name_map) {
  if (!FLAGS_labelmapfile.empty()) {
    std::ifstream file(FLAGS_labelmapfile);
    string line;

    vector<int> label_val;
    vector<string> display_value;
    while (getline(file, line)) {
      if (line.find("label:") != std::string::npos) {
        for (int i = 0; i < line.size(); i++) {
          if (isdigit(line[i])) {
            string s1 = line.substr(i, line.size() - i);
            std::stringstream s2(s1);
            int index;
            s2 >> index;
            label_val.push_back(index);
            break;
          }
        }
      }
      if (line.find("display_name:") != std::string::npos) {
        int first_index = line.find_first_of('\"');
        int last_index = line.find_last_of('\"');
        string s1 = line.substr(first_index + 1, last_index - first_index - 1);
        display_value.push_back(s1);
      }
    }
    for (int i = 0; i < label_val.size(); i++)
      label_name_map->insert(pair<int, string>(label_val[i], display_value[i]));
  }
}
INSTANTIATE_ALL_CLASS(SsdProcessor);
#endif
