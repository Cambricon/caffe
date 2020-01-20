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

#ifndef EXAMPLES_YOLO_V2_POST_PROCESS_YOLOV2_PROCESSOR_HPP_
#define EXAMPLES_YOLO_V2_POST_PROCESS_YOLOV2_PROCESSOR_HPP_
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "post_processor.hpp"

using std::map;
using std::vector;
using std::string;
using std::stringstream;

class PredictionResult{
  public:
    float x;
    float y;
    float w;
    float h;
    float objScore;
    float classScore;
    float confidence;
    int classType;
};

template<typename Dtype, template <typename> class Qtype> class Runner;

template<typename Dtype, template <typename> class Qtype>
class YoloV2Processor : public PostProcessor<Dtype, Qtype> {
  public:
  YoloV2Processor() {}
  virtual ~YoloV2Processor() {}
  inline float sigmoid(float x) { return 1. / (1. + exp(-x)); }
  void IntersectBBox(const NormalizedBBox& bbox1,
                     const NormalizedBBox& bbox2,
                     NormalizedBBox* intersect_bbox);
  float BBoxSize(const NormalizedBBox& bbox,
                 const bool normalized = true);
  float JaccardOverlap(const NormalizedBBox& bbox1,
                       const NormalizedBBox& bbox2,
                       const bool normalized);
  void setNormalizedBBox(NormalizedBBox* bbox,
                         float x,
                         float y,
                         float w,
                         float h);
  void class_index_and_score(float* input,
                             int classes,
                             float confidence_threshold_,
                             map<int, float> *prob_index);
  void get_region_box(float* x,
                      PredictionResult* predict,
                      vector<float> biases,
                      int n,
                      int index,
                      int i,
                      int j,
                      int w,
                      int h);
  void ApplyNms(vector<PredictionResult>* boxes,
                vector<int>* idxes,
                float threshold,
                vector< vector<float> >* result,
                int b,
                int num_classes_);
  vector<vector<float> > detection_out(float* net_output,
                                       int out_n,
                                       int out_c,
                                       int out_h,
                                       int out_w);
  void get_point_position(const vector<float> pos,
                        cv::Point* p1, cv::Point* p2, int h, int w);
  void correct_region_boxes(vector<vector<float>>* boxes,
                        const cv::Mat image);
  void WriteVisualizeBBox_offline(const vector<cv::Mat>& images,
                   const vector<vector<float>>& detections,
                   const vector<string>& labels_,
                   const vector<string>& img_names,
                   const int from, const int to);
  void WriteVisualizeBBox_online(const vector<cv::Mat>& images,
                   const vector<vector<vector<float>>> detections,
                   const vector<string>& labelToDisplayName,
                   const vector<string>& imageNames);
  void readLabels(vector<string>* labels);

  protected:
  vector<string> label_to_display_name;
};

#endif  // EXAMPLES_YOLO_V2_POST_PROCESS_YOLOV2_PROCESSOR_HPP_
