/*
All modification made by Cambricon Corporation: © 2018 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2018, the respective contributors
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

#if defined(USE_MLU) && defined(USE_OPENCV)
#include "glog/logging.h"
#include "cnrt.h" // NOLINT
#include "yolov2_on_post.hpp"
#include "runner.hpp"
#include "on_runner.hpp"
#include "command_option.hpp"
#include "on_data_provider.hpp"
#include "common_functions.hpp"
using std::vector;
using std::string;
using caffe::Blob;

template <typename Dtype, template <typename> class Qtype>
void YoloV2OnPostProcessor<Dtype, Qtype>::runParallel() {
}

template <typename Dtype, template <typename> class Qtype>
void YoloV2OnPostProcessor<Dtype, Qtype>::runSerial() {
  if (!this->initSerialMode) {
    this->readLabels(&this->label_to_display_name);

    this->initSerialMode = true;
  }

  vector<cv::Mat> imgs;
  vector<string> img_names;
  vector<vector<vector<float> > > multi_detections = getResults(&imgs, &img_names);

  this->WriteVisualizeBBox_online(imgs, multi_detections,
                         this->label_to_display_name, img_names);
}

template <typename Dtype, template <typename> class Qtype>
vector<vector<vector<float> > > YoloV2OnPostProcessor<Dtype, Qtype>::getResults(
                                            vector<cv::Mat> *imgs,
                                            vector<string> *img_names) {
  OnRunner<Dtype, Qtype> * runner = static_cast<OnRunner<Dtype, Qtype>*>(this->runner_);
  caffe::Net<float>* netBuff = runner->net();

  Blob<float>* outputLayer = netBuff->output_blobs()[0];
  const float* outputData = outputLayer->cpu_data();
  vector<vector<vector<float> > > detections(runner->n());
  for (int i = 0; i < outputLayer->height(); ++i) {
    if (outputData[0] == -1) {
      // Skip invalid detection.
      outputData += 7;
      continue;
    }
    vector<float> temp(outputData, outputData + 7);
    detections[static_cast<int>(outputData[0])].push_back(temp);
    outputData += 7;
  }

  vector<string> origin_img = runner->popValidInputNames();
  for (const auto& img_name : origin_img) {
    if (img_name != "null") {
      cv::Mat img;
      if (FLAGS_yuv) {
        img = convertYuv2Mat(img_name, runner->w(), runner->h());
      } else {
        img = cv::imread(img_name, -1);
      }
      imgs->push_back(img);
      img_names->push_back(img_name);
    }
  }

  return detections;
}

INSTANTIATE_ON_CLASS(YoloV2OnPostProcessor);
#endif  // defined(USE_MLU) && defined(USE_OPENCV)
