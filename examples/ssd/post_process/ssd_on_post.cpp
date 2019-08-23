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
#include <queue>
#include <string>
#include <sstream>
#include <thread> // NOLINT
#include <utility>
#include <vector>
#include <iomanip>
#include "cnrt.h" // NOLINT
#include "glog/logging.h"
#include "ssd_on_post.hpp"
#include "runner.hpp"
#include "on_runner.hpp"
#include "command_option.hpp"
#include "on_data_provider.hpp"
#include "common_functions.hpp"
using std::vector;
using std::string;

template <typename Dtype, template <typename> class Qtype>
void SsdOnPostProcessor<Dtype, Qtype>::runParallel() {
}

template <typename Dtype, template <typename> class Qtype>
void SsdOnPostProcessor<Dtype, Qtype>::runSerial() {
  if (!this->initSerialMode) {
    this->readLabels(&this->labelNameMap);

    this->initSerialMode = true;
  }

  vector<cv::Mat> imgs;
  vector<string> img_names;
  vector<cv::Scalar> colors;
  vector<vector<vector<float> > > detections = getResults(&imgs, &img_names, &colors);

  this->WriteVisualizeBBox_offline(imgs, detections, FLAGS_confidencethreshold,
                                   colors, this->labelNameMap, img_names);
}

template <typename Dtype, template <typename> class Qtype>
vector<vector<vector<float> > > SsdOnPostProcessor<Dtype, Qtype>::getResults(
                                         vector<cv::Mat> *imgs,
                                         vector<string> *img_names,
                                         vector<cv::Scalar> *colors) {
  OnRunner<Dtype, Qtype> * runner = static_cast<OnRunner<Dtype, Qtype>*>(this->runner_);

  caffe::Net<float>* netBuff = runner->net();
  auto outputBlob = netBuff->output_blobs()[0];
  float* data = outputBlob->mutable_cpu_data();

  vector<vector<vector<float> > > detections(runner->n());
  if (caffe::Caffe::mode() == caffe::Caffe::CPU) {
    int numDet = outputBlob->height();
    for (int k = 0; k < numDet; ++k) {
      if (data[0] == -1) {
        // Skip invalid detection.
        data += 7;
        continue;
      }
      vector<float> detection(data, data + 7);
      detections[static_cast<int>(data[0])].push_back(detection);
      data += 7;
    }
  } else {
    for (int k = 0; k < runner->outCount() / runner->outWidth(); ++k) {
      if (data[4] == 0) {
        // the score(outputCpuPtr[4]) must be 0 if invalid detection, so skip it.
        data += runner->outWidth();
        continue;
      }
      int batch = k / runner->outChannel();
      vector<float> detection(7, 0);
      detection[0] = batch;
      detection[1] = data[5];
      detection[2] = data[4];
      detection[3] = data[0];
      detection[4] = data[1];
      detection[5] = data[2];
      detection[6] = data[3];
      detections[batch].push_back(detection);
      data += runner->outWidth();
    }
  }

  vector<string> origin_img = runner->popValidInputNames();
  for (auto img_name : origin_img) {
    if (img_name != "null") {
      cv::Mat img;
      if (FLAGS_yuv) {
        cv::Mat img = convertYuv2Mat(img_name, runner->w(), runner->h());
      } else {
        img = cv::imread(img_name, -1);
      }
      int pos = img_name.find_last_of('/');
      string file_name(img_name.substr(pos+1));
      imgs->push_back(img);
      img_names->push_back(file_name);
    }
  }
  *colors = this->getColors(this->labelNameMap.size());

  return detections;
}

INSTANTIATE_ON_CLASS(SsdOnPostProcessor);
#endif  // defined(USE_MLU) && defined(USE_OPENCV)
