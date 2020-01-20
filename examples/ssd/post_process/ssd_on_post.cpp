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
  OnRunner<Dtype, Qtype> *runner = static_cast<OnRunner<Dtype, Qtype>*>(this->runner_);
  ::setupConfig(this->threadId_, runner->deviceId(), runner->deviceSize());
  this->readLabels(&this->labelNameMap);
  int TASK_NUM = SimpleInterface::thread_num;
  caffe::Net<float>* netBuff = runner->net();
  int outputCount = netBuff->output_blobs()[0]->count();
  outCpuPtrs_ = new Dtype[outputCount];
  size_t tensorSize;
  MLU_CHECK(cnmlGetTensorSize_V2(netBuff->output_blobs()[0]->mlu_tensor(), &tensorSize));
  std::vector<std::future<void>> futureVector;
  while (true) {
    Dtype* outputMluPtr = runner->popValidOutputData();
    Dtype* outputSyncPtr = runner->popValidOutputSyncData();
    if (nullptr == outputMluPtr) break;  // no more work, exit

    auto outputBlob = netBuff->output_blobs()[0];
    Timer timer;
    CNRT_CHECK(cnrtMemcpy(outputSyncPtr, outputMluPtr,
          tensorSize, CNRT_MEM_TRANS_DIR_DEV2HOST));
    cnrtDataType_t cpuDtype = to_cnrt_dtype(outputBlob->cpu_type());
    cnrtDataType_t mluDtype = to_cnrt_dtype(outputBlob->mlu_type());
    int dim_values[4] = {outputBlob->mlu_shape()[0], outputBlob->mlu_shape()[1],
                         outputBlob->mlu_shape()[2], outputBlob->mlu_shape()[3]};
    int dim_order[4] = {0, 3, 1, 2};
    if (mluDtype != cpuDtype) {
      CNRT_CHECK(cnrtTransOrderAndCast(reinterpret_cast<void*>(outputSyncPtr),
                                       mluDtype,
                                       reinterpret_cast<void*>(outCpuPtrs_),
                                       cpuDtype,
                                       nullptr,
                                       outputBlob->mlu_shape().size(),
                                       dim_values,
                                       dim_order));
    } else {
      CNRT_CHECK(cnrtTransDataOrder(reinterpret_cast<void*>(outputSyncPtr),
                                    cpuDtype,
                                    reinterpret_cast<void*>(outCpuPtrs_),
                                    outputBlob->mlu_shape().size(),
                                    dim_values,
                                    dim_order));
    }
    resultDataPtr_ = outCpuPtrs_;
    timer.log("copyout time ...");
    runner->pushFreeOutputData(outputMluPtr);
    runner->pushFreeOutputSyncData(outputSyncPtr);

    Timer postProcess;
    vector<cv::Mat> imgs;
    vector<string> img_names;
    vector<cv::Scalar> colors;
    vector<vector<vector<float>>> detections = getResults(&imgs, &img_names, &colors);

    Timer dumpTimer;
    if (FLAGS_dump) {
      const float threshold = FLAGS_confidencethreshold;
      const int size = imgs.size();
      if (TASK_NUM > size)
        TASK_NUM = size;
      const int delta = size / TASK_NUM;
      int from = 0;
      int to = delta;
      for (int i = 0; i < TASK_NUM; i++) {
        from = delta * i;
        if (i == TASK_NUM - 1) {
          to = size;
        } else {
          to = delta * (i + 1);
        }

        auto func = this->tp_->add([](const vector<cv::Mat>& imgs,
                    const vector<vector<vector<float>>>& detections,
                    const float threshold,
                    const vector<cv::Scalar>& colors,
                    const map<int, string>& labelNameMap,
                    const vector<string>& img_names,
                    const int& from,
                    const int& to,
                    SsdProcessor<Dtype, Qtype>* object) {
          object->WriteVisualizeBBox(imgs,
                    detections,
                    threshold,
                    colors,
                    labelNameMap,
                    img_names,
                    from,
                    to);
        }, imgs, detections, threshold, colors,
           this->labelNameMap, img_names, from, to, this);

        futureVector.push_back(std::move(func));
      }
    }
    dumpTimer.log("dump out time ...");
    postProcess.log("post process time ...");
  }
  for (int i = 0; i < futureVector.size(); i++) {
    futureVector[i].get();
  }
}

template <typename Dtype, template <typename> class Qtype>
void SsdOnPostProcessor<Dtype, Qtype>::runSerial() {
  OnRunner<Dtype, Qtype> * runner = static_cast<OnRunner<Dtype, Qtype>*>(this->runner_);
  if (!this->initSerialMode) {
    this->readLabels(&this->labelNameMap);
    this->initSerialMode = true;
  }
  caffe::Net<float>* netBuff = runner->net();
  auto outputBlob = netBuff->output_blobs()[0];
  resultDataPtr_ = outputBlob->mutable_cpu_data();
  vector<cv::Mat> imgs;
  vector<string> img_names;
  vector<cv::Scalar> colors;
  vector<vector<vector<float> > > detections = getResults(&imgs, &img_names, &colors);

  this->WriteVisualizeBBox(imgs, detections, FLAGS_confidencethreshold,
                                   colors, this->labelNameMap, img_names, 0, imgs.size());
}

template <typename Dtype, template <typename> class Qtype>
vector<vector<vector<float> > > SsdOnPostProcessor<Dtype, Qtype>::getResults(
                                         vector<cv::Mat> *imgs,
                                         vector<string> *img_names,
                                         vector<cv::Scalar> *colors) {
  OnRunner<Dtype, Qtype> * runner = static_cast<OnRunner<Dtype, Qtype>*>(this->runner_);
  float* data = resultDataPtr_;
  vector<vector<vector<float>>> detections;
  if (caffe::Caffe::mode() == caffe::Caffe::CPU) {
    int numDet = runner->outHeight();
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
    // BangOp implementation
    int count = runner->outChannel();
    for (int i = 0; i < runner->outNum(); i++) {
      int output_num = data[i * count];
      vector<vector<float>> batch_detection;
      for (int k = 0; k < output_num; k++) {
         int index = i * count + 64 + k * 7;
         if (static_cast<int>(data[index]) != i) continue;
         vector<float> single_detection(data + index, data + index + 7);
         batch_detection.push_back(single_detection);
      }
      detections.push_back(batch_detection);
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
