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
#include "yolov2_on_post.hpp"
#include "cnrt.h"  // NOLINT
#include "command_option.hpp"
#include "common_functions.hpp"
#include "glog/logging.h"
#include "on_data_provider.hpp"
#include "on_runner.hpp"
#include "runner.hpp"

using std::vector;
using std::string;
using caffe::Blob;

template <typename Dtype, template <typename> class Qtype>
void YoloV2OnPostProcessor<Dtype, Qtype>::runParallel() {
#if 1
  OnRunner<Dtype, Qtype> *runner = static_cast<OnRunner<Dtype, Qtype>*>(this->runner_);
  ::setupConfig(this->threadId_, runner->deviceId(), runner->deviceSize());
  this->readLabels(&this->label_to_display_name);
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
    vector<vector<float>> boxes = getResults(&imgs, &img_names);

    if (FLAGS_dump) {
      Timer dumpTimer;
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

        auto func = tp_->add([](const vector<cv::Mat>& imgs,
               const vector<vector<float> >& boxes,
               const vector<string>& label_to_display_name,
               const vector<string>& img_names, const int& from,
               const int& to, YoloV2Processor<Dtype, Qtype>* object) {
          object->WriteVisualizeBBox_offline(imgs, boxes,
              label_to_display_name, img_names, from, to);
        }, imgs, boxes, this->label_to_display_name,
        img_names, from, to, this);

        futureVector.push_back(std::move(func));
      }
      dumpTimer.log("dump imgs time ...");
    }
    postProcess.log("post process time ...");
  }
  for (int i = 0; i < futureVector.size(); i++) {
    futureVector[i].get();
  }
#endif
}

template <typename Dtype, template <typename> class Qtype>
void YoloV2OnPostProcessor<Dtype, Qtype>::runSerial() {
  OnRunner<Dtype, Qtype>* runner = static_cast<OnRunner<Dtype, Qtype>*>(this->runner_);
  if (!this->initSerialMode) {
    this->readLabels(&this->label_to_display_name);
    this->initSerialMode = true;
  }

  auto outputBlob = runner->net()->output_blobs()[0];
  resultDataPtr_ = outputBlob->mutable_cpu_data();
  vector<cv::Mat> imgs;
  vector<string> img_names;
  vector<vector<float>> detections = getResults(&imgs, &img_names);

  this->WriteVisualizeBBox_offline(imgs, detections,
                  this->label_to_display_name, img_names, 0, imgs.size());
}

template <typename Dtype, template <typename> class Qtype>
vector<vector<float>> YoloV2OnPostProcessor<Dtype, Qtype>::getResults(
                                            vector<cv::Mat>* imgs,
                                            vector<string>* img_names) {
  OnRunner<Dtype, Qtype>* runner = static_cast<OnRunner<Dtype, Qtype>*>(this->runner_);
  int outN = runner->outNum();
  int outC = runner->outChannel();
  int outH = runner->outHeight();
  int outW = runner->outWidth();

  float* outputData = resultDataPtr_;
  vector<vector<float>> detections;
  if (FLAGS_Bangop != 0) {  // mlu detection
     detections = this->detection_out(outputData, outN, outC, outH, outW);
  } else {  // cpu detection, batchsize = 1
    int Index = runner->net()->output_blobs()[0]->channels();
    if (outputData[0] == 0) {
      for (int i = 0; i < Index; ++i) {
        if (outputData[0] == -1) {
          // Skip invalid detection.
          outputData += 7;
          continue;
        }
        vector<float> temp(outputData, outputData + 7);
        temp[0] = 0;
        detections.push_back(temp);
        outputData += 7;
      }
    }
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
