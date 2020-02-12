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
#include "yolov2_off_post.hpp"
#include "off_runner.hpp"
#include "command_option.hpp"
#include "common_functions.hpp"
#include "threadPool.h"

using std::vector;
using std::string;

template<typename Dtype, template <typename> class Qtype>
void YoloV2OffPostProcessor<Dtype, Qtype>::runParallel() {
  OffRunner<Dtype, Qtype> * infr = static_cast<OffRunner<Dtype, Qtype>*>(this->runner_);
  setDeviceId(infr->deviceId());

  this->readLabels(&this->label_to_display_name);

  outCpuPtrs_ = new(Dtype);
  outCpuPtrs_[0] = new float[infr->outCounts()[0]];
  syncCpuPtrs_ = new(Dtype);
  syncCpuPtrs_[0] = new char[infr->outputSizeArray()[0]];

  int TASK_NUM = SimpleInterface::thread_num;
  std::vector<std::future<void>> futureVector;
  int dim_values[4] = {infr->outNum(), infr->outHeight(),
    infr->outWidth(), infr->outChannel()};
  int dim_order[4] = {0, 3, 1, 2};

  while (true) {
    Dtype* mluOutData = infr->popValidOutputData();
    if (mluOutData == nullptr) break;  // no more work

    Timer copyout;
    CNRT_CHECK(cnrtMemcpy(syncCpuPtrs_[0], mluOutData[0],
                          infr->outputSizeArray()[0],
                          CNRT_MEM_TRANS_DIR_DEV2HOST));
    cnrtDataType_t cpuDtype = CNRT_FLOAT32;
    cnrtDataType_t mluDtype = infr->mluOutputDtype()[0];
    if (mluDtype != cpuDtype) {
      CNRT_CHECK(cnrtTransOrderAndCast(syncCpuPtrs_[0], mluDtype,
            outCpuPtrs_[0], cpuDtype,
            nullptr, 4, dim_values, dim_order));
    } else {
      CNRT_CHECK(cnrtTransDataOrder(syncCpuPtrs_[0], cpuDtype,
            outCpuPtrs_[0], 4, dim_values, dim_order));
    }
    copyout.log("copyout time ...");

    infr->pushFreeOutputData(mluOutData);

    Timer postProcess;
    vector<cv::Mat> imgs;
    vector<string> img_names;
    vector<vector<float> > boxes = getResults(&imgs, &img_names);

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
}

template<typename Dtype, template <typename> class Qtype>
void YoloV2OffPostProcessor<Dtype, Qtype>::runSerial() {
  OffRunner<Dtype, Qtype> * infr = static_cast<OffRunner<Dtype, Qtype>*>(this->runner_);

  if (!this->initSerialMode) {
    this->readLabels(&this->label_to_display_name);

    outCpuPtrs_ = new(Dtype);
    outCpuPtrs_[0] = new float[infr->outCounts()[0]];
    syncCpuPtrs_ = new(Dtype);
    syncCpuPtrs_[0] = new char[infr->outputSizeArray()[0]];

    this->initSerialMode = true;
  }

  Dtype* mluOutData = infr->popValidOutputData();
  int dim_values[4] = {infr->outNum(), infr->outHeight(),
      infr->outWidth(), infr->outChannel()};
  int dim_order[4] = {0, 3, 1, 2};
  CNRT_CHECK(cnrtMemcpy(syncCpuPtrs_[0], mluOutData[0],
                        infr->outputSizeArray()[0],
                        CNRT_MEM_TRANS_DIR_DEV2HOST));
  cnrtDataType_t cpuDtype = CNRT_FLOAT32;
  cnrtDataType_t mluDtype = infr->mluOutputDtype()[0];
  if (cpuDtype != mluDtype) {
    CNRT_CHECK(cnrtTransOrderAndCast(syncCpuPtrs_[0], mluDtype,
          outCpuPtrs_[0], cpuDtype,
          nullptr, 4, dim_values, dim_order));
  } else {
    CNRT_CHECK(cnrtTransDataOrder(syncCpuPtrs_[0], cpuDtype,
          outCpuPtrs_[0], 4, dim_values, dim_order));
  }
  infr->pushFreeOutputData(mluOutData);

  vector<cv::Mat> imgs;
  vector<string> img_names;
  vector<vector<float> > boxes = getResults(&imgs, &img_names);
  if (FLAGS_dump) {
    this->WriteVisualizeBBox_offline(imgs, boxes,
             this->label_to_display_name, img_names, 0, imgs.size());
  }
}

template<typename Dtype, template <typename> class Qtype>
vector<vector<float> > YoloV2OffPostProcessor<Dtype, Qtype>::getResults(
                                          vector<cv::Mat> *imgs,
                                          vector<string> *img_names) {
  OffRunner<Dtype, Qtype> * infr = static_cast<OffRunner<Dtype, Qtype>*>(this->runner_);
  int outN = infr->outNum();
  int outC = infr->outChannel();
  int outH = infr->outHeight();
  int outW = infr->outWidth();

  float* data = reinterpret_cast<float*>(outCpuPtrs_[0]);
  vector<vector<float> > boxes = this->detection_out(data, outN, outC,
                                                     outH, outW);
  auto&& origin_img = infr->popValidInputNames();
  for (auto& img_name : origin_img) {
    if (img_name != "null") {
      cv::Mat img;
      if (FLAGS_yuv) {
        img = convertYuv2Mat(img_name, infr->w(), infr->h());
      } else {
        img = cv::imread(img_name, -1);
      }
      int pos = img_name.find_last_of('/');
      string file_name(img_name.substr(pos+1));
      imgs->push_back(img);
      img_names->push_back(file_name);
    }
  }
  return boxes;
}

INSTANTIATE_OFF_CLASS(YoloV2OffPostProcessor);
#endif  // defined(USE_MLU) && defined(USE_OPENCV)
