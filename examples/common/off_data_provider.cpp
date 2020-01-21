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
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "include/off_data_provider.hpp"
#include "include/off_runner.hpp"
#include "include/pipeline.hpp"
#include "include/command_option.hpp"
#include "include/common_functions.hpp"

using std::string;
using std::vector;

template <typename Dtype, template <typename> class Qtype>
void OffDataProvider<Dtype, Qtype>::runParallel() {
  OffRunner<Dtype, Qtype> *runner = static_cast<OffRunner<Dtype, Qtype>*>(this->runner_);

  setDeviceId(runner->deviceId());
  this->deviceId_ = runner->deviceId();

  allocateMemory(FLAGS_fifosize);

  Pipeline<Dtype, Qtype>::waitForNotification();
#ifdef PRE_READ
  for (int i = 0; i < this->inImages_.size(); i++) {
    Timer preprocessor;
    vector<cv::Mat> rawImages = this->inImages_[i];
    vector<string> imageNameVec = this->imageName_[i];
#else
  while (this->imageList.size()) {
    Timer preprocessor;
    this->inImages_.clear();
    this->imageName_.clear();
    this->readOneBatch();
    if (this->inImages_.empty()) break;
    vector<cv::Mat>& rawImages = this->inImages_[0];
    vector<string>& imageNameVec = this->imageName_[0];
#endif
    vector<vector<cv::Mat> > preprocessedImages;
    Timer prepareInput;
    this->WrapInputLayer(&preprocessedImages, reinterpret_cast<float*>(cpuData_[0]));
    this->Preprocess(rawImages, &preprocessedImages);
    prepareInput.log("prepare input data ...");

    void** mluData = runner->popFreeInputData();
    Timer copyin;
    cnrtDataType_t mluDtype = runner->mluInputDtype()[0];
    cnrtDataType_t cpuDtype;
    if (FLAGS_yuv) {
      cpuDtype = CNRT_UINT8;
    } else {
      cpuDtype = CNRT_FLOAT32;
    }

    int dim_values[4] = {runner->n(), runner->c(), runner->h(), runner->w()};
    int dim_order[4] = {0, 2, 3, 1};  // NCHW --> NHWC
    if (mluDtype != cpuDtype) {
      CNRT_CHECK(cnrtTransOrderAndCast(reinterpret_cast<void*>(cpuData_[0]), cpuDtype,
            reinterpret_cast<void*>(syncCpuData_[0]), mluDtype,
            nullptr, 4, dim_values, dim_order));
    } else {
      CNRT_CHECK(cnrtTransDataOrder(reinterpret_cast<void*>(cpuData_[0]), mluDtype,
            reinterpret_cast<void*>(syncCpuData_[0]), 4, dim_values, dim_order));
    }
    CNRT_CHECK(cnrtMemcpy(reinterpret_cast<void*>(mluData[0]),
                          reinterpret_cast<void*>(syncCpuData_[0]),
                          runner->inputSizeArray()[0],
                          CNRT_MEM_TRANS_DIR_HOST2DEV));
    copyin.log("copyin time ...");
    preprocessor.log("preprocessor time ...");

    runner->pushValidInputDataAndNames(mluData, imageNameVec);
  }

  LOG(INFO) << "DataProvider: no data ...";
  // tell runner there is no more images
}

template <typename Dtype, template <typename> class Qtype>
void OffDataProvider<Dtype, Qtype>::runSerial() {
  OffRunner<Dtype, Qtype> *runner = static_cast<OffRunner<Dtype, Qtype>*>(this->runner_);

  if (!this->initSerialMode) {
    allocateMemory(1);
    this->initSerialMode = true;
  }

  if (this->imageList.size()) {
    this->inImages_.clear();
    this->imageName_.clear();
    this->readOneBatch();
    vector<cv::Mat>& rawImages = this->inImages_[0];
    vector<string>& imageNameVec = this->imageName_[0];

    vector<vector<cv::Mat> > preprocessedImages;
    this->WrapInputLayer(&preprocessedImages, reinterpret_cast<float*>(cpuData_[0]));
    this->Preprocess(rawImages, &preprocessedImages);

    void** mluData = runner->popFreeInputData();
    cnrtDataType_t mluDtype = runner->mluInputDtype()[0];
    cnrtDataType_t cpuDtype;
    if (FLAGS_yuv) {
      cpuDtype = CNRT_UINT8;
    } else {
      cpuDtype = CNRT_FLOAT32;
    }
    int dim_values[4] = {runner->n(), runner->c(), runner->h(), runner->w()};
    int dim_order[4] = {0, 2, 3, 1};  // NCHW --> NHWC

    if (mluDtype != cpuDtype) {
      CNRT_CHECK(cnrtTransOrderAndCast(reinterpret_cast<void*>(cpuData_[0]), cpuDtype,
          reinterpret_cast<void*>(syncCpuData_[0]), mluDtype,
          nullptr, 4, dim_values, dim_order));
    } else {
      CNRT_CHECK(cnrtTransDataOrder(reinterpret_cast<void*>(cpuData_[0]), mluDtype,
          reinterpret_cast<void*>(syncCpuData_[0]), 4, dim_values, dim_order));
    }

    CNRT_CHECK(cnrtMemcpy(reinterpret_cast<void*>(mluData[0]),
                          reinterpret_cast<void*>(syncCpuData_[0]),
                          runner->inputSizeArray()[0],
                          CNRT_MEM_TRANS_DIR_HOST2DEV));
    runner->pushValidInputData(mluData);
    runner->pushValidInputNames(imageNameVec);
    for (auto images : imageNameVec) {
      std::cout << images << std::endl;
    }
  } else {
    LOG(INFO) << "DataProvider: no data ...";
    // tell runner there is no more images
  }
}

template <typename Dtype, template <typename> class Qtype>
void OffDataProvider<Dtype, Qtype>::allocateMemory(int queueLength) {
  OffRunner<Dtype, Qtype> *runner = static_cast<OffRunner<Dtype, Qtype>*>(this->runner_);
  for (int i = 0; i < queueLength; i++) {
    int inBlobNum = runner->inBlobNum();
    int outBlobNum = runner->outBlobNum();
    void** inputMluPtrS =
      reinterpret_cast<void**>(malloc(sizeof(void*) * inBlobNum));
    void** outputMluPtrS =
      reinterpret_cast<void**>(malloc(sizeof(void*) * outBlobNum));

    // malloc input
    for (int i = 0; i < inBlobNum; i++) {
      CNRT_CHECK(cnrtMalloc(&(inputMluPtrS[i]), runner->inputSizeArray()[i]));
    }
    for (int i = 0; i < outBlobNum; i++) {
      CNRT_CHECK(cnrtMalloc(&(outputMluPtrS[i]), runner->outputSizeArray()[i]));
    }
    runner->pushFreeInputData(inputMluPtrS);
    runner->pushFreeOutputData(outputMluPtrS);
    pushInPtrVector(inputMluPtrS);
    pushOutPtrVector(outputMluPtrS);
  }
  this->inNum_ = runner->n();
  this->inChannel_ = runner->c();
  this->inHeight_ = runner->h();
  this->inWidth_ = runner->w();
  this->inGeometry_ = cv::Size(this->inWidth_, this->inHeight_);
  this->SetMean();

  cpuData_ = new(void*);
  cpuData_[0] = new float[this->inNum_ * this->inChannel_ *
                          this->inHeight_ * this->inWidth_];
  syncCpuData_ = new(void*);
  syncCpuData_[0] = new char[runner->inputSizeArray()[0]];
}

void setDeviceId(int deviceID) {
  unsigned devNum;
  CNRT_CHECK(cnrtGetDeviceCount(&devNum));
  if (deviceID >= 0) {
    CHECK_NE(devNum, 0) << "No device found";
    CHECK_LT(deviceID, devNum) << "Valid device count: " <<devNum;
  } else {
    LOG(FATAL) << "Invalid device number";
  }
  cnrtDev_t dev;
  LOG(INFO) << "Using MLU device " << deviceID;
  CNRT_CHECK(cnrtGetDeviceHandle(&dev, deviceID));
  CNRT_CHECK(cnrtSetCurrentDevice(dev));
}

INSTANTIATE_OFF_CLASS(OffDataProvider);

#endif  // USE_MLU && USE_OPENCV
