/*
All modification made by Cambricon Corporation: © 2018--2019 Cambricon Corporation
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
#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <map>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "blocking_queue.hpp"
#include "cnrt.h"  // NOLINT
#include "common_functions.hpp"
#define MAX_NBOXES 1024
using std::queue;
using std::string;
using std::stringstream;
using std::vector;

DEFINE_string(offlinemodel, "",
              "The prototxt file used to find net configuration");
DEFINE_string(meanfile, "",
              "The mean file used to subtract from the input image.");
DEFINE_string(
    meanvalue, "",
    "If specified, can be one value or can be same as image channels"
    " - would subtract from the corresponding channel). Separated by ','."
    "Either mean_file or mean_value should be provided, not both.");
DEFINE_string(images, "", "The input file list");
DEFINE_string(outputdir, ".", "The directoy used to save output images");
DEFINE_string(labels, "", "infomation about mapping from label to name");
DEFINE_int32(mludevice, 0, "set using mlu device number, default: 0");
DEFINE_int32(fix8, 0, "FP16 or FIX8, fix8 mode, default: 0");
DEFINE_int32(int8, -1,
             "invalid(-1), fp16(0) or int8(1) mode. Default is invalid(-1)."
             "If specified, use int8 value, else, use fix8 value");
DEFINE_string(logdir, "",
              "path to dump log file, to terminal "
              "stderr by default");
DEFINE_int32(dump, 0, "0 or 1, dump output images or not.");
DEFINE_double(confidence, 0.25,
              "Only keep detections with scores  equal "
              "to or higher than the confidence.");
DEFINE_double(nmsthresh, 0.45,
              "Identify the optimal cell among all candidates "
              " when the object lies in multiple cells of a grid");

class Detector {
 public:
  Detector(const string& modelFile, const string& meanFile,
           const string& meanValues);
  ~Detector();

  vector<vector<vector<float>>> detect(const vector<cv::Mat>& images);
  int getBatchSize() { return batchSize; }
  int inputDim() { return inDimValues_[0][1]; }
  float mluTime() { return mlu_time; }
  void readImages(queue<string>* imagesQueue, int inputNumber,
                  vector<cv::Mat>* images, vector<string>* imageNames);

 private:
  void setMean(const string& meanFile, const string& meanValues);
  void wrapInputLayer(vector<vector<cv::Mat>>* inputImages);
  void preProcess(const vector<cv::Mat>& images,
                  vector<vector<cv::Mat>>* inputImages);

 private:
  cnrtModel_t model;
  cv::Size inputGeometry;
  int batchSize;
  int numberChannels;
  cv::Mat meanValue;
  void** inputCpuPtrS;
  void** outputCpuPtrS;
  void** inputMluPtrS;
  void** outputMluPtrS;
  void** inputSyncPtrS;
  void** outputSyncPtrS;
  void** param;
  cnrtQueue_t cnrt_queue;
  int inputNum, outputNum;
  cnrtFunction_t function;
  int64_t* inputSizeArray;
  int64_t* outputSizeArray;
  cnrtDataType_t* inputDataTypeArray;
  cnrtDataType_t* outputDataTypeArray;
  vector<int> inCounts_, outCounts_;
  vector<int> inDimNums_, outDimNums_;
  vector<int*> inDimValues_, outDimValues_;
  float mlu_time;
  cnrtRuntimeContext_t rt_ctx;
};

Detector::Detector(const string& modelFile, const string& meanFile,
                   const string& meanValues) {
  // offline model
  // 1. init runtime_lib and device
  unsigned devNum;
  cnrtGetDeviceCount(&devNum);
  if (FLAGS_mludevice >= 0) {
    CHECK_NE(devNum, 0) << "No device found";
    CHECK_LT(FLAGS_mludevice, devNum) << "valid device count: " << devNum;
  } else {
    LOG(FATAL) << "Invalid device number";
  }
  cnrtDev_t dev;
  cnrtGetDeviceHandle(&dev, FLAGS_mludevice);
  cnrtSetCurrentDevice(dev);
  // 2. load model and get function
  LOG(INFO) << "load file: " << modelFile;
  cnrtLoadModel(&model, modelFile.c_str());
  string name = "subnet0";
  cnrtCreateFunction(&function);
  cnrtExtractFunction(&function, model, name.c_str());
  // 3. get function's I/O DataDesc
  CNRT_CHECK(cnrtGetInputDataSize(&inputSizeArray,
        &inputNum, function));
  CNRT_CHECK(cnrtGetOutputDataSize(&outputSizeArray,
        &outputNum, function));
  CNRT_CHECK(cnrtGetInputDataType(&inputDataTypeArray,
        &inputNum, function));
  CNRT_CHECK(cnrtGetOutputDataType(&outputDataTypeArray,
        &outputNum, function));
  // 4. allocate I/O data space on CPU memory and prepare Input data
  inputCpuPtrS = reinterpret_cast<void**>(malloc(sizeof(void*) * inputNum));
  outputCpuPtrS = reinterpret_cast<void**>(malloc(sizeof(void*) * outputNum));
  inputSyncPtrS = reinterpret_cast<void**>(malloc(sizeof(void*) * inputNum));
  outputSyncPtrS = reinterpret_cast<void**>(malloc(sizeof(void*) * outputNum));

  inCounts_.resize(inputNum, 1);
  outCounts_.resize(outputNum, 1);
  inDimNums_.resize(inputNum, 0);
  outDimNums_.resize(outputNum, 0);
  inDimValues_.resize(inputNum, nullptr);
  outDimValues_.resize(outputNum, nullptr);
  /* input shape : 1, 3, 416, 416 */
  for (int i = 0; i < inputNum; i++) {
    CNRT_CHECK(cnrtGetInputDataShape(&(inDimValues_[i]),
        &(inDimNums_[i]), i, function));
    for (int j = 0; j < inDimNums_[i]; ++j) {
      this->inCounts_[i] *= inDimValues_[i][j];
      LOG(INFO) << "shape " << inDimValues_[i][j];
    }
    if (i == 0) {
      batchSize = inDimValues_[i][0];
      numberChannels = inDimValues_[i][3];
      inputGeometry = cv::Size(inDimValues_[i][2], inDimValues_[i][1]);
    }
    inputCpuPtrS[i] =
        reinterpret_cast<void*>(malloc(sizeof(float) * inCounts_[i]));
    inputSyncPtrS[i] =
        reinterpret_cast<void*>(malloc(inputSizeArray[i]));
  }

  for (int i = 0; i < outputNum; i++) {
    CNRT_CHECK(cnrtGetOutputDataShape(&(outDimValues_[i]),
        &(outDimNums_[i]), i, function));
    for (int j = 0; j < outDimNums_[i]; ++j) {
      outCounts_[i] *= outDimValues_[i][j];
      LOG(INFO) << "shape " << outDimValues_[i][j];
    }
    outputCpuPtrS[i] =
        reinterpret_cast<void*>(malloc(sizeof(float) * outCounts_[i]));
    outputSyncPtrS[i] =
        reinterpret_cast<void**>(malloc(outputSizeArray[i]));
  }

  // 5. allocate I/O data space on MLU memory and copy Input data
  inputMluPtrS =
    reinterpret_cast<void**>(malloc(sizeof(void*) * inputNum));
  outputMluPtrS =
    reinterpret_cast<void**>(malloc(sizeof(void*) * outputNum));
  for (int i = 0; i < inputNum; i++) {
    cnrtMalloc(&(inputMluPtrS[i]), inputSizeArray[i]);
  }
  for (int i = 0; i < outputNum; i++) {
    cnrtMalloc(&(outputMluPtrS[i]), outputSizeArray[i]);
  }
  cnrtCreateQueue(&cnrt_queue);
  setMean(meanFile, meanValues);
  if (cnrtCreateRuntimeContext(&rt_ctx, function, nullptr) != CNRT_RET_SUCCESS) {
    LOG(FATAL)<< "Failed to create runtime context";
  }

  // set device ordinal. if not set, a random device will be used
  cnrtSetRuntimeContextDeviceId(rt_ctx, FLAGS_mludevice);
  // Instantiate the runtime context on actual MLU device
  // All cnrtSetRuntimeContext* interfaces must be caller prior to cnrtInitRuntimeContext
  if (cnrtInitRuntimeContext(rt_ctx, nullptr) != CNRT_RET_SUCCESS) {
    LOG(FATAL)<< "Failed to initialize runtime context";
  }
}

Detector::~Detector() {
  if (inputCpuPtrS != NULL) {
    for (int i = 0; i < inputNum; i++) {
      if (inputCpuPtrS[i] != NULL) free(inputCpuPtrS[i]);
    }
    free(inputCpuPtrS);
  }
  if (inputSyncPtrS != NULL) {
    for (int i = 0; i < inputNum; i++) {
      if (inputSyncPtrS[i] != NULL) free(inputSyncPtrS[i]);
    }
    free(inputSyncPtrS);
  }
  if (outputCpuPtrS != NULL) {
    for (int i = 0; i < outputNum; i++) {
      if (outputCpuPtrS[i] != NULL) free(outputCpuPtrS[i]);
    }
    free(outputCpuPtrS);
  }
  if (outputSyncPtrS != NULL) {
    for (int i = 0; i < outputNum; i++) {
      if (outputSyncPtrS[i] != NULL) free(outputSyncPtrS[i]);
    }
    free(outputSyncPtrS);
  }
  if (inputMluPtrS != NULL) {
    for (int i = 0; i < inputNum; i++) {
      if (inputMluPtrS[i] != NULL) {
        cnrtFree(inputMluPtrS[i]);
      }
    }
    free(inputMluPtrS);
  }
  if (outputMluPtrS != NULL) {
    for (int i = 0; i < outputNum; i++) {
      if (outputMluPtrS[i] != NULL) {
        cnrtFree(outputMluPtrS[i]);
      }
    }
    free(outputMluPtrS);
  }

  cnrtDestroyQueue(cnrt_queue);
  cnrtDestroyFunction(function);
  // unload model
  cnrtUnloadModel(model);
  cnrtDestroyRuntimeContext(rt_ctx);
}

vector<vector<vector<float>>> Detector::detect(const vector<cv::Mat>& images) {
  vector<vector<cv::Mat>> inputImages;
  wrapInputLayer(&inputImages);
  preProcess(images, &inputImages);
  cnrtNotifier_t notifierBeginning, notifierEnd;
  cnrtCreateNotifier(&notifierBeginning);
  cnrtCreateNotifier(&notifierEnd);
  float eventTimeUse = 0;

  cnrtDataType_t inputCpuDtype = CNRT_FLOAT32;
  cnrtDataType_t inputMluDtype = inputDataTypeArray[0];
  int dimValuesCpu[4] = {inDimValues_[0][0], inDimValues_[0][3],
                         inDimValues_[0][1], inDimValues_[0][2]};
  int dimOrder[4] = {0, 2, 3, 1};  // NCHW --> NHWC
  if (inputCpuDtype != inputMluDtype) {
    CNRT_CHECK(cnrtTransOrderAndCast(inputCpuPtrS[0], inputCpuDtype,
          inputSyncPtrS[0], inputMluDtype, nullptr, 4, dimValuesCpu, dimOrder));
  } else {
    CNRT_CHECK(cnrtTransDataOrder(inputCpuPtrS[0], inputCpuDtype,
          inputSyncPtrS[0], 4, dimValuesCpu, dimOrder));
  }
  CNRT_CHECK(cnrtMemcpy(inputMluPtrS[0], inputSyncPtrS[0],
        inputSizeArray[0], CNRT_MEM_TRANS_DIR_HOST2DEV));

  param =
      reinterpret_cast<void**>(malloc(sizeof(void*) * (inputNum + outputNum)));
  for (int i = 0; i < inputNum; i++) param[i] = inputMluPtrS[i];
  for (int i = 0; i < outputNum; i++) param[i + inputNum] = outputMluPtrS[i];


  cnrtPlaceNotifier(notifierBeginning, cnrt_queue);
  CNRT_CHECK(cnrtInvokeRuntimeContext(rt_ctx, param, cnrt_queue, nullptr));
  cnrtPlaceNotifier(notifierEnd, cnrt_queue);
  if (cnrtSyncQueue(cnrt_queue) == CNRT_RET_SUCCESS) {
    cnrtNotifierDuration(notifierBeginning, notifierEnd, &eventTimeUse);
    printfMluTime(eventTimeUse);
    mlu_time += eventTimeUse;
  } else {
    LOG(INFO) << " SyncQueue error ";
  }

  CNRT_CHECK(cnrtMemcpy(outputSyncPtrS[0], outputMluPtrS[0],
        outputSizeArray[0], CNRT_MEM_TRANS_DIR_DEV2HOST));
  cnrtDataType_t outputCpuDtype = CNRT_FLOAT32;
  cnrtDataType_t outputMluDtype = outputDataTypeArray[0];
  int dimValuesMlu[4] = {outDimValues_[0][0], outDimValues_[0][1],
                         outDimValues_[0][2], outDimValues_[0][3]};
  int dimOrderMlu[4] = {0, 3, 1, 2};  // NHWC --> NCHW
  if (outputCpuDtype != outputMluDtype) {
    CNRT_CHECK(cnrtTransOrderAndCast(outputSyncPtrS[0], outputMluDtype,
          outputCpuPtrS[0], outputCpuDtype, nullptr, 4, dimValuesMlu, dimOrderMlu));
  } else {
    CNRT_CHECK(cnrtTransDataOrder(outputSyncPtrS[0], outputCpuDtype,
          outputCpuPtrS[0], 4, dimValuesMlu, dimOrderMlu));
  }

  /* copy the output layer to a vector*/
  vector<vector<vector<float>>> final_boxes;
  const float* outputData = reinterpret_cast<float*>(outputCpuPtrS[0]);
  vector<float> single_box;
  vector<vector<float>> batch_box;
  int count = outDimValues_[0][3];
  for (int b = 0; b < batchSize; b++) {
    batch_box.clear();
    int num_boxes = static_cast<int>(outputData[b * count]);
    if (num_boxes > 1024) {
        LOG(INFO) << "num_boxes : " << num_boxes;
        num_boxes = 1024;
    }
    for (int k = 0; k < num_boxes; k++) {
      int index = b * count + 64 + k * 7;
      single_box.clear();
      float max_limit = 1;
      float min_limit = 0;
      float bl = std::max(
          min_limit, std::min(max_limit, outputData[index + 3]));  // x1
      float br = std::max(
          min_limit, std::min(max_limit, outputData[index + 5]));  // x2
      float bt = std::max(
          min_limit, std::min(max_limit, outputData[index + 4]));  // y1
      float bb = std::max(
          min_limit, std::min(max_limit, outputData[index + 6]));  // y2
      single_box.push_back(bl);
      single_box.push_back(bt);
      single_box.push_back(br);
      single_box.push_back(bb);
      single_box.push_back(outputData[index + 2]);
      single_box.push_back(outputData[index + 1]);
      if ((br - bl) > 0 && (bb - bt) > 0) {
        batch_box.push_back(single_box);
      }
    }
    final_boxes.push_back(batch_box);
  }
  return final_boxes;
}

/* Load the mean file in binaryproto format. */
void Detector::setMean(const string& meanFile, const string& meanValues) {
  cv::Scalar channelMean;
  if (!meanValues.empty()) {
    if (!meanFile.empty()) {
      LOG(INFO) << "Cannot specify mean file";
      LOG(INFO) << " and mean value at the same time; ";
      LOG(INFO) << "Mean value will be specified ";
    }
    stringstream ss(meanValues);
    vector<float> values;
    string item;
    while (getline(ss, item, ',')) {
      float value = std::atof(item.c_str());
      values.push_back(value);
    }
    CHECK(values.size() == 1 || values.size() == numberChannels)
        << "Specify either 1 mean_value or as many as channels: "
        << numberChannels;
    vector<cv::Mat> channels;
    for (int i = 0; i < numberChannels; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(inputGeometry.height, inputGeometry.width, CV_32FC1,
                      cv::Scalar(values[i]));
      channels.push_back(channel);
    }
    cv::merge(channels, meanValue);
  } else {
    LOG(INFO) << "Cannot support mean file";
  }
}

void Detector::wrapInputLayer(vector<vector<cv::Mat>>* inputImages) {
  int width = inputGeometry.width;
  int height = inputGeometry.height;
  float* inputData = reinterpret_cast<float*>(inputCpuPtrS[0]);
  for (int i = 0; i < batchSize; ++i) {
    (*inputImages).push_back(vector<cv::Mat>());
    for (int j = 0; j < numberChannels; ++j) {
      cv::Mat channel(height, width, CV_32FC1, inputData);
      (*inputImages)[i].push_back(channel);
      inputData += width * height;
    }
  }
}

void Detector::preProcess(const vector<cv::Mat>& images,
                          vector<vector<cv::Mat>>* inputImages) {
  CHECK(images.size() == inputImages->size())
      << "Size of imgs and input_imgs doesn't match";
  for (int i = 0; i < images.size(); ++i) {
    cv::Mat sample;
    int num_channels_ = inDimValues_[0][3];
    cv::Size input_geometry;
    input_geometry = cv::Size(inDimValues_[0][1], inDimValues_[0][2]);  // 416*416
    if (images[i].channels() == 3 && num_channels_ == 1)
      cv::cvtColor(images[i], sample, cv::COLOR_BGR2GRAY);
    else if (images[i].channels() == 4 && num_channels_ == 1)
      cv::cvtColor(images[i], sample, cv::COLOR_BGRA2GRAY);
    else if (images[i].channels() == 4 && num_channels_ == 3)
      cv::cvtColor(images[i], sample, cv::COLOR_BGRA2BGR);
    else if (images[i].channels() == 1 && num_channels_ == 3)
      cv::cvtColor(images[i], sample, cv::COLOR_GRAY2BGR);
    else if (images[i].channels() == 3 && num_channels_ == 4)
      cv::cvtColor(images[i], sample, cv::COLOR_BGR2BGRA);
    else if (images[i].channels() == 1 && num_channels_ == 4)
      cv::cvtColor(images[i], sample, cv::COLOR_GRAY2BGRA);
    else
      sample = images[i];

    // 2.resize the image
    cv::Mat sample_temp;
    int input_dim = inDimValues_[0][1];
    cv::Mat sample_resized(input_dim, input_dim, CV_8UC4,
                           cv::Scalar(128, 128, 128));
    if (sample.size() != input_geometry) {
      // resize
      float img_w = sample.cols;
      float img_h = sample.rows;
      int new_w = static_cast<int>(
          img_w * std::min(static_cast<float>(input_dim) / img_w,
                           static_cast<float>(input_dim) / img_h));
      int new_h = static_cast<int>(
          img_h * std::min(static_cast<float>(input_dim) / img_w,
                           static_cast<float>(input_dim) / img_h));
      cv::resize(sample, sample_temp, cv::Size(new_w, new_h), CV_INTER_CUBIC);
      sample_temp.copyTo(sample_resized(
          cv::Range((static_cast<float>(input_dim) - new_h) / 2,
                    (static_cast<float>(input_dim) - new_h) / 2 + new_h),
          cv::Range((static_cast<float>(input_dim) - new_w) / 2,
                    (static_cast<float>(input_dim) - new_w) / 2 + new_w)));
    } else {
      sample_resized = sample;
    }

    // 3.BGR->RGB
    cv::Mat sample_rgb;
    if (num_channels_ == 4)
      cv::cvtColor(sample_resized, sample_rgb, cv::COLOR_BGRA2RGBA);
    else
      cv::cvtColor(sample_resized, sample_rgb, cv::COLOR_BGR2RGB);

    // 4.convert to float
    cv::Mat sample_float;
    if (num_channels_ == 3)
      sample_rgb.convertTo(sample_float, CV_32FC3, 1);
    else if (num_channels_ == 4)
      sample_rgb.convertTo(sample_float, CV_32FC4, 1);
    else
      sample_rgb.convertTo(sample_float, CV_32FC1, 1);

    cv::Mat sampleNormalized;
    bool int8 = (FLAGS_int8 != -1) ? FLAGS_int8 : FLAGS_fix8;
    if (int8 || (FLAGS_meanvalue.empty() && FLAGS_meanfile.empty()))
      sampleNormalized = sample_float;
    else
      cv::subtract(sample_float, meanValue, sampleNormalized);

    /* This operation will write the separate BGR planes directly to the
     * input layer of the network because it is wrapped by the cv::Mat
     * objects in input_channels. */
    cv::split(sampleNormalized, (*inputImages)[i]);
  }
}

void Detector::readImages(queue<string>* imagesQueue, int inputNumber,
                          vector<cv::Mat>* images, vector<string>* imageNames) {
  int leftNumber = imagesQueue->size();
  string file = imagesQueue->front();
  for (int i = 0; i < inputNumber; i++) {
    if (i < leftNumber) {
      file = imagesQueue->front();
      imageNames->push_back(file);
      imagesQueue->pop();
      if (file.find(" ") != string::npos) file = file.substr(0, file.find(" "));
      cv::Mat image = cv::imread(file, -1);
      images->push_back(image);
    } else {
      cv::Mat image = cv::imread(file, -1);
      images->push_back(image);
      imageNames->push_back("null");
    }
  }
}

static void WriteVisualizeBBox_offline(
    const vector<cv::Mat>& images,
    const vector<vector<vector<float>>> detections,
    const vector<string>& labelToDisplayName, const vector<string>& imageNames,
    int input_dim) {
  // Retrieve detections.
  const int imageNumber = images.size();

  for (int i = 0; i < imageNumber; ++i) {
    if (imageNames[i] == "null") continue;
    vector<vector<float>> result = detections[i];
    cv::Mat image = images[i];
    std::string name = imageNames[i];
    int positionMap = imageNames[i].rfind("/");
    if (positionMap > 0 && positionMap < imageNames[i].size()) {
      name = name.substr(positionMap + 1);
    }
    LOG(INFO) << "detect image: " << name;
    positionMap = name.rfind(".");
    if (positionMap > 0 && positionMap < name.size()) {
      name = name.substr(0, positionMap);
    }
    name = FLAGS_outputdir + "/" + name + ".txt";
    std::ofstream fileMap(name);

    float scaling_factors = std::min(
        static_cast<float>(input_dim) / static_cast<float>(image.cols),
        static_cast<float>(input_dim) / static_cast<float>(image.rows));
    for (int j = 0; j < result.size(); j++) {
      result[j][0] =
          result[j][0] * input_dim -
          static_cast<float>(input_dim - scaling_factors * image.cols) / 2.0;
      result[j][2] =
          result[j][2] * input_dim -
          static_cast<float>(input_dim - scaling_factors * image.cols) / 2.0;
      result[j][1] =
          result[j][1] * input_dim -
          static_cast<float>(input_dim - scaling_factors * image.rows) / 2.0;
      result[j][3] =
          result[j][3] * input_dim -
          static_cast<float>(input_dim - scaling_factors * image.rows) / 2.0;

      for (int k = 0; k < 4; k++) {
        result[j][k] = result[j][k] / scaling_factors;
      }
    }
    for (int j = 0; j < result.size(); j++) {
      result[j][0] = result[j][0] < 0 ? 0 : result[j][0];
      result[j][2] = result[j][2] < 0 ? 0 : result[j][2];
      result[j][1] = result[j][1] < 0 ? 0 : result[j][1];
      result[j][3] = result[j][3] < 0 ? 0 : result[j][3];
      result[j][0] = result[j][0] > image.cols ? image.cols : result[j][0];
      result[j][2] = result[j][2] > image.cols ? image.cols : result[j][2];
      result[j][1] = result[j][1] > image.rows ? image.rows : result[j][1];
      result[j][3] = result[j][3] > image.rows ? image.rows : result[j][3];
    }
    // getPointPosition(result, &p1, &p2, image.rows, image.cols);
    for (int j = 0; j < result.size(); j++) {
      int x0 = static_cast<int>(result[j][0]);
      int y0 = static_cast<int>(result[j][1]);
      int x1 = static_cast<int>(result[j][2]);
      int y1 = static_cast<int>(result[j][3]);
      cv::Point p1(x0, y0);
      cv::Point p2(x1, y1);
      cv::rectangle(image, p1, p2, cv::Scalar(0, 255, 0), 2);
      stringstream ss;
      ss << round(result[j][4] * 1000) / 1000.0;
      std::string str =
          labelToDisplayName[static_cast<int>(result[j][5])] + ":" + ss.str();
      cv::Point p5(x0, y0 - 5);
      cv::putText(image, str, p5, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                  cv::Scalar(255, 0, 0), 1);
      fileMap << labelToDisplayName[static_cast<int>(result[j][5])] << " "
              << ss.str() << " "
              << static_cast<float>(result[j][0]) / image.cols << " "
              << static_cast<float>(result[j][1]) / image.rows << " "
              << static_cast<float>(result[j][2]) / image.cols << " "
              << static_cast<float>(result[j][3]) / image.rows << " "
              << image.cols << " " << image.rows << "\n";
    }
    fileMap.close();
    stringstream ss;
    string outFile;
    int position = imageNames[i].find_last_of('/');
    string fileName(imageNames[i].substr(position + 1));
    string path = FLAGS_outputdir + "/" + "yolov3_";
    ss << path << fileName;
    ss >> outFile;
    cv::imwrite(outFile.c_str(), image);
  }
}

int main(int argc, char** argv) {
  {
    const char* env = getenv("log_prefix");
    if (!env || strcmp(env, "true") != 0) FLAGS_log_prefix = false;
  }
  ::google::InitGoogleLogging(argv[0]);
#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif
  gflags::SetUsageMessage(
      "Do detection using yolov3 mode.\n"
      "Usage:\n"
      "    yolov3_offline [FLAGS] model_file weights_file list_file\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (argc == 0) {
    gflags::ShowUsageWithFlagsRestrict(argv[0],
                                       "examples/yolo_v3/yolov3_offline");
    return 1;
  }
  if (FLAGS_logdir != "") {
    FLAGS_log_dir = FLAGS_logdir;
  } else {
    //  log to terminal's stderr if no log path specified
    FLAGS_alsologtostderr = 1;
  }

  cnrtInit(0);
  /* Create Detector class */
  Detector* detector =
      new Detector(FLAGS_offlinemodel, FLAGS_meanfile, FLAGS_meanvalue);
  /* Load labels. */
  std::vector<string> labels;
  std::ifstream labelsHandler(FLAGS_labels.c_str());
  CHECK(labelsHandler) << "Unable to open labels file " << FLAGS_labels;
  string line;
  while (std::getline(labelsHandler, line)) labels.push_back(line);
  labelsHandler.close();

  /* Load image files */
  queue<string> imageListQueue;
  int figuresNumber = 0;
  string lineTemp;
  std::ifstream filesHandler(FLAGS_images.c_str(), std::ios::in);
  CHECK(!filesHandler.fail()) << "Image file is invalid!";
  while (getline(filesHandler, lineTemp)) {
    imageListQueue.push(lineTemp);
    figuresNumber++;
  }
  filesHandler.close();
  LOG(INFO) << "there are " << figuresNumber << " figures in " << FLAGS_images;

  /* Detecting images */
  float timeUse;
  float totalTime = 0;
  struct timeval tpStart, tpEnd;
  int batchesNumber =
      ceil(static_cast<float>(figuresNumber) / detector->getBatchSize());
  for (int i = 0; i < batchesNumber; i++) {
    gettimeofday(&tpStart, NULL);
    vector<cv::Mat> images;
    vector<string> imageNames;
    /* Firstly read images from file list */
    detector->readImages(&imageListQueue, detector->getBatchSize(), &images,
                         &imageNames);
    /* Secondly fill images into input blob and do net forwarding */
    vector<vector<vector<float>>> detections = detector->detect(images);

    if (FLAGS_dump) {
      if (!FLAGS_outputdir.empty()) {
        WriteVisualizeBBox_offline(images, detections, labels, imageNames,
                                   detector->inputDim());
      }
    }
    gettimeofday(&tpEnd, NULL);
    timeUse = 1000000 * (tpEnd.tv_sec - tpStart.tv_sec) + tpEnd.tv_usec -
              tpStart.tv_usec;
    totalTime += timeUse;
    LOG(INFO) << "Detecting execution time: " << timeUse << " us";
    for (int num = 0; num < detector->getBatchSize(); num++) {
      LOG(INFO) << "detection size : " << detections[num].size();
    }
    LOG(INFO) << "\n";
    images.clear();
    imageNames.clear();
  }
  LOG(INFO) << "Total execution time: " << totalTime << " us";
  LOG(INFO) << "mluTime time: " << detector->mluTime() << " us";
  int batchsize = detector->getBatchSize();
  printPerf(figuresNumber, totalTime, detector->mluTime(), 1, batchsize);
  saveResult(figuresNumber, (-1), (-1), (-1),
      detector->mluTime(), totalTime, 1, batchsize);
  delete detector;
  cnrtDestroy();
  return 0;
}
#else
#include <glog/logging.h>
int main(int argc, char* argv[]) {
  LOG(FATAL) << "This program should be compiled with USE_MLU!";
  return 0;
}
#endif  // USE_MLU  && USE OPENCV
