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
#include <condition_variable>  // NOLINT
#include <iomanip>
#include <iosfwd>
#include <map>
#include <memory>
#include <opencv2/core/core.hpp>  // NOLINT
#include <opencv2/highgui/highgui.hpp>  // NOLINT
#include <opencv2/imgproc/imgproc.hpp>  // NOLINT
#include <queue>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "blocking_queue.hpp"
#include "cnrt.h"  // NOLINT
#include "command_option.hpp"
#include "common_functions.hpp"
#include "simple_interface.hpp"
#include "threadPool.h"

using std::map;
using std::pair;
using std::queue;
using std::string;
using std::stringstream;
using std::thread;
using std::vector;

std::condition_variable condition;
std::mutex condition_m;
int start;

#define PRE_READ

DEFINE_int32(dump, 1, "0 or 1, dump output images or not.");
DEFINE_string(outputdir, ".", "The directoy used to save output images");

static void WriteVisualizeBBox_offline(
    const vector<cv::Mat>& images,
    const vector<vector<vector<float>>> detections,
    const vector<string>& labelToDisplayName, const vector<string>& imageNames,
    int input_dim, const int from, const int to) {
  // Retrieve detections.
  for (int i = from; i < to; ++i) {
    if (imageNames[i] == "null") continue;
    cv::Mat image;
    image = images[i];
    vector<vector<float>> result = detections[i];
    std::string name = imageNames[i];
    int positionMap = imageNames[i].rfind("/");
    if (positionMap > 0 && positionMap < imageNames[i].size()) {
      name = name.substr(positionMap + 1);
    }
    positionMap = name.find(".");
    if (positionMap > 0 && positionMap < name.size()) {
      name = name.substr(0, positionMap);
    }
    string filename = FLAGS_outputdir + "/" + name + ".txt";
    std::ofstream fileMap(filename);
    float scaling_factors = std::min(
        static_cast<float>(input_dim) / static_cast<float>(images[i].cols),
        static_cast<float>(input_dim) / static_cast<float>(images[i].rows));
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
      cv::Point p5(x0, y0 + 10);
      cv::putText(image, str, p5, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                  cv::Scalar(255, 0, 0), 0.5);

      fileMap << labelToDisplayName[static_cast<int>(result[j][5])] << " "
              << ss.str() << " "
              << static_cast<float>(result[j][0]) / image.cols << " "
              << static_cast<float>(result[j][1]) / image.rows << " "
              << static_cast<float>(result[j][2]) / image.cols << " "
              << static_cast<float>(result[j][3]) / image.rows << " "
              << image.cols << " " << image.rows << std::endl;
    }
    fileMap.close();
    stringstream ss;
    string outFile;
    ss << FLAGS_outputdir << "/yolov3_offline_" << name << ".jpg";
    ss >> outFile;
    cv::imwrite(outFile.c_str(), image);
  }
}

void setDeviceId(int dev_id) {
  unsigned devNum;
  CNRT_CHECK(cnrtGetDeviceCount(&devNum));
  if (dev_id >= 0) {
    CHECK_NE(devNum, 0) << "No device found";
    CHECK_LT(dev_id, devNum) << "Valid device count: " << devNum;
  } else {
    LOG(FATAL) << "Invalid device number";
  }
  cnrtDev_t dev;
  LOG(INFO) << "Using MLU device " << dev_id;
  CNRT_CHECK(cnrtGetDeviceHandle(&dev, dev_id));
  CNRT_CHECK(cnrtSetCurrentDevice(dev));
}

class PostProcessor;

class Inferencer {
  public:
  Inferencer(const cnrtRuntimeContext_t rt_ctx, const int& id);
  ~Inferencer();
  int n() { return inNum_; }
  int c() { return inChannel_; }
  int h() { return inHeight_; }
  int w() { return inWidth_; }
  bool simpleFlag() { return simple_flag_; }
  vector<int*> outDimValues() { return outDimValues_; }
  void pushValidInputData(void** data);
  void pushFreeInputData(void** data);
  void** popValidInputData();
  void** popFreeInputData();
  void pushValidOutputData(void** data);
  void pushFreeOutputData(void** data);
  void** popValidOutputData();
  void** popFreeOutputData();
  void pushValidInputNames(vector<string> rawImages);
  void pushValidInputDataAndNames(void** data, const vector<string>& images);
  vector<string> popValidInputNames();
  vector<void*> outCpuPtrs_;
  vector<void*> outSyncPtrs_;
  void simpleRun();
  inline int inBlobNum() { return inBlobNum_; }
  inline int outBlobNum() { return outBlobNum_; }
  inline int threadId() { return threadId_; }
  inline int deviceId() { return deviceId_; }
  inline int deviceSize() { return deviceSize_; }
  inline float inferencingTime() { return inferencingTime_; }
  inline void setThreadId(int id) { threadId_ = id; }
  inline void setPostProcessor(PostProcessor* p) { postProcessor_ = p; }
  inline int64_t* inputSizeArray() { return inputSizeArray_; }
  inline int64_t* outputSizeArray() { return outputSizeArray_; }
  inline cnrtDataType_t* inputDataTypeArray() { return inputDataTypeArray_; }
  inline cnrtDataType_t* outputDataTypeArray() { return outputDataTypeArray_; }

  private:
  void getIODataDesc();

  private:
  BlockingQueue<void**> validInputFifo_;
  BlockingQueue<void**> freeInputFifo_;
  BlockingQueue<void**> validOutputFifo_;
  BlockingQueue<void**> freeOutputFifo_;
  BlockingQueue<vector<string>> imagesFifo_;

  cnrtModel_t model_;
  cnrtQueue_t queue_;
  cnrtFunction_t function_;
  cnrtRuntimeContext_t rt_ctx_;
  cnrtDim3_t dim_;

  int64_t* inputSizeArray_;
  int64_t* outputSizeArray_;
  cnrtDataType_t* inputDataTypeArray_;
  cnrtDataType_t* outputDataTypeArray_;
  vector<int> inCounts_, outCounts_;
  vector<int> inDimNums_, outDimNums_;
  vector<int*> inDimValues_, outDimValues_;

  bool simple_flag_;
  int inBlobNum_, outBlobNum_;
  unsigned int inNum_, inChannel_, inHeight_, inWidth_;
  unsigned int outNum_, outChannel_, outHeight_, outWidth_;
  int threadId_;
  int deviceId_;
  int deviceSize_;
  int parallel_ = 1;
  float inferencingTime_;
  PostProcessor* postProcessor_;
  std::mutex infr_mutex_;
};

class PostProcessor {
  public:
  explicit PostProcessor(const int& deviceId)
      : threadId_(0), deviceId_(deviceId) {
    tp_ = new zl::ThreadPool(SimpleInterface::thread_num);
  }
  ~PostProcessor() {
    delete tp_;
  }
  void run();
  inline void setThreadId(int id) { threadId_ = id; }
  inline void setInferencer(Inferencer* p) { inferencer_ = p; }
  inline int top1() { return top1_; }
  inline int top5() { return top5_; }

  private:
  Inferencer* inferencer_;
  int threadId_;
  int deviceId_;
  int top1_;
  int top5_;
  zl::ThreadPool* tp_;
};

class DataProvider {
  public:
  DataProvider(const string& meanFile, const string& meanValue,
               const int& deviceId, const queue<string>& images)
      : threadId_(0), deviceId_(deviceId), imageList(images) {}
  ~DataProvider() {
    delete [] reinterpret_cast<float*>(cpuData_[0]);
    setDeviceId(deviceId_);
    delete cpuData_;
    if (inputSyncData_ != nullptr) {
      for (int i = 0; i < inferencer_->inBlobNum(); ++i) {
        if (inputSyncData_[i] != nullptr) free(inputSyncData_[i]);
      }
      free(inputSyncData_);
    }
    for (auto ptr : inPtrVector_) {
      for (int i = 0; i < this->inferencer_->inBlobNum(); i++) {
        cnrtFree(ptr[i]);
      }
      if (ptr != nullptr) free(ptr);
    }
    for (auto ptr : outPtrVector_) {
      for (int i = 0; i < this->inferencer_->outBlobNum(); i++) {
        cnrtFree(ptr[i]);
      }
      if (ptr != nullptr) free(ptr);
    }
  }
  void run();
  void SetMean(const string&, const string&);
  void preRead();
  void WrapInputLayer(vector<vector<cv::Mat>>* wrappedImages);
  void Preprocess(const vector<cv::Mat>& srcImages,
                  vector<vector<cv::Mat>>* dstImages);
  inline void setThreadId(int id) { threadId_ = id; }
  inline void setInferencer(Inferencer* p) {
    inferencer_ = p;
    inNum_ = p->n();  // make preRead happy
  }
  inline void pushInPtrVector(void** data) { inPtrVector_.push_back(data); }
  inline void pushOutPtrVector(void** data) { outPtrVector_.push_back(data); }

  private:
  int inNum_, inChannel_, inHeight_, inWidth_;
  int threadId_;
  int deviceId_;
  cv::Mat mean_;
  queue<string> imageList;
  Inferencer* inferencer_;
  cv::Size inGeometry_;
  void** cpuData_;
  void** inputSyncData_;
  vector<vector<cv::Mat>> inImages_;
  vector<vector<string>> imageName_;
  vector<void**> inPtrVector_;
  vector<void**> outPtrVector_;
};

void DataProvider::preRead() {
  while (imageList.size()) {
    vector<cv::Mat> rawImages;
    vector<string> imageNameVec;
    int imageLeft = imageList.size();
    string file = imageList.front();
    cv::Size imgsize = cv::Size(inferencer_->w(), inferencer_->h());
    for (int i = 0; i < inNum_; i++) {
      if (i < imageLeft) {
        file = imageList.front();
        imageNameVec.push_back(file);
        imageList.pop();
        if (file.find(" ") != string::npos)
          file = file.substr(0, file.find(" "));
        cv::Mat img = readImage(file, imgsize, FLAGS_yuv);
        rawImages.push_back(img);
      } else {
        cv::Mat img = readImage(file, imgsize, FLAGS_yuv);
        rawImages.push_back(img);
        imageNameVec.push_back("null");
      }
    }
    inImages_.push_back(rawImages);
    imageName_.push_back(imageNameVec);
  }
}

void DataProvider::run() {
  setDeviceId(deviceId_);
  for (int i = 0; i < FLAGS_fifosize; i++) {
    int inputNum = inferencer_->inBlobNum();
    int outputNum = inferencer_->outBlobNum();
    void** inputMluPtrS =
      reinterpret_cast<void**>(malloc(sizeof(void*) * inputNum));
    void** outputMluPtrS =
      reinterpret_cast<void**>(malloc(sizeof(void*) * outputNum));

    // malloc input
    for (int i = 0; i < inputNum; i++) {
      CNRT_CHECK(cnrtMalloc(&(inputMluPtrS[i]), inferencer_->inputSizeArray()[i]));
    }
    for (int i = 0; i < outputNum; i++) {
      CNRT_CHECK(cnrtMalloc(&(outputMluPtrS[i]), inferencer_->outputSizeArray()[i]));
    }
    inferencer_->pushFreeInputData(inputMluPtrS);
    inferencer_->pushFreeOutputData(outputMluPtrS);
    pushInPtrVector(inputMluPtrS);
    pushOutPtrVector(outputMluPtrS);
  }

  inNum_ = inferencer_->n();
  inChannel_ = inferencer_->c();
  inHeight_ = inferencer_->h();
  inWidth_ = inferencer_->w();
  inGeometry_ = cv::Size(inWidth_, inHeight_);
  SetMean(FLAGS_meanfile, FLAGS_meanvalue);
  cpuData_ = new (void*);
  cpuData_[0] = new float[inNum_ * inChannel_ * inHeight_ * inWidth_];
  inputSyncData_ =
    reinterpret_cast<void**>(malloc(sizeof(void*) * inferencer_->inBlobNum()));
  inputSyncData_[0] = malloc(inferencer_->inputSizeArray()[0]);

  std::unique_lock<std::mutex> lk(condition_m);
  LOG(INFO) << "Waiting ...";
  condition.wait(lk, [] { return start; });
  lk.unlock();

#ifdef PRE_READ
  for (int i = 0; i < inImages_.size(); i++) {
    vector<cv::Mat> rawImages = inImages_[i];
    vector<string> imageNameVec = imageName_[i];
#else
  while (imageList.size()) {
    vector<cv::Mat> rawImages;
    vector<string> imageNameVec;
    int imageLeft = imageList.size();
    string file = imageList.front();
    cv::Size imgsize = cv::Size(inferencer_->w(), inferencer_->h());
    for (int i = 0; i < inNum_; i++) {
      if (i < imageLeft) {
        file = imageList.front();
        imageNameVec.push_back(file);
        imageList.pop();
        if (file.find(" ") != string::npos)
          file = file.substr(0, file.find(" "));
        cv::Mat img = readImage(file, imgsize, FLAGS_yuv);
        rawImages.push_back(img);
      } else {
        cv::Mat img = readImage(file, imgsize, FLAGS_yuv);
        rawImages.push_back(img);
        imageNameVec.push_back("null");
      }
    }
#endif
    Timer prepareInput;
    vector<vector<cv::Mat>> preprocessedImages;
    WrapInputLayer(&preprocessedImages);
    Preprocess(rawImages, &preprocessedImages);
    prepareInput.log("prepare input data ...");

    void** mluData = inferencer_->popFreeInputData();
    Timer copyin;
    cnrtDataType_t mluDtype = inferencer_->inputDataTypeArray()[0];
    cnrtDataType_t cpuDtype;
    if (FLAGS_yuv) {
      cpuDtype = CNRT_UINT8;
    } else {
      cpuDtype = CNRT_FLOAT32;
    }
    int dimValuesCpu[4] = {inNum_, inChannel_,
                           inHeight_, inWidth_};
    int dimOrder[4] = {0, 2, 3, 1};  // NCHW --> NHWC
    if (cpuDtype != mluDtype) {
      CNRT_CHECK(cnrtTransOrderAndCast(cpuData_[0],
                                       cpuDtype,
                                       inputSyncData_[0],
                                       mluDtype,
                                       NULL,
                                       4,
                                       dimValuesCpu,
                                       dimOrder));
    } else {
        CNRT_CHECK(cnrtTransDataOrder(cpuData_[0],
                                      cpuDtype,
                                      inputSyncData_[0],
                                      4,
                                      dimValuesCpu,
                                      dimOrder));
    }
    CNRT_CHECK(cnrtMemcpy(reinterpret_cast<void*>(mluData[0]),
                          inputSyncData_[0],
                          inferencer_->inputSizeArray()[0],
                          CNRT_MEM_TRANS_DIR_HOST2DEV));
    copyin.log("copyin time ...");

    inferencer_->pushValidInputDataAndNames(mluData, imageNameVec);
  }

  LOG(INFO) << "DataProvider: no data ...";
  // tell inferencer there is no more images to process
}

void DataProvider::WrapInputLayer(vector<vector<cv::Mat>>* wrappedImages) {
  //  Parameter images is a vector [ ----   ] <-- images[in_n]
  //                                |
  //                                |-> [ --- ] <-- channels[3]
  // This method creates Mat objects, and places them at the
  // right offset of input stream
  int width = inferencer_->w();
  int height = inferencer_->h();
  int channels = FLAGS_yuv ? 1 : inferencer_->c();
  float* data = reinterpret_cast<float*>(cpuData_[0]);

  for (int i = 0; i < inferencer_->n(); ++i) {
    wrappedImages->push_back(vector<cv::Mat>());
    for (int j = 0; j < channels; ++j) {
      if (FLAGS_yuv) {
        cv::Mat channel(height, width, CV_8UC1, reinterpret_cast<char*>(data));
        (*wrappedImages)[i].push_back(channel);
        data += width * height / 4;
      } else {
        cv::Mat channel(height, width, CV_32FC1, data);
        (*wrappedImages)[i].push_back(channel);
        data += width * height;
      }
    }
  }
}

void DataProvider::Preprocess(const vector<cv::Mat>& sourceImages,
                              vector<vector<cv::Mat>>* destImages) {
  // Convert the input image to the input image format of the network.
  CHECK(sourceImages.size() == destImages->size())
      << "Size of sourceImages and destImages doesn't match";
  for (int i = 0; i < sourceImages.size(); ++i) {
    if (FLAGS_yuv) {
      cv::Mat sample_yuv;
      sourceImages[i].convertTo(sample_yuv, CV_8UC1);
      cv::split(sample_yuv, (*destImages)[i]);
      continue;
    }
    cv::Mat sample;
    int num_channels_ = inferencer_->c();
    if (sourceImages[i].channels() == 3 && num_channels_ == 1)
      cv::cvtColor(sourceImages[i], sample, cv::COLOR_BGR2GRAY);
    else if (sourceImages[i].channels() == 4 && num_channels_ == 1)
      cv::cvtColor(sourceImages[i], sample, cv::COLOR_BGRA2GRAY);
    else if (sourceImages[i].channels() == 4 && num_channels_ == 3)
      cv::cvtColor(sourceImages[i], sample, cv::COLOR_BGRA2BGR);
    else if (sourceImages[i].channels() == 1 && num_channels_ == 3)
      cv::cvtColor(sourceImages[i], sample, cv::COLOR_GRAY2BGR);
    else if (sourceImages[i].channels() == 3 && num_channels_ == 4)
      cv::cvtColor(sourceImages[i], sample, cv::COLOR_BGR2BGRA);
    else if (sourceImages[i].channels() == 1 && num_channels_ == 4)
      cv::cvtColor(sourceImages[i], sample, cv::COLOR_GRAY2BGRA);
    else
      sample = sourceImages[i];

    // 2.resize the image
    cv::Mat sample_temp;
    int input_dim = inferencer_->h();
    cv::Mat sample_resized(input_dim, input_dim, CV_8UC4,
                           cv::Scalar(128, 128, 128));
    if (sample.size() != inGeometry_) {
      // resize the raw picture and copyTo the center of a 416*416 backgroud
      // feature map
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
    // 3.BGR(A)->RGB(A)
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
      sample_resized.convertTo(sample_float, CV_32FC1, 1);
    // This operation will write the separate BGR planes directly to the
    // input layer of the network because it is wrapped by the cv::Mat
    // objects in input_channels. */
    cv::split(sample_float, (*destImages)[i]);
  }
}

void DataProvider::SetMean(const string& meanFile, const string& meanValue) {
  if (FLAGS_yuv) return;
  if (FLAGS_meanfile.empty() && FLAGS_meanvalue.empty()) return;
  cv::Scalar channel_mean;
  if (!meanValue.empty()) {
    if (!meanFile.empty()) {
      LOG(INFO) << "Cannot specify mean file";
      LOG(INFO) << " and mean value at the same time; ";
      LOG(INFO) << "Mean value will be specified ";
    }
    stringstream ss(meanValue);
    vector<float> values;
    string item;
    while (getline(ss, item, ',')) {
      float value = std::atof(item.c_str());
      values.push_back(value);
    }
    CHECK(values.size() == 1 || values.size() == inChannel_)
        << "Specify either one mean value or as many as channels: "
        << inChannel_;
    vector<cv::Mat> channels;
    for (int i = 0; i < inChannel_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(inGeometry_.height, inGeometry_.width, CV_32FC1,
                      cv::Scalar(values[i]));
      channels.push_back(channel);
    }
    cv::merge(channels, mean_);
  } else {
    LOG(WARNING) << "Cannot support mean file";
  }
}

Inferencer::Inferencer(const cnrtRuntimeContext_t rt_ctx, const int& id)
    : simple_flag_(true) {
  this->rt_ctx_ = rt_ctx;
  this->threadId_ = id;
  this->inferencingTime_ = 0;

  cnrtGetRuntimeContextInfo(rt_ctx, CNRT_RT_CTX_FUNCTION,
                            reinterpret_cast<void**>(&this->function_));
  cnrtGetRuntimeContextInfo(rt_ctx, CNRT_RT_CTX_MODEL_PARALLEL,
                            reinterpret_cast<void**>(&this->parallel_));
  cnrtGetRuntimeContextInfo(rt_ctx, CNRT_RT_CTX_DEV_ORDINAL,
                            reinterpret_cast<void**>(&this->deviceId_));

  getIODataDesc();
}
// get function's I/O DataDesc,
// allocate I/O data space on CPU memory and prepare Input data;
void Inferencer::getIODataDesc() {
  CNRT_CHECK(cnrtGetInputDataSize(&inputSizeArray_,
        &inBlobNum_, function_));
  CNRT_CHECK(cnrtGetOutputDataSize(&outputSizeArray_,
        &outBlobNum_, function_));
  CNRT_CHECK(cnrtGetInputDataType(&inputDataTypeArray_,
        &inBlobNum_, function_));
  CNRT_CHECK(cnrtGetOutputDataType(&outputDataTypeArray_,
        &outBlobNum_, function_));
  LOG(INFO) << "input blob num is " << inBlobNum_;

  inCounts_.resize(inBlobNum_, 1);
  outCounts_.resize(outBlobNum_, 1);
  inDimNums_.resize(inBlobNum_, 0);
  outDimNums_.resize(outBlobNum_, 0);
  inDimValues_.resize(inBlobNum_, nullptr);
  outDimValues_.resize(outBlobNum_, nullptr);

  for (int i = 0; i < inBlobNum_; i++) {
    CNRT_CHECK(cnrtGetInputDataShape(&(inDimValues_[i]),
        &(inDimNums_[i]), i, function_));
    for (int j = 0; j < inDimNums_[i]; ++j) {
      this->inCounts_[i] *= inDimValues_[i][j];
      LOG(INFO) << "shape " << inDimValues_[i][j];
    }
    if (i == 0) {
      inNum_ = inDimValues_[i][0];
      inChannel_ = inDimValues_[i][3];
      inWidth_ = inDimValues_[i][1];
      inHeight_ = inDimValues_[i][2];
    }
  }

  for (int i = 0; i < outBlobNum_; i++) {
    CNRT_CHECK(cnrtGetOutputDataShape(&(outDimValues_[i]),
        &(outDimNums_[i]), i, function_));
    for (int j = 0; j < outDimNums_[i]; ++j) {
      outCounts_[i] *= outDimValues_[i][j];
      LOG(INFO) << "shape " << outDimValues_[i][j];
    }
    if (0 == i) {
      outNum_ = outDimValues_[i][0];
      outChannel_ = outDimValues_[i][3];
      outHeight_ = outDimValues_[i][1];
      outWidth_ = outDimValues_[i][2];
    }
    outCpuPtrs_.push_back(malloc(sizeof(float) * outCounts_[i]));
    outSyncPtrs_.push_back(malloc(outputSizeArray_[i]));
  }
}

Inferencer::~Inferencer() {
  setDeviceId(deviceId_);
  for (int i = 0; i < outCpuPtrs_.size(); i++) {
    if (outCpuPtrs_[i] != nullptr) {
      free(outCpuPtrs_[i]);
      outCpuPtrs_[i] = nullptr;
    }
  }
  for (int i = 0; i < outSyncPtrs_.size(); i++) {
    if (outSyncPtrs_[i] != nullptr) {
      free(outSyncPtrs_[i]);
      outSyncPtrs_[i] = nullptr;
    }
  }
}

void** Inferencer::popFreeInputData() { return freeInputFifo_.pop(); }

void** Inferencer::popValidInputData() { return validInputFifo_.pop(); }

void Inferencer::pushFreeInputData(void** data) { freeInputFifo_.push(data); }

void Inferencer::pushValidInputData(void** data) { validInputFifo_.push(data); }

void** Inferencer::popFreeOutputData() { return freeOutputFifo_.pop(); }

void** Inferencer::popValidOutputData() { return validOutputFifo_.pop(); }

void Inferencer::pushFreeOutputData(void** data) { freeOutputFifo_.push(data); }

void Inferencer::pushValidOutputData(void** data) {
  validOutputFifo_.push(data);
}

void Inferencer::pushValidInputNames(vector<string> images) {
  imagesFifo_.push(images);
}

vector<string> Inferencer::popValidInputNames() { return imagesFifo_.pop(); }

void Inferencer::pushValidInputDataAndNames(void** data, const vector<string>& images) {
  std::lock_guard<std::mutex> lk(infr_mutex_);
  pushValidInputData(data);
  pushValidInputNames(images);
}

void Inferencer::simpleRun() {
// #define PINGPONG
#ifdef PINGPONG
#define RES_SIZE 2
#else
#define RES_SIZE 1
#endif

  // set device to runtime context binded device
  cnrtSetCurrentContextDevice(rt_ctx_);

  cnrtQueue_t queue[RES_SIZE];
  cnrtNotifier_t notifierBeginning[RES_SIZE];
  cnrtNotifier_t notifierEnd[RES_SIZE];

  for (int i = 0; i < RES_SIZE; i++) {
    CHECK(cnrtCreateQueue(&queue[i]) == CNRT_RET_SUCCESS)
        << "CNRT create queue error, thread_id " << threadId();
    cnrtCreateNotifier(&notifierBeginning[i]);
    cnrtCreateNotifier(&notifierEnd[i]);
  }
  float eventInterval[RES_SIZE] = {0};
  void** mluInData[RES_SIZE];
  void** mluOutData[RES_SIZE];

  auto do_pop = [&](int index, void** param) {
    mluInData[index] = popValidInputData();
    if (mluInData[index] == nullptr) return false;
    mluOutData[index] = popFreeOutputData();
    for (int i = 0; i < inBlobNum(); i++) {
      param[i] = mluInData[index][i];
    }
    for (int i = 0; i < outBlobNum(); i++) {
      param[inBlobNum() + i] = mluOutData[index][i];
    }

    return true;
  };

  auto do_invoke = [&](int index, void** param) {
    cnrtPlaceNotifier(notifierBeginning[index], queue[index]);
    CNRT_CHECK(cnrtInvokeRuntimeContext(rt_ctx_, param, queue[index], nullptr));
  };

  auto do_sync = [&](int index) {
    cnrtPlaceNotifier(notifierEnd[index], queue[index]);
    if (cnrtSyncQueue(queue[index]) == CNRT_RET_SUCCESS) {
      cnrtNotifierDuration(notifierBeginning[index], notifierEnd[index],
                           &eventInterval[index]);
      inferencingTime_ += eventInterval[index];
      printfMluTime(eventInterval[index]);
    } else {
      LOG(ERROR) << " SyncQueue error";
    }
    pushValidOutputData(mluOutData[index]);
    pushFreeInputData(mluInData[index]);
  };

#ifdef PINGPONG
  bool pong_valid = false;
  while (true) {
    void* param[inBlobNum() + outBlobNum()];

    // pop - ping
    if (do_pop(0, static_cast<void**>(param)) == false) {
      if (pong_valid) do_sync(1);
      break;
    }
    // invoke - ping
    do_invoke(0, static_cast<void**>(param));

    // sync - pong
    if (pong_valid) do_sync(1);

    // pop - pong
    if (do_pop(1, static_cast<void**>(param)) == false) {
      do_sync(0);
      break;
    }

    // invoke - pong
    do_invoke(1, static_cast<void**>(param));
    pong_valid = true;

    // sync - ping
    do_sync(0);
  }
#else
  while (true) {
    void* param[inBlobNum() + outBlobNum()];
    // pop - ping
    if (do_pop(0, static_cast<void**>(param)) == false) {
      break;
    }
    // invoke - ping
    do_invoke(0, static_cast<void**>(param));

    // sync - ping
    do_sync(0);
  }
#endif

  for (int i = 0; i < RES_SIZE; i++) {
    cnrtDestroyNotifier(&notifierBeginning[i]);
    cnrtDestroyNotifier(&notifierEnd[i]);
    cnrtDestroyQueue(queue[i]);
  }

  // tell postprocessor to exit
  pushValidOutputData(nullptr);
}


void PostProcessor::run() {
  setDeviceId(deviceId_);
  Inferencer* infr = inferencer_;  // avoid line wrap

  vector<string> labelNameMap;
  if (!FLAGS_labels.empty()) {
    std::ifstream labels(FLAGS_labels);
    string line;
    while (std::getline(labels, line)) {
      labelNameMap.push_back(line);
    }
    labels.close();
  }

  int TASK_NUM = SimpleInterface::thread_num;
  std::vector<std::future<void>> futureVector;
  while (true) {
    void** mluOutData = infr->popValidOutputData();
    if (nullptr == mluOutData) break;  // no more data to process

    CNRT_CHECK(cnrtMemcpy(reinterpret_cast<void*>(infr->outSyncPtrs_[0]),
                          mluOutData[0],
                          infr->outputSizeArray()[0],
                          CNRT_MEM_TRANS_DIR_DEV2HOST));
    cnrtDataType_t cpuDtype = CNRT_FLOAT32;
    cnrtDataType_t mluDtype = infr->outputDataTypeArray()[0];
    int dimValuesMlu[4] = {infr->outDimValues()[0][0], infr->outDimValues()[0][1],
                           infr->outDimValues()[0][2], infr->outDimValues()[0][3]};
    int dimOrder[4] = {0, 3, 1, 2};  // NHWC --> NCHW
    if (cpuDtype != mluDtype) {
      CNRT_CHECK(cnrtTransOrderAndCast(reinterpret_cast<void*>(infr->outSyncPtrs_[0]),
                                       mluDtype,
                                       reinterpret_cast<void*>(infr->outCpuPtrs_[0]),
                                       cpuDtype,
                                       nullptr,
                                       4,
                                       dimValuesMlu,
                                       dimOrder));
    } else {
      CNRT_CHECK(cnrtTransDataOrder(infr->outSyncPtrs_[0],
                                    mluDtype,
                                    infr->outCpuPtrs_[0],
                                    4,
                                    dimValuesMlu,
                                    dimOrder));
    }
    infr->pushFreeOutputData(mluOutData);

    Timer dumpTimer;
    if (FLAGS_dump) {
      vector<vector<vector<float>>> final_boxes;
      float* outputData = reinterpret_cast<float*>(infr->outCpuPtrs_[0]);
      float max_limit = 1;
      float min_limit = 0;
      int batchSize = infr->outDimValues()[0][0];
      int count = infr->outDimValues()[0][3];

      for (int i = 0; i < batchSize; i++) {
        int num_boxes = static_cast<int>(outputData[i * count]);
        if (num_boxes > 1024) {
            LOG(INFO) << "num_boxes : " << num_boxes;
            num_boxes = 1024;
        }
        vector<vector<float>> batch_box;
        for (int k = 0; k < num_boxes; k++) {
          int index = i * count + 64 + k * 7;
          vector<float> single_box;
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

      vector<string> origin_img = infr->popValidInputNames();
      vector<cv::Mat> imgs;
      vector<string> img_names;
      for (auto img_name : origin_img) {
        if (img_name != "null") {
          cv::Mat img;
          if (FLAGS_yuv) {
            cv::Size size = cv::Size(infr->w(), infr->h());
            img = yuv420sp2Bgr24(convertYuv2Mat(img_name, size));
          } else {
            img = cv::imread(img_name, -1);
          }
          imgs.push_back(img);
          img_names.push_back(img_name);
        }
      }
      int input_dim = FLAGS_yuv ? 416 : infr->h();
      const int size = imgs.size();
      if (TASK_NUM > size) TASK_NUM = size;
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
        auto func = tp_->add(
            [](const vector<cv::Mat>& imgs,
              const vector<vector<vector<float>>>& final_boxes,
              const vector<string>& labelNameMap,
              const vector<string>& img_names, const int input_dim,
              const int& from, const int& to) {
            WriteVisualizeBBox_offline(imgs, final_boxes, labelNameMap,
                img_names, input_dim, from, to); },
            imgs, final_boxes, labelNameMap, img_names, input_dim, from, to);
        futureVector.push_back(std::move(func));
      }
    }
    dumpTimer.log("dump imgs time ...");
  }
  for (int i = 0; i < futureVector.size(); i++) {
    futureVector[i].get();
  }
}

class Pipeline {
  public:
  Pipeline(const string& offlinemodel, const string& meanFile,
           const string& meanValue, const int& id, const int& deviceId,
           const int& devicesize,
           const vector<queue<string>>& images);
  ~Pipeline();
  void run();
  inline DataProvider* dataProvider() { return data_provider_; }
  inline Inferencer* inferencer() { return inferencer_; }
  inline PostProcessor* postProcessor() { return postProcessor_; }

  private:
  vector<DataProvider*> data_providers_;
  DataProvider* data_provider_;
  Inferencer* inferencer_;
  PostProcessor* postProcessor_;
};
Pipeline::Pipeline(const string& offlinemodel, const string& meanFile,
                   const string& meanValue, const int& id, const int& deviceId,
                   const int& devicesize,
                   const vector<queue<string>>& images)
    : data_providers_(SimpleInterface::data_provider_num_),
      data_provider_(nullptr),
      inferencer_(nullptr),
      postProcessor_(nullptr) {
  auto& simpleInterface = SimpleInterface::getInstance();
  auto dev_runtime_contexts = simpleInterface.get_runtime_contexts();
  inferencer_ = new Inferencer(dev_runtime_contexts[id % devicesize], id);
  postProcessor_ = new PostProcessor(deviceId);

  postProcessor_->setInferencer(inferencer_);
  postProcessor_->setThreadId(id);
  inferencer_->setPostProcessor(postProcessor_);
  inferencer_->setThreadId(id);

  int data_provider_num = SimpleInterface::data_provider_num_;
  for (int i = 0; i < data_provider_num; i++) {
    data_providers_[i] = new DataProvider(meanFile, meanValue, deviceId,
        images[data_provider_num * id + i]);
    data_providers_[i]->setInferencer(inferencer_);
    data_providers_[i]->setThreadId(id);
#ifdef PRE_READ
    data_providers_[i]->preRead();
#endif
  }
}

Pipeline::~Pipeline() {
  for (auto data_provider : data_providers_) {
    delete data_provider;
  }

  if (inferencer_) {
    delete inferencer_;
  }

  if (postProcessor_) {
    delete postProcessor_;
  }
}

void Pipeline::run() {
  int data_provider_num = 1;
  data_provider_num = data_providers_.size();
  vector<thread*> threads(data_provider_num + 2, nullptr);

  for (int i = 0; i < data_provider_num; i++) {
    threads[i] = new thread(&DataProvider::run, data_providers_[i]);
  }
  threads[data_provider_num] =
      new thread(&Inferencer::simpleRun, inferencer_);
  threads[data_provider_num + 1] =
      new thread(&PostProcessor::run, postProcessor_);

  for (int i = 0; i < data_provider_num; i++) {
    threads[i]->join();
    delete threads[i];
  }

  // push a nullptr for simple compile when the thread of data provider finished
  // tasks
  inferencer_->pushValidInputData(nullptr);

  for (int i = 0; i < 2; i++) {
    threads[data_provider_num + i]->join();
    delete threads[data_provider_num + i];
  }
}

int main(int argc, char* argv[]) {
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
      "    yolov3_offline_multicore [FLAGS] model_file list_file\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (argc == 0) {
    gflags::ShowUsageWithFlagsRestrict(
        argv[0], "examples/yolo_v3/yolov3_offline_multicore");
    return 1;
  }

  auto& simpleInterface = SimpleInterface::getInstance();
  // if simple_compile option has been specified to 1 by user, simple compile
  // thread);
  int provider_num = 1;
  simpleInterface.setFlag(true);
  provider_num = SimpleInterface::data_provider_num_;

  if (FLAGS_logdir != "") {
    FLAGS_log_dir = FLAGS_logdir;
  } else {
    //  log to terminal's stderr if no log path specified
    FLAGS_alsologtostderr = 1;
  }

  std::ifstream files_tmp(FLAGS_images.c_str(), std::ios::in);
  // get device ids
  std::stringstream sdevice(FLAGS_mludevice);
  vector<int> deviceIds_;
  std::string item;
  while (getline(sdevice, item, ',')) {
    int device = std::atoi(item.c_str());
    deviceIds_.push_back(device);
  }
  int totalThreads = FLAGS_threads * deviceIds_.size();
  int imageNum = 0;
  vector<string> files;
  std::string line_tmp;
  vector<queue<string>> imageList(totalThreads * provider_num);
  if (files_tmp.fail()) {
    LOG(ERROR) << "open " << FLAGS_images << " file fail!";
    return 1;
  } else {
    while (getline(files_tmp, line_tmp)) {
      imageList[imageNum % totalThreads].push(line_tmp);
      imageNum++;
    }
  }
  files_tmp.close();
  LOG(INFO) << "there are " << imageNum << " figures in " << FLAGS_images;

  cnrtInit(0);
  simpleInterface.loadOfflinemodel(FLAGS_offlinemodel, deviceIds_,
      FLAGS_channel_dup);

  vector<thread*> stageThreads;
  vector<Pipeline*> pipelineVector;
  for (int i = 0; i < totalThreads; i++) {
    Pipeline* pipeline;
    if (imageList.size()) {
      pipeline =
        new Pipeline(FLAGS_offlinemodel, FLAGS_meanfile,
            FLAGS_meanvalue, i, deviceIds_[i % deviceIds_.size()],
            deviceIds_.size(), imageList);
    }

    stageThreads.push_back(new thread(&Pipeline::run, pipeline));
    pipelineVector.push_back(pipeline);
  }

  float execTime;
  struct timeval tpend, tpstart;
  gettimeofday(&tpstart, NULL);
  {
    std::lock_guard<std::mutex> lk(condition_m);
    LOG(INFO) << "Notify to start ...";
  }
  start = 1;
  condition.notify_all();
  for (int i = 0; i < stageThreads.size(); i++) {
    stageThreads[i]->join();
    delete stageThreads[i];
  }
  gettimeofday(&tpend, NULL);
  execTime = 1000000 * (tpend.tv_sec - tpstart.tv_sec) + tpend.tv_usec -
             tpstart.tv_usec;
  LOG(INFO) << "yolov3_detection() execution time: " << execTime << " us";
  float mluTime = 0;
  for (int i = 0; i < pipelineVector.size(); i++) {
    mluTime += pipelineVector[i]->inferencer()->inferencingTime();
  }

  /*  LOG(INFO) << "Hardware fps: " << imageNum / mluTime * totalThreads * 1e6;
    LOG(INFO) << "End2end throughput fps: " << imageNum / execTime * 1e6; */
  int batchsize = pipelineVector[0]->inferencer()->n();
  printPerf(imageNum, execTime, mluTime, totalThreads, batchsize);
  saveResult(imageNum, (-1), (-1), (-1), mluTime, execTime, totalThreads, batchsize);


  for (auto iter : pipelineVector) {
    if (iter != nullptr) {
      delete iter;
    }
  }
  simpleInterface.destroyRuntimeContext();
  cnrtDestroy();
}

#else
#include "caffe/common.hpp"
int main(int argc, char* argv[]) {
  LOG(FATAL) << "This program should be compiled with the defintion"
             << " of both USE_MLU and USE_OPENCV!";
  return 0;
}
#endif  // USE_MLU
