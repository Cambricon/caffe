/*
All modification made by Cambricon Corporation: © 2018--2019 Cambricon Corporation
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

#if defined(USE_OPENCV)
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "caffe/proto/caffe.pb.h"
#include "caffe/blob.hpp"
#include "caffe/util/io.hpp"

#include "include/on_data_provider.hpp"
#include "include/on_runner.hpp"
#include "include/pipeline.hpp"
#include "include/command_option.hpp"
#include "include/common_functions.hpp"

using std::string;
using std::vector;
using caffe::BlobProto;

template <typename Dtype, template <typename> class Qtype>
void OnDataProvider<Dtype, Qtype>::runParallel() {
  OnRunner<Dtype, Qtype> *runner = static_cast<OnRunner<Dtype, Qtype> *>(this->runner_);
#ifdef USE_MLU
  setupConfig(this->threadId_, runner->deviceId(), runner->deviceSize());

  caffe::Net<Dtype>* netBuff =  runner->net();
  // data_parallel has already been applied to count
  int inputCount = netBuff->input_blobs()[0]->count();
  inputCpuPtr_ = new Dtype[inputCount];

  this->inNum_ = runner->n();
  this->inChannel_ = runner->c();
  this->inHeight_ = runner->h();
  this->inWidth_ = runner->w();
  this->inGeometry_ = cv::Size(this->inWidth_, this->inHeight_);

  this->SetMean();

  Pipeline<Dtype, Qtype>::waitForNotification();

#ifdef PRE_READ
  for (int i = 0; i < this->inImages_.size(); i++) {
    vector<cv::Mat> imgs = this->inImages_[i];
    vector<string> imgNames = this->imageName_[i];
#else
  while (this->imageList.size()) {
    this->inImages_.clear();
    this->imageName_.clear();
    this->readOneBatch();
    if (this->inImages_.empty()) break;
    vector<cv::Mat>& imgs = this->inImages_[0];
    vector<string>& imgNames  = this->imageName_[0];
#endif
    std::vector<std::vector<cv::Mat> > preprocessedImages;
    this->WrapInputLayer(&preprocessedImages, inputCpuPtr_);
    this->Preprocess(imgs, &preprocessedImages);

    auto inputBlob = netBuff->input_blobs()[0];
    auto inputMluTensorPtr = inputBlob->mlu_tensor();
    auto inputCpuTensorPtr = inputBlob->cpu_tensor();
    float* inputMluPtr = runner->popFreeInputData();
    Timer timer;
    cnmlMemcpyBatchTensorToDevice(inputCpuTensorPtr,
                                  inputCpuPtr_,
                                  inputMluTensorPtr,
                                  inputMluPtr,
                                  runner->dataParallel());
    timer.log("copy in time");
    runner->pushValidInputData(inputMluPtr);
    runner->pushValidInputNames(imgNames);
  }
  LOG(INFO) << "DataProvider: no more data. Exit!";
#endif
  // tell runner to exit
  runner->pushValidInputData(nullptr);
}

template <typename Dtype, template <typename> class Qtype>
void OnDataProvider<Dtype, Qtype>::runSerial() {
  OnRunner<Dtype, Qtype> *runner = static_cast<OnRunner<Dtype, Qtype> *>(this->runner_);

  if (!this->initSerialMode) {
    this->inNum_ = runner->n();
    this->inChannel_ = runner->c();
    this->inHeight_ = runner->h();
    this->inWidth_ = runner->w();
    this->inGeometry_ = cv::Size(this->inWidth_, this->inHeight_);

    this->SetMean();

    this->initSerialMode = true;
  }

  if (this->imageList.size()) {
    this->inImages_.clear();
    this->imageName_.clear();
    this->readOneBatch();
    vector<cv::Mat>& imgs = this->inImages_[0];
    vector<string>& imgNames  = this->imageName_[0];
    Dtype* inputMluPtr = runner->popFreeInputData();

    std::vector<std::vector<cv::Mat> > preprocessedImages;
    this->WrapInputLayer(&preprocessedImages, inputMluPtr);
    this->Preprocess(imgs, &preprocessedImages);

    runner->pushValidInputData(inputMluPtr);
    runner->pushValidInputNames(imgNames);
  } else {
    LOG(INFO) << "DataProvider: no more data. Exit!";
  }
}


template <typename Dtype, template <typename> class Qtype>
void OnDataProvider<Dtype, Qtype>::SetMeanFile() {
  cv::Scalar channel_mean;
  CHECK(this->meanValue_.empty()) <<
    "Cannot specify mean file and mean value at the same time";
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(this->meanFile_.c_str(), &blob_proto);
  /* Convert from BlobProto to Blob<float> */
  CHECK_EQ(blob_proto.channels(), this->inChannel_)
    << "Number of channels of mean file doesn't match input layer.";
  /* The format of the mean file is planar 32-bit float BGR or grayscale. */
  vector<cv::Mat> channels;
  int offset = blob_proto.height() * blob_proto.width();
  vector<int> tmp_array;
  tmp_array.resize(offset);
  for (int i = 0; i < this->inChannel_; ++i) {
    for (int j = 0; j < offset; ++j) {
      tmp_array[j] = blob_proto.data(i * offset + j);
    }
    /* Extract an individual channel. */
    cv::Mat channel(blob_proto.height(), blob_proto.width(), CV_32FC1,
                    tmp_array.data());
    channels.push_back(channel);
  }
  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);
  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
  channel_mean = cv::mean(mean);
  this->mean_ = cv::Mat(this->inGeometry_, mean.type(), channel_mean);
}

void setupConfig(int threadID, int deviceID, int deviceSize) {
  // note that these configurations are thread independent
  // i.e. you have to set it for all the threads which need MLU support
  // class Caffe is a thread local storage
  // to figure out how it works
  // see src/caffe/common.cpp and include/caffe/common.hpp
#ifdef USE_MLU
  caffe::Caffe::set_mlu_device(deviceID);
  caffe::Caffe::set_rt_core(FLAGS_mcore);
  caffe::Caffe::set_functype(FLAGS_functype);
  caffe::Caffe::set_mode(FLAGS_mmode);
  caffe::Caffe::setReshapeMode(caffe::Caffe::ReshapeMode::SETUPONLY);
  caffe::Caffe::setDataParallel(FLAGS_dataparallel);
  caffe::Caffe::setModelParallel(FLAGS_modelparallel);
  caffe::Caffe::setChannelId(threadID / deviceSize % 4);
  stringstream ss(FLAGS_datastrategy);
  vector<int> strategy;
  string value;
  while (getline(ss, value, ',')) {
    strategy.push_back(std::atoi(value.c_str()));
  }
  CHECK(strategy.size() == 2) <<
    "only support two values: input and output strategy";
  if (strategy[0] != -1 || strategy[1] != -1) {
    caffe::Caffe::setDataStrategy(strategy);
  }
#endif
}

INSTANTIATE_ON_CLASS(OnDataProvider);

#endif  // USE_OPENCV
