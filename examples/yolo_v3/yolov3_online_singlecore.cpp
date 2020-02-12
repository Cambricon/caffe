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
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>
#include "common_functions.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::vector;
using std::string;
using std::queue;

DEFINE_string(model, "",
    "The prototxt file used to find net configuration");
DEFINE_string(weights, "",
    "The binary file used to set net parameter");
DEFINE_string(meanfile, "",
              "The mean file used to subtract from the input image.");
DEFINE_string(
    meanvalue, "",
    "If specified, can be one value or can be same as image channels"
    " - would subtract from the corresponding channel). Separated by ','."
    "Either mean_file or mean_value should be provided, not both.");
DEFINE_string(mmode, "MFUS",
    "CPU, MLU or MFUS, MFUS mode");
DEFINE_string(mcore, "MLU100",
    "1H8, 1H16, MLU100 for different Cambricon hardware pltform");
DEFINE_int32(fix8, 0,
    "FP16 or FIX8, fix8 mode, default: 0");
DEFINE_int32(int8, -1, "invalid(-1), fp16(0) or int8(1) mode. Default is invalid(-1)."
             "If specified, use int8 value, else, use fix8 value");
DEFINE_int32(mludevice, 0,
    "set using mlu device number, default: 0");
DEFINE_string(images, "", "The input file list");
DEFINE_string(outputdir, ".", "The directoy used to save output images");
DEFINE_string(labels, "", "infomation about mapping from label to name");
DEFINE_string(logdir, "",
              "path to dump log file, to terminal "
              "stderr by default");
DEFINE_int32(dump, 1, "0 or 1, dump output images or not.");

class Detector {
  public:
  Detector(const string& modelFile,
           const string& weightsFile,
           const string& meanFile,
           const string& meanValues);
  ~Detector();
  vector<vector<vector<float>>> detect(const vector<cv::Mat>& images);
  vector<int> inputShape;
  int getBatchSize() { return batchSize; }
  void readImages(queue<string>* imagesQueue, int inputNumber,
                  vector<cv::Mat>* images, vector<string>* imageNames);
  inline float runTime() { return runTime_; }

  private:
  void setMean(const string& meanFile, const string& meanValues);
  void wrapInputLayer(vector<vector<cv::Mat>>* inputImages);
  void preProcess(const vector<cv::Mat>& images,
                  vector<vector<cv::Mat>>* inputImages);

  private:
  Net<float>* network;
  cv::Size inputGeometry;
  int batchSize;
  int numberChannels;
  cv::Mat meanValue;
  int inputNum, outputNum;
  float runTime_;
};

Detector::Detector(const string& modelFile,
                   const string& weightsFile,
                   const string& meanFile,
                   const string& meanValues):runTime_(0) {
  /* Load the network. */
  network = new Net<float>(modelFile, TEST);
  network->CopyTrainedLayersFrom(weightsFile);

  outputNum = network->num_outputs();
  Blob<float>* inputLayer = network->input_blobs()[0];
  batchSize = inputLayer->num();
  numberChannels = inputLayer->channels();
  inputShape = inputLayer->shape();
  CHECK(numberChannels == 3 || numberChannels == 1)
    << "Input layer should have 1 or 3 channels.";
  inputGeometry = cv::Size(inputLayer->width(), inputLayer->height());
  /* Load the binaryproto mean file. */
  setMean(meanFile, meanValues);
}

Detector::~Detector() {
  delete network;
}

vector<vector<vector<float>>> Detector::detect(const vector<cv::Mat>& images) {
  vector<vector<cv::Mat>> inputImages;
  wrapInputLayer(&inputImages);
  preProcess(images, &inputImages);
  float timeUse;
  struct timeval tpEnd, tpStart;
  gettimeofday(&tpStart, NULL);

#ifdef USE_MLU
  float eventTimeUse;
  cnrtNotifier_t notifierBeginning, notifierEnd;
  if(caffe::Caffe::mode() != caffe::Caffe::CPU) {
    cnrtCreateNotifier(&notifierBeginning);
    cnrtCreateNotifier(&notifierEnd);
    cnrtPlaceNotifier(notifierBeginning, caffe::Caffe::queue());
  }
#endif

  network->Forward();

#ifdef USE_MLU
  if (caffe::Caffe::mode() != caffe::Caffe::CPU) {
    cnrtPlaceNotifier(notifierEnd, caffe::Caffe::queue());
    cnrtSyncQueue(caffe::Caffe::queue());
    cnrtNotifierDuration(notifierBeginning, notifierEnd, &eventTimeUse);
    this->runTime_ +=  eventTimeUse;
    printfMluTime(eventTimeUse);
  }
#endif

  gettimeofday(&tpEnd, NULL);
  timeUse = 1000000 * (tpEnd.tv_sec - tpStart.tv_sec)
             + tpEnd.tv_usec - tpStart.tv_usec;
  LOG(INFO) << "Forward execution time: " << timeUse << " us";

  /* copy the output layer to a vector*/
  vector<vector<vector<float>>> final_boxes;
  Blob<float>* outputLayer = network->output_blobs()[0];
  const float* outputData = outputLayer->cpu_data();
  const vector<int> shape = outputLayer->shape();
  vector<float> single_box;
  vector<vector<float>> batch_box;
  int count = outputLayer->channels();
  for (int b=0; b < batchSize; b++) {
    batch_box.clear();
    int num_boxes = static_cast<int>(outputData[b * count]);
    if (num_boxes > 1024) {
        LOG(INFO) << "num_boxes : " << num_boxes;
        num_boxes = 1024;
    }
    for (int k =0; k < num_boxes; k++) {
      single_box.clear();
      int index = b * count + 64 + k * 7;
      float max_limit = 1;
      float min_limit = 0;
      float bl = std::max(min_limit,
                 std::min(max_limit, outputData[index + 3])); //x1
      float br = std::max(min_limit,
                 std::min(max_limit, outputData[index + 5])); //x2
      float bt = std::max(min_limit,
                 std::min(max_limit, outputData[index + 4])); //y1
      float bb = std::max(min_limit,
                 std::min(max_limit, outputData[index + 6])); //y2
      single_box.push_back(bl);
      single_box.push_back(bt);
      single_box.push_back(br);
      single_box.push_back(bb);
      single_box.push_back(outputData[index + 2]);
      single_box.push_back(outputData[index + 1]);
      if ((br-bl)> 0 && (bb-bt) > 0) {
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
    if (!meanFile.empty()) {
      BlobProto blobProto;
      ReadProtoFromBinaryFileOrDie(meanFile.c_str(), &blobProto);
      /* Convert from BlobProto to Blob<float> */
      Blob<float> meanBlob;
      meanBlob.FromProto(blobProto);
      CHECK_EQ(meanBlob.channels(), numberChannels)
          << "Number of channels of mean file doesn't match input layer.";
      /* The format of the mean file is planar 32-bit float BGR or grayscale. */
      vector<cv::Mat> channels;
      float* data = meanBlob.mutable_cpu_data();
      for (int i = 0; i < numberChannels; ++i) {
        /* Extract an individual channel. */
        cv::Mat channel(meanBlob.height(), meanBlob.width(), CV_32FC1, data);
        channels.push_back(channel);
        data += meanBlob.height() * meanBlob.width();
      }
      /* Merge the separate channels into a single image. */
      cv::Mat mean;
      cv::merge(channels, mean);
      /* Compute the global mean pixel value and create a mean image
       * filled with this value. */
      channelMean = cv::mean(mean);
      meanValue = cv::Mat(inputGeometry, mean.type(), channelMean);
    }
  }
}

void Detector::wrapInputLayer(vector<vector<cv::Mat>>* inputImages) {
  int width = inputGeometry.width;
  int height = inputGeometry.height;
  Blob<float>* inputLayer = network->input_blobs()[0];
  float* inputData = inputLayer->mutable_cpu_data();
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
    int num_channels_ = inputShape[1];
    if (images[i].channels() == 3 && num_channels_ == 1)
      cv::cvtColor(images[i], sample, cv::COLOR_BGR2GRAY);
    else if (images[i].channels() == 4 && num_channels_ == 1)
      cv::cvtColor(images[i], sample, cv::COLOR_BGRA2GRAY);
    else if (images[i].channels() == 4 && num_channels_ == 3)
      cv::cvtColor(images[i], sample, cv::COLOR_BGRA2BGR);
    else if (images[i].channels() == 1 && num_channels_ == 3)
      cv::cvtColor(images[i], sample, cv::COLOR_GRAY2BGR);
    else
      sample = images[i];

    // 2.resize the image
    cv::Mat sample_temp;
    int input_dim = inputShape[2];
    cv::Mat sample_resized(input_dim, input_dim, CV_8UC3,
                           cv::Scalar(128, 128, 128));
    if (sample.size() != inputGeometry) {
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
    cv::cvtColor(sample_resized, sample_rgb, cv::COLOR_BGR2RGB);
    // 4.convert to float
    cv::Mat sample_float;
    if (num_channels_ == 3)
      // 1/255.0
      sample_rgb.convertTo(sample_float, CV_32FC3, 1);
    else
      sample_rgb.convertTo(sample_float, CV_32FC1, 1);
    /* This operation will write the separate BGR planes directly to the
     * input layer of the network because it is wrapped by the cv::Mat
     * objects in input_channels. */
    cv::split(sample_float, (*inputImages)[i]);
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

static void WriteVisualizeBBox_online(
    const vector<cv::Mat>& images,
    const vector<vector<vector<float>>> detections,
    const vector<string>& labelToDisplayName,
    const vector<string>& imageNames,
    int input_dim) {
  // Retrieve detections.
  const int imageNumber = images.size();

  for (int i = 0; i < imageNumber; ++i) {
    if (imageNames[i] == "null") continue;
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
    vector<vector<float>> result = detections[i];
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
      result[j][0] =
          result[j][0] > image.cols ? image.cols : result[j][0];
      result[j][2] =
          result[j][2] > image.cols ? image.cols : result[j][2];
      result[j][1] =
          result[j][1] > image.rows ? image.rows : result[j][1];
      result[j][3] =
          result[j][3] > image.rows ? image.rows : result[j][3];
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
      stringstream s;
      s << j << "-";
      std::string str =
          s.str() + labelToDisplayName[static_cast<int>(result[j][5])] + ":" + ss.str();
      cv::Point p5(x0, y0 - 5);
      cv::putText(image, str, p5, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                 cv::Scalar(255, 0, 0), 1);
      fileMap << labelToDisplayName[static_cast<int>(result[j][5])]
              << " " << ss.str()
              << " " << static_cast<float>(result[j][0]) / image.cols
              << " " << static_cast<float>(result[j][1]) / image.rows
              << " " << static_cast<float>(result[j][2]) / image.cols
              << " " << static_cast<float>(result[j][3]) / image.rows
              << " " << image.cols
              << " " << image.rows << "\n";
    }
    fileMap.close();
    stringstream ss;
    string outFile;
    int position = imageNames[i].find_last_of('/');
    string fileName(imageNames[i].substr(position + 1));
    string path = FLAGS_outputdir + "/" + "yolov3_" + FLAGS_mmode + "_";
    ss << path << fileName;
    ss >> outFile;
    cv::imwrite(outFile.c_str(), image);
  }
}

int main(int argc, char** argv) {
{
  const char * env = getenv("log_prefix");
  if (!env || strcmp(env, "true") != 0)
    FLAGS_log_prefix = false;
}
  ::google::InitGoogleLogging(argv[0]);
#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif
  gflags::SetUsageMessage(
      "Do detection using yolov3 mode.\n"
      "Usage:\n"
      "    yolov3_online [FLAGS] model_file weights_file list_file\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (argc == 0) {
    gflags::ShowUsageWithFlagsRestrict(argv[0],
                                       "examples/yolo_v3/yolov3_online");
    return 1;
  }
  if (FLAGS_logdir != "") {
    FLAGS_log_dir = FLAGS_logdir;
  } else {
    //  log to terminal's stderr if no log path specified
    FLAGS_alsologtostderr = 1;
  }

  if (FLAGS_mmode == "CPU") {
    Caffe::set_mode(Caffe::CPU);
  } else {
#ifdef USE_MLU
    cnmlInit(0);
    Caffe::set_rt_core(FLAGS_mcore);
    Caffe::set_mlu_device(FLAGS_mludevice);
    Caffe::set_mode(FLAGS_mmode);
    Caffe::setReshapeMode(Caffe::ReshapeMode::SETUPONLY);
#else
    LOG(FATAL) << "No other available modes, please recompile with USE_MLU!";
#endif
  }


  /* Create Detector class */
  Detector* detector =
      new Detector(FLAGS_model, FLAGS_weights, FLAGS_meanfile, FLAGS_meanvalue);
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
    vector<vector<vector<float>>> detections =
        detector->detect(images);
    if (FLAGS_dump) {
      WriteVisualizeBBox_online(images, detections,
                                labels, imageNames, detector->inputShape[2]);
    }
    for (int j =0; j< detector->getBatchSize(); j++)
      LOG(INFO) << "detection size: " << detections[j].size();

    gettimeofday(&tpEnd, NULL);
    timeUse = 1000000 * (tpEnd.tv_sec - tpStart.tv_sec) + tpEnd.tv_usec -
              tpStart.tv_usec;
    totalTime += timeUse;
    LOG(INFO) << "Detecting execution time: " << timeUse << " us ";
    LOG(INFO) << "\n";
    images.clear();
  }
  LOG(INFO) << "yolov3_detection() execution time: " << totalTime << " us";
  printPerf(figuresNumber, totalTime, detector->runTime(), 1, detector->getBatchSize());
  saveResult(figuresNumber, (-1), (-1), (-1), (-1), totalTime);
  delete detector;
#ifdef USE_MLU
  if (FLAGS_mmode != "CPU") {
    Caffe::freeQueue();
    cnmlExit();
  }
#endif
  return 0;
}
#else
#include "caffe/common.hpp"
int main(int argc, char* argv[]) {
  LOG(FATAL) << "This program should be compiled with USE_MLU!";
  return 0;
}
#endif  // USE_MLU  && USE OPENCV
