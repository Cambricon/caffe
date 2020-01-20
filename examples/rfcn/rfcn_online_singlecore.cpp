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

#ifdef USE_OPENCV
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sys/time.h>
#include <algorithm>
#include <fstream>
#include <iosfwd>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <queue>
#include "common_functions.hpp"

using std::string;
using std::vector;
using std::string;
using std::set;
using std::queue;
using boost::shared_ptr;
using std::min;
using std::max;
using caffe::Net;
using caffe::Blob;
using caffe::TEST;
using caffe::Caffe;
using cv::Mat;
using cv::Size;
using cv::Point;
using cv::Scalar;
using std::stringstream;

// macro define log
#define BACKEND(x, l)                             \
  string(x).substr(string(x).find_last_of(l) + 1, \
                   string(x).size() - string(x).find_last_of("/") - 1)
#define LOG_PRE \
  LOG(INFO) << "[" << BACKEND(__FILE__, "/") << ": " << __LINE__ << "] "

DEFINE_string(model, "",
    "The prototxt file used to find net configuration");
DEFINE_string(weights, "",
    "Optional; the pretrained weights to initialize finetuning, "
    "separated by ','. Cannot be set simultaneously with snapshot.");
DEFINE_string(images, "",
    "The input file list");
DEFINE_string(labels, "",
    "infomation about mapping from label to name");
DEFINE_int32(fix8, 0,
    "FP16 or FIX8, fix8 mode, default: 0");
DEFINE_int32(int8, -1,  "invalid(-1), fp16(0) or int8(1) mode. Default is invalid(-1)."
    "If specified, use int8 value, else, use fix8 value");
DEFINE_string(mmode, "MFUS",
    "CPU, MLU or MFUS, MFUS mode");
DEFINE_string(mcore, "MLU100",
    "1H8, 1H16, MLU100 for different Cambricon hardware pltform");
DEFINE_string(outputdir, "./",
    "The directoy used to save output images");
DEFINE_int32(Bangop, 0, "Use Bang Operator or not");
DEFINE_string(meanvalue, "",
    "If specified, can be one value or can be same as image channels"
    " - would subtract from the corresponding channel). Separated by ','."
    "Either mean_file or mean_value should be provided, not both.");



const std::string objects[21] = {"__background__",  // NOLINT
                             "aeroplane", "bicycle", "bird", "boat",
                             "bottle", "bus", "car", "cat", "chair",
                             "cow", "diningtable", "dog", "horse",
                             "motorbike", "person", "pottedplant",
                             "sheep", "sofa", "train", "tvmonitor"};

class Detector {
  public:
  Detector(const string& modelFile,
           const string& trainedFile);
  ~Detector();

  void detect(const cv::Mat& images);
  vector<vector<float> > getResults();
  int getBatchSize() {return batchSize;}
  void readImages(queue<string>* imagesQueue,
                  cv::Mat* images,
                  string* imageNames);
  void writeVisualizeBboxOnline(const vector<vector<float>>& detections,
                                const string& imageNames,
                                const cv::Mat& images);

  private:
  void preProcess(const cv::Mat& images);
  void scaleInputShape(const cv::Mat& images);

  private:
  Net<float>* network;
  cv::Size inputGeometry;
  int numberChannels;
  int batchSize;
  float scale;
};

Detector::Detector(const string& modelFile,
                   const string& trainedFile) {
  /* Load the network. */
  network = new Net<float>(modelFile, TEST);
  network->CopyTrainedLayersFrom(trainedFile);

  /* faster rcnn network should have exactly two input */
  Blob<float> *inputLayer = network->input_blobs()[0];
  batchSize = inputLayer->num();
  numberChannels = inputLayer->channels();
  CHECK(numberChannels == 3 || numberChannels == 1)
      << "Input layer should have 1 or 3 channels.";
  inputGeometry = cv::Size(inputLayer->width(), inputLayer->height());
}

Detector::~Detector() {
  delete network;
}

void Detector::detect(const cv::Mat& images) {
  scaleInputShape(images);
  preProcess(images);
  LOG(INFO) << "start forward .. ";

  float timeUse;
  struct timeval tpStart, tpEnd;
  gettimeofday(&tpStart, NULL);

  network->Forward();

  gettimeofday(&tpEnd, NULL);
  timeUse = 1000000 * (tpEnd.tv_sec - tpStart.tv_sec) + tpEnd.tv_usec -
             tpStart.tv_usec;
  LOG_PRE << "net_->ForwardPrefilled() execution time: " << timeUse << " us";
}

vector<vector<float> > Detector::getResults() {
  int inputHeight = inputGeometry.height;
  int inputWidth = inputGeometry.width;

  shared_ptr<Blob<float> > roi = network->blob_by_name("rois");
  shared_ptr<Blob<float> > score = network->blob_by_name("cls_prob");
  shared_ptr<Blob<float> > box = network->blob_by_name("bbox_pred");

  float* roiData = roi->mutable_cpu_data();
  float* scoreData = score->mutable_cpu_data();
  float* boxData = box->mutable_cpu_data();

  std::vector<float> rois(roi->count());
  if (FLAGS_mmode != "CPU") {
    for (int i = 0; i < roi->count() / 5; i++) {
      rois[i * 5] = 0;
      for (int j = 0; j < 4; j++) {
        rois[i * 5 + j + 1] = roiData[i * 5 + j];
      }
    }
  } else {
    for (int i = 0; i < roi->count(); i++) {
      rois[i] = roiData[i];
    }
  }

  for (int i = 0; i < roi->count(); i++)
    rois[i] /= scale;

  int batch = 304;
  int **boxes = reinterpret_cast<int **>(malloc(sizeof(int *) * batch));
  float **result = reinterpret_cast<float **>(malloc(sizeof(float *) * batch));
  int *size = reinterpret_cast<int *>(malloc(sizeof(int) * batch));
  int **use = reinterpret_cast<int **>(malloc(sizeof(int *) * batch));
  float **scoreCpu = reinterpret_cast<float **>(malloc(sizeof(float *) * batch));
  for (int i = 0; i < batch; i++) {
    boxes[i] = reinterpret_cast<int *>(malloc((sizeof(int)) * 8));
    scoreCpu[i] = reinterpret_cast<float *>(malloc((sizeof(float)) * 21));
    result[i] = reinterpret_cast<float *>(malloc((sizeof(float)) * 6));
    use[i] = reinterpret_cast<int *>(malloc(sizeof(int) * 21));
    for (int j = 0; j < 21; j++) {
      if ((FLAGS_mmode != "CPU") && !FLAGS_Bangop)
        scoreCpu[i][j] = scoreData[i * 32 + j];
      else
        scoreCpu[i][j] = scoreData[i * 21 + j];
      use[i][j] = 1;
    }
  }

  int mluOption;
  if ((FLAGS_mmode != "CPU") && !FLAGS_Bangop)
    mluOption = 1;
  else
    mluOption = 0;

  for (int i = 0; i < batch; i++) {
    int width = rois[i * 5 + 3] - rois[i * 5 + 1] + 1;
    int height = rois[i * 5 + 4] - rois[i * 5 + 2] + 1;
    float c_x = rois[i * 5 + 1] + static_cast<float>(width) / 2;
    float c_y = rois[i * 5 + 2] + static_cast<float>(height) / 2;
    float pc_x = c_x + width * boxData[i * (mluOption ? 16 : 8) + 4];
    float pc_y = c_y + height * boxData[i * (mluOption ? 16 : 8) + 5];
    float pc_w = exp(boxData[i * (mluOption ? 16 : 8) + 6]) * width;
    float pc_h = exp(boxData[i * (mluOption ? 16 : 8) + 7]) * height;
    boxes[i][4] = ((pc_x - 0.5 * pc_w) > 0 ? (pc_x - 0.5 * pc_w) : 0);
    boxes[i][5] = ((pc_y - 0.5 * pc_h) > 0 ? (pc_y - 0.5 * pc_h) : 0);
    boxes[i][6] =
        ((pc_x + 0.5 * pc_w) < inputWidth ? (pc_x + 0.5 * pc_w) : inputWidth - 1);
    boxes[i][7] =
        ((pc_y + 0.5 * pc_h) < inputHeight ? (pc_y + 0.5 * pc_h) : inputHeight - 1);
    size[i] = (boxes[i][7] - boxes[i][5]) * (boxes[i][6] - boxes[i][4]);
  }

  vector<vector<float> > resultInfo;
  for (int t = 1; t < 21; t++) {
    for (int i = 0; i < batch; i++) {
      if (use[i][t]) {
        for (int j = i + 1; j < batch; j++) {
          int overlap_x = min(boxes[i][6], boxes[j][6]) - max(boxes[i][4], boxes[j][4]);
          int overlap_y = min(boxes[i][7], boxes[j][7]) - max(boxes[i][5], boxes[j][5]);
          int overlap =
              (overlap_x > 0 ? overlap_x : 0) * (overlap_y > 0 ? overlap_y : 0);
          float nms = static_cast<float>(overlap) /
                      static_cast<float>((size[i] + size[j] - overlap));
          if (nms < 0.3) continue;
          if (scoreCpu[j][t] > scoreCpu[i][t]) {
            use[i][t] = 0;
            break;
          } else if (scoreCpu[j][t] <= scoreCpu[i][t]) {
            use[j][t] = 0;
          }
        }
        if (use[i][t] && scoreCpu[i][t] > 0.1) {
          vector<float> tmp(6);
          tmp[0] = boxes[i][4];
          tmp[1] = boxes[i][5];
          tmp[2] = boxes[i][6];
          tmp[3] = boxes[i][7];
          tmp[4] = t;
          tmp[5] = scoreCpu[i][t];
          resultInfo.push_back(tmp);
        }
      }
    }
  }

  for (int i = 0; i < batch; i++) {
    free(boxes[i]);
    free(scoreCpu[i]);
    free(result[i]);
    free(use[i]);
  }
  free(boxes);
  free(result);
  free(size);
  free(use);
  free(scoreCpu);

  return resultInfo;
}

void Detector::writeVisualizeBboxOnline(const vector<vector<float>>& detections,
                                        const string& imageNames,
                                        const cv::Mat& images) {
  cv::Mat image = images;
  vector<vector<float> > result = detections;
  std::string name = imageNames;
  int pos_image = imageNames.rfind("/");
  if (pos_image > 0 && pos_image < imageNames.size()) {
    name = name.substr(pos_image);
  }
  pos_image = name.rfind(".");
  if (pos_image > 0 && pos_image < name.size()) {
    name = name.substr(0, pos_image);
  }
  name = FLAGS_outputdir + name + ".txt";
  std::ofstream file_map(name);
  for (int j = 0; j < result.size(); j++) {
    Point p1, p2;
    p1.x = result[j][0];
    p1.y = result[j][1];
    p2.x = result[j][2];
    p2.y = result[j][3];
    rectangle(image, p1, p2, Scalar(0, 255, 0), 8, 8, 0);
    stringstream s0;
    LOG(INFO) << "obj is " << objects[static_cast<int>(result[j][4])];
    s0 << result[j][5];
    string s00 = s0.str();
    putText(image, objects[static_cast<int>(result[j][4])],
            Point(p1.x, (p1.y + p2.y) / 2 - 10), 2, 0.5, Scalar(255, 0, 0), 0,
            8, 0);
    putText(image, s00.c_str(), Point(p1.x, (p1.y + p2.y) / 2 + 10), 2, 0.5,
            Scalar(255, 0, 0), 0, 8, 0);
    LOG(INFO) << "detection result: " << result[j][4] << " " << result[j][5]
              << " " << result[j][0] << " " << result[j][1] << " "
              << result[j][2] << " " << result[j][3];
    // the order for mAP is: label score x_min y_min x_max y_max
    file_map << objects[static_cast<int>(result[j][4])] << " "
             << static_cast<float>(result[j][5]) << " "
             << static_cast<float>(p1.x) / image.cols << " "
             << static_cast<float>(p1.y) / image.rows << " "
             << static_cast<float>(p2.x) / image.cols << " "
             << static_cast<float>(p2.y) / image.rows << std::endl;
  }
  file_map.close();
  stringstream s;
  size_t position = imageNames.find_last_of("/");
  s << FLAGS_outputdir;
  s << '/';
  s << "Detect_";
  s << imageNames.substr(position + 1, imageNames.length() - position - 1);
  string s1;
  s >> s1;
  cv::imwrite(s1, image);
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Detector::preProcess(const cv::Mat& images) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (images.channels() == 3 && numberChannels == 1)
    cv::cvtColor(images, sample, cv::COLOR_BGR2GRAY);
  else if (images.channels() == 4 && numberChannels == 1)
    cv::cvtColor(images, sample, cv::COLOR_BGRA2GRAY);
  else if (images.channels() == 4 && numberChannels == 3)
    cv::cvtColor(images, sample, cv::COLOR_BGRA2BGR);
  else if (images.channels() == 1 && numberChannels == 3)
    cv::cvtColor(images, sample, cv::COLOR_GRAY2BGR);
  else
    sample = images;
  Blob<float> *inputLayer = network->input_blobs()[0];

  float means[3] = {0, 0, 0};
  bool int8 = (FLAGS_int8 != -1) ? FLAGS_int8 : FLAGS_fix8;
  if (!(int8 || FLAGS_meanvalue.empty())) {
    stringstream ss(FLAGS_meanvalue);
    vector<float> values;
    string item;
    while (getline(ss, item, ',')) {
      float value = std::atof(item.c_str());
      values.push_back(value);
    }
    CHECK(values.size() == 1 || values.size() == numberChannels) <<
      "Specify either 1 mean_value or as many as channels: " << numberChannels;
    std::vector<cv::Mat> channels;
    for (int i = 0; i < numberChannels; ++i) {
      means[i] = values[i];
    }
  }

  cv::Mat sampleResized(inputGeometry.height, inputGeometry.width, CV_8UC3);
  if (sample.size() != inputGeometry)
    cv::resize(sample, sampleResized, Size(), scale, scale, CV_INTER_LINEAR);
  else
    sampleResized = sample;

  // put data in
  int width = inputLayer->shape()[3];
  int height = inputLayer->shape()[2];
  int channels = inputLayer->shape()[1];
  for (int j = 0; j < height; j++) {
    for (int k = 0; k < width; k++) {
      for (int c = 0; c < channels; c++) {
        inputLayer->mutable_cpu_data()[inputLayer->offset(0, c, j, k)] =
            static_cast<float>(sampleResized.ptr(j)[k * 3 + c]) - means[c];
      }
    }
  }
}

void Detector::scaleInputShape(const cv::Mat& images) {
  int lowBound, upBound;
  lowBound = inputGeometry.height;
  upBound = inputGeometry.width;

  if (images.cols > images.rows) {
    if (images.cols * lowBound / images.rows < upBound) {
      inputGeometry = cv::Size(
        static_cast<int>(images.cols * lowBound / images.rows), lowBound);
      scale = static_cast<float>(lowBound) / static_cast<float>(images.rows);
    } else {
      inputGeometry = cv::Size(
          upBound, static_cast<int>(images.rows * upBound / images.cols));
      scale = static_cast<float>(upBound) / static_cast<float>(images.cols);
    }
  } else {
    if (images.rows * lowBound / images.cols < upBound) {
      inputGeometry = cv::Size(
        lowBound, static_cast<int>(images.rows * lowBound / images.cols));
      scale = static_cast<float>(lowBound) / static_cast<float>(images.cols);
    } else {
      inputGeometry = cv::Size(
          static_cast<int>(images.cols * upBound / images.rows), upBound);
      scale = static_cast<float>(upBound) / static_cast<float>(images.rows);
    }
  }

  LOG(INFO) << "scale: " << scale;

  if (network->input_blobs().size() == 2) {
    /* set image info */
    Blob<float>* inputLayer1 = network->input_blobs()[1];
    float* shapeData = inputLayer1->mutable_cpu_data();

    shapeData[0] = inputGeometry.height;
    shapeData[1] = inputGeometry.width;
    shapeData[2] = static_cast<float>(inputGeometry.height) /
                   static_cast<float>(images.rows);
  }
}

void Detector::readImages(queue<string>* imagesQueue,
                          cv::Mat* images, string* imageNames) {
  int leftNumber = imagesQueue->size();
  string file;

  if (leftNumber > 0) {
    file = imagesQueue->front();
    *imageNames = file;
    imagesQueue->pop();
    if (file.find(" ") != string::npos)
      file = file.substr(0, file.find(" "));
    cv::Mat image = cv::imread(file, -1);
    Size tempSize = Size(inputGeometry.width, inputGeometry.height);
    Mat img = Mat(tempSize, CV_8UC3);
    cv::resize(image, img, tempSize);
    *images = img;
  }
}

int main(int argc, char **argv) {
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;
#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif
  gflags::SetUsageMessage("Do detection using rfcn-demo_mlu.\n"
        "Usage:\n"
        "    rfcn-demo_mlu [FLAGS] model_file list_file\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (argc == 0) {
    gflags::ShowUsageWithFlagsRestrict(argv[0],
        "examples/rfcn/rfcn-demo_mlu");
    return 1;
  }

  /* Set CPU or MLU running mode */
  if (FLAGS_mmode == "CPU") {
    Caffe::set_mode(Caffe::CPU);
  } else {
#ifdef USE_MLU
    cnmlInit(0);
    Caffe::set_rt_core(FLAGS_mcore);
    Caffe::set_mlu_device(0);
    Caffe::setReshapeMode(Caffe::ReshapeMode::SETUPONLY);
    Caffe::set_mode(FLAGS_mmode);
    Caffe::setDetectOpMode(FLAGS_Bangop);
#else
    LOG(FATAL) << "No other available modes, please recompile with USE_MLU!";
#endif
  }

  /* Create detector class */
  Detector *detector = new Detector(FLAGS_model,
                                    FLAGS_weights);

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

  /* Detecting images */
  float timeUse;
  struct timeval tpStart, tpEnd;
  LOG_PRE << "Loop classifying images start" << std::endl;
  gettimeofday(&tpStart, NULL);
  for (int i = 0; i < figuresNumber; i++) {
    cv::Mat images;
    string imageNames;

    /* Firstly read one image from file list */
    detector->readImages(&imageListQueue, &images, &imageNames);

    /* Secondly do detecting process */
    detector->detect(images);

    /* Thirdly do post-process */
    vector<vector<float> > detections = detector->getResults();

    /* Lastly write virsual box into image */
    detector->writeVisualizeBboxOnline(detections, imageNames, images);
  }
  gettimeofday(&tpEnd, NULL);
  timeUse = 1000000 * (tpEnd.tv_sec - tpStart.tv_sec) + tpEnd.tv_usec -
             tpStart.tv_usec;
  LOG(INFO) << "Detecting execution time: " << timeUse << " us";
  LOG(INFO) << "End2end throughput fps: " << figuresNumber / timeUse * 1e6;
  saveResult(figuresNumber, (-1), (-1), (-1), (-1), timeUse);

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
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
