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
DEFINE_int32(mludevice, 0,
    "set using mlu device number, default: 0");
DEFINE_string(datastrategy, "-1,-1",
              "Use it to control input and output data_strategy"
              " 0: cpu_mlu_balance; 1: cpu_priority; "
              " 2: mlu_priority;    3: no_preprocess ");
DEFINE_string(images, "", "The input file list");
DEFINE_string(outputdir, ".", "The directoy used to save output images");
DEFINE_string(bboxanchordir, "./bbox_anchor/", "The directoy used to read"
                             " anchor_tensors and x_y_offset");
DEFINE_string(labels, "", "infomation about mapping from label to name");
DEFINE_string(logdir, "",
              "path to dump log file, to terminal "
              "stderr by default");
DEFINE_int32(dump, 1, "0 or 1, dump output images or not.");
DEFINE_double(confidence, 0.25, "Only keep detections with scores  equal "
                                         "to or higher than the confidence.");
DEFINE_double(nmsthresh, 0.45, "Identify the optimal cell among all candidates "
                               " when the object lies in multiple cells of a grid");


class Detector {
  public:
  Detector(const string& modelFile,
           const string& weightsFile,
           const string& meanFile,
           const string& meanValues);
  ~Detector();

  void restore_Boxes(vector<vector<float>> tmp_boxes,
                     vector<vector<vector<float>>>* final_boxes,
                     const vector<cv::Mat>& imgs);
  vector<vector<vector<float>>> detect(
    const vector<cv::Mat>& images,
    const vector<vector<vector<float>>>& x_y_offsets,
    const vector<vector<vector<float>>>& anchor_tensors);
  int inputDim() { return inputShape[2]; }
  int getBatchSize() { return batchSize; }
  void readImages(queue<string>* imagesQueue, int inputNumber,
                  vector<cv::Mat>* images, vector<string>* imageNames);

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

  vector<int> inputShape;
  vector<vector<vector<float>>> tensors;
};

Detector::Detector(const string& modelFile,
                   const string& weightsFile,
                   const string& meanFile,
                   const string& meanValues) {
  /* Load the network. */
  network = new Net<float>(modelFile, TEST);
  network->CopyTrainedLayersFrom(weightsFile);
  CHECK_EQ(network->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(network->num_outputs(), 3) << "Network should have exactly one output.";

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

// obtain 1x255x13x13 blob data
// return 155 * 169
vector<vector<float>> get_blob_data(
    const vector<int>& yolov3_shape, const float* result_buffer,
    int batch) {
  int batchs = yolov3_shape[0] / batch;
  int channels = yolov3_shape[1];
  int width = yolov3_shape[2];
  int height = yolov3_shape[3];
  vector<vector<float>> output_data(channels, vector<float>(width * height));
  for (int n = 0; n < batchs; ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          output_data[c][h * width + w] =
              result_buffer[c * height * width + h * width + w];
        }
      }
    }
  }
  return output_data;
}

// transpose two vector  255*169->169*255
void transpose(const vector<vector<float>>& A, vector<vector<float>>* B) {
  if (A.size() == 0) {
    LOG(FATAL) << "input vector is equal to 0" << std::endl;
  }
  vector<vector<float>> C(A[0].size());
  for (int i = 0; i < C.size(); i++) C[i].resize(A.size());
  for (int i = 0; i < A[0].size(); i++) {
    for (int j = 0; j < A.size(); j++) {
      C[i][j] = A[j][i];
    }
  }
  *B = std::move(C);
}

// reshape two vector
void matrixReshape(vector<vector<float>> A, vector<vector<float>>* B,
                   int r, int c) {
  int m = A.size(), n = A[0].size();
  if (m * n != r * c) {
    *B = std::move(A);
  } else {
    vector<vector<float>> C(r);
    for (int x = 0; x < C.size(); x++) C[x].resize(c);
    for (int i = 0; i < r; ++i) {
      for (int j = 0; j < c; ++j) {
        int k = i * c + j;
        C[i][j] = static_cast<float>(A[k / n][k % n]);
      }
    }
    *B = std::move(C);
  }
}

// sigmoid two vector
void sigmoid(vector<vector<float>>* B, int col) {
  vector<vector<float>>::iterator it_begin = B->begin();
  for (; it_begin != B->end(); ++it_begin) {
    for (int i = col; i < col + 1; i++) {
      (*it_begin)[i] = 1 / (1 + std::exp(-(*it_begin)[i]));
    }
  }
}

void matrixAdd(vector<vector<float>>* a, const vector<vector<float>>& b,
               int numberOfRows, int selectColsLeft, int selectColsRight) {
  for (int i = 0; i < numberOfRows; i++) {
    for (int j = selectColsLeft; j < selectColsRight; j++) {
      (*a)[i][j] = (*a)[i][j] + b[i][j];
    }
  }
}

// sigmoid two vector
void sigmoid(vector<vector<float>>* B, int colsLeft, int colsRight) {
  vector<vector<float>>::iterator it_begin = B->begin();
  for (; it_begin != B->end(); ++it_begin) {
    for (int i = colsLeft; i < colsRight; i++) {
      (*it_begin)[i] = 1 / (1 + std::exp(-(*it_begin)[i]));
    }
  }
}

void matrixMulti(vector<vector<float>>* a, const vector<vector<float>>& b,
                 int numberOfRows, int selectColsLeft, int selectColsRight) {
  for (int i = 0; i < numberOfRows; i++)
    for (int j = selectColsLeft; j < selectColsRight; j++)
      (*a)[i][j] = std::exp((*a)[i][j]) * b[i][j - selectColsLeft];
}

void matrixMulti(vector<vector<float>>* a, int b, int numberOfRows,
                 int selectColsLeft, int selectColsRight) {
  for (int i = 0; i < numberOfRows; i++)
    for (int j = selectColsLeft; j < selectColsRight; j++)
      (*a)[i][j] = (*a)[i][j] * b;
}

void transform_tensor(vector<vector<float>> tensor_output,
                      vector<vector<float>>* tensor_data,
                      int num_classes,
                      vector<vector<float>> x_y_offset,
                      vector<vector<float>> anchor_tensor) {
  int input_dim = 416;
  int stride = input_dim / std::sqrt(tensor_output[0].size());  // 32
  int gride_size = input_dim / stride;  // 13
  int bbox_attrs = 5 + num_classes;  // 85
  int anchor_num = 3;

  vector<vector<float>> tensor_trans;
  transpose(tensor_output, &tensor_trans);  // 255*169->169*255

  matrixReshape(tensor_trans, tensor_data, gride_size * gride_size * anchor_num,
                bbox_attrs);  // 169*255->507*85

  sigmoid(tensor_data, 0);
  sigmoid(tensor_data, 1);
  sigmoid(tensor_data, 4);

  matrixAdd(tensor_data, x_y_offset, tensor_data->size(), 0, 2);
  matrixMulti(tensor_data, anchor_tensor, tensor_data->size(), 2, 4);
  sigmoid(tensor_data, 5, 85);
  matrixMulti(tensor_data, stride, tensor_data->size(), 0, 4);
}

void concatenate(vector<vector<float>>* all_boxes,
                 const vector<vector<float>>& boxes_13,
                 const vector<vector<float>>& boxes_26,
                 const vector<vector<float>>& boxes_52) {
  vector<vector<float>> temp(
      (boxes_13.size() + boxes_26.size() + boxes_52.size()),
      vector<float>(boxes_13[0].size(), 0));
  for (int j = 0; j < temp.size(); j++) {
    for (int k = 0; k < temp[0].size(); k++) {
      if (j < boxes_13.size()) {
        temp[j][k] = boxes_13[j][k];
      } else {
        if (j >= boxes_13.size() && j < (boxes_13.size() + boxes_26.size())) {
          temp[j][k] = boxes_26[j - boxes_13.size()][k];
        } else {
          temp[j][k] = boxes_52[j - (boxes_13.size() + boxes_26.size())][k];
        }
      }
    }
  }
  *all_boxes = std::move(temp);
}

void fill_zeros(vector<vector<float>>* all_boxes, int cols, float confidence) {
  for (int i = 0; i < all_boxes->size(); i++) {
    if ((*all_boxes)[i][cols] > confidence)
      continue;
    else
      (*all_boxes)[i][cols] = 0;
  }
}

vector<vector<float>> filter_boxes(vector<vector<float>>* all_boxes,
                                   vector<float>* max_class_score,
                                   vector<float>* max_class_idx) {
  vector<vector<float>> temp(all_boxes->size(), vector<float>(5 + 2, 0));
  for (int i = 0; i < all_boxes->size(); i++) {
    for (int j = 0; j < 7; j++) {
      if (j < 5)
        temp[i][j] = (*all_boxes)[i][j];
      else if (j == 5)
        temp[i][j] = (*max_class_score)[i];
      else if (j == 6)
        temp[i][j] = (*max_class_idx)[i];
      else
        LOG(FATAL) << " filter_boxes index error ";
    }
  }
  vector<vector<float>> vec;
  for (int m = 0; m < temp.size(); m++) {
    if (temp[m][4] == 0)
      continue;
    else
      vec.push_back(temp[m]);
  }
  return vec;
}

void unique_vector(vector<vector<float>>* input_vector,
                   vector<float>* output_vector) {
  for (int i = 0; i < input_vector->size(); i++) {
    (*output_vector).push_back((*input_vector)[i][6]);
  }
  sort((*output_vector).begin(), (*output_vector).end());
  auto new_end = unique((*output_vector).begin(), (*output_vector).end());
  (*output_vector).erase(new_end, (*output_vector).end());
}

float findMax(vector<float> vec) {
  float max = -999;
  for (auto v : vec) {
    if (max < v) max = v;
  }
  return max;
}

int getPositionOfMax(vector<float> vec, float max) {
  auto distance = find(vec.begin(), vec.end(), max);
  return distance - vec.begin();
}

void nms_by_classes(vector<vector<float>> sort_boxes, vector<float>* ious,
                    int start) {
  for (int i = start + 1; i < sort_boxes.size(); i++) {
    float first_x1 = sort_boxes[start][0];
    float first_y1 = sort_boxes[start][1];
    float first_x2 = sort_boxes[start][2];
    float first_y2 = sort_boxes[start][3];

    float next_x1 = sort_boxes[i][0];
    float next_y1 = sort_boxes[i][1];
    float next_x2 = sort_boxes[i][2];
    float next_y2 = sort_boxes[i][3];

    float inter_x1 = std::max(first_x1, next_x1);
    float inter_y1 = std::max(first_y1, next_y1);
    float inter_x2 = std::min(first_x2, next_x2);
    float inter_y2 = std::min(first_y2, next_y2);
    float inter_area, first_area, next_area, union_area, iou;
    if ((inter_x2 - inter_x1 + 1 > 0) && (inter_y2 - inter_y1 + 1 > 0))
      inter_area = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1);
    else
      inter_area = 0;
    first_area = (first_x2 - first_x1 + 1) * (first_y2 - first_y1 + 1);
    next_area = (next_x2 - next_x1 + 1) * (next_y2 - next_y1 + 1);
    union_area = first_area + next_area - inter_area;
    iou = inter_area / union_area;
    (*ious).push_back(iou);
  }
}

void get_detection(vector<vector<float>> all_boxes,
                   vector<vector<float>>* final_boxes, int num_classes,
                   float confidence, float nms_thresh) {
  fill_zeros(&all_boxes, 4, confidence);
  vector<vector<float>> boxes_copy;
  boxes_copy = all_boxes;
  for (int i = 0; i < all_boxes.size(); i++) {
    all_boxes[i][0] = boxes_copy[i][0] - boxes_copy[i][2] / 2;
    all_boxes[i][1] = boxes_copy[i][1] - boxes_copy[i][3] / 2;
    all_boxes[i][2] = boxes_copy[i][0] + boxes_copy[i][2] / 2;
    all_boxes[i][3] = boxes_copy[i][1] + boxes_copy[i][3] / 2;
  }

  vector<float> max_class_idx;
  vector<float> max_class_score;
  for (int j = 0; j < all_boxes.size(); j++) {
    vector<float>::iterator biggest =
        std::max_element(std::begin(all_boxes[j]) + 5, std::end(all_boxes[j]));
    max_class_score.push_back(*biggest);
    max_class_idx.push_back(
        std::distance(std::begin(all_boxes[j]) + 5, biggest));
  }

  vector<vector<float>> boxes_cleaned;
  boxes_cleaned = filter_boxes(&all_boxes, &max_class_score, &max_class_idx);
  vector<float> unique_classes;
  unique_vector(&boxes_cleaned, &unique_classes);
  vector<vector<float>> curr_classes;
  for (auto v : unique_classes) {
    for (int m = 0; m < boxes_cleaned.size(); m++) {
      if (boxes_cleaned[m][6] == v)
        curr_classes.push_back(boxes_cleaned[m]);
      else
        continue;
    }
    vector<float> object_score;
    for (int n = 0; n < curr_classes.size(); n++) {
      object_score.push_back(curr_classes[n][4]);
    }
    vector<float> sort_score, sort_idx;
    for (int i = 0; i < object_score.size(); i++) {
      float maxNumber = findMax(object_score);
      sort_score.push_back(maxNumber);
      int maxIndex = getPositionOfMax(object_score, maxNumber);
      sort_idx.push_back(maxIndex);
      object_score[maxIndex] = -999;
    }
    vector<vector<float>> sort_boxes;
    for (int j = 0; j < sort_idx.size(); j++) {
      sort_boxes.push_back(curr_classes[sort_idx[j]]);
    }
    vector<float> ious;
    for (int k = 0; k < sort_boxes.size(); k++) {
      ious.clear();
      nms_by_classes(sort_boxes, &ious, k);
      int dele_number = 0;

      for (int s = 0; s < ious.size(); s++) {
        if (ious[s] > nms_thresh) {
          sort_boxes.erase(sort_boxes.begin() + k - dele_number + 1 + s);
          dele_number = dele_number + 1;
        } else {
          continue;
        }
      }
    }
    for (int t = 0; t < sort_boxes.size(); t++) {
      (*final_boxes).push_back(sort_boxes[t]);
    }
    curr_classes.clear();
    object_score.clear();
    sort_score.clear();
    sort_idx.clear();
    sort_boxes.clear();
  }
}

void Detector::restore_Boxes(vector<vector<float>> tmp_boxes,
                             vector<vector<vector<float>>>* final_boxes,
                             const vector<cv::Mat>& imgs) {
  for (int j = 0; j < imgs.size(); j++) {
    int input_dim = inputShape[2];
    float scaling_factors = std::min(
        static_cast<float>(input_dim) / static_cast<float>(imgs[j].cols),
        static_cast<float>(input_dim) / static_cast<float>(imgs[j].rows));
    for (int i = 0; i < tmp_boxes.size(); i++) {
      tmp_boxes[i][0] =
          tmp_boxes[i][0] -
          static_cast<float>(input_dim - scaling_factors * imgs[j].cols) / 2.0;
      tmp_boxes[i][2] =
          tmp_boxes[i][2] -
          static_cast<float>(input_dim - scaling_factors * imgs[j].cols) / 2.0;
      tmp_boxes[i][1] =
          tmp_boxes[i][1] -
          static_cast<float>(input_dim - scaling_factors * imgs[j].rows) / 2.0;
      tmp_boxes[i][3] =
          tmp_boxes[i][3] -
          static_cast<float>(input_dim - scaling_factors * imgs[j].rows) / 2.0;
      for (int k = 0; k < 4; k++) {
        tmp_boxes[i][k] = tmp_boxes[i][k] / scaling_factors;
      }
    }
    for (int i = 0; i < tmp_boxes.size(); i++) {
      tmp_boxes[i][0] = tmp_boxes[i][0] < 0 ? 0 : tmp_boxes[i][0];
      tmp_boxes[i][2] = tmp_boxes[i][2] < 0 ? 0 : tmp_boxes[i][2];
      tmp_boxes[i][1] = tmp_boxes[i][1] < 0 ? 0 : tmp_boxes[i][1];
      tmp_boxes[i][3] = tmp_boxes[i][3] < 0 ? 0 : tmp_boxes[i][3];
      tmp_boxes[i][0] =
          tmp_boxes[i][0] > imgs[j].cols ? imgs[j].cols : tmp_boxes[i][0];
      tmp_boxes[i][2] =
          tmp_boxes[i][2] > imgs[j].cols ? imgs[j].cols : tmp_boxes[i][2];
      tmp_boxes[i][1] =
          tmp_boxes[i][1] > imgs[j].rows ? imgs[j].rows : tmp_boxes[i][1];
      tmp_boxes[i][3] =
          tmp_boxes[i][3] > imgs[j].rows ? imgs[j].rows : tmp_boxes[i][3];
    }
    (*final_boxes).push_back(tmp_boxes);
  }
}

vector<vector<vector<float>>> Detector::detect(
    const vector<cv::Mat>& images,
    const vector<vector<vector<float>>>& x_y_offsets,
    const vector<vector<vector<float>>>& anchor_tensors) {
  vector<vector<cv::Mat>> inputImages;
  wrapInputLayer(&inputImages);
  preProcess(images, &inputImages);

  float timeUse;
  struct timeval tpEnd, tpStart;
  gettimeofday(&tpStart, NULL);

  network->Forward();

  gettimeofday(&tpEnd, NULL);
  timeUse = 1000000 * (tpEnd.tv_sec - tpStart.tv_sec)
             + tpEnd.tv_usec - tpStart.tv_usec;
  LOG(INFO) << "Forward execution time: " << timeUse << " us";

  /* copy the output layer to a vector*/
  /* 255 * 169  */
  /* 255 * 676  */
  /* 255 * 2704 */
  vector<vector<vector<float>>> final_boxes;
  for (int m = 0; m < inputShape[0]; m++) {
    tensors.clear();
    vector<vector<float>> tensor;
    for (int i = 0; i < outputNum; i++) {
      Blob<float>* outputLayer = network->output_blobs()[i];
      const float* outputData = outputLayer->cpu_data();
      const vector<int> shape = outputLayer->shape();
      int singleCount = shape[1] * shape[2] * shape[3];

      tensor = get_blob_data(shape, outputData + m * singleCount, inputShape[0]);
      tensors.push_back(tensor);
    }

    vector<vector<vector<float>>> three_boxes;
    three_boxes.resize(3);
    for (int i = 0; i < 3; i++) {
      transform_tensor(tensors[i], &three_boxes[i], 80, x_y_offsets[i],
                       anchor_tensors[i]);
    }
    vector<vector<float>> all_boxes, tmp_boxes;

    // 10647*85
    concatenate(&all_boxes, three_boxes[0], three_boxes[1], three_boxes[2]);
    get_detection(all_boxes, &tmp_boxes, 80, FLAGS_confidence, FLAGS_nmsthresh);
    final_boxes.push_back(tmp_boxes);
  }
  // restore_Boxes(tmp_boxes, &final_boxes, images);
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
    cv::Size input_geometry;
    input_geometry = cv::Size(inputShape[2], inputShape[3]);  // 416*416
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

void readtxt(vector<vector<vector<float>>>* x_y_offsets,
             vector<vector<vector<float>>>* anchor_tensors,
             vector<int> size_rows, vector<string> anchor_strs) {
  int size = x_y_offsets->size();
  for (int i = 0; i < size; i++) {
    std::ifstream infile_1;
    string filename_1 = FLAGS_bboxanchordir + "/x_y_offset_"
        + anchor_strs[i] + ".txt";
    infile_1.open(filename_1);
    for (int m = 0; m < size_rows[i]; m++) {
      for (int n = 0; n < 2; n++) {
        infile_1 >> (*x_y_offsets)[i][m][n];
      }
    }
    infile_1.close();
    std::ifstream infile_2;
    string filename_2 = FLAGS_bboxanchordir +  "/anchors_tensor_"
        + anchor_strs[i] + ".txt";
    infile_2.open(filename_2);
    for (int m = 0; m < size_rows[i]; m++) {
      for (int n = 0; n < 2; n++) {
        infile_2 >> (*anchor_tensors)[i][m][n];
      }
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
    vector<vector<float>> result = detections[i];
    cv::Mat image = images[i];
    std::string name = imageNames[i];
    int positionMap = imageNames[i].rfind("/");
    if (positionMap > 0 && positionMap < imageNames[i].size()) {
      name = name.substr(positionMap + 1);
    }
    positionMap = name.rfind(".");
    if (positionMap > 0 && positionMap < name.size()) {
      name = name.substr(0, positionMap);
    }
    name = FLAGS_outputdir + "/" + name + ".txt";
    std::ofstream fileMap(name);

    float scaling_factors = std::min(
        static_cast<float>(input_dim) / static_cast<float>(images[i].cols),
        static_cast<float>(input_dim) / static_cast<float>(images[i].rows));

    for (int j = 0; j < result.size(); j++) {
      result[j][0] =
          result[j][0] -
          static_cast<float>(input_dim - scaling_factors * images[i].cols) /
              2.0;
      result[j][2] =
          result[j][2] -
          static_cast<float>(input_dim - scaling_factors * images[i].cols) /
              2.0;
      result[j][1] =
          result[j][1] -
          static_cast<float>(input_dim - scaling_factors * images[i].rows) /
              2.0;
      result[j][3] =
          result[j][3] -
          static_cast<float>(input_dim - scaling_factors * images[i].rows) /
              2.0;

      for (int k = 0; k < 4; k++) {
        result[j][k] = result[j][k] / scaling_factors;
      }
    }
    for (int j = 0; j < result.size(); j++) {
      // bounding boxes bundary check
      result[j][0] = result[j][0] < 0 ? 0 : result[j][0];
      result[j][2] = result[j][2] < 0 ? 0 : result[j][2];
      result[j][1] = result[j][1] < 0 ? 0 : result[j][1];
      result[j][3] = result[j][3] < 0 ? 0 : result[j][3];

      result[j][0] =
          result[j][0] > images[i].cols ? images[i].cols : result[j][0];
      result[j][2] =
          result[j][2] > images[i].cols ? images[i].cols : result[j][2];
      result[j][1] =
          result[j][1] > images[i].rows ? images[i].rows : result[j][1];
      result[j][3] =
          result[j][3] > images[i].rows ? images[i].rows : result[j][3];
    }
    for (int j = 0; j < result.size(); j++) {
      cv::Point p1(static_cast<int>(result[j][0]),
                   static_cast<int>(result[j][1]));
      cv::Point p2(static_cast<int>(result[j][2]),
                   static_cast<int>(result[j][3]));
      cv::rectangle(image, p1, p2, cv::Scalar(0, 0, 255), 1, 1, 0);
      cv::Point p3(static_cast<int>(result[j][0]),
                   static_cast<int>(result[j][1]) - 20);
      cv::Point p4(static_cast<int>(result[j][0]) + 100,
                   static_cast<int>(result[j][1]));
      cv::rectangle(image, p3, p4, cv::Scalar(255, 0, 0), -1, 4);
      stringstream ss;
      ss << round(result[j][4] * 1000) / 1000.0;
      std::string str =
          labelToDisplayName[static_cast<int>(result[j][6])] + ":" + ss.str();
      cv::Point p5(static_cast<int>(result[j][0]),
                   static_cast<int>(result[j][1]) - 1);
      cv::putText(image, str, p5, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                  cv::Scalar(255, 215, 0), 1);

      fileMap << labelToDisplayName[static_cast<int>(result[j][6])] << " "
              << ss.str() << " " << static_cast<float>(p1.x) / image.cols << " "
              << static_cast<float>(p1.y) / image.rows << " "
              << static_cast<float>(p2.x) / image.cols << " "
              << static_cast<float>(p2.y) / image.rows << " " << image.cols
              << " " << image.rows << std::endl;
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
    cv::imwrite((FLAGS_outputdir + "/" + imageNames[i].c_str()), image);
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
    std::stringstream ss(FLAGS_datastrategy);
    vector<int> strategy;
    string value;
    while (getline(ss, value, ',')) {
      strategy.push_back(std::atoi(value.c_str()));
    }
    CHECK(strategy.size() == 2) <<
        "only support two values: input and output strategy";
    if (strategy[0] != -1 || strategy[1] != -1) {
      Caffe::setDataStrategy(strategy);
    }
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

  vector<int> size_row = {507, 2028, 8112};
  vector<string> anchor_str = {"13", "26", "52"};
  vector<vector<vector<float>>> x_y_offsets;
  vector<vector<vector<float>>> anchor_tensors;
  x_y_offsets.push_back(vector<vector<float>>(507, vector<float>(2)));
  x_y_offsets.push_back(vector<vector<float>>(2028, vector<float>(2)));
  x_y_offsets.push_back(vector<vector<float>>(8112, vector<float>(2)));
  anchor_tensors.push_back(vector<vector<float>>(507, vector<float>(2)));
  anchor_tensors.push_back(vector<vector<float>>(2028, vector<float>(2)));
  anchor_tensors.push_back(vector<vector<float>>(8112, vector<float>(2)));

  readtxt(&x_y_offsets, &anchor_tensors, size_row, anchor_str);

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
        detector->detect(images, x_y_offsets, anchor_tensors);
    if (FLAGS_dump) {
      if (!FLAGS_outputdir.empty()) {
        WriteVisualizeBBox_online(images, detections, labels, imageNames,
                detector->inputDim());
      }
    }
    gettimeofday(&tpEnd, NULL);
    timeUse = 1000000 * (tpEnd.tv_sec - tpStart.tv_sec) + tpEnd.tv_usec -
              tpStart.tv_usec;
    totalTime += timeUse;
    LOG(INFO) << "Detecting execution time: " << timeUse << " us";
    for (int num = 0; num < detector->getBatchSize(); num++) {
      LOG(INFO) << "objs is : " << detections[num].size();
    }
    images.clear();
    imageNames.clear();
  }
  LOG(INFO) << "yolov3_detection() execution time: " << totalTime << " us";
  LOG(INFO) << "End2end throughput fps: " << figuresNumber / totalTime * 1e6;
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
