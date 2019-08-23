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
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <condition_variable>  // NOLINT
#include <iomanip>
#include <iosfwd>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "blocking_queue.hpp"
#include "cnrt.h"  // NOLINT
#include "common_functions.hpp"

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

DEFINE_string(offlinemodel, "", "prototxt file used to find net configuration");
DEFINE_string(meanfile, "", "file provides mean value(s) of image input.");
DEFINE_string(meanvalue, "",
              "mean value of input image. "
              "One value or image channel number of values shoudl be provided "
              "Either mean file or mean value should be provided");
DEFINE_string(mludevice, "0",
              "set using mlu device id, set multidevice seperated by ','"
              "eg 0,1 when you use device id is 0 and 1, default: 0");
DEFINE_int32(dataparallel, 1,
             "dataparallel, data * model parallel should "
             "be lower than or equal to 32 ");
DEFINE_int32(threads, 1, "thread number");
DEFINE_string(images, "", "image list file");
DEFINE_int32(fix8, 0, "fp16 or fix8 mode. Default is fp16(0)");
DEFINE_int32(int8, -1, "invalid(-1), fp16(0) or int8(1) mode. Default is invalid(-1)."
    "If specified, use int8 value, else, use fix8 value");
DEFINE_double(scale, 1, "scale for input data, mobilenet...");
DEFINE_int32(dump, 0, "0 or 1, dump output images or not.");
DEFINE_string(labels, "", "infomation about mapping from label to name");

DEFINE_double(confidence, 0.25,
              "Only keep detections with scores  equal "
              "to or higher than the confidence.");
DEFINE_double(nmsthresh, 0.45,
              "Identify the optimal cell among all candidates "
              " when the object lies in multiple cells of a grid");

DEFINE_string(logdir, "",
              "path to dump log file, to terminal "
              "stderr by default");
DEFINE_int32(fifosize, 2,
             "set FIFO size of mlu input and output buffer, default is 2");
DEFINE_string(outputdir, ".",
              "The directory used to save output images and txt.");
DEFINE_string(bboxanchordir, "./bbox_anchor/", "The directoy used to read"
                             " anchor_tensors and x_y_offset");

static void matrixMulti(vector<vector<float>>* a,
                        const vector<vector<float>>& b, int numberOfRows,
                        int selectColsLeft, int selectColsRight) {
  for (int i = 0; i < numberOfRows; i++)
    for (int j = selectColsLeft; j < selectColsRight; j++)
      (*a)[i][j] = std::exp((*a)[i][j]) * b[i][j - selectColsLeft];
}

static void matrixMulti(vector<vector<float>>* a, int b, int numberOfRows,
                        int selectColsLeft, int selectColsRight) {
  for (int i = 0; i < numberOfRows; i++)
    for (int j = selectColsLeft; j < selectColsRight; j++)
      (*a)[i][j] = (*a)[i][j] * b;
}

static void matrixAdd(vector<vector<float>>* a, const vector<vector<float>>& b,
                      int numberOfRows, int selectColsLeft,
                      int selectColsRight) {
  for (int i = 0; i < numberOfRows; i++) {
    for (int j = selectColsLeft; j < selectColsRight; j++) {
      (*a)[i][j] = (*a)[i][j] + b[i][j];
    }
  }
}

// sigmoid two vector
static void sigmoid(vector<vector<float>>* B, int col) {
  vector<vector<float>>::iterator it_begin = B->begin();
  for (; it_begin != B->end(); ++it_begin) {
    for (int i = col; i < col + 1; i++) {
      (*it_begin)[i] = 1 / (1 + std::exp(-(*it_begin)[i]));
    }
  }
}

// sigmoid two vector
static void sigmoid(vector<vector<float>>* B, int colsLeft, int colsRight) {
  vector<vector<float>>::iterator it_begin = B->begin();
  for (; it_begin != B->end(); ++it_begin) {
    for (int i = colsLeft; i < colsRight; i++) {
      (*it_begin)[i] = 1 / (1 + std::exp(-(*it_begin)[i]));
    }
  }
}

// reshape two vector
static void matrixReshape(vector<vector<float>> A, vector<vector<float>>* B,
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

static vector<vector<float>> get_blob_data(const vector<int>& yolov3_shape,
                                           const float* result_buffer) {
  int batchs = yolov3_shape[0];
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
static void transpose(const vector<vector<float>>& A,
                      vector<vector<float>>* B) {
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

static void transform_tensor(vector<vector<float>> tensor_output,
                             vector<vector<float>>* tensor_data,
                             int num_classes, vector<vector<float>> x_y_offset,
                             vector<vector<float>> anchor_tensor) {
  int input_dim = 416;
  int stride = input_dim / std::sqrt(tensor_output[0].size());  // 32
  int gride_size = input_dim / stride;                          // 13
  int bbox_attrs = 5 + num_classes;                             // 85
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

static void concatenate(vector<vector<float>>* all_boxes,
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

static void fill_zeros(vector<vector<float>>* all_boxes, int cols,
                       float confidence) {
  for (int i = 0; i < all_boxes->size(); i++) {
    if ((*all_boxes)[i][cols] > confidence)
      continue;
    else
      (*all_boxes)[i][cols] = 0;
  }
}

static vector<vector<float>> filter_boxes(vector<vector<float>>* all_boxes,
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

static void unique_vector(vector<vector<float>>* input_vector,
                          vector<float>* output_vector) {
  for (int i = 0; i < input_vector->size(); i++) {
    (*output_vector).push_back((*input_vector)[i][6]);
  }
  sort((*output_vector).begin(), (*output_vector).end());
  auto new_end = unique((*output_vector).begin(), (*output_vector).end());
  (*output_vector).erase(new_end, (*output_vector).end());
}

static float findMax(vector<float> vec) {
  float max = -999;
  for (auto v : vec) {
    if (max < v) max = v;
  }
  return max;
}

static int getPositionOfMax(vector<float> vec, float max) {
  auto distance = find(vec.begin(), vec.end(), max);
  return distance - vec.begin();
}

static void nms_by_classes(vector<vector<float>> sort_boxes,
                           vector<float>* ious, int start) {
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

static void get_detection(vector<vector<float>> all_boxes,
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

static void readtxt(vector<vector<vector<float>>>* x_y_offsets,
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
    positionMap = name.rfind(".");
    if (positionMap > 0 && positionMap < name.size()) {
      name = name.substr(0, positionMap);
    }
    // this is used to cancel "yolov3_offline_" in name
    std::string prefix = "yolov3_offline_";
    name = name.substr(prefix.size());
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
    cv::imwrite((FLAGS_outputdir + "/" + imageNames[i].c_str()), image);
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
  Inferencer(const string& offlinemodel, const int& dp,
             const int& deviceId, const int& devicesize);
  ~Inferencer();
  int n() { return inNum_; }
  int c() { return inChannel_; }
  int h() { return inHeight_; }
  int w() { return inWidth_; }
  vector<vector<int>> outputshape() { return outputShape_; }
  cnrtDataDescArray_t inDescs() { return inDescs_; }
  cnrtDataDescArray_t outDescs() { return outDescs_; }
  void pushValidInputData(void** data);
  void pushFreeInputData(void** data);
  void** popValidInputData();
  void** popFreeInputData();
  void pushValidOutputData(void** data);
  void pushFreeOutputData(void** data);
  void** popValidOutputData();
  void** popFreeOutputData();
  void pushValidInputNames(vector<string> rawImages);
  vector<string> popValidInputNames();
  void run();
  inline int modelParallel() { return modelParallel_; }
  inline int dataParallel() { return dataParallel_; }
  inline int inBlobNum() { return inBlobNum_; }
  inline int outBlobNum() { return outBlobNum_; }
  inline int threadId() { return threadId_; }
  inline int deviceId() { return deviceId_; }
  inline int deviceSize() { return deviceSize_; }
  inline float inferencingTime() { return inferencingTime_; }
  inline void setThreadId(int id) { threadId_ = id; }
  inline void setPostProcessor(PostProcessor* p) { postProcessor_ = p; }
  inline void pushInPtrVector(void** data) { inPtrVector_.push_back(data); }
  inline void pushOutPtrVector(void** data) { outPtrVector_.push_back(data); }

  inline vector<int> outCount() {
    vector<int> storage;
    for (int i = 0; i < 3; i++) storage.push_back(outCount_[i]);
    return storage;
  }

  private:
  BlockingQueue<void**> validInputFifo_;
  BlockingQueue<void**> freeInputFifo_;
  BlockingQueue<void**> validOutputFifo_;
  BlockingQueue<void**> freeOutputFifo_;
  BlockingQueue<vector<string>> imagesFifo_;

  cnrtModel_t model_;
  cnrtDataDescArray_t inDescs_, outDescs_;
  cnrtQueue_t queue_;
  cnrtFunction_t function_;
  cnrtDim3_t dim_;

  int inBlobNum_, outBlobNum_;
  unsigned int inNum_, inChannel_, inHeight_, inWidth_;
  unsigned int outNum_[3], outChannel_[3], outHeight_[3], outWidth_[3];
  vector<vector<int>> outputShape_;
  int outCount_[3];
  int threadId_;
  int deviceId_;
  int deviceSize_;
  int dataParallel_;
  int modelParallel_;
  float inferencingTime_;
  PostProcessor* postProcessor_;
  vector<void**> inPtrVector_;
  vector<void**> outPtrVector_;
};

class PostProcessor {
  public:
  explicit PostProcessor(const int& deviceId)
      : threadId_(0), deviceId_(deviceId) {}
  ~PostProcessor() {
    delete[] reinterpret_cast<float*>(outCpuPtrs_[0]);
    delete[] reinterpret_cast<float*>(outCpuPtrs_[1]);
    delete[] reinterpret_cast<float*>(outCpuPtrs_[2]);
    delete outCpuPtrs_;
  }
  void run();
  vector<vector<vector<float>>> detectionOutput(float*, float*, float*);
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
  void** outCpuPtrs_;
  vector<vector<vector<float>>> tensors;
  vector<vector<vector<float>>> x_y_offsets;
  vector<vector<vector<float>>> anchor_tensors;
};

class DataProvider {
  public:
  DataProvider(const string& meanFile, const string& meanValue,
               const int& deviceId, const queue<string>& images)
      : threadId_(0), deviceId_(deviceId), imageList(images) {}
  ~DataProvider() {
    delete[] reinterpret_cast<float*>(cpuData_[0]);
    delete cpuData_;
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

  private:
  int inNum_, inChannel_, inHeight_, inWidth_;
  int threadId_;
  int deviceId_;
  cv::Mat mean_;
  queue<string> imageList;
  Inferencer* inferencer_;
  cv::Size inGeometry_;
  void** cpuData_;
  vector<vector<cv::Mat>> inImages_;
  vector<vector<string>> imageName_;
};

void DataProvider::preRead() {
  while (imageList.size()) {
    vector<cv::Mat> rawImages;
    vector<string> imageNameVec;
    int imageLeft = imageList.size();
    string file = imageList.front();
    for (int i = 0; i < inNum_; i++) {
      if (i < imageLeft) {
        file = imageList.front();
        imageNameVec.push_back(file);
        imageList.pop();
        if (file.find(" ") != string::npos)
          file = file.substr(0, file.find(" "));
        cv::Mat img = cv::imread(file, -1);
        rawImages.push_back(img);
      } else {
        cv::Mat img = cv::imread(file, -1);
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
  int channel = threadId_ / inferencer_->deviceSize() % 4;
  cnrtSetCurrentChannel((cnrtChannelType_t)channel);
  for (int i = 0; i < FLAGS_fifosize; i++) {
    void** inputMluPtrS;
    void** outputMluPtrS;
    cnrtMallocBatchByDescArray(&inputMluPtrS, inferencer_->inDescs(),
                               inferencer_->inBlobNum(),
                               inferencer_->dataParallel());
    cnrtMallocBatchByDescArray(&outputMluPtrS, inferencer_->outDescs(),
                               inferencer_->outBlobNum(),
                               inferencer_->dataParallel());
    inferencer_->pushFreeInputData(inputMluPtrS);
    inferencer_->pushFreeOutputData(outputMluPtrS);
    inferencer_->pushInPtrVector(inputMluPtrS);
    inferencer_->pushOutPtrVector(outputMluPtrS);
  }

  inNum_ = inferencer_->n();
  inChannel_ = inferencer_->c();
  inHeight_ = inferencer_->h();
  inWidth_ = inferencer_->w();
  inGeometry_ = cv::Size(inWidth_, inHeight_);
  SetMean(FLAGS_meanfile, FLAGS_meanvalue);

  cpuData_ = new (void*);
  cpuData_[0] = new float[inNum_ * inChannel_ * inHeight_ * inWidth_];

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

    for (int i = 0; i < inNum_; i++) {
      if (i < imageLeft) {
        file = imageList.front();
        imageNameVec.push_back(file);
        imageList.pop();
        if (file.find(" ") != string::npos)
          file = file.substr(0, file.find(" "));
        cv::Mat img = cv::imread(file, -1);
        rawImages.push_back(img);
      } else {
        cv::Mat img = cv::imread(file, -1);
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
    CNRT_CHECK(cnrtMemcpyBatchByDescArray(
        mluData, cpuData_, inferencer_->inDescs(), inferencer_->inBlobNum(),
        inferencer_->dataParallel(), CNRT_MEM_TRANS_DIR_HOST2DEV));
    copyin.log("copyin time ...");
    inferencer_->pushValidInputData(mluData);
    inferencer_->pushValidInputNames(imageNameVec);
  }

  LOG(INFO) << "DataProvider: no data ...";
  // tell inferencer there is no more images to process
  inferencer_->pushValidInputData(nullptr);
}

void DataProvider::WrapInputLayer(vector<vector<cv::Mat>>* wrappedImages) {
  //  Parameter images is a vector [ ----   ] <-- images[in_n]
  //                                |
  //                                |-> [ --- ] <-- channels[3]
  // This method creates Mat objects, and places them at the
  // right offset of input stream
  int width = inferencer_->w();
  int height = inferencer_->h();
  float* data = reinterpret_cast<float*>(cpuData_[0]);

  for (int i = 0; i < inferencer_->n(); ++i) {
    wrappedImages->push_back(vector<cv::Mat>());
    for (int j = 0; j < 3; ++j) {
      cv::Mat channel(height, width, CV_32FC1, data);
      (*wrappedImages)[i].push_back(channel);
      data += width * height;
    }
  }
}

void DataProvider::Preprocess(const vector<cv::Mat>& sourceImages,
                              vector<vector<cv::Mat>>* destImages) {
  /* Convert the input image to the input image format of the network. */
  CHECK(sourceImages.size() == destImages->size())
      << "Size of sourceImages and destImages doesn't match";
  for (int i = 0; i < sourceImages.size(); ++i) {
    cv::Mat sample;
    int num_channels_ = inferencer_->c();
    cv::Size input_geometry;
    input_geometry = cv::Size(inferencer_->h(), inferencer_->w());  // 416*416
    if (sourceImages[i].channels() == 3 && inChannel_ == 1)
      cv::cvtColor(sourceImages[i], sample, cv::COLOR_BGR2GRAY);
    else if (sourceImages[i].channels() == 4 && inChannel_ == 1)
      cv::cvtColor(sourceImages[i], sample, cv::COLOR_BGRA2GRAY);
    else if (sourceImages[i].channels() == 4 && inChannel_ == 3)
      cv::cvtColor(sourceImages[i], sample, cv::COLOR_BGRA2BGR);
    else if (sourceImages[i].channels() == 1 && inChannel_ == 3)
      cv::cvtColor(sourceImages[i], sample, cv::COLOR_GRAY2BGR);
    else
      sample = sourceImages[i];

    // 2.resize the image
    cv::Mat sample_temp;
    int input_dim = inferencer_->h();
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

    cv::Mat sampleNormalized;
    bool int8 = (FLAGS_int8 != -1) ? FLAGS_int8 : FLAGS_fix8;
    if ((int8) || (FLAGS_meanvalue.empty() && FLAGS_meanfile.empty())) {
      sampleNormalized = sample_float;
    } else {
      cv::subtract(sample_float, mean_, sampleNormalized);
      if (FLAGS_scale != 1) {
        sampleNormalized *= FLAGS_scale;
      }
    }
    /* This operation will write the separate BGR planes directly to the
     * input layer of the network because it is wrapped by the cv::Mat
     * objects in input_channels. */
    cv::split(sampleNormalized, (*destImages)[i]);
  }
}

void DataProvider::SetMean(const string& meanFile, const string& meanValue) {
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
    LOG(INFO) << "Cannot support mean file";
  }
}

Inferencer::Inferencer(const string& offlinemodel, const int& dp,
                       const int& deviceId, const int& devicesize) {
  modelParallel_ = 1;
  deviceId_ = deviceId;
  deviceSize_ = devicesize;
  inferencingTime_ = 0;
  // 1. set current device
  setDeviceId(deviceId_);
  // 2. load model and get function
  LOG(INFO) << "load file: " << offlinemodel.c_str();
  cnrtLoadModel(&model_, offlinemodel.c_str());
  int mp;
  cnrtQueryModelParallelism(model_, &mp);
  if (FLAGS_dataparallel * mp <= 32) {
    dataParallel_ = dp;
    modelParallel_ = mp;
  } else {
    dataParallel_ = 1;
    modelParallel_ = 1;
    LOG(ERROR)
        << "dataparallel * modelparallel should <= 32, changed them to 1";
  }
  const string name = "subnet0";
  cnrtCreateFunction(&function_);
  cnrtExtractFunction(&function_, model_, name.c_str());

  // 3. get function's I/O DataDesc
  cnrtGetInputDataDesc(&inDescs_, &inBlobNum_, function_);
  cnrtGetOutputDataDesc(&outDescs_, &outBlobNum_, function_);
#if !defined(CROSS_COMPILE) && !defined(CROSS_COMPILE_ARM64)
  uint64_t stack_size;
  cnrtQueryModelStackSize(model_, &stack_size);
  unsigned int current_device_size;
  cnrtGetStackMem(&current_device_size);
  if (stack_size > current_device_size) {
    cnrtSetStackMem(stack_size + 50);
  }
#endif  // CROSS_COMPILE && CROSS_COMPILE_ARM64
  // 4. allocate I/O data space on CPU memory and prepare Input data
  int in_count;

  LOG(INFO) << "input blob num is " << inBlobNum_;
  for (int i = 0; i < inBlobNum_; i++) {
    unsigned int inN, inC, inH, inW;
    cnrtDataDesc_t desc = inDescs_[i];
    cnrtGetHostDataCount(desc, &in_count);
    cnrtSetHostDataLayout(desc, CNRT_FLOAT32, CNRT_NCHW);
    cnrtGetDataShape(desc, &inN, &inC, &inH, &inW);
    in_count *= dataParallel_;
    inN *= dataParallel_;
    LOG(INFO) << "shape " << inN;
    LOG(INFO) << "shape " << inC;
    LOG(INFO) << "shape " << inH;
    LOG(INFO) << "shape " << inW;
    if (i == 0) {
      inNum_ = inN;
      inChannel_ = inC;
      inWidth_ = inW;
      inHeight_ = inH;
    } else {
      cnrtGetHostDataCount(desc, &in_count);
    }
  }

  for (int i = 0; i < outBlobNum_; i++) {
    vector<int> shape;
    cnrtDataDesc_t desc = outDescs_[i];
    cnrtSetHostDataLayout(desc, CNRT_FLOAT32, CNRT_NCHW);
    cnrtGetHostDataCount(desc, &outCount_[i]);
    cnrtGetDataShape(desc, &outNum_[i], &outChannel_[i], &outHeight_[i],
                     &outWidth_[i]);
    outCount_[i] *= dataParallel_;
    outNum_[i] *= dataParallel_;
    shape.push_back(outNum_[i]);
    shape.push_back(outChannel_[i]);
    shape.push_back(outHeight_[i]);
    shape.push_back(outWidth_[i]);
    outputShape_.push_back(shape);
    LOG(INFO) << "output shape " << outNum_[i];
    LOG(INFO) << "output shape " << outChannel_[i];
    LOG(INFO) << "output shape " << outHeight_[i];
    LOG(INFO) << "output shape " << outWidth_[i];
  }
}

Inferencer::~Inferencer() {
  setDeviceId(deviceId_);
  cnrtDestroyFunction(function_);
  cnrtUnloadModel(model_);
  for (auto ptr : inPtrVector_) cnrtFreeArray(ptr, inBlobNum_);
  for (auto ptr : outPtrVector_) cnrtFreeArray(ptr, outBlobNum_);
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

void Inferencer::run() {
  setDeviceId(deviceId_);
  int channel = threadId_ / deviceSize_ % 4;
  cnrtSetCurrentChannel((cnrtChannelType_t)channel);

  cnrtFunction_t func;
  cnrtCreateFunction(&func);
  cnrtCopyFunction(&func, function_);

  // initialize function memory
  cnrtInitFuncParam_t initFuncParam;
  bool muta = false;
  int data_parallel = dataParallel_;
  uint32_t affinity = 0x01;
  initFuncParam.muta = &muta;
  initFuncParam.affinity = &affinity;
  initFuncParam.data_parallelism = &data_parallel;
  initFuncParam.end = CNRT_PARAM_END;
  cnrtInitFunctionMemory_V2(func, &initFuncParam);

  CHECK(cnrtCreateQueue(&queue_) == CNRT_RET_SUCCESS)
      << "CNRT create queue error, thread_id " << threadId_;

  //  create start_event and end_event
  cnrtNotifier_t notifierBeginning, notifierEnd;
  cnrtCreateNotifier(&notifierBeginning);
  cnrtCreateNotifier(&notifierEnd);
  float eventInterval;

  while (true) {
    void** mluInData = validInputFifo_.pop();
    if (mluInData == nullptr) break;  // no more images

    void** mluOutData = freeOutputFifo_.pop();
    void* param[inBlobNum_ + outBlobNum_];
    for (int i = 0; i < inBlobNum_; i++) {
      param[i] = mluInData[i];
    }
    for (int i = 0; i < outBlobNum_; i++) {
      param[inBlobNum_ + i] = mluOutData[i];
    }
    cnrtDim3_t dim = {1, 1, 1};
    cnrtInvokeFuncParam_t invokeFuncParam;
    invokeFuncParam.data_parallelism = &data_parallel;
    invokeFuncParam.affinity = &affinity;
    invokeFuncParam.end = CNRT_PARAM_END;
    cnrtFunctionType_t funcType = (cnrtFunctionType_t)0;
#if defined(CROSS_COMPILE) || defined(CROSS_COMPILE_ARM64)
    struct timeval tpend, tpstart;
    gettimeofday(&tpstart, NULL);
#endif
    cnrtPlaceNotifier(notifierBeginning, queue_);
    CNRT_CHECK(cnrtInvokeFunction_V2(func, dim, param, funcType, queue_,
                                  &invokeFuncParam));
    cnrtPlaceNotifier(notifierEnd, queue_);
#if defined(CROSS_COMPILE) || defined(CROSS_COMPILE_ARM64)
    gettimeofday(&tpend, NULL);
#endif
    if (cnrtSyncQueue(queue_) == CNRT_RET_SUCCESS) {
      cnrtNotifierDuration(notifierBeginning, notifierEnd, &eventInterval);
#if defined(CROSS_COMPILE) || defined(CROSS_COMPILE_ARM64)
      eventInterval = 1000000 * (tpend.tv_sec - tpstart.tv_sec) +
        tpend.tv_usec - tpstart.tv_usec;
#endif
      inferencingTime_ += eventInterval;
      printfMluTime(eventInterval);
    } else {
      LOG(ERROR) << "SyncQueue error";
    }

    pushValidOutputData(mluOutData);
    pushFreeInputData(mluInData);
  }

  cnrtDestroyNotifier(&notifierBeginning);
  cnrtDestroyNotifier(&notifierEnd);
  pushValidOutputData(nullptr);  // tell postprocessor to exit
}

vector<vector<vector<float>>> PostProcessor::detectionOutput(float* conv13,
                                                     float* conv26,
                                                     float* conv52) {
  vector<vector<int>> outputShape = inferencer_->outputshape();
  int batches = inferencer_->n() / inferencer_->dataParallel();
  outputShape[0][0] = outputShape[0][0] / inferencer_->dataParallel() / batches;
  outputShape[1][0] = outputShape[1][0] / inferencer_->dataParallel() / batches;
  outputShape[2][0] = outputShape[2][0] / inferencer_->dataParallel() / batches;

  int singleCount13 = outputShape[0][1] * outputShape[0][2] * outputShape[0][3];
  int singleCount26 = outputShape[1][1] * outputShape[1][2] * outputShape[1][3];
  int singleCount52 = outputShape[2][1] * outputShape[2][2] * outputShape[2][3];

  vector<vector<vector<float>>> final_boxes;
  for (int m = 0; m < batches; m++) {
    tensors.clear();
    tensors.push_back(get_blob_data(outputShape[0], conv13 + m * singleCount13));
    tensors.push_back(get_blob_data(outputShape[1], conv26 + m * singleCount26));
    tensors.push_back(get_blob_data(outputShape[2], conv52 + m * singleCount52));

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
  return final_boxes;
}

void PostProcessor::run() {
  setDeviceId(deviceId_);
  Inferencer* infr = inferencer_;  // avoid line wrap
  int channel = threadId_ / infr->deviceSize() % 4;
  cnrtSetCurrentChannel((cnrtChannelType_t)channel);

  vector<string> labelNameMap;
  if (!FLAGS_labels.empty()) {
    std::ifstream labels(FLAGS_labels);
    string line;
    while (std::getline(labels, line)) {
      labelNameMap.push_back(line);
    }
    labels.close();
  }


  outCpuPtrs_ = new void*[3];
  outCpuPtrs_[0] = new float[infr->outCount()[0]];
  outCpuPtrs_[1] = new float[infr->outCount()[1]];
  outCpuPtrs_[2] = new float[infr->outCount()[2]];

  vector<int> size_row = {507, 2028, 8112};
  vector<string> anchor_str = {"13", "26", "52"};
  x_y_offsets.push_back(vector<vector<float>>(507, vector<float>(2)));
  x_y_offsets.push_back(vector<vector<float>>(2028, vector<float>(2)));
  x_y_offsets.push_back(vector<vector<float>>(8112, vector<float>(2)));
  anchor_tensors.push_back(vector<vector<float>>(507, vector<float>(2)));
  anchor_tensors.push_back(vector<vector<float>>(2028, vector<float>(2)));
  anchor_tensors.push_back(vector<vector<float>>(8112, vector<float>(2)));

  readtxt(&x_y_offsets, &anchor_tensors, size_row, anchor_str);

  while (true) {
    void** mluOutData = infr->popValidOutputData();
    if (nullptr == mluOutData) break;  // no more data to process
    CNRT_CHECK(cnrtMemcpyBatchByDescArray(
        outCpuPtrs_, mluOutData, infr->outDescs(), infr->outBlobNum(),
        infr->dataParallel(), CNRT_MEM_TRANS_DIR_DEV2HOST));
    infr->pushFreeOutputData(mluOutData);

    int batches = inferencer_->n() / inferencer_->dataParallel();
    vector<vector<vector<float>>> final_boxes;
    vector<vector<vector<vector<float>>>> boxes(infr->dataParallel());
    float* outData0 = reinterpret_cast<float*>(outCpuPtrs_[0]);
    float* outData1 = reinterpret_cast<float*>(outCpuPtrs_[1]);
    float* outData2 = reinterpret_cast<float*>(outCpuPtrs_[2]);
    for (int dp = 0; dp < infr->dataParallel(); dp++) {
      boxes[dp] = detectionOutput(
          outData0 + (dp * infr->outCount()[0] / infr->dataParallel()),
          outData1 + (dp * infr->outCount()[1] / infr->dataParallel()),
          outData2 + (dp * infr->outCount()[2] / infr->dataParallel()));
    }
    for (int dp = 0; dp < infr->dataParallel(); dp++) {
      for (int batch = 0; batch < batches; batch++) {
          final_boxes.push_back(boxes[dp][batch]);
      }
    }
    vector<string> origin_img = infr->popValidInputNames();
    vector<cv::Mat> imgs;
    vector<string> img_names;
    for (auto img_name : origin_img) {
      if (img_name != "null") {
        cv::Mat img = cv::imread(img_name, -1);
        int pos = img_name.find_last_of('/');
        string file_name(img_name.substr(pos + 1));
        imgs.push_back(img);
        img_names.push_back("yolov3_offline_" + file_name);
      }
    }
    Timer dumpTimer;
    if (FLAGS_dump)
      WriteVisualizeBBox_offline(imgs, final_boxes, labelNameMap, img_names,
                                 infr->h());
    dumpTimer.log("dump imgs time ...");
  }
}

class Pipeline {
  public:
  Pipeline(const string& offlinemodel, const string& meanFile,
           const string& meanValue, const int& id,
           const int& deviceId, const int& devicesize,
           const int& dataparallel, queue<string> images);
  ~Pipeline();
  void run();
  inline DataProvider* dataProvider() { return data_provider_; }
  inline Inferencer* inferencer() { return inferencer_; }
  inline PostProcessor* postProcessor() { return postProcessor_; }

  private:
  DataProvider* data_provider_;
  Inferencer* inferencer_;
  PostProcessor* postProcessor_;
};

Pipeline::Pipeline(const string& offlinemodel, const string& meanFile,
                   const string& meanValue, const int& id,
                   const int& deviceId, const int& devicesize,
                   const int& dataparallel, queue<string> images) {
  inferencer_ = new Inferencer(offlinemodel, dataparallel, deviceId, devicesize);
  data_provider_ = new DataProvider(meanFile, meanValue, deviceId, images);
  postProcessor_ = new PostProcessor(deviceId);

  data_provider_->setInferencer(inferencer_);
  postProcessor_->setInferencer(inferencer_);
  inferencer_->setPostProcessor(postProcessor_);

  data_provider_->setThreadId(id);
  postProcessor_->setThreadId(id);
  inferencer_->setThreadId(id);

#ifdef PRE_READ
  data_provider_->preRead();
#endif
}

Pipeline::~Pipeline() {
  delete inferencer_;
  delete data_provider_;
  delete postProcessor_;
}

void Pipeline::run() {
  vector<thread*> threads(3, nullptr);
  threads[0] = new thread(&DataProvider::run, data_provider_);
  threads[1] = new thread(&Inferencer::run, inferencer_);
  threads[2] = new thread(&PostProcessor::run, postProcessor_);
  for (auto th : threads) th->join();
  for (auto th : threads) delete th;
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
  vector<queue<string>> imageList(totalThreads);
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
  vector<thread*> stageThreads;
  vector<Pipeline*> pipelineVector;
  for (int i = 0; i < totalThreads; i++) {
    if (imageList[i].size()) {
      Pipeline* pipeline = new Pipeline(
          FLAGS_offlinemodel, FLAGS_meanfile, FLAGS_meanvalue, i,
          deviceIds_[i % deviceIds_.size()], deviceIds_.size(),
          FLAGS_dataparallel, imageList[i]);
      stageThreads.push_back(new thread(&Pipeline::run, pipeline));
      pipelineVector.push_back(pipeline);
    }
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
  }
  gettimeofday(&tpend, NULL);
  execTime = 1000000 * (tpend.tv_sec - tpstart.tv_sec) + tpend.tv_usec -
             tpstart.tv_usec;
  LOG(INFO) << "yolov3_detection() execution time: " << execTime << " us";
  float mluTime = 0;
  for (int i = 0; i < pipelineVector.size(); i++) {
    mluTime += pipelineVector[i]->inferencer()->inferencingTime();
  }
  printPerf(imageNum, execTime, mluTime, totalThreads);

  for (auto iter : pipelineVector) {
    if (iter != nullptr) {
      delete iter;
    }
  }
  cnrtDestroy();
}

#else
#include <glog/logging.h>
int main(int argc, char* argv[]) {
  LOG(FATAL) << "This program should be compiled with the defintion"
             << " of both USE_MLU and USE_OPENCV!";
  return 0;
}
#endif  // USE_MLU
