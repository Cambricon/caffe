/*
All modification made by Cambricon Corporation: © 2018-2019 Cambricon Corporation
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

#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <opencv2/highgui/highgui_c.h>
#include <stdint.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <stdio.h>
#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/image_io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

using std::fstream;
using std::ios;
using std::max;
using std::string;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;

namespace caffe {

void ImageToBuffer(const cv::Mat* img, char* buffer) {
  int idx = 0;
  for (int c = 0; c < 3; ++c) {
    for (int h = 0; h < img->rows; ++h) {
      for (int w = 0; w < img->cols; ++w) {
        buffer[idx++] = img->at<cv::Vec3b>(h, w)[c];
      }
    }
  }
}
void ImageChannelToBuffer(const cv::Mat* img, char* buffer, int c) {
  int idx = 0;
  for (int h = 0; h < img->rows; ++h) {
    for (int w = 0; w < img->cols; ++w) {
      buffer[idx++] = img->at<cv::Vec3b>(h, w)[c];
    }
  }
}

void GrayImageToBuffer(const cv::Mat* img, char* buffer) {
  int idx = 0;
  for (int h = 0; h < img->rows; ++h) {
    for (int w = 0; w < img->cols; ++w) {
      buffer[idx++] = img->at<unsigned char>(h, w);
    }
  }
}
void BufferToGrayImage(const char* buffer, const int height, const int width,
                       cv::Mat* img) {
  int idx = 0;
  img->create(height, width, CV_8U);
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      img->at<unsigned char>(h, w) = buffer[idx++];
    }
  }
}
void BufferToGrayImage(const float* buffer, const int height, const int width,
                       cv::Mat* img) {
  int idx = 0;
  img->create(height, width, CV_8U);
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      img->at<unsigned char>(h, w) = (unsigned char)(buffer[idx++]);
    }
  }
}
void BufferToGrayImage(const double* buffer, const int height, const int width,
                       cv::Mat* img) {
  int idx = 0;
  img->create(height, width, CV_8U);
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      img->at<unsigned char>(h, w) = (unsigned char)(buffer[idx++]);
    }
  }
}

void BufferToColorImage(const char* buffer, const int height, const int width,
                        cv::Mat* img) {
  img->create(height, width, CV_8UC3);
  for (int c = 0; c < 3; c++) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        img->at<cv::Vec3b>(h, w)[c] =
            buffer[c * width * height + h * width + w];
      }
    }
  }
}

bool ReadVideoToVolumeDatum(const char* filename, const int start_frm,
                            const int label, const int length, const int height,
                            const int width, const int sampling_rate,
                            VolumeDatum* datum) {
  if (!ReadVideoToVolumeDatumHelper(filename, start_frm, label, length, height,
                                    width, sampling_rate, datum)) {
    return ReadVideoToVolumeDatumHelperSafe(filename, start_frm, label, length,
                                            height, width, sampling_rate,
                                            datum);
  } else {
    return true;
  }
}

bool ReadVideoToVolumeDatumHelper(const char* filename, const int start_frm,
                                  const int label, const int length,
                                  const int height, const int width,
                                  const int sampling_rate, VolumeDatum* datum) {
  cv::VideoCapture cap;
  cv::Mat img, img_origin;
  char* buffer = NULL;
  int offset = 0;
  int channel_size = 0;
  int image_size = 0;
  int data_size = 0;
  int use_start_frm = start_frm;

  cap.open(filename);
  if (!cap.isOpened()) {
    LOG(ERROR) << "Cannot open " << filename;
    return false;
  }

  datum->set_channels(3);
  datum->set_length(length);
  datum->set_label(label);
  datum->clear_data();
  datum->clear_float_data();

  int num_of_frames = cap.get(CV_CAP_PROP_FRAME_COUNT);
  if (num_of_frames < length * sampling_rate) {
    LOG(INFO) << filename << " does not have enough frames; having "
              << num_of_frames;
    return false;
  }

  // If start_frame == -1, then use random temporal jitering
  if (start_frm < 0) {
    use_start_frm =
        caffe_rng_rand() % (num_of_frames - length * sampling_rate + 1);
  }

  offset = 0;
  CHECK_GE(use_start_frm, 0) << "start frame must be greater or equal to 0";

  if (use_start_frm) cap.set(CV_CAP_PROP_POS_FRAMES, use_start_frm);

  int end_frm = use_start_frm + length * sampling_rate;
  CHECK_LE(end_frm, num_of_frames)
      << "end frame must be less or equal to num of frames";

  for (int i = use_start_frm; i < end_frm; i += sampling_rate) {
    if (sampling_rate > 1) cap.set(CV_CAP_PROP_POS_FRAMES, i);
    if (height > 0 && width > 0) {
      cap.read(img_origin);
      if (!img_origin.data) {
        LOG(INFO) << filename << " has no data at frame " << i;
        if (buffer != NULL) delete[] buffer;
        return false;
      }
      cv::resize(img_origin, img, cv::Size(width, height));
    } else {
      cap.read(img);
    }
    if (!img.data) {
      LOG(ERROR) << "Could not open or find file " << filename;
      if (buffer != NULL) delete[] buffer;
      return false;
    }

    if (i == use_start_frm) {
      datum->set_height(img.rows);
      datum->set_width(img.cols);
      image_size = img.rows * img.cols;
      channel_size = image_size * length;
      data_size = channel_size * 3;
      buffer = new char[data_size];
    }
    for (int c = 0; c < 3; c++) {
      ImageChannelToBuffer(&img, buffer + c * channel_size + offset, c);
    }
    offset += image_size;
  }
  CHECK(offset == channel_size) << "wrong offset size" << std::endl;
  datum->set_data(buffer, data_size);
  delete[] buffer;
  cap.release();
  return true;
}

bool ReadVideoToVolumeDatumHelperSafe(const char* filename, const int start_frm,
                                      const int label, const int length,
                                      const int height, const int width,
                                      const int sampling_rate,
                                      VolumeDatum* datum) {
  cv::VideoCapture cap;
  cv::Mat img, img_origin;
  char* buffer = NULL;
  int offset = 0;
  int channel_size = 0;
  int image_size = 0;
  int data_size = 0;
  int use_start_frm = start_frm;

  cap.open(filename);
  if (!cap.isOpened()) {
    LOG(ERROR) << "Cannot open " << filename;
    return false;
  }

  datum->set_channels(3);
  datum->set_length(length);
  datum->set_label(label);
  datum->clear_data();
  datum->clear_float_data();

  int num_of_frames = cap.get(CV_CAP_PROP_FRAME_COUNT);
  if (num_of_frames < length * sampling_rate) {
    LOG(INFO) << filename << " does not have enough frames; having "
              << num_of_frames;
    return false;
  }
  if (start_frm < 0) {
    use_start_frm =
        caffe_rng_rand() % (num_of_frames - length * sampling_rate + 1);
  }

  offset = 0;
  CHECK_GE(use_start_frm, 0) << "start frame must be greater or equal to 0";

  // Instead of random acess, do sequentically access (avoid key-frame issue)
  // This will keep use_start_frm frames
  int sequential_counter = 0;
  while (sequential_counter < use_start_frm) {
    cap.read(img_origin);
    sequential_counter++;
  }

  int end_frm = use_start_frm + length * sampling_rate;
  CHECK_LE(end_frm, num_of_frames)
      << "end frame must be less or equal to num of frames";

  for (int i = use_start_frm; i < end_frm; i++) {
    if (sampling_rate > 1) {
      // If sampling_rate > 1, purposely keep some frames
      if ((i - use_start_frm) % sampling_rate != 0) {
        cap.read(img_origin);
        continue;
      }
    }
    if (height > 0 && width > 0) {
      cap.read(img_origin);
      if (!img_origin.data) {
        LOG(INFO) << filename << " has no data at frame " << i;
        if (buffer != NULL) delete[] buffer;
        return false;
      }
      cv::resize(img_origin, img, cv::Size(width, height));
    } else {
      cap.read(img);
    }
    if (!img.data) {
      LOG(ERROR) << "Could not open or find file " << filename;
      if (buffer != NULL) delete[] buffer;
      return false;
    }

    if (i == use_start_frm) {
      datum->set_height(img.rows);
      datum->set_width(img.cols);
      image_size = img.rows * img.cols;
      channel_size = image_size * length;
      data_size = channel_size * 3;
      buffer = new char[data_size];
    }
    for (int c = 0; c < 3; c++) {
      ImageChannelToBuffer(&img, buffer + c * channel_size + offset, c);
    }
    offset += image_size;
  }
  CHECK(offset == channel_size) << "wrong offset size" << std::endl;
  datum->set_data(buffer, data_size);
  delete[] buffer;
  cap.release();
  return true;
}

bool ReadImageSequenceToVolumeDatum(const char* img_dir, const int start_frm,
                                    const int label, const int length,
                                    const int height, const int width,
                                    const int sampling_rate,
                                    VolumeDatum* datum) {
  char fn_im[256];
  cv::Mat img, img_origin;
  char* buffer = NULL;
  int offset = 0;
  int channel_size = 0;
  int image_size = 0;
  int data_size = 0;

  datum->set_channels(3);
  datum->set_length(length);
  datum->set_label(label);
  datum->clear_data();
  datum->clear_float_data();

  offset = 0;
  int end_frm = start_frm + length * sampling_rate;
  for (int i = start_frm; i < end_frm; i += sampling_rate) {
    sprintf(fn_im, "%s/%06d.jpg", img_dir, i);
    if (height > 0 && width > 0) {
      img_origin = cv::imread(fn_im, CV_LOAD_IMAGE_COLOR);
      if (!img_origin.data) {
        LOG(ERROR) << "Could not open or find file " << fn_im;
        return false;
      }
      cv::resize(img_origin, img, cv::Size(width, height));
      img_origin.release();
    } else {
      img = cv::imread(fn_im, CV_LOAD_IMAGE_COLOR);
    }

    if (!img.data) {
      LOG(ERROR) << "Could not open or find file " << fn_im;
      return false;
    }

    if (i == start_frm) {
      datum->set_height(img.rows);
      datum->set_width(img.cols);
      image_size = img.rows * img.cols;
      channel_size = image_size * length;
      data_size = channel_size * 3;
      buffer = new char[data_size];
    }
    for (int c = 0; c < 3; c++) {
      ImageChannelToBuffer(&img, buffer + c * channel_size + offset, c);
    }
    offset += image_size;
  }
  CHECK(offset == channel_size) << "wrong offset size" << std::endl;
  datum->set_data(buffer, data_size);
  delete[] buffer;
  return true;
}

template <>
bool load_blob_from_binary<float>(const string fn_blob, Blob<float>* blob) {
  FILE* f;
  f = fopen(fn_blob.c_str(), "rb");
  if (f == NULL) return false;
  int n, c, l, w, h;
  float* buff;
  size_t size;
  size = fread(&n, sizeof(int), 1, f);
  size = fread(&c, sizeof(int), 1, f);
  size = fread(&l, sizeof(int), 1, f);
  size = fread(&h, sizeof(int), 1, f);
  size = fread(&w, sizeof(int), 1, f);
  vector<int> shape(5);
  shape[0] = n;
  shape[1] = c;
  shape[2] = l;
  shape[3] = h;
  shape[4] = w;

  blob->Reshape(shape);
  buff = blob->mutable_cpu_data();

  size = fread(buff, sizeof(float), n * c * l * h * w, f);
  LOG(INFO) << "size: " << size;
  fclose(f);
  return true;
}

template <>
bool load_blob_from_binary<double>(const string fn_blob, Blob<double>* blob) {
  FILE* f;
  f = fopen(fn_blob.c_str(), "rb");
  if (f == NULL) return false;
  int n, c, l, w, h;
  double* buff;
  size_t size;
  size = fread(&n, sizeof(int), 1, f);
  size = fread(&c, sizeof(int), 1, f);
  size = fread(&l, sizeof(int), 1, f);
  size = fread(&h, sizeof(int), 1, f);
  size = fread(&w, sizeof(int), 1, f);
  vector<int> shape(5);
  shape[0] = n;
  shape[1] = c;
  shape[2] = l;
  shape[3] = h;
  shape[4] = w;

  blob->Reshape(shape);
  buff = blob->mutable_cpu_data();

  size = fread(buff, sizeof(double), n * c * l * h * w, f);
  LOG(INFO) << "size: " << size;
  fclose(f);
  return true;
}

template <>
bool load_blob_from_uint8_binary<float>(const string fn_blob,
                                        Blob<float>* blob) {
  FILE* f;
  f = fopen(fn_blob.c_str(), "rb");
  if (f == NULL) return false;
  int n, c, l, w, h;
  float* buff;
  size_t size;
  size = fread(&n, sizeof(int), 1, f); 
  size = fread(&c, sizeof(int), 1, f); 
  size = fread(&l, sizeof(int), 1, f); 
  size = fread(&h, sizeof(int), 1, f); 
  size = fread(&w, sizeof(int), 1, f); 

  vector<int> shape(5);
  shape[0] = n;
  shape[1] = c;
  shape[2] = l;
  shape[3] = h;
  shape[4] = w;
  blob->Reshape(shape);

  buff = blob->mutable_cpu_data();

  int count = n * c * l * h * w;
  unsigned char* temp_buff = new unsigned char[count];

  size = fread(temp_buff, sizeof(unsigned char), count, f);
  fclose(f);

  for (int i = 0; i < count; i++) buff[i] = static_cast<float>(temp_buff[i]);

  LOG(INFO) << "size: " << size;
  delete[] temp_buff;
  return true;
}

template <>
bool load_blob_from_uint8_binary<double>(const string fn_blob,
                                         Blob<double>* blob) {
  FILE* f;
  f = fopen(fn_blob.c_str(), "rb");
  if (f == NULL) return false;
  int n, c, l, w, h;
  double* buff;
  size_t size;
  size = fread(&n, sizeof(int), 1, f);
  size = fread(&c, sizeof(int), 1, f);
  size = fread(&l, sizeof(int), 1, f);
  size = fread(&h, sizeof(int), 1, f);
  size = fread(&w, sizeof(int), 1, f);

  vector<int> shape(5);
  shape[0] = n;
  shape[1] = c;
  shape[2] = l;
  shape[3] = h;
  shape[4] = w;
  blob->Reshape(shape);

  buff = blob->mutable_cpu_data();

  int count = n * c * l * h * w;
  unsigned char* temp_buff = new unsigned char[count];

  size = fread(temp_buff, sizeof(unsigned char), count, f);
  fclose(f);
  LOG(INFO) << "size: " << size;
  for (int i = 0; i < count; i++) buff[i] = static_cast<double>(temp_buff[i]);

  delete[] temp_buff;
  return true;
}

template <>
bool save_blob_to_binary<float>(Blob<float>* blob, const string fn_blob,
                                int num_index) {
  FILE* f;
  float* buff;
  int n, c, l, w, h;
  f = fopen(fn_blob.c_str(), "wb");
  if (f == NULL) return false;

  c = blob->shape(1);
  if (blob->num_axes() > 2)
    l = blob->shape(2);
  else
    l = 1;
  if (blob->num_axes() > 3)
    h = blob->shape(3);
  else
    h = 1;
  if (blob->num_axes() > 4)
    w = blob->shape(4);
  else
    w = 1;

  if (num_index < 0) {
    n = blob->shape(0);
    buff = blob->mutable_cpu_data();
  } else {
    n = 1;
    buff = blob->mutable_cpu_data() + num_index * c * l * h * w;
  }

  fwrite(&n, sizeof(int), 1, f);
  fwrite(&c, sizeof(int), 1, f);
  fwrite(&l, sizeof(int), 1, f);
  fwrite(&h, sizeof(int), 1, f);
  fwrite(&w, sizeof(int), 1, f);
  fwrite(buff, sizeof(float), n * c * l * h * w, f);
  fclose(f);
  return true;
}

template <>
bool save_blob_to_binary<double>(Blob<double>* blob, const string fn_blob,
                                 int num_index) {
  FILE* f;
  double* buff;
  int n, c, l, w, h;
  f = fopen(fn_blob.c_str(), "wb");
  if (f == NULL) return false;

  c = blob->shape(1);
  if (blob->num_axes() > 2)
    l = blob->shape(2);
  else
    l = 1;
  if (blob->num_axes() > 3)
    h = blob->shape(3);
  else
    h = 1;
  if (blob->num_axes() > 4)
    w = blob->shape(4);
  else
    w = 1;
  if (num_index < 0) {
    n = blob->shape(0);
    buff = blob->mutable_cpu_data();
  } else {
    n = 1;
    buff = blob->mutable_cpu_data() + num_index * c * l * h * w;
  }

  fwrite(&n, sizeof(int), 1, f);
  fwrite(&c, sizeof(int), 1, f);
  fwrite(&l, sizeof(int), 1, f);
  fwrite(&h, sizeof(int), 1, f);
  fwrite(&w, sizeof(int), 1, f);
  fwrite(buff, sizeof(double), n * c * l * h * w, f);
  fclose(f);
  return true;
}
}  // namespace caffe
