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

#ifndef INCLUDE_CAFFE_UTIL_IMAGE_IO_HPP_
#define INCLUDE_CAFFE_UTIL_IMAGE_IO_HPP_

#include <string>

#include "caffe/proto/caffe.pb.h"
#include "google/protobuf/message.h"

#include <opencv2/highgui/highgui_c.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/blob.hpp"

using std::string;
using ::google::protobuf::Message;

namespace caffe {

void ImageToBuffer(const cv::Mat* img, char* buffer);
void ImageChannelToBuffer(const cv::Mat* img, char* buffer, int c);
void BufferToGrayImage(const char* buffer, const int h, const int w,
                       cv::Mat* img);
void BufferToGrayImage(const float* buffer, const int h, const int w,
                       cv::Mat* img);
void BufferToGrayImage(const double* buffer, const int h, const int w,
                       cv::Mat* img);
void BufferToColorImage(const char* buffer, const int height, const int width,
                        cv::Mat* img);

bool ReadVideoToVolumeDatum(const char* filename, const int start_frm,
                            const int label, const int length, const int height,
                            const int width, const int sampling_rate,
                            VolumeDatum* datum);

bool ReadVideoToVolumeDatumHelper(const char* filename, const int start_frm,
                                  const int label, const int length,
                                  const int height, const int width,
                                  const int sampling_rate, VolumeDatum* datum);
bool ReadVideoToVolumeDatumHelperSafe(const char* filename, const int start_frm,
                                      const int label, const int length,
                                      const int height, const int width,
                                      const int sampling_rate,
                                      VolumeDatum* datum);

inline bool ReadVideoToVolumeDatum(const char* filename, const int start_frm,
                                   const int label, const int length,
                                   const int sampling_rate,
                                   VolumeDatum* datum) {
  return ReadVideoToVolumeDatum(filename, start_frm, label, length, 0, 0,
                                sampling_rate, datum);
}

bool ReadImageSequenceToVolumeDatum(const char* img_dir, const int start_frm,
                                    const int label, const int length,
                                    const int height, const int width,
                                    const int sampling_rate,
                                    VolumeDatum* datum);

inline bool ReadImageSequenceToVolumeDatum(const char* img_dir,
                                           const int start_frm, const int label,
                                           const int length,
                                           const int sampling_rate,
                                           VolumeDatum* datum) {
  return ReadImageSequenceToVolumeDatum(img_dir, start_frm, label, length, 0, 0,
                                        sampling_rate, datum);
}

template <typename Dtype>
bool load_blob_from_binary(const string fn_blob, Blob<Dtype>* blob);

template <typename Dtype>
bool load_blob_from_uint8_binary(const string fn_blob, Blob<Dtype>* blob);

template <typename Dtype>
bool save_blob_to_binary(Blob<Dtype>* blob, const string fn_blob,
                         int num_index);

template <typename Dtype>
inline bool save_blob_to_binary(Blob<Dtype>* blob, const string fn_blob) {
  return save_blob_to_binary(blob, fn_blob, -1);
}

}  // namespace caffe

#endif  // INCLUDE_CAFFE_UTIL_IMAGE_IO_HPP_
