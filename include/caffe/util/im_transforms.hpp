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

#ifdef USE_OPENCV
#ifndef INCLUDE_CAFFE_UTIL_IM_TRANSFORMS_HPP_
#define INCLUDE_CAFFE_UTIL_IM_TRANSFORMS_HPP_

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

// Generate random number given the probablities for each number.
int roll_weighted_die(const std::vector<float>& probabilities);

template <typename T>
bool is_border(const cv::Mat& edge, T color);

// Auto cropping image.
template <typename T>
cv::Rect CropMask(const cv::Mat& src, T point, int padding = 2);

cv::Mat colorReduce(const cv::Mat& image, int div = 64);

void fillEdgeImage(const cv::Mat& edgesIn, cv::Mat* filledEdgesOut);

void CenterObjectAndFillBg(const cv::Mat& in_img, const bool fill_bg,
                           cv::Mat* out_img);

cv::Mat AspectKeepingResizeAndPad(const cv::Mat& in_img,
                                  const int new_width, const int new_height,
                                  const int pad_type = cv::BORDER_CONSTANT,
                                  const cv::Scalar pad = cv::Scalar(0, 0, 0),
                                  const int interp_mode = cv::INTER_LINEAR);

cv::Mat AspectKeepingResizeBySmall(const cv::Mat& in_img,
                                   const int new_width, const int new_height,
                                   const int interp_mode = cv::INTER_LINEAR);

void constantNoise(const int n, const vector<uchar>& val, cv::Mat* image);

void UpdateBBoxByResizePolicy(const ResizeParameter& param,
                              const int old_width, const int old_height,
                              NormalizedBBox* bbox);

cv::Mat ApplyResize(const cv::Mat& in_img, const ResizeParameter& param);

cv::Mat ApplyNoise(const cv::Mat& in_img, const NoiseParameter& param);

}  // namespace caffe

#endif  // IM_TRANSFORMS_HPP
#endif  // INCLUDE_CAFFE_UTIL_IM_TRANSFORMS_HPP_
