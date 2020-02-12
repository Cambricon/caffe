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

#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffe/util/vol2col.hpp"

namespace caffe {

template <typename Dtype>
void vol2col_cpu(const Dtype* data_im, const int channels, const int length,
                 const int height, const int width, const int ksize,
                 const int kdepth, const int pad, const int temporal_pad,
                 const int stride, const int temporal_stride, Dtype* data_col) {
  int length_col = (length + 2 * temporal_pad - kdepth) / temporal_stride + 1;
  int height_col = (height + 2 * pad - ksize) / stride + 1;
  int width_col = (width + 2 * pad - ksize) / stride + 1;

  int channels_col = channels * kdepth * ksize * ksize;
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % ksize;
    int h_offset = (c / ksize) % ksize;
    int l_offset = (c / ksize / ksize) % kdepth;
    int c_im = c / ksize / ksize / kdepth;
    for (int l = 0; l < length_col; ++l) {
      for (int h = 0; h < height_col; ++h) {
        for (int w = 0; w < width_col; ++w) {
          int l_pad = l * temporal_stride - temporal_pad + l_offset;
          int h_pad = h * stride - pad + h_offset;
          int w_pad = w * stride - pad + w_offset;

          if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width &&
              l_pad >= 0 && l_pad < length)
            data_col[((c * length_col + l) * height_col + h) * width_col + w] =
                data_im[((c_im * length + l_pad) * height + h_pad) * width +
                        w_pad];
          else
            data_col[((c * length_col + l) * height_col + h) * width_col + w] =
                0;
        }
      }
    }
  }
}

// Explicit instantiation
template void vol2col_cpu<float>(const float* data_im, const int channels,
                                 const int length, const int height,
                                 const int width, const int ksize,
                                 const int kdepth, const int pad,
                                 const int temporal_pad, const int stride,
                                 const int temporal_stride, float* data_col);
template void vol2col_cpu<double>(const double* data_im, const int channels,
                                  const int length, const int height,
                                  const int width, const int ksize,
                                  const int kdepth, const int pad,
                                  const int temporal_pad, const int stride,
                                  const int temporal_stride, double* data_col);

template <typename Dtype>
void col2vol_cpu(const Dtype* data_col, const int channels, const int length,
                 const int height, const int width, const int ksize,
                 const int kdepth, const int pad, const int temporal_pad,
                 const int stride, const int temporal_stride, Dtype* data_im) {
  memset(data_im, 0, sizeof(Dtype) * length * height * width * channels);
  int length_col = (length + 2 * temporal_pad - kdepth) / temporal_stride + 1;
  int height_col = (height + 2 * pad - ksize) / stride + 1;
  int width_col = (width + 2 * pad - ksize) / stride + 1;
  int channels_col = channels * kdepth * ksize * ksize;
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % ksize;
    int h_offset = (c / ksize) % ksize;
    int l_offset = (c / ksize / ksize) % kdepth;
    int c_im = c / ksize / ksize / kdepth;
    for (int l = 0; l < length_col; ++l) {
      for (int h = 0; h < height_col; ++h) {
        for (int w = 0; w < width_col; ++w) {
          int l_pad = l * temporal_stride - temporal_pad + l_offset;
          int h_pad = h * stride - pad + h_offset;
          int w_pad = w * stride - pad + w_offset;
          if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width &&
              l_pad >= 0 && l_pad < length)
            data_im[((c_im * length + l_pad) * height + h_pad) * width +
                    w_pad] +=
                data_col[((c * length_col + l) * height_col + h) * width_col +
                         w];
        }
      }
    }
  }
}

// Explicit instantiation
template void col2vol_cpu<float>(const float* data_col, const int channels,
                                 const int length, const int height,
                                 const int width, const int ksize,
                                 const int kdepth, const int pad,
                                 const int temporal_pad, const int stride,
                                 const int temporal_stride, float* data_im);
template void col2vol_cpu<double>(const double* data_col, const int channels,
                                  const int length, const int height,
                                  const int width, const int ksize,
                                  const int kdepth, const int pad,
                                  const int temporal_pad, const int stride,
                                  const int temporal_stride, double* data_im);

}  // namespace caffe
