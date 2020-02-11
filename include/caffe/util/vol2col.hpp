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

#ifndef INCLUDE_CAFFE_UTIL_VOL2COL_HPP_
#define INCLUDE_CAFFE_UTIL_VOL2COL_HPP_

namespace caffe {

template <typename Dtype>
void vol2col_cpu(const Dtype* data_im, const int channels, const int length,
                 const int height, const int width, const int ksize,
                 const int kdepth, const int pad, const int temporal_pad,
                 const int stride, const int temporal_stride, Dtype* data_col);

template <typename Dtype>
void col2vol_cpu(const Dtype* data_col, const int channels, const int length,
                 const int height, const int width, const int ksize,
                 const int kdepth, const int pad, const int temporal_pad,
                 const int stride, const int temporal_stride, Dtype* data_im);

template <typename Dtype>
void vol2col_gpu(const Dtype* data_im, const int channels, const int length,
                 const int height, const int width, const int ksize,
                 const int kdepth, const int pad, const int temporal_pad,
                 const int stride, const int temporal_stride, Dtype* data_col);

template <typename Dtype>
void col2vol_gpu(const Dtype* data_col, const int channels, const int length,
                 const int height, const int width, const int ksize,
                 const int kdepth, const int pad, const int temporal_pad,
                 const int stride, const int temporal_stride, Dtype* data_im);

}  // namespace caffe

#endif  // INCLUDE_CAFFE_UTIL_VOL2COL_HPP_
