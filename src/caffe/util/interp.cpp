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

#include <algorithm>
#include <cmath>
#include "caffe/common.hpp"
#include "caffe/util/interp.hpp"

namespace caffe {

template <typename Dtype>
void caffe_cpu_interp2(const int channels, const Dtype *data1,
     const int x1, const int y1, const int height1, const int width1,
     const int Height1, const int Width1, Dtype *data2, const int x2,
     const int y2, const int height2, const int width2, const int Height2,
     const int Width2, bool packed) {
  CHECK(x1 >= 0 && y1 >= 0 && height1 > 0 && width1 > 0
        && x2 >= 0 && y2 >= 0 && height2 > 0 && width2 > 0);
  CHECK(Width1 >= width1 + x1 && Height1 >= height1 + y1 &&
        Width2 >= width2 + x2 && Height2 >= height2 + y2);
  // special case: just copy
  if (height1 == height2 && width1 == width2) {
    for (int h2 = 0; h2 < height2; ++h2) {
      const int h1 = h2;
      for (int w2 = 0; w2 < width2; ++w2) {
        const int w1 = w2;
        if (packed) {
          const Dtype* pos1 = &data1[channels *
                ((y1 + h1) * Width1 + (x1 + w1))];
          Dtype* pos2 = &data2[channels * ((y2 + h2) * Width2 + (x2 + w2))];
          for (int c = 0; c < channels; ++c) {
            pos2[0] = pos1[0];
            pos1++;
            pos2++;
          }
        } else {
          const Dtype* pos1 = &data1[(y1 + h1) * Width1 + (x1 + w1)];
          Dtype* pos2 = &data2[(y2 + h2) * Width2 + (x2 + w2)];
          for (int c = 0; c < channels; ++c) {
            pos2[0] = pos1[0];
            pos1 += Width1 * Height1;
            pos2 += Width2 * Height2;
          }
        }
      }
    }
    return;
  }
  const float rheight = (height2 > 1) ?
        static_cast<float>(height1 - 1) / (height2 - 1) : 0.f;
  const float rwidth = (width2 > 1) ?
        static_cast<float>(width1 - 1) / (width2 - 1) : 0.f;
  for (int h2 = 0; h2 < height2; ++h2) {
    const float h1r = rheight * h2;
    const int h1 = h1r;
    const int h1p = (h1 < height1 - 1) ? 1 : 0;
    const Dtype h1lambda = h1r - h1;
    const Dtype h0lambda = Dtype(1.) - h1lambda;
    for (int w2 = 0; w2 < width2; ++w2) {
      const float w1r = rwidth * w2;
      const int w1 = w1r;
      const int w1p = (w1 < width1 - 1) ? 1 : 0;
      const Dtype w1lambda = w1r - w1;
      const Dtype w0lambda = Dtype(1.) - w1lambda;
      if (packed) {
        const Dtype* pos1 = &data1[channels * ((y1 + h1) * Width1 + (x1 + w1))];
        Dtype* pos2 = &data2[channels * ((y2 + h2) * Width2 + (x2 + w2))];
        for (int c = 0; c < channels; ++c) {
        pos2[0] =
            h0lambda * (w0lambda * pos1[0] + w1lambda * pos1[channels * w1p]) +
            h1lambda * (w0lambda * pos1[channels * h1p * Width1] +
            w1lambda * pos1[channels * (h1p * Width1 + w1p)]);
          pos1++;
          pos2++;
        }
      } else {
        const Dtype* pos1 = &data1[(y1 + h1) * Width1 + (x1 + w1)];
        Dtype* pos2 = &data2[(y2 + h2) * Width2 + (x2 + w2)];
        for (int c = 0; c < channels; ++c) {
          pos2[0] =
            h0lambda * (w0lambda * pos1[0] + w1lambda * pos1[w1p]) +
            h1lambda * (w0lambda * pos1[h1p * Width1] +
            w1lambda * pos1[h1p * Width1 + w1p]);
          pos1 += Width1 * Height1;
          pos2 += Width2 * Height2;
        }
      }
    }
  }
}


// Backward (adjoint) operation 1 <- 2 (accumulates)
template <typename Dtype>
void caffe_cpu_interp2_backward(const int channels,
    Dtype *data1, const int x1, const int y1, const int height1,
    const int width1, const int Height1, const int Width1,
    const Dtype *data2, const int x2, const int y2,
    const int height2, const int width2, const int Height2,
    const int Width2, bool packed) {
  CHECK(x1 >= 0 && y1 >= 0 && height1 > 0 && width1 > 0
        && x2 >= 0 && y2 >= 0 && height2 > 0 && width2 > 0);
  CHECK(Width1 >= width1 + x1 && Height1 >= height1 +
     y1 && Width2 >= width2 + x2 && Height2 >= height2 + y2);
  if (height1 == height2 && width1 == width2) {
    for (int h2 = 0; h2 < height2; ++h2) {
      const int h1 = h2;
      for (int w2 = 0; w2 < width2; ++w2) {
        const int w1 = w2;
        if (packed) {
          Dtype* pos1 = &data1[channels * ((y1 + h1) * Width1 + (x1 + w1))];
          const Dtype* pos2 = &data2[channels *
                ((y2 + h2) * Width2 + (x2 + w2))];
          for (int c = 0; c < channels; ++c) {
            pos1[0] += pos2[0];
            pos1++;
            pos2++;
          }
       } else {
          Dtype* pos1 = &data1[(y1 + h1) * Width1 + (x1 + w1)];
          const Dtype* pos2 = &data2[(y2 + h2) * Width2 + (x2 + w2)];
          for (int c = 0; c < channels; ++c) {
            pos1[0] += pos2[0];
            pos1 += Width1 * Height1;
            pos2 += Width2 * Height2;
          }
        }
      }
    }
    return;
  }
  const float rheight = (height2 > 1) ?
        static_cast<float>(height1 - 1) / (height2 - 1) : 0.f;
  const float rwidth = (width2 > 1) ?
        static_cast<float>(width1 - 1) / (width2 - 1) : 0.f;
  for (int h2 = 0; h2 < height2; ++h2) {
    const float h1r = rheight * h2;
    const int h1 = h1r;
    const int h1p = (h1 < height1 - 1) ? 1 : 0;
    const Dtype h1lambda = h1r - h1;
    const Dtype h0lambda = Dtype(1.) - h1lambda;
    for (int w2 = 0; w2 < width2; ++w2) {
      const float w1r = rwidth * w2;
      const int w1 = w1r;
      const int w1p = (w1 < width1 - 1) ? 1 : 0;
      const Dtype w1lambda = w1r - w1;
      const Dtype w0lambda = Dtype(1.) - w1lambda;
      if (packed) {
        Dtype* pos1 = &data1[channels * ((y1 + h1) * Width1 + (x1 + w1))];
        const Dtype* pos2 = &data2[channels * ((y2 + h2) * Width2 + (x2 + w2))];
        for (int c = 0; c < channels; ++c) {
          pos1[0] += h0lambda * w0lambda * pos2[0];
          pos1[channels * w1p] += h0lambda * w1lambda * pos2[0];
          pos1[channels * h1p * Width1] += h1lambda * w0lambda * pos2[0];
          pos1[channels * (h1p * Width1 + w1p)] +=
              h1lambda * w1lambda * pos2[0];
          pos1++;
          pos2++;
        }
      } else {
        Dtype* pos1 = &data1[(y1 + h1) * Width1 + (x1 + w1)];
        const Dtype* pos2 = &data2[(y2 + h2) * Width2 + (x2 + w2)];
        for (int c = 0; c < channels; ++c) {
          pos1[0] += h0lambda * w0lambda * pos2[0];
          pos1[w1p] += h0lambda * w1lambda * pos2[0];
          pos1[h1p * Width1] += h1lambda * w0lambda * pos2[0];
          pos1[h1p * Width1 + w1p] += h1lambda * w1lambda * pos2[0];
          pos1 += Width1 * Height1;
          pos2 += Width2 * Height2;
        }
      }
    }
  }
}

// Create Gaussian pyramid of an image. Assume output space is pre-allocated.
// IN : [channels height width]
template <typename Dtype>
void caffe_cpu_pyramid2(const int channels,
    const Dtype *data, const int height, const int width,
    Dtype *data_pyr, const int levels, bool packed) {
  CHECK(height > 0 && width > 0 && levels >= 0);
  int height1 = height, width1 = width;
  int height2 = height, width2 = width;
  const Dtype *data1 = data;
  Dtype *data2 = data_pyr;
  for (int l = 0; l < levels; ++l) {
    height2 /= 2;
    width2 /= 2;
    if (height2 == 0 || width2 == 0) {
      break;
    }
    for (int h2 = 0; h2 < height2; ++h2) {
      const int h1 = 2 * h2;
      for (int w2 = 0; w2 < width2; ++w2) {
        const int w1 = 2 * w2;
        if (packed) {
          const Dtype* pos1 = &data1[channels * (h1 * width1 + w1)];
          Dtype* pos2 = &data2[channels * (h2 * width2 + w2)];
          for (int c = 0; c < channels; ++c) {
           pos2[0] =  static_cast<Dtype>(.25) *
              (pos1[0] + pos1[channels] +
               pos1[channels * width1] + pos1[channels * (width1 + 1)]);
            pos1++;
            pos2++;
          }
        } else {
          const Dtype* pos1 = &data1[h1 * width1 + w1];
          Dtype* pos2 = &data2[h2 * width2 + w2];
          for (int c = 0; c < channels; ++c) {
           pos2[0] =  static_cast<Dtype>(.25) *
              (pos1[0] + pos1[1] +
               pos1[width1] + pos1[width1 + 1]);
            pos1 += width1 * height1;
            pos2 += width2 * height2;
           }
        }
      }
    }
    data1 = data2;
    height1 = height2;
    width1 = width2;
    data2 += channels * height2 * width2;
  }
}

template void caffe_cpu_interp2<float>(const int,
  const float *, const int, const int, const int, const int,
  const int, const int, float *, const int, const int,
  const int, const int, const int, const int, bool);

template void caffe_cpu_interp2<double>(const int,
  const double *, const int, const int, const int, const int,
  const int, const int, double *, const int, const int,
  const int, const int, const int, const int, bool);

template void caffe_cpu_interp2_backward<float>(const int,
  float *, const int, const int, const int, const int, const int,
  const int, const float *, const int, const int, const int,
  const int, const int, const int, bool);

template void caffe_cpu_interp2_backward<double>(const int,
  double *, const int, const int, const int, const int,
  const int, const int, const double *, const int,
  const int, const int, const int, const int, const int, bool);

template void caffe_cpu_pyramid2<float>(const int,
  const float *, const int, const int, float *, const int, bool);

template void caffe_cpu_pyramid2<double>(const int,
  const double *, const int, const int, double *, const int, bool);

}  // namespace caffe
