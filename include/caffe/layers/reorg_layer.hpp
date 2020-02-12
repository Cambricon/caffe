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

#ifndef INCLUDE_CAFFE_LAYERS_REORG_LAYER_HPP_
#define INCLUDE_CAFFE_LAYERS_REORG_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/*
 * @brief Reshapes the input Blob into an arbitrary-sized output Blob.
 *
 * Note: similarly to FlattenLayer, this layer does not change the input values
 * (see FlattenLayer, Blob::ShareData and Blob::ShareDiff).
 */
template <typename Dtype>
class ReorgLayer : public Layer<Dtype> {
  public:
  explicit ReorgLayer(const LayerParameter &param) : Layer<Dtype>(param) {}

  virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                          const vector<Blob<Dtype> *> &top);

  virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                       const vector<Blob<Dtype> *> &top);

  virtual inline const char *type() const { return "Reorg"; }

  virtual inline int ExactNumBottomBlobs() const { return 1; }

  virtual inline int ExactNumTopBlobs() const { return 1; }

  protected:
  virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                           const vector<Blob<Dtype> *> &top);

  virtual void Backward_cpu(const vector<Blob<Dtype> *> &top,
                            const vector<bool> &propagate_down,
                            const vector<Blob<Dtype> *> &bottom);

  virtual void Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                           const vector<Blob<Dtype> *> &top);

  virtual void Backward_gpu(const vector<Blob<Dtype> *> &top,
                            const vector<bool> &propagate_down,
                            const vector<Blob<Dtype> *> &bottom);

  int stride_;
  bool reverse_;
  int batch_num_;
  int channels_;
  int reorged_channels_;
  int height_, width_;
  int reorged_height_, reorged_width_;
  Blob<Dtype> diff_;
};
template <typename Dtype>
void reorg_cpu(Dtype *x, int w, int h, int c, int batch, int stride,
               int forward, Dtype *out) {
  int b, i, j, k;
  int out_c = c / (stride * stride);
  CHECK_GT(out_c, 0);
  for (b = 0; b < batch; ++b) {
    for (k = 0; k < c; ++k) {
      for (j = 0; j < h; ++j) {
        for (i = 0; i < w; ++i) {
          int in_index = i + w * (j + h * (k + c * b));
          int c2 = k % out_c;
          int offset = k / out_c;
          int w2 = i * stride + offset % stride;
          int h2 = j * stride + offset / stride;
          int out_index =
              w2 + w * stride * (h2 + h * stride * (c2 + out_c * b));
          if (forward)
            out[out_index] = x[in_index];
          else
            out[in_index] = x[out_index];
        }
      }
    }
  }
}

template <typename Dtype>
void reorg_cpu(const Dtype *bottom_data, const int b_w, const int b_h,
               const int b_c, const int b_n, const int stride,
               const bool forward, Dtype *top_data) {
  int t_c = b_c / (stride * stride);
  int t_w = b_w * stride;
  int t_h = b_h * stride;
  for (int n = 0; n < b_n; n++) {
    for (int c = 0; c < b_c; c++) {
      for (int h = 0; h < b_h; h++) {
        for (int w = 0; w < b_w; w++) {
          int bottom_index = w + b_w * (h + b_h * (c + b_c * n));
          int c2 = c % t_c;
          int offset = c / t_c;
          int w2 = w * stride + offset % stride;
          int h2 = h * stride + offset / stride;
          int top_index = w2 + t_w * (h2 + t_h * (c2 + t_c * n));
          if (forward)
            top_data[top_index] = bottom_data[bottom_index];
          else
            top_data[bottom_index] = bottom_data[top_index];
        }
      }
    }
  }
}

}  // namespace caffe

#endif  // INCLUDE_CAFFE_LAYERS_REORG_LAYER_HPP_
