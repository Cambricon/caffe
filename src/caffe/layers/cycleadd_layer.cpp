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
#include <vector>
#include "caffe/layers/cycleadd_layer.hpp"
#include "caffe/util/math_functions.hpp"
namespace caffe {

template <typename Dtype>
void CycleAddLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[1]->num(), 1) << "number of bottom[1] not equal to 1";
  CHECK_EQ(bottom[1]->height(), 1) << "height of bottom[1] not be equal to 1";
  CHECK_EQ(bottom[1]->width(), 1) << "width of bottom[1] not equal to 1";
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels())
     << "channel of bottom[0] and botoom[1] not equal";
  if (bottom[0] != top[0]) {
    top[0]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void CycleAddLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data_a = bottom[0]->cpu_data();
  const Dtype* bottom_data_b = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count0 = bottom[0]->count(1);
  const int count1 = bottom[1]->channels();
  const int interval = count0/count1;

  for (int n = 0; n < bottom[0]->num(); n++) {
    for (int i = 0; i < count1; i++) {
      for (int j = 0; j < interval; j++) {
        top_data[n * count0 + i * interval + j ] =
          bottom_data_a[n * count0 + i * interval + j] + bottom_data_b[i];
      }
    }
  }
}

INSTANTIATE_CLASS(CycleAddLayer);
}  // namespace caffe
