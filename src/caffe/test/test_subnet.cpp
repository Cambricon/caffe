/*
All modification made by Cambricon Corporation: © 2018 Cambricon Corporation
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

#ifdef USE_MLU

#include <memory>
#include <vector>
#include <unordered_set>
#include <unordered_map>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/mlu_absval_layer.hpp"
#include "caffe/mlu/subnet.hpp"

#include "caffe/test/test_caffe_main.hpp"

using std::unordered_set;
using std::unordered_map;

namespace caffe {

template <typename TypeParam>
class SubNetTest : public MFUSDeviceTest<TypeParam> {
typedef typename TypeParam::Dtype Dtype;

  protected:
  SubNetTest() {
    LayerParameter layer_param;
    layers_.resize(2);
    layers_[0].reset(new MLUAbsValLayer<Dtype>(layer_param));
    layers_[1].reset(new MLUAbsValLayer<Dtype>(layer_param));

    vector<Blob<Dtype>*> bottom1;
    vector<Blob<Dtype>*> bottom2;
    vector<Blob<Dtype>*> top1;
    vector<Blob<Dtype>*> top2;
    vector<Blob<Dtype>*> blobs_;
    blobs_.push_back(new Blob<Dtype>(2, 3, 6, 9));
    blobs_.push_back(new Blob<Dtype>(2, 3, 6, 9));
    blobs_.push_back(new Blob<Dtype>(2, 3, 6, 9));


    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(blobs_[0]);

    bottom1.push_back(blobs_[0]);
    bottom2.push_back(blobs_[1]);
    top1.push_back(blobs_[1]);
    top2.push_back(blobs_[2]);
    net_outputs_.push_back(blobs_[2]);

    bottom_vecs_.push_back(bottom1);
    bottom_vecs_.push_back(bottom2);
    top_vecs_.push_back(top1);
    top_vecs_.push_back(top2);

    net_data_.reset(new NetData<Dtype>(&layers_,
        &bottom_vecs_,
        &top_vecs_,
        &net_outputs_));
    vector<int> layer_index(2, 0);
    layer_index[1] = 1;
    subnet_ = new SubNet<Dtype>(1, net_data_, layer_index);
  }
  vector<shared_ptr<Layer<Dtype> > > layers_;
  vector<vector<Blob<Dtype>* > > bottom_vecs_;
  vector<vector<Blob<Dtype>* > > top_vecs_;
  vector<Blob<Dtype>* > net_outputs_;
  shared_ptr<NetData<Dtype> > net_data_;
  SubNet<Dtype>* subnet_;
};

TYPED_TEST_CASE(SubNetTest, TestMFUSDevices);

TYPED_TEST(SubNetTest, TestInitialization) {
  const vector<int> layer_index = this->subnet_->layers();
  EXPECT_EQ(layer_index.size(), 2);
  EXPECT_EQ(layer_index[0], 0);
  EXPECT_EQ(layer_index[1], 1);
}

TYPED_TEST(SubNetTest, TestForward) {
  this->subnet_->addInput(this->bottom_vecs_[0][0]);
  this->subnet_->addOutput(this->top_vecs_[1][0]);
  this->subnet_->Reshape();
  this->subnet_->Forward(0, 1);
  for (int i = 0; i < 324; i++) {
    EXPECT_EQ(this->top_vecs_[1][0]->cpu_data()[i],
         fabs(this->bottom_vecs_[0][0]->cpu_data()[i]));
  }
}

}  // namespace caffe

#endif  // USE_MLU
