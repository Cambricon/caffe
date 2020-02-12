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

#include <memory>
#include <vector>

#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"
#include "caffe/mlu/netdata.hpp"
#include "caffe/filler.hpp"
#include "caffe/net.hpp"
#include "caffe/util/io.hpp"
#include "caffe/layers/absval_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class NetDataTest : public ::testing::Test {
  protected:
  NetDataTest() {
    LayerParameter layer_param;
    layers_.resize(1);
    layers_[0].reset(new AbsValLayer<Dtype>(layer_param));
    Blob<Dtype>* blob = new Blob<Dtype>();
    bottom_vecs_.resize(1, vector<Blob<Dtype>* >(1, blob));
    top_vecs_.resize(1, vector<Blob<Dtype>* >(1, blob));
    output_blobs_.resize(1, blob);
    net_data_ = new NetData<Dtype>(&layers_,
        &bottom_vecs_,
        &top_vecs_,
        &output_blobs_);
  }
  NetData<Dtype>* net_data_;
  vector<vector<Blob<Dtype>* > > bottom_vecs_;
  vector<vector<Blob<Dtype>* > > top_vecs_;
  vector<Blob<Dtype>* > output_blobs_;
  vector<shared_ptr<Layer<Dtype> > > layers_;
};

TYPED_TEST_CASE(NetDataTest, TestDtypes);

TYPED_TEST(NetDataTest, TestInitialization) {
  EXPECT_EQ(this->net_data_->size(), this->net_data_->bottom_vecs().size());
  EXPECT_EQ(this->net_data_->size(), this->net_data_->top_vecs().size());
  EXPECT_EQ(this->net_data_->size(), this->net_data_->layers().size());
}

TYPED_TEST(NetDataTest, TestAddInputs) {
  this->net_data_->addInputs(this->net_data_->top_vecs()[0][0]);
  EXPECT_EQ(this->net_data_->inputs().size(), 1);
}

}  // namespace caffe
