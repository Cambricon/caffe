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
#include "caffe/layer.hpp"
#include "caffe/layers/absval_layer.hpp"
#include "caffe/mlu/spliter.hpp"

#include "caffe/test/test_caffe_main.hpp"

using std::unordered_set;
using std::unordered_map;

namespace caffe {

template <typename Dtype>
class SpliterTest : public ::testing::Test {
  protected:
  SpliterTest() {
    Caffe::set_mode(Caffe::MFUS);
    LayerParameter layer_param;
    layers_.resize(2);
    layers_[0].reset(new AbsValLayer<Dtype>(layer_param));
    layers_[1].reset(new AbsValLayer<Dtype>(layer_param));

    vector<Blob<Dtype>*> bottom1;
    vector<Blob<Dtype>*> bottom2;
    vector<Blob<Dtype>*> top1;
    vector<Blob<Dtype>*> top2;
    vector<Blob<Dtype>*> blobs;

    blobs.push_back(new Blob<Dtype>);
    blobs.push_back(new Blob<Dtype>);
    bottom2.push_back(blobs[0]);
    top1.push_back(blobs[0]);
    top2.push_back(blobs[1]);
    net_outputs_.push_back(blobs[1]);

    bottom_vecs_.push_back(bottom1);
    bottom_vecs_.push_back(bottom2);
    top_vecs_.push_back(top1);
    top_vecs_.push_back(top2);


    net_data_.reset(new NetData<Dtype>(&layers_,
        &bottom_vecs_,
        &top_vecs_,
        &net_outputs_));
    spliter_ = new Spliter<Dtype>(net_data_);
  }
  vector<shared_ptr<Layer<Dtype> > > layers_;
  vector<vector<Blob<Dtype>* > > bottom_vecs_;
  vector<vector<Blob<Dtype>* > > top_vecs_;
  vector<Blob<Dtype>* > net_outputs_;
  shared_ptr<NetData<Dtype> > net_data_;
  vector<shared_ptr<SubNet<Dtype> > > subnets_;
  Spliter<Dtype>* spliter_;
};

TYPED_TEST_CASE(SpliterTest, TestDtypes);

TYPED_TEST(SpliterTest, TestInitialization) {
  this->net_data_->addInputs(this->net_data_->top_vecs()[0][0]);
  this->spliter_->split(&this->subnets_);
  EXPECT_EQ(this->subnets_.size(), 1);
}


}  // namespace caffe

#endif  // USE_MLU
