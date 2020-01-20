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

#ifdef USE_MLU
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/mlu/edge.hpp"
#include "caffe/mlu/graph.hpp"
#include "caffe/mlu/node.hpp"

#include "caffe/test/test_caffe_main.hpp"

using std::unordered_set;
using std::unordered_map;

namespace caffe {


/*
  test net architecture:

      one_side_bottom
            |
          Layer 0
        /         \
       /           \
  Layer 1       Layer 2
       \           /
        \         /
          Layer 3
            |
      one_side_top
*/

template <typename Dtype>
class GraphTest : public ::testing::Test {
  protected:
  GraphTest() {
    shared_ptr<Layer<Dtype> > p0((Layer<Dtype>*)(new int));
    shared_ptr<Layer<Dtype> > p1((Layer<Dtype>*)(new int));
    shared_ptr<Layer<Dtype> > p2((Layer<Dtype>*)(new int));
    shared_ptr<Layer<Dtype> > p3((Layer<Dtype>*)(new int));
    layers_.push_back(p0);
    layers_.push_back(p1);
    layers_.push_back(p2);
    layers_.push_back(p3);
    for (int i = 0; i < 5; i++)
      blobs_.push_back(new Blob<Dtype>());
    vector<Blob<Dtype>*> bottom0;
    vector<Blob<Dtype>*> bottom1;
    vector<Blob<Dtype>*> bottom2;
    vector<Blob<Dtype>*> bottom3;
    vector<Blob<Dtype>*> top0;
    vector<Blob<Dtype>*> top1;
    vector<Blob<Dtype>*> top2;
    vector<Blob<Dtype>*> top3;
    bottom0.push_back(blobs_[0]);
    top0.push_back(blobs_[1]);
    bottom1.push_back(blobs_[1]);
    bottom2.push_back(blobs_[1]);
    top1.push_back(blobs_[2]);
    top2.push_back(blobs_[3]);
    bottom3.push_back(blobs_[2]);
    bottom3.push_back(blobs_[3]);
    top3.push_back(blobs_[4]);
    bottom_vecs_.push_back(bottom0);
    bottom_vecs_.push_back(bottom1);
    bottom_vecs_.push_back(bottom2);
    bottom_vecs_.push_back(bottom3);
    top_vecs_.push_back(top0);
    top_vecs_.push_back(top1);
    top_vecs_.push_back(top2);
    top_vecs_.push_back(top3);
    graph_ = new Graph<Dtype>(layers_, bottom_vecs_, top_vecs_);
  }
  virtual ~GraphTest() {
  }
  vector<shared_ptr<Layer<Dtype> > > layers_;
  vector<vector<Blob<Dtype>*> > bottom_vecs_;
  vector<vector<Blob<Dtype>*> > top_vecs_;
  vector<Blob<Dtype>*> blobs_;
  Graph<Dtype>* graph_;

  unordered_set<Edge<Dtype> > from_;
  unordered_set<Edge<Dtype> > to_;
  unordered_set<Blob<Dtype>* > one_side_bottoms_;
  unordered_set<Blob<Dtype>* > one_side_tops_;
};

TYPED_TEST_CASE(GraphTest, TestDtypes);

TYPED_TEST(GraphTest, TestInitialization) {
  EXPECT_FALSE(this->graph_->empty());
  EXPECT_EQ(this->graph_->nodes_num(), 4);
  EXPECT_EQ(this->graph_->edge_num(), 4);
}

TYPED_TEST(GraphTest, TestNeighbors) {
  this->from_.clear();
  this->to_.clear();
  this->one_side_bottoms_.clear();
  this->one_side_tops_.clear();

  this->graph_->getNeighbors(0, &this->from_,
      &this->to_,
      &this->one_side_bottoms_,
      &this->one_side_tops_);

  EXPECT_EQ(this->from_.size(), 2);
  EXPECT_EQ(this->to_.size(), 0);
  EXPECT_EQ(this->one_side_bottoms_.size(), 1);
  EXPECT_EQ(this->one_side_tops_.size(), 0);

  this->from_.clear();
  this->to_.clear();
  this->one_side_bottoms_.clear();
  this->one_side_tops_.clear();

  this->graph_->getNeighbors(1, &this->from_,
      &this->to_,
      &this->one_side_bottoms_,
      &this->one_side_tops_);

  EXPECT_EQ(this->from_.size(), 1);
  EXPECT_EQ(this->to_.size(), 1);
  EXPECT_EQ(this->one_side_bottoms_.size(), 0);
  EXPECT_EQ(this->one_side_tops_.size(), 0);

  this->from_.clear();
  this->to_.clear();
  this->one_side_bottoms_.clear();
  this->one_side_tops_.clear();

  this->graph_->getNeighbors(3, &this->from_,
      &this->to_,
      &this->one_side_bottoms_,
      &this->one_side_tops_);

  EXPECT_EQ(this->from_.size(), 0);
  EXPECT_EQ(this->to_.size(), 2);
  EXPECT_EQ(this->one_side_bottoms_.size(), 0);
  EXPECT_EQ(this->one_side_tops_.size(), 1);
}

/*
  test net architecture after erase index 0 and 1:

                  Layer 2
                   /
                  /
          Layer 3
            |
      one_side_top
*/

TYPED_TEST(GraphTest, TestErase) {
  unordered_set<int> s;
  s.insert(0);
  s.insert(1);
  this->graph_->erase(s);
  EXPECT_EQ(this->graph_->edge_num(), 1);

  this->from_.clear();
  this->to_.clear();
  this->one_side_bottoms_.clear();
  this->one_side_tops_.clear();

  this->graph_->getNeighbors(2, &this->from_,
      &this->to_,
      &this->one_side_bottoms_,
      &this->one_side_tops_);

  EXPECT_EQ(this->from_.size(), 1);
  EXPECT_EQ(this->to_.size(), 0);
  EXPECT_EQ(this->one_side_bottoms_.size(), 0);
  EXPECT_EQ(this->one_side_tops_.size(), 0);

  this->from_.clear();
  this->to_.clear();
  this->one_side_bottoms_.clear();
  this->one_side_tops_.clear();

  this->graph_->getNeighbors(3, &this->from_,
      &this->to_,
      &this->one_side_bottoms_,
      &this->one_side_tops_);

  EXPECT_EQ(this->from_.size(), 0);
  EXPECT_EQ(this->to_.size(), 1);
  EXPECT_EQ(this->one_side_bottoms_.size(), 0);
  EXPECT_EQ(this->one_side_tops_.size(), 1);
}

}  // namespace caffe

#endif  // USE_MLU
