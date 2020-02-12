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

#ifndef INCLUDE_CAFFE_MLU_NETDATA_HPP_
#define INCLUDE_CAFFE_MLU_NETDATA_HPP_
#ifdef USE_MLU

#include <sstream>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "cnml.h"  // NOLINT

namespace caffe {

// To hold some "global" data of Net.

template <typename Dtype>
class NetData {
  public:
  NetData() = delete;
  explicit NetData(vector<shared_ptr<Layer<Dtype> > >* layers,
                   vector<vector<Blob<Dtype>*> >* bottom_vecs,
                   vector<vector<Blob<Dtype>*> >* top_vecs,
                   vector<Blob<Dtype>*>* output_blobs)
      : bottom_vecs_(bottom_vecs),
        top_vecs_(top_vecs),
        output_blobs_(output_blobs) {
    CHECK_EQ(layers->size(), bottom_vecs->size());
    CHECK_EQ(layers->size(), top_vecs->size());
    layers_ = layers;
    findInputs();
  }

  int size() { return layers_->size(); }
  void print();

  shared_ptr<Layer<Dtype> > layers(int i) { return (*layers_)[i]; }
  vector<Blob<Dtype>*>& bottom_vecs(int i) { return (*bottom_vecs_)[i]; }
  vector<Blob<Dtype>*>& top_vecs(int i) { return (*top_vecs_)[i]; }
  vector<shared_ptr<Layer<Dtype> > >& layers() { return *layers_; }
  vector<vector<Blob<Dtype>*> >& bottom_vecs() { return *bottom_vecs_; }
  vector<vector<Blob<Dtype>*> >& top_vecs() { return *top_vecs_; }

  void addInputs(Blob<Dtype>* in) { input_blobs_.push_back(in); }
  vector<Blob<Dtype>*>& inputs() {
    CHECK_NE(input_blobs_.size(), 0);
    return input_blobs_;
  }
  vector<Blob<Dtype>*>& outputs() { return *output_blobs_; }

  private:
  void findInputs();
  // pointers to data owned by Net.
  vector<shared_ptr<Layer<Dtype> > >* layers_;
  vector<vector<Blob<Dtype>*> >* bottom_vecs_;
  vector<vector<Blob<Dtype>*> >* top_vecs_;
  vector<Blob<Dtype>*> input_blobs_;
  vector<Blob<Dtype>*>* output_blobs_;

  DISABLE_COPY_AND_ASSIGN(NetData);
};


/**
 * @brief Find out all input blobs of a network.
 *
 * Net has already provided the output blobs. We need the input blobs
 * information to reduce reshape and split subnets.
 *
 * We walk layers and assume a blob is taken as input:
 * 1. For tops - if a layer has no bottoms, thus all tops of it are inputs.
 * 2. For bottoms - if they have never been used as tops, thus they are inputs.
 */
template <typename Dtype>
void NetData<Dtype>::findInputs() {
  unordered_set<Blob<Dtype>*> inputs;
  auto add_blobs = [](unordered_set<Blob<Dtype>*>& dest,
                      vector<Blob<Dtype>*>& src) {
    for (auto blob : src) {
      dest.insert(blob);
    }
  };

  // find top inputs
  for (int i = 0; i < this->size(); i++) {
    if (this->bottom_vecs(i).size() == 0) {
      add_blobs(inputs, this->top_vecs(i));
    }
  }

  // find bottom inputs
  unordered_set<Blob<Dtype>*> all_bottoms;
  unordered_set<Blob<Dtype>*> all_tops;
  for (int i = 0; i < this->size(); i++) {
    add_blobs(all_bottoms, this->bottom_vecs(i));
    add_blobs(all_tops, this->top_vecs(i));
  }
  for (auto blob : all_bottoms) {
    if (all_tops.find(blob) == all_tops.end()) {
      inputs.insert(blob);
    }
  }

  // record in NetData
  for (auto blob : inputs) {
    this->addInputs(blob);
  }
}
template <typename Dtype>
void NetData<Dtype>::print() {
  auto vec2str = [](vector<Blob<Dtype>*>& blobs, std::stringstream& str) {
    for (auto blob : blobs) {
      str << blob << ", ";
    }
  };

  std::stringstream inputs;
  std::stringstream outputs;
  vec2str(input_blobs_, inputs);
  vec2str(*output_blobs_, outputs);

  LOG(INFO) << "Network inputs(" << input_blobs_.size()
            << "): " << inputs.str();
  LOG(INFO) << "Network outputs(" << output_blobs_->size()
            << "): " << outputs.str();
}

}  // namespace caffe

#endif  // USE_MLU
#endif  // INCLUDE_CAFFE_MLU_NETDATA_HPP_
