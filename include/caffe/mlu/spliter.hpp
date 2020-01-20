/*
All modification made by Cambricon Corporation: © 2018--2019 Cambricon Corporation
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

#ifndef INCLUDE_CAFFE_MLU_SPLITER_HPP_
#define INCLUDE_CAFFE_MLU_SPLITER_HPP_
#ifdef USE_MLU

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/mlu/edge.hpp"
#include "caffe/mlu/graph.hpp"
#include "caffe/mlu/subnet.hpp"
#include "cnml.h" // NOLINT

namespace caffe {

/**
 * @brief Spliter splits a neural network into several SubNets according to
 *        whether these SubNet is supported by MLU.
 *
 * During the process, the input/output Blobs of the SubNet are also drawed.
 * After spliting, the Reshape/Forward/Backward is conducted inside every
 * SubNets with the layer index of SubNet in a incremental order.
 */
template <typename Dtype>
class Spliter {
  public:
  Spliter() = delete;
  explicit Spliter(shared_ptr<NetData<Dtype> > net) : net_(net) {
    CHECK(Caffe::mode() == Caffe::MFUS);
  }

  void split(vector<shared_ptr<SubNet<Dtype> > >* subnets);
  ~Spliter() {}

  private:
  shared_ptr<NetData<Dtype> > net_;

  void findSubNet(vector<shared_ptr<SubNet<Dtype> > >* subnets);
  void buildSubNet(Graph<Dtype>* graph, shared_ptr<SubNet<Dtype> > subnet);
  bool isMfusSupported(int i) { return net_->layers(i)->mfus_supported(); }
  int filterBlob(Blob<Dtype>* blob, shared_ptr<SubNet<Dtype> > subnet,
                 shared_ptr<NetData<Dtype> > net);
};

template <typename Dtype>
void Spliter<Dtype>::split(vector<shared_ptr<SubNet<Dtype> > >* subnets) {
  CHECK(Caffe::mode() == Caffe::MFUS);
  CHECK_EQ(subnets->size(), 0);

  Graph<Dtype> graph(net_->layers(), net_->bottom_vecs(), net_->top_vecs());

  findSubNet(subnets);
  for (auto &subnet : *subnets) {
    buildSubNet(&graph, subnet);
  }

  LOG(INFO) << "split done: " << subnets->size() << " subnets in total.";
  for (auto &subnet : *subnets) {
    subnet->print();
  }
}

template <typename Dtype>
void Spliter<Dtype>::findSubNet(vector<shared_ptr<SubNet<Dtype> > >* subnets) {
  int start = 0;
  int end = 0;
  int subnet_count = 0;
  for (; start < net_->size(); start = end, subnet_count++) {
    vector<int> layers_index;
    layers_index.push_back(start);
    for (end = start + 1; end < net_->size(); end++) {
      if (isMfusSupported(start) == isMfusSupported(end)) {
        layers_index.push_back(end);
      } else {
        break;
      }
    }
    shared_ptr<SubNet<Dtype> > subnet(new SubNet<Dtype>(
        isMfusSupported(start), net_, layers_index, subnet_count));
    subnets->push_back(subnet);
  }
}

template <typename Dtype>
int Spliter<Dtype>::filterBlob(Blob<Dtype>* blob,
                               shared_ptr<SubNet<Dtype> > subnet,
                               shared_ptr<NetData<Dtype> > net) {
  int flag = 0;
  vector<shared_ptr<Layer<Dtype> > > layers = net->layers();
  vector<vector<Blob<Dtype>*> > bottom_vecs = net->bottom_vecs();
  for (auto layer : subnet->layers()) {
    if ((layers[layer]->layer_param().type() == "PriorBox") &&
        (blob == bottom_vecs[layer][0] || blob == bottom_vecs[layer][1])) {
      flag = flag | 1;
    } else if ((layers[layer]->layer_param().type() == "Crop") &&
               blob == bottom_vecs[layer][1]) {
      flag = flag | 1;
    } else if (layers[layer]->layer_param().type() == "UnPooling" &&
               blob == bottom_vecs[layer][1] &&
               bottom_vecs[layer].size() == 2) {
      flag = flag | 1;
    } else if (layers[layer]->layer_param().type() != "PriorBox" &&
               layers[layer]->layer_param().type() != "Crop" &&
               layers[layer]->layer_param().type() != "UnPooling") {
      for (auto blob_i : bottom_vecs[layer]) {
        if (blob_i == blob) flag = flag | 2;
      }
    }
  }
  if (flag == 1)
    return 1;
  else
    return 0;
}

template <typename Dtype>
void Spliter<Dtype>::buildSubNet(Graph<Dtype>* graph,
                                 shared_ptr<SubNet<Dtype> > subnet) {
  unordered_set<int> layers;
  for (auto layer : subnet->layers()) {
    layers.insert(layer);
  }
  // get subnet edge data from graph
  unordered_set<Blob<Dtype>*> subnet_inputs;
  unordered_set<Blob<Dtype>*> subnet_outputs;
  unordered_set<Blob<Dtype>*> one_side_bottoms;
  unordered_set<Blob<Dtype>*> one_side_tops;
  unordered_set<Blob<Dtype>*> all_blobs;
  unordered_set<Edge<Dtype> > edges_in;
  unordered_set<Edge<Dtype> > edges_out;
  for (auto layer : layers) {
    graph->getNeighbors(layer, &edges_out, &edges_in, &one_side_bottoms,
                       &one_side_tops);
  }

  // find input/output edges that are acrossing subnets
  for (auto edge : edges_in) {
    all_blobs.insert(edge.blob());
    if (layers.find(edge.neighbor()) == layers.end() &&
        !filterBlob(edge.blob(), subnet, net_)) {
      subnet_inputs.insert(edge.blob());
    }
  }
  for (auto edge : edges_out) {
    all_blobs.insert(edge.blob());
    if (layers.find(edge.neighbor()) == layers.end()) {
      subnet_outputs.insert(edge.blob());
    }
  }

  // find input/output of network that are belonging to this subnet
  for (auto blob : net_->inputs()) {
    if ((all_blobs.find(blob) != all_blobs.end() ||
         one_side_bottoms.find(blob) != one_side_bottoms.end()) &&
        !filterBlob(blob, subnet, net_)) {
      subnet_inputs.insert(blob);
    }
  }
  for (auto blob : net_->outputs()) {
    if (all_blobs.find(blob) != all_blobs.end() ||
        one_side_tops.find(blob) != one_side_tops.end()) {
      subnet_outputs.insert(blob);
    }
  }

  // flush input/output to subnet
  for (auto blob : subnet_inputs) {
    subnet->addInput(blob);
  }
  for (auto blob : subnet_outputs) {
    subnet->addOutput(blob);
  }
}

}  // namespace caffe

#endif  // USE_MLU
#endif  // INCLUDE_CAFFE_MLU_SPLITER_HPP_
