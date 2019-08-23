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

#ifndef INCLUDE_CAFFE_MLU_EDGE_HPP_
#define INCLUDE_CAFFE_MLU_EDGE_HPP_
#ifdef USE_MLU

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/mlu/node.hpp"

namespace caffe {

/**
 * @brief Edge records the "edge" in neuron network.
 *
 */
template <typename Dtype>
class Edge {
  public:
  Edge() = delete;
  explicit Edge(int cur, int neighbor, Blob<Dtype>* blob)
      : layer_cur_(cur), layer_neighbor_(neighbor), blob_(blob) {}

  int cur() const { return layer_cur_; }
  int neighbor() const { return layer_neighbor_; }
  Blob<Dtype>* blob() const { return blob_; }
  friend bool operator==(const Edge<Dtype>& a, const Edge<Dtype>& b) {
    return a.layer_cur_ == b.layer_cur_ &&
           a.layer_neighbor_ == b.layer_neighbor_ && a.blob_ == b.blob_;
  }

  private:
  int layer_cur_;
  int layer_neighbor_;
  Blob<Dtype>* blob_;
};

}  // namespace caffe

namespace std {

/**
 * @brief Hash function for Edge neighbor make it able for containers.
 */
template <>
template <typename Dtype>
class hash<caffe::Edge<Dtype> > {
  public:
  size_t operator()(const caffe::Edge<Dtype>& e) const {
    return hash<int>()(e.cur()) ^ hash<int>()(e.neighbor()) ^
           hash<void*>()(e.blob());
  }
};

}  // namespace std

namespace caffe {

/**
 * @brief Edges maintains all the edges in one network.
 *        The connection(edge) is addressed through the index of one layer,
 *        the addressing result is the layers that connected neighbor the layer.
 */
template <typename Dtype>
class Edges {
  public:
  void add(int layer_cur, int layer_neighbor, Blob<Dtype>* bridge_blob);
  void erase(const Edge<Dtype>& edge);
  void clear() { map_.clear(); }
  int empty() { return map_.empty(); }
  const unordered_set<Edge<Dtype> >& operator[](int layer) {
    return map_[layer];
  }
  int size() {
    int count = 0;
    for (auto index : map_) count += index.second.size();
    return count;
  }

  private:
  // layer_cur neighbor Edge map
  unordered_map<int, unordered_set<Edge<Dtype> > > map_;
};  // class Edges

template <typename Dtype>
void Edges<Dtype>::add(int layer_cur, int layer_neighbor,
                       Blob<Dtype>* bridge_blob) {
  Edge<Dtype> e(layer_cur, layer_neighbor, bridge_blob);
  if (map_.find(layer_cur) == map_.end()) {
    unordered_set<Edge<Dtype> > s;
    map_[layer_cur] = s;
  }
  map_[layer_cur].insert(e);
}

template <typename Dtype>
void Edges<Dtype>::erase(const Edge<Dtype>& edge) {
  if (map_.find(edge.cur()) != map_.end()) map_[edge.cur()].erase(edge);
}

}  // namespace caffe

#endif  // USE_MLU
#endif  // INCLUDE_CAFFE_MLU_EDGE_HPP_
