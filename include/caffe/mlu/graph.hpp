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

#ifndef INCLUDE_CAFFE_MLU_GRAPH_HPP_
#define INCLUDE_CAFFE_MLU_GRAPH_HPP_
#ifdef USE_MLU

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/mlu/edge.hpp"
#include "caffe/mlu/node.hpp"

namespace caffe {

/**
 * @brief Graph holds the topology when spliting neuron network into SubNets.
 *        Graph is the whole network initially, and subnets keep removed from
 *        Graph till it's empty.
 *
 *  Graph's tasks include:
 *  1. build the graph from layers and blobs. In the graph, nodes are layers
 *     while edges are the blobs that bridging nodes. Break the non-in-place
 *     layer links that linked by "in place" blobs.
 *  2. get neighboring layers of one specific layer. The blobs that bridging
 *     layers are also returned since they could be input or output of a subnet.
 */
template <typename Dtype>
class Graph {
  public:
  Graph() = delete;
  explicit Graph(const vector<shared_ptr<Layer<Dtype> > >& layers,
                 const vector<vector<Blob<Dtype>*> >& bottom_vecs,
                 const vector<vector<Blob<Dtype>*> >& top_vecs);
  explicit Graph(const vector<int>& layers_index,
                 const vector<shared_ptr<Layer<Dtype> > >& layers,
                 const vector<vector<Blob<Dtype>*> >& bottom_vecs,
                 const vector<vector<Blob<Dtype>*> >& top_vecs);
  /**
   * @brief Remove all the nodes by index in index_set and
   * cascadedly remove all the edges connected to it.
   */
  void erase(const unordered_set<int>& index_set);
  /**
   * @brief Get the neighboring layers of one specific layer.
   *        the *from* and *to* is in the view of the requested layer.
   */
  void getNeighbors(int layer_index, unordered_set<Edge<Dtype> >* from,
                    unordered_set<Edge<Dtype> >* to,
                    unordered_set<Blob<Dtype>*>* one_side_bottoms,
                    unordered_set<Blob<Dtype>*>* one_side_tops);

  inline bool empty() { return nodes_.empty(); }

  inline int nodes_num() { return nodes_.size(); }

  inline int edge_num() { return edges_.size(); }

  ~Graph() {
    clear();
    for (auto node_pair : nodes_) {
      delete node_pair.second;
    }
  }

  private:
  unordered_map<int, Node<Dtype>*> nodes_;
  Edges<Dtype> edges_;
  Edges<Dtype> edges_r_;
  /**
   * @brief one_side_bottoms_ and one_side_tops_ are blobs that connect
   *        ONLY one layer as eithor bottom or top. These blobs are not
   *        recored in Edges where blobs are both bottom and top.
   *        As these blobs are mostly inputs or outputs of network,
   *        they deserve special attention.
   */
  unordered_map<int, unordered_set<Blob<Dtype>*> > one_side_bottoms_;
  unordered_map<int, unordered_set<Blob<Dtype>*> > one_side_tops_;

  inline void init(const vector<int>& layers_index,
                   const vector<shared_ptr<Layer<Dtype> > >& layers,
                   const vector<vector<Blob<Dtype>*> >& bottom_vecs,
                   const vector<vector<Blob<Dtype>*> >& top_vecs);
  inline void clear();
  void buildGraph();
  // below are helpers of buildGraph()
  unordered_set<Blob<Dtype>*> all_blobs_;
  unordered_map<Blob<Dtype>*, unordered_set<int> > inblob_to_layers_;
  unordered_map<Blob<Dtype>*, unordered_set<int> > outblob_to_layers_;
  unordered_map<Blob<Dtype>*, int> blob_count_;

  void collectMappings();
  void buildEdgesAndBlobs();

  DISABLE_COPY_AND_ASSIGN(Graph);
};  // class Graph

template <typename Dtype>
void Graph<Dtype>::erase(const unordered_set<int>& index_set) {
  for (auto index : index_set) {
    if (nodes_.find(index) == nodes_.end()) continue;
    unordered_set<Edge<Dtype> > from;
    unordered_set<Edge<Dtype> > to;
    unordered_set<Blob<Dtype>* > one_side_bottoms;
    unordered_set<Blob<Dtype>* > one_side_tops;
    getNeighbors(index, &from, &to, &one_side_bottoms, &one_side_tops);
    for (auto edge_from : from) {
      inblob_to_layers_[edge_from.blob()].erase(edge_from.neighbor());
      outblob_to_layers_[edge_from.blob()].erase(edge_from.cur());
      edges_.erase(edge_from);
      Edge<Dtype> edge_from_r(edge_from.neighbor(), edge_from.cur(),
                              edge_from.blob());
      edges_r_.erase(edge_from_r);
    }
    for (auto edge_to : to) {
      inblob_to_layers_[edge_to.blob()].erase(edge_to.cur());
      outblob_to_layers_[edge_to.blob()].erase(edge_to.neighbor());
      edges_.erase(edge_to);
      Edge<Dtype> edge_to_r(edge_to.neighbor(), edge_to.cur(), edge_to.blob());
      edges_r_.erase(edge_to_r);
    }
    for (auto bottom : nodes_[index]->bottom()) {
      blob_count_[bottom]--;
      if (blob_count_[bottom] == 0) all_blobs_.erase(bottom);
    }
    for (auto top : nodes_[index]->top()) {
      blob_count_[top]--;
      if (blob_count_[top] == 0) all_blobs_.erase(top);
    }
    nodes_.erase(index);
    one_side_bottoms_.erase(index);
    one_side_tops_.erase(index);
  }
}

template <typename Dtype>
Graph<Dtype>::Graph(const vector<shared_ptr<Layer<Dtype> > >& layers,
                    const vector<vector<Blob<Dtype>*> >& bottom_vecs,
                    const vector<vector<Blob<Dtype>*> >& top_vecs) {
  CHECK_EQ(layers.size(), bottom_vecs.size());
  CHECK_EQ(layers.size(), top_vecs.size());
  int size = layers.size();
  vector<int> layers_index(size);
  for (int i = 0; i < size; i++) {
    layers_index[i] = i;
  }

  init(layers_index, layers, bottom_vecs, top_vecs);
}

template <typename Dtype>
Graph<Dtype>::Graph(const vector<int>& layers_index,
                    const vector<shared_ptr<Layer<Dtype> > >& layers,
                    const vector<vector<Blob<Dtype>*> >& bottom_vecs,
                    const vector<vector<Blob<Dtype>*> >& top_vecs) {
  CHECK_EQ(layers_index.size(), bottom_vecs.size());
  CHECK_EQ(layers_index.size(), top_vecs.size());
  init(layers_index, layers, bottom_vecs, top_vecs);
}

template <typename Dtype>
void Graph<Dtype>::getNeighbors(int layer_index,
                                unordered_set<Edge<Dtype> >* from,
                                unordered_set<Edge<Dtype> >* to,
                                unordered_set<Blob<Dtype>*>* one_side_bottoms,
                                unordered_set<Blob<Dtype>*>* one_side_tops) {
  CHECK(nodes_.find(layer_index) != nodes_.end());
  for (auto edge : edges_[layer_index]) {
    from->insert(edge);
  }
  for (auto edge : edges_r_[layer_index]) {
    to->insert(edge);
  }
  for (auto blob : one_side_bottoms_[layer_index]) {
    one_side_bottoms->insert(blob);
  }
  for (auto blob : one_side_tops_[layer_index]) {
    one_side_tops->insert(blob);
  }
}

template <typename Dtype>
inline void Graph<Dtype>::init(const vector<int>& layers_index,
                               const vector<shared_ptr<Layer<Dtype> > >& layers,
                               const vector<vector<Blob<Dtype>*> >& bottom_vecs,
                               const vector<vector<Blob<Dtype>*> >& top_vecs) {
  for (int i : layers_index) {
    for (auto bottom : bottom_vecs[i]) blob_count_[bottom]++;
    for (auto top : top_vecs[i]) blob_count_[top]++;
    nodes_[i] = new Node<Dtype>(i, layers[i], &bottom_vecs[i], &top_vecs[i]);
  }
  buildGraph();
}

template <typename Dtype>
inline void Graph<Dtype>::clear() {
  edges_.clear();
  edges_r_.clear();
  one_side_bottoms_.clear();
  one_side_tops_.clear();

  all_blobs_.clear();
  inblob_to_layers_.clear();
  outblob_to_layers_.clear();
}

template <typename Dtype>
void Graph<Dtype>::buildGraph() {
  clear();
  collectMappings();
  buildEdgesAndBlobs();
}

template <typename Dtype>
void Graph<Dtype>::collectMappings() {
  auto add_layers = [](unordered_map<Blob<Dtype>*, unordered_set<int> >& map,
                       Blob<Dtype>* blob, int layer) {
    if (map.find(blob) == map.end()) {
      unordered_set<int> s;
      map[blob] = s;
    }
    map[blob].insert(layer);
  };

  for (auto node_pair : nodes_) {
    auto node = node_pair.second;
    for (auto blob : node->bottom()) {
      all_blobs_.insert(blob);
      add_layers(inblob_to_layers_, blob, node->index());
    }
    for (auto blob : node->top()) {
      all_blobs_.insert(blob);
      add_layers(outblob_to_layers_, blob, node->index());
    }
  }
}

template <typename Dtype>
void Graph<Dtype>::buildEdgesAndBlobs() {
  auto add_edge = [this](Edges<Dtype>& edges, Blob<Dtype>* blob,
                         unordered_set<int>& layers_cur,
                         unordered_set<int>& layers_neighbor) {
    for (auto cur : layers_cur) {
      for (auto neighbor : layers_neighbor) {
        edges.add(cur, neighbor, blob);
      }
    }
  };

  auto add_one_side_blob = [](
      unordered_set<int>& layers, Blob<Dtype>* blob,
      unordered_map<int, unordered_set<Blob<Dtype>*> >& one_side_blobs) {
    for (auto layer : layers) {
      if (one_side_blobs.find(layer) == one_side_blobs.end()) {
        unordered_set<Blob<Dtype>*> s;
        one_side_blobs[layer] = s;
      }
      one_side_blobs[layer].insert(blob);
    }
  };

  for (auto blob : all_blobs_) {
    bool found_in = inblob_to_layers_.find(blob) != inblob_to_layers_.end();
    bool found_out = outblob_to_layers_.find(blob) != outblob_to_layers_.end();
    if (found_in && found_out) {
      auto layers_to = inblob_to_layers_[blob];
      auto layers_from = outblob_to_layers_[blob];
      add_edge(edges_, blob, layers_from, layers_to);
      add_edge(edges_r_, blob, layers_to, layers_from);
    } else if (found_in) {
      add_one_side_blob(inblob_to_layers_[blob], blob, one_side_bottoms_);
    } else if (found_out) {
      add_one_side_blob(outblob_to_layers_[blob], blob, one_side_tops_);
    } else {
      CHECK(false);
    }
  }
}

}  // namespace caffe

#endif  // USE_MLU
#endif  // INCLUDE_CAFFE_MLU_GRAPH_HPP_
