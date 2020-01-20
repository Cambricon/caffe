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

#ifndef INCLUDE_CAFFE_LAYERS_PROPOSAL_LAYER_HPP_
#define INCLUDE_CAFFE_LAYERS_PROPOSAL_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

using namespace std;  // NOLINT
namespace caffe {

/**
 * @brief Compute elementwise operations, such as product and sum,
 *        along multiple input Blobs.
 */
template <typename Dtype>
class ProposalLayer : public Layer<Dtype> {
  public:
  explicit ProposalLayer(const LayerParameter& param) : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Proposal"; }
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual inline void set_int8_context(bool int8_mode) { int8_context = int8_mode; }
  void get_anchor();
  Dtype* init_anchor_;

  protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom) {}

  Dtype stride_, im_min_w_, im_min_h_, top_num_, nms_thresh_, nms_num_;
  int A_;
  bool nproposal_mode_ = false;
  void Sort(vector<Dtype>* scores, vector<int>* id, int start, int length);
  void nSort(vector<Dtype>* list_cpu, int start, int end, int num_top);
  void CreateAnchor(vector<Dtype> * anchor, int A, int W, int H, Dtype stride);
  void CreateBox(vector<Dtype> * box, Dtype* scores, vector<Dtype>* newsocres,
                 vector<Dtype >anchor, const Dtype * delt,
                 int A, int W, int H, Dtype im_w, Dtype im_h);
  void RemoveSmallBox(vector<Dtype>* box, vector<int> id,
                      vector< int >* keep, int * keep_num, int total,
                      Dtype w_min_size, Dtype h_min_size, vector<Dtype>* newscores);
  void GetNewScoresByKeep(const Dtype * scores, vector<Dtype> * new_scores,
          vector<int >keep, int keep_num);
  void GetTopScores(vector<Dtype>* scores, vector<Dtype>* box,
                    vector< int >* id, int * size, int THRESH);
  void NMS(const vector<Dtype>& box, vector<int>* id, int * id_size,
          Dtype THRESH, int MAX_NUM);
  void GetNewBox(const vector<Dtype>& box, Dtype * new_box, vector<int> id,
          int id_size, vector<Dtype>* scores);
          // int A, Dtype * init_anchor, Dtype stride, Dtype im_w, Dtype im_h,
  int Proposal(const Dtype * bbox_pred, Dtype * scores, int H, int W,
          int A, Dtype stride, Dtype im_w, Dtype im_h,
          Dtype im_min_w, Dtype im_min_h, Dtype top_thresh, Dtype nms_thresh,
          int nms_num, Dtype * new_box);
  bool int8_context;
  bool shuffle_channel_;
};

}  // namespace caffe

#endif  // INCLUDE_CAFFE_LAYERS_PROPOSAL_LAYER_HPP_
