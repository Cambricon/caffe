/*
All modification made by Cambricon Corporation: © 2018--2019 Cambricon Corporation
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


#ifndef INCLUDE_CAFFE_MLU_SUBNET_HPP_
#define INCLUDE_CAFFE_MLU_SUBNET_HPP_
#ifdef USE_MLU

#include <algorithm>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/compile.hpp"
#include "caffe/layer.hpp"
#include "caffe/mlu/fusion.hpp"
#include "caffe/mlu/netdata.hpp"
#include "caffe/mlu/util.hpp"

namespace caffe {

/**
 * @brief SubNet is part of a neural network.
 *
 * With in one SubNet, all layers have same attribution that whether
 * the layer is supported by MLU. Thus we can divide any network into
 * multiple subnets where each subnet has same attribute. When people
 * call Reshape/Forward upon one network, the subnet is actually doing
 * the work.
 */
template <typename Dtype>
class SubNet {
  public:
  SubNet() = delete;
  explicit SubNet(bool mfus_supported, shared_ptr<NetData<Dtype> > net,
                  const vector<int>& layers_index, int subnet_index = INT_MAX)
      : mfus_supported_(mfus_supported),
        ops_fused_(false),
        net_(net),
        layers_index_(layers_index),
        subnet_index_(subnet_index) {
    std::stringstream layers;
    layers << "subnet[" << subnet_index_ << "] layers : ";
    for (auto i : layers_index_) {
      layers << i << "(" << net_->layers(i)->layer_param().name() << ") ";
    }
    contents_ = layers.str();
  }

  const vector<int>& layers() const { return layers_index_; }
  void addInput(Blob<Dtype>* in) { uniquePushBack(&input_blobs_, in); }
  void addOutput(Blob<Dtype>* out) { uniquePushBack(&output_blobs_, out); }
  void print();

  void Reshape();
  void Forward(int start, int end);
  void Backward();

  void fuseLayers();
  void setHardwareReshape(hardwareReshape_t hard_reshape);
  void setDataPreprocessStrategy();

  void addOffline(const cnmlModel_t& model, std::stringstream& ss,
                  int* off_index,
                  const unordered_map<Blob<Dtype>*, int>& blob2index,
                  const unordered_map<Blob<Dtype>*, string>& blob2name,
                  hardwareReshape_t hardware_reshape);
  vector<string> inputBlobNames(
      const unordered_map<Blob<Dtype>*, string>& blob2name);
  vector<string> outputBlobNames(
      const unordered_map<Blob<Dtype>*, string>& blob2name);
  bool mfus_supported();

  private:
  bool mfus_supported_;
  MFusion<Dtype> fuse_;
  bool ops_fused_;
  shared_ptr<NetData<Dtype> > net_;
  vector<int> layers_index_;
  int subnet_index_;
  vector<Blob<Dtype>*> input_blobs_;
  vector<Blob<Dtype>*> output_blobs_;
  string contents_;  // human-readable subnet layers.

  void Reshape_cpu();
  void Reshape_mfus();
  void Forward_cpu();
  void Forward_mfus();
  inline void reshapeLayer(int layer_index);

  DISABLE_COPY_AND_ASSIGN(SubNet);
};

template <typename Dtype>
void SubNet<Dtype>::print() {
  LOG(INFO) << contents_;
}

template <typename Dtype>
void SubNet<Dtype>::Reshape() {
  CHECK(Caffe::mode() == Caffe::MFUS);
  LOG(INFO) << "reshaping subnet[" << subnet_index_ << "]...";
  if (mfus_supported_) {
    Reshape_mfus();
  } else {
    Reshape_cpu();
  }
}

template <typename Dtype>
void SubNet<Dtype>::Reshape_cpu() {
  for (auto i : layers_index_) {
    reshapeLayer(i);
  }
}

template <typename Dtype>
void SubNet<Dtype>::fuseLayers() {
  if (!mfus_supported_) {
    return;
  }
  if (ops_fused_) {
    // only fuse once
    return;
  }

  fuse_.reset();

  fuse_.addInputs(input_blobs_);
  fuse_.addOutputs(output_blobs_);
  LOG(INFO) << "subnet[" << subnet_index_ << "] fusing...";
  for (auto i : layers_index_) {
    net_->layers(i)->fuse(&fuse_);
    if (net_->layers(i)->externalOutput()) {
      net_->layers(i)->addExternalOutput(&fuse_, net_->top_vecs()[i]);
    }
  }

  ops_fused_ = true;
}

template <typename Dtype>
void SubNet<Dtype>::Reshape_mfus() {
  ops_fused_ = false;
  fuse_.reset();
  fuse_.addInputs(input_blobs_);
  fuse_.addOutputs(output_blobs_);
  for (auto i : layers_index_) {
    reshapeLayer(i);
  }
}

template <typename Dtype>
void SubNet<Dtype>::setHardwareReshape(hardwareReshape_t hardware_reshape) {
  cnmlDataOrder_t order_type;
  cnmlDataType_t data_type;
  bool set_flag = false;
  switch (hardware_reshape) {
    case hardware_reshape_NCHW:
      order_type = CNML_NCHW;
      data_type = CNML_DATA_FLOAT32;
      set_flag = true;
      break;
    case hardware_reshape_NHWC:
      order_type = CNML_NHWC;
      data_type = CNML_DATA_FLOAT32;
      set_flag = true;
      break;
    case hardware_reshape_close:
      break;
  }
  if (set_flag) {
    for (auto blob : input_blobs_) {
      cnmlEnableHardwareReshape(blob->mlu_tensor(), data_type, order_type);
    }
    for (auto blob : output_blobs_) {
      cnmlEnableHardwareReshape(blob->mlu_tensor(), data_type, order_type);
    }
  }
}
template <typename Dtype>
void SubNet<Dtype>::setDataPreprocessStrategy() {
  if (Caffe::getInDataStrategy() != -1) {
    for (auto blob : input_blobs_) {
      blob->setCpuTensorOrder(Caffe::in_dataorder());
      cnmlSetDataPreprocessStrategy(blob->mlu_tensor(),
              blob->cpu_tensor(), Caffe::in_datastrategy());
    }
  }
  if (Caffe::getOutDataStrategy() != -1) {
    for (auto blob : output_blobs_) {
      blob->setCpuTensorOrder(Caffe::out_dataorder());
      cnmlSetDataPreprocessStrategy(blob->mlu_tensor(),
              blob->cpu_tensor(), Caffe::out_datastrategy());
    }
  }
  for (auto i : layers_index_) {
    if (net_->layers(i)->externalOutput() &&
          net_->layers(i)->layer_param().type() == "MLUYUVtoRGB") {
      auto blob = net_->top_vecs()[i][0];
      cnmlSetDataPreprocessStrategy(blob->mlu_tensor(),
                                    blob->cpu_tensor(), CNML_NO_PREPROCESS);
    }
  }
}

template <typename Dtype>
void SubNet<Dtype>::Forward(int start, int end) {
  CHECK(Caffe::mode() == Caffe::MFUS);
  CHECK_NE(layers_index_.size(), 0);
  if (end < layers_index_.front() || start > layers_index_.back()) {
    LOG(INFO) << "Forward(" << start << ", " << end << ") is not in subnet("
              << layers_index_.front() << ", " << layers_index_.back()
              << "), skip";
  } else {
    LOG(INFO) << "forwarding subnet[" << subnet_index_ << "]...";
    if (mfus_supported_) {
      Forward_mfus();
    } else {
      Forward_cpu();
    }
  }
}

template <typename Dtype>
void SubNet<Dtype>::Forward_cpu() {
  for (auto layer_index : layers_index_) {
    auto layer = net_->layers(layer_index);
    auto bottom = net_->bottom_vecs(layer_index);
    auto top = net_->top_vecs(layer_index);
    layer->Forward(bottom, top);
  }
}

template <typename Dtype>
void SubNet<Dtype>::Forward_mfus() {
  CHECK(Caffe::mode() == Caffe::MFUS);
  setDataPreprocessStrategy();
  fuseLayers();
  fuse_.compile();
  fuse_.forward();
}

template <typename Dtype>
void SubNet<Dtype>::addOffline(
    const cnmlModel_t& model, std::stringstream& ss, int* off_index,
    const unordered_map<Blob<Dtype>*, int>& blob2index,
    const unordered_map<Blob<Dtype>*, string>& blob2name,
    hardwareReshape_t hardware_reshape) {
  CHECK(Caffe::mode() == Caffe::MFUS);
  if (!mfus_supported_) {
    ss << "[On CPU] " << contents_ << std::endl;
    return;
  }
  std::stringstream func_name;
  func_name << "subnet" << *off_index;
  fuseLayers();

  setDataPreprocessStrategy();
  setHardwareReshape(hardware_reshape);
  fuse_.sortIO(blob2index);
  fuse_.compile();
  MLU_CHECK(cnmlAddFusionOpToModel(model, fuse_.op(), func_name.str().c_str()));
  ss << "[On MLU] [call via func \"" << func_name.str() << "\"] " << contents_
     << std::endl;
  ss << "                   func \"" << func_name.str()
     << "\"  inputs:" << fuse_.inputBlobs(blob2name) << std::endl;
  ss << "                   func \"" << func_name.str()
     << "\" outputs:" << fuse_.outputBlobs(blob2name) << std::endl;
  (*off_index)++;
}

template <typename Dtype>
vector<string> SubNet<Dtype>::inputBlobNames(
    const unordered_map<Blob<Dtype>*, string>& blob2name) {
  return fuse_.inputBlobNames(blob2name);
}

template <typename Dtype>
vector<string> SubNet<Dtype>::outputBlobNames(
    const unordered_map<Blob<Dtype>*, string>& blob2name) {
  return fuse_.outputBlobNames(blob2name);
}

template <typename Dtype>
bool SubNet<Dtype>::mfus_supported() {
  return mfus_supported_;
}

template <typename Dtype>
void SubNet<Dtype>::Backward() {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
inline void SubNet<Dtype>::reshapeLayer(int layer_index) {
  auto layer = net_->layers(layer_index);
  auto bottom = net_->bottom_vecs(layer_index);
  auto top = net_->top_vecs(layer_index);
  layer->Reshape_dispatch(bottom, top);
}

}  // namespace caffe

#endif  // USE_MLU
#endif  // INCLUDE_CAFFE_MLU_SUBNET_HPP_
