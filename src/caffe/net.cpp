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

#include <sys/time.h>
#include <algorithm>
#include <chrono> // NOLINT
#include <cstdio>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#ifdef USE_HDF5
#include "hdf5.h"  //  NOLINT(build/include)
#endif
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/parallel.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/insert_splits.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/upgrade_proto.hpp"

#include "caffe/util/format.hpp"

#ifdef USE_MLU
#include "caffe/mlu/reshape_helper.hpp"
#include "caffe/mlu/spliter.hpp"
#include "caffe/mlu/subnet.hpp"
#endif  // USE_MLU

int forward_iteration;

namespace caffe {

template <typename Dtype>
Net<Dtype>::Net(const NetParameter& param) {
  Init(param);
  blob_info();
}

template <typename Dtype>
Net<Dtype>::Net(const string& param_file, Phase phase, const int level,
                const vector<string>* stages) {
  NetParameter param;
  ReadNetParamsFromTextFileOrDie(param_file, &param);
  // Set phase, stages and level
  param.mutable_state()->set_phase(phase);
  if (stages != NULL) {
    for (int i = 0; i < stages->size(); i++) {
      param.mutable_state()->add_stage((*stages)[i]);
    }
  }
  param.mutable_state()->set_level(level);
  Init(param);
  blob_info();
}
template <typename Dtype>
Net<Dtype>::Net(void* buffer, int buffer_size, Phase phase) {
  NetParameter param;
  ReadNetParamsFromTextMemOrDie(buffer, buffer_size, &param);
  param.mutable_state()->set_phase(phase);
  LOG(INFO) << "Read params from buffer " << std::endl;
  Init(param);
  blob_info();
  LOG(INFO) << "net init finished" << std::endl;
}

template <typename Dtype>
void Net<Dtype>::Init(const NetParameter& in_param) {
  // Set phase from the state.
  phase_ = in_param.state().phase();
  // Filter layers based on their include/exclude rules and
  // the current NetState.
  NetParameter filtered_param;
  FilterNet(in_param, &filtered_param);
  LOG_IF(INFO, Caffe::root_solver())
      << "Initializing net from parameters: " << std::endl
      << filtered_param.DebugString();
  // Create a copy of filtered_param with splits added where necessary.
  NetParameter param;
  // Split_layer is used to find the outputs of net.
  // It is not needed anymore.
  // InsertSplits(filtered_param, &param);
  param = filtered_param;
#ifdef USE_MLU
  opt_level_ = in_param.opt_level();
  CHECK_GE(opt_level_, 0)
    << "opt_level should be in: 0,1,2";
  CHECK_LE(opt_level_, 2)
    << "opt_level should be in: 0,1,2";
  // Optimize net
  if (opt_level_ > 0 &&
      (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS)) {
    //  optimize Ssd structure
    if (opt_level_ >= 1) {
      NetParameter optimized_param_a;
      OptimizeSsd(param, &optimized_param_a);
      param = optimized_param_a;
    }
    //  optimize ConvBnScale structure
    if (opt_level_ == 2) {
      NetParameter optimized_param_b;
      OptimizeConvBnScale(param, &optimized_param_b);
      param = optimized_param_b;
    }

    LOG(INFO) << "[opt_level set] optimized parameter: " <<
        param.DebugString();
  }
#endif

  // Basically, build all the layers and set up their connections.
  name_ = param.name();
  map<string, int> blob_name_to_idx;
  set<string> available_blobs;
  memory_used_ = 0;
  // For each layer, set up its input and output
  bottom_vecs_.resize(param.layer_size());
  top_vecs_.resize(param.layer_size());
  bottom_id_vecs_.resize(param.layer_size());
  param_id_vecs_.resize(param.layer_size());
  top_id_vecs_.resize(param.layer_size());
  bottom_need_backward_.resize(param.layer_size());
  for (int layer_id = 0; layer_id < param.layer_size(); ++layer_id) {
    // Inherit phase from net if unset.
    if (!param.layer(layer_id).has_phase())
      param.mutable_layer(layer_id)->set_phase(phase_);
#ifdef USE_MLU
    if (param.layer(layer_id).blobs_dtype_size() &&
        (param.layer(layer_id).blobs_dtype(0).type() == DT_INT8)) {
      if (!int8_mode_flag_) {
        int8_mode_flag_ = true;
      }
    }
    // top_mlu_dtype
    if (Caffe::topDataType() != DT_INVALID) {
      param.set_top_mlu_dtype(Caffe::topDataType());
    }
    if (param.has_top_mlu_dtype() && param.top_mlu_dtype() != DT_INVALID) {
      param.mutable_layer(layer_id)->set_top_mlu_dtype(param.top_mlu_dtype());
    }
    // debug_dtype
    if (param.debug_dtype()) {
      param.mutable_layer(layer_id)->set_debug_dtype(param.debug_dtype());
    }

#endif
    // Setup layer.
    const LayerParameter& layer_param = param.layer(layer_id);
    if (layer_param.propagate_down_size() > 0) {
      CHECK_EQ(layer_param.propagate_down_size(), layer_param.bottom_size())
          << "propagate_down param must be specified "
          << "either 0 or bottom_size times ";
    }
    layers_.push_back(LayerRegistry<Dtype>::CreateLayer(layer_param));
    layer_names_.push_back(layer_param.name());
    LOG_IF(INFO, Caffe::root_solver()) << "Creating Layer "
                                       << layer_param.name();
    bool need_backward = false;
    int num_bottom = layer_param.bottom_size();
    // Figure out this layer's input and output
    for (int bottom_id = 0; bottom_id < num_bottom; ++bottom_id) {
      const int blob_id = AppendBottom(param, layer_id, bottom_id,
                                       &available_blobs, &blob_name_to_idx);
      // If a blob needs backward, this layer should provide it.
      need_backward |= blob_need_backward_[blob_id];
    }
    int num_top = layer_param.top_size();
    for (int top_id = 0; top_id < num_top; ++top_id) {
      AppendTop(param, layer_id, top_id, &available_blobs, &blob_name_to_idx);
      // Collect Input layer tops as Net inputs.
      if (layer_param.type() == "Input") {
        const int blob_id = blobs_.size() - 1;
        net_input_blob_indices_.push_back(blob_id);
        net_input_blobs_.push_back(blobs_[blob_id].get());
      }
    }
    // If the layer specifies that AutoTopBlobs() -> true and the LayerParameter
    // specified fewer than the required number (as specified by
    // ExactNumTopBlobs() or MinTopBlobs()), allocate them here.
    Layer<Dtype>* layer = layers_[layer_id].get();
    if (layer->AutoTopBlobs()) {
      const int needed_num_top =
          std::max(layer->MinTopBlobs(), layer->ExactNumTopBlobs());
      for (; num_top < needed_num_top; ++num_top) {
        // Add "anonymous" top blobs -- do not modify available_blobs or
        // blob_name_to_idx as we don't want these blobs to be usable as input
        // to other layers.
        AppendTop(param, layer_id, num_top, NULL, NULL);
      }
    }
    // After this layer is connected, set it up.
    layers_[layer_id]->SetUp(bottom_vecs_[layer_id], top_vecs_[layer_id]);
    LOG_IF(INFO, Caffe::root_solver()) << "Setting up "
                                       << layer_names_[layer_id];

#ifdef USE_MLU
    auto layer_type = static_cast<string>(layers_[layer_id]->type());
    if (layer_type == "Proposal" ||
        layer_type == "Region" ||
        layer_type == "ROIPooling" ||
        layer_type == "PSROIPooling" ||
        layer_type == "DetectionOutput" ||
        layer_type == "SsdDetection" ||
        layer_type == "ImageDetect" ||
        layer_type == "DetectionOut" ||
        layer_type == "Yolov3Detection" ||
        layer_type == "Region")
      layers_[layer_id]->set_int8_context(int8_mode_flag_);
#endif

    for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
      if (blob_loss_weights_.size() <= top_id_vecs_[layer_id][top_id]) {
        blob_loss_weights_.resize(top_id_vecs_[layer_id][top_id] + 1, Dtype(0));
      }
      blob_loss_weights_[top_id_vecs_[layer_id][top_id]] = layer->loss(top_id);
      LOG_IF(INFO, Caffe::root_solver())
          << "Top shape: " << top_vecs_[layer_id][top_id]->shape_string();
      if (layer->loss(top_id)) {
        LOG_IF(INFO, Caffe::root_solver()) << "    with loss weight "
                                           << layer->loss(top_id);
      }
      memory_used_ += top_vecs_[layer_id][top_id]->count();
    }
    LOG_IF(INFO, Caffe::root_solver()) << "Memory required for data: "
                                       << memory_used_ * sizeof(Dtype);
    const int param_size = layer_param.param_size();
    const int num_param_blobs = layers_[layer_id]->blobs().size();
    CHECK_LE(param_size, num_param_blobs)
        << "Too many params specified for layer " << layer_param.name();
    ParamSpec default_param_spec;
    for (int param_id = 0; param_id < num_param_blobs; ++param_id) {
      const ParamSpec* param_spec = (param_id < param_size)
                                        ? &layer_param.param(param_id)
                                        : &default_param_spec;
      const bool param_need_backward = param_spec->lr_mult() != 0;
      need_backward |= param_need_backward;
      layers_[layer_id]->set_param_propagate_down(param_id,
                                                  param_need_backward);
    }
    for (int param_id = 0; param_id < num_param_blobs; ++param_id) {
      AppendParam(param, layer_id, param_id);
    }
    // Finally, set the backward flag
    layer_need_backward_.push_back(need_backward);
    if (need_backward) {
      for (int top_id = 0; top_id < top_id_vecs_[layer_id].size(); ++top_id) {
        blob_need_backward_[top_id_vecs_[layer_id][top_id]] = true;
      }
    }
  }
  // Go through the net backwards to determine which blobs contribute to the
  // loss.  We can skip backward computation for blobs that don't contribute
  // to the loss.
  // Also checks if all bottom blobs don't need backward computation (possible
  // because the skip_propagate_down param) and so we can skip bacward
  // computation for the entire layer
  set<string> blobs_under_loss;
  set<string> blobs_skip_backp;
  for (int layer_id = layers_.size() - 1; layer_id >= 0; --layer_id) {
    bool layer_contributes_loss = false;
    bool layer_skip_propagate_down = true;
    for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
      const string& blob_name = blob_names_[top_id_vecs_[layer_id][top_id]];
      if (layers_[layer_id]->loss(top_id) ||
          (blobs_under_loss.find(blob_name) != blobs_under_loss.end())) {
        layer_contributes_loss = true;
      }
      if (blobs_skip_backp.find(blob_name) == blobs_skip_backp.end()) {
        layer_skip_propagate_down = false;
      }
      if (layer_contributes_loss && !layer_skip_propagate_down) break;
    }
    // If this layer can skip backward computation, also all his bottom blobs
    // don't need backpropagation
    if (layer_need_backward_[layer_id] && layer_skip_propagate_down) {
      layer_need_backward_[layer_id] = false;
      for (int bottom_id = 0; bottom_id < bottom_vecs_[layer_id].size();
           ++bottom_id) {
        bottom_need_backward_[layer_id][bottom_id] = false;
      }
    }
    if (!layer_contributes_loss) {
      layer_need_backward_[layer_id] = false;
    }
    if (Caffe::root_solver()) {
      if (layer_need_backward_[layer_id]) {
        LOG(INFO) << layer_names_[layer_id] << " needs backward computation.";
      } else {
        LOG(INFO) << layer_names_[layer_id]
                  << " does not need backward computation.";
      }
    }
    for (int bottom_id = 0; bottom_id < bottom_vecs_[layer_id].size();
         ++bottom_id) {
      // erase indicates whether the blob is the output blob of the net
      const string& bottom_blob_name =
          blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
      bool erase = false;
      if (available_blobs.find(bottom_blob_name) != available_blobs.end()) {
        erase = true;
        for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
          const string& top_blob_name =
              blob_names_[top_id_vecs_[layer_id][top_id]];
          if (bottom_blob_name == top_blob_name) {
            erase = false;
          }
        }
      }
      if (erase) {
        available_blobs.erase(bottom_blob_name);
      }
      if (layer_contributes_loss) {
        const string& blob_name =
            blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
        blobs_under_loss.insert(blob_name);
      } else {
        bottom_need_backward_[layer_id][bottom_id] = false;
      }
      if (!bottom_need_backward_[layer_id][bottom_id]) {
        const string& blob_name =
            blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
        blobs_skip_backp.insert(blob_name);
      }
    }
  }
  // Handle force_backward if needed.
  if (param.force_backward()) {
    for (int layer_id = 0; layer_id < layers_.size(); ++layer_id) {
      layer_need_backward_[layer_id] = true;
      for (int bottom_id = 0;
           bottom_id < bottom_need_backward_[layer_id].size(); ++bottom_id) {
        bottom_need_backward_[layer_id][bottom_id] =
            bottom_need_backward_[layer_id][bottom_id] ||
            layers_[layer_id]->AllowForceBackward(bottom_id);
        blob_need_backward_[bottom_id_vecs_[layer_id][bottom_id]] =
            blob_need_backward_[bottom_id_vecs_[layer_id][bottom_id]] ||
            bottom_need_backward_[layer_id][bottom_id];
      }
      for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
           ++param_id) {
        layers_[layer_id]->set_param_propagate_down(param_id, true);
      }
    }
  }
  // In the end, all remaining blobs are considered output blobs.
  for (set<string>::iterator it = available_blobs.begin();
       it != available_blobs.end(); ++it) {
    LOG_IF(INFO, Caffe::root_solver()) << "This network produces output "
                                       << *it;
    net_output_blobs_.push_back(blobs_[blob_name_to_idx[*it]].get());
    net_output_blob_indices_.push_back(blob_name_to_idx[*it]);
  }
  for (size_t blob_id = 0; blob_id < blob_names_.size(); ++blob_id) {
    blob_names_index_[blob_names_[blob_id]] = blob_id;
  }
  for (size_t layer_id = 0; layer_id < layer_names_.size(); ++layer_id) {
    layer_names_index_[layer_names_[layer_id]] = layer_id;
  }
  ShareWeights();
  debug_info_ = in_param.debug_info();
  LOG_IF(INFO, Caffe::root_solver()) << "Network initialization done.";

#ifdef USE_MLU
  // For CNML fusion mode
  // build subnets which are depend on layers execution on CPU or MLU
  InitSubnet();
#endif  // USE_MLU

  set_net_param_without_weights(in_param);
}

#ifdef USE_MLU

template <typename Dtype>
void Net<Dtype>::InitSubnet() {
  net_data_ = shared_ptr<NetData<Dtype>>(new NetData<Dtype>(
      &layers_, &bottom_vecs_, &top_vecs_, &net_output_blobs_));
  reshape_helper_ =
      shared_ptr<ReshapeHelper<Dtype>>(new ReshapeHelper<Dtype>(net_data_));
  net_data_->print();

  if (Caffe::mode() == Caffe::MFUS) {
    Spliter<Dtype> spliter(net_data_);
    spliter.split(&subnets_);

    // when the last layer of subnet is deconv, the optimization level of
    // this layer should be set to 1
    for (auto subnet : subnets_) {
      vector<int> subnetlayer_index = subnet->layers();
      int lastindex = subnetlayer_index.size() - 1;
      Layer<Dtype>* lastlayer = layers_[subnetlayer_index[lastindex]].get();

      if (static_cast<string>(lastlayer->type()) == "Deconvolution")
        lastlayer->set_optimization_level(1);
    }
  } else if (Caffe::mode() == Caffe::MLU) {
    // when the last layer of net is deconv, the optimization level of
    // this layer should be set to 1
    int lastindex = layers_.size() - 1;
    Layer<Dtype>* lastlayer = layers_[lastindex].get();
    if (static_cast<string>(lastlayer->type()) == "Deconvolution")
      lastlayer->set_optimization_level(1);
  }
}

template <typename Dtype>
void Net<Dtype>::genOfflineModel(const std::string& name) {
  Caffe::set_mode(Caffe::MFUS);
  Caffe::setReshapeMode(Caffe::ReshapeMode::SETUPONLY);
  Reshape();

  unordered_map<Blob<Dtype>*, string> blob2name;
  unordered_map<Blob<Dtype>*, int> blob2index;
  for (int i = blobs_.size() - 1; i >= 0; i--) {
    blob2name[blobs_[i].get()] = blob_names_[i];
    blob2index[blobs_[i].get()] = i;
  }

  std::stringstream info;
  cnmlModel_t model;
  string filename = name + ".cambricon";
  cnmlCreateModel(&model, name.c_str());
  int subnet_off_index = 0;
  vector<vector<string>> input_blob_array;
  vector<vector<string>> output_blob_array;
  for (auto subnet : subnets_) {
    subnet->addOffline(model, info, &subnet_off_index, blob2index, blob2name);
    input_blob_array.push_back(subnet->inputBlobNames(blob2name));
    output_blob_array.push_back(subnet->outputBlobNames(blob2name));
  }
  cnmlSaveModel(model, filename.c_str());
  cnmlDestroyModel(model);

  if (set_cpu_info_) {
    AppendCpuInfo(filename, input_blob_array, output_blob_array);
  }
  LOG(INFO) << "Offline model is generated!" << std::endl
            << "************************ Offline model information BEGIN "
            << "************************" << std::endl
            << "file name : " << filename << std::endl
            << "model name: " << name << std::endl
            << "model details as follow" << std::endl
            << info.str()
            << "************************* Offline model information END "
            << "*************************";
}

template <typename Dtype>
bool Net<Dtype>::genOfflineModelToMem(void* buffer,
                uint64_t* buffer_size,
                uint64_t* model_size) {
  Caffe::set_mode(Caffe::MFUS);
  Caffe::setReshapeMode(Caffe::ReshapeMode::SETUPONLY);
  Reshape();
  unordered_map<Blob<Dtype>*, string> blob2name;
  unordered_map<Blob<Dtype>*, int> blob2index;
  for (int i = blobs_.size() - 1; i >= 0; i--) {
    blob2name[blobs_[i].get()] = blob_names_[i];
    blob2index[blobs_[i].get()] = i;
  }

  std::stringstream info;
  cnmlModel_t model;
  string name = "offline";
  if (CNML_STATUS_SUCCESS != cnmlCreateModel(&model, name.c_str())) {
    return false;
  }
  int subnet_off_index = 0;
  for (auto subnet : subnets_) {
    subnet->addOffline(model, info, &subnet_off_index, blob2index, blob2name);
  }
  LOG(INFO) << "Start to Save Model" << std::endl;
  if (CNML_STATUS_SUCCESS !=
      cnmlSaveModelToMem(model, buffer, *buffer_size, model_size)) {
    return false;
  }
  if (CNML_STATUS_SUCCESS != cnmlDestroyModel(model)) {
    return false;
  }
  return true;
}

template <typename Dtype>
bool Net<Dtype>::genOfflineModelToMem(void** buffer, uint64_t* model_size) {
  Caffe::set_mode(Caffe::MFUS);
  Caffe::setReshapeMode(Caffe::ReshapeMode::SETUPONLY);
  Reshape();
  unordered_map<Blob<Dtype>*, string> blob2name;
  unordered_map<Blob<Dtype>*, int> blob2index;
  for (int i = blobs_.size() - 1; i >= 0; i--) {
    blob2name[blobs_[i].get()] = blob_names_[i];
    blob2index[blobs_[i].get()] = i;
  }

  std::stringstream info;
  cnmlModel_t model;
  string name = "offline";
  if (CNML_STATUS_SUCCESS != cnmlCreateModel(&model, name.c_str())) {
    return false;
  }
  int subnet_off_index = 0;
  for (auto subnet : subnets_) {
    subnet->addOffline(model, info, &subnet_off_index, blob2index, blob2name);
  }
  uint64_t ms = 0, realsize = 0;
  cnmlGetModelSize(model, &ms);
  void * buf = malloc(ms);

  LOG(INFO) << "Start to Save Model" << std::endl;
  if (CNML_STATUS_SUCCESS != cnmlSaveModelToMem(model, buf, ms, &realsize)) {
    free(buf);
    return false;
  }
  if (CNML_STATUS_SUCCESS != cnmlDestroyModel(model)) {
    free(buf);
    return false;
  }

  *buffer = buf;
  *model_size = ms;
  return true;
}

template <typename Dtype>
void Net<Dtype>::OptimizeConvBnScale(NetParameter param,
                           NetParameter* const param_optimized) {
  param_optimized->CopyFrom(param);
  bool need_opt = false;
  auto layer_optimized = GetConvBnScaleStruct(param);
  for (const auto& item : layer_optimized) {
    if (item.size()) {
      need_opt = true;
      break;
    }
  }
  if (need_opt) {
    vector<int> layer_passed(param.layer_size(), 0);
    //  log optimized structs
    //  tag layers to delete
    for (int i = 0; i < param.layer_size(); i++) {
      if (layer_optimized[i].size()) {
        param.mutable_layer(i)->mutable_convolution_param()->set_bias_term(true);
        LOG(INFO) << "[opt_level set] ConvBnScale optimization "
          <<"structure detected";
        LOG(INFO) << "=================conv layer "
                  << param.layer(i).name();
        for (int j = 0; j < layer_optimized[i].size(); j++) {
          int k = layer_optimized[i][j];
          layer_passed[k] = 1;
          LOG(INFO) << param.layer(k).name();
        }
        LOG(INFO) << "=====================";
      }
    }
    DeleteOptimizedLayers(param, param_optimized, layer_passed);
    //  a special value to notify CopyFromTrainedLayers
    //  that ConvBnScale optimization is detected.
    //  -1 means only ConBnScale optimization detected
    //  -2 means both ConvBnScale optimization is detected and int8 enabled
    //  we have to recalculate int8 info for weights
    //  bottom data stays unchanged
    //  so we don't have to recalculate bottom_mlu_dtype
    //  Ssd Optimization need no int8 recalculation cauz it's just a prototxt level
    //  optimization.
    opt_level_ = -1;
    if (IsInt8Net(*param_optimized)) {
      LOG(INFO) << "[opt_level set] Int8 enabled, weights of int8 info will be "
        << "recalculated";
      opt_level_ = -2;
    }
  } else {
    LOG(INFO) << "[opt_level set] the model doesn't need"
      << "ConvBnScale optimization";
  }
}

template <typename Dtype>
void Net<Dtype>::OptimizeSsd(const NetParameter& param,
                           NetParameter* const param_optimized) {
  param_optimized->CopyFrom(param);
  // Optimize SSD net for MLU
  string data_name = "";
  if (NetNeedsSsdOptimization(*param_optimized)) {
    param_optimized->clear_layer();
    for (int i = 0; i < param.layer_size(); i++) {
      vector<std::string> bottom_names;
      vector<int> dropped_layers(param.layer_size(), 0);
      if (param.layer(i).type() == "PriorBox" && data_name == "") {
        data_name = param.layer(i).bottom(1);
      }
      if (param.layer(i).type() == "DetectionOutput" ||
          param.layer(i).type() == "DetectionPoseOutput") {
        // drop layers
        dropped_layers[i] = 1;
        // record bottom names
        for (int j = 0; j < param.layer(i).bottom_size(); j++)
          bottom_names.push_back(param.layer(i).bottom(j));
        // if data name not in detection out layer's bottoms
        // push back it to bottom_names
        // so that we can make sure prior box is bottom_names[-2]
        auto data_name_exist = find(bottom_names.begin(), bottom_names.end(),
                                    data_name);
        if (data_name_exist == bottom_names.end()) {
          bottom_names.push_back(data_name);
        }
        // create SsdDetectionOut
        LayerParameter ssdDetectionLayer;
        ssdDetectionLayer.set_name(param.layer(i).name());
        ssdDetectionLayer.set_type("SsdDetection");
        ssdDetectionLayer.add_top(param.layer(i).top(0));
        if (param.layer(i).type() == "DetectionOutput")
          ssdDetectionLayer.mutable_detection_output_param()
              ->CopyFrom(param.layer(i).detection_output_param());
        else
          ssdDetectionLayer.mutable_detection_pose_output_param()
              ->CopyFrom(param.layer(i).detection_pose_output_param());
        if (param.layer(i).type() == "DetectionPoseOutput")
          ssdDetectionLayer.set_type("SsdDetectionPose");
        vector<string> ssd_bottoms;

        // hack loc parameters
        ParseNetSsdLocParameter(param, &ssd_bottoms, &dropped_layers, bottom_names[0]);
        // hack conf parameters
        ParseNetSsdConfParameter(param, &ssd_bottoms, &dropped_layers, bottom_names[1]);
        // hack pose parameters
        if (param.layer(i).type() == "DetectionPoseOutput")
          ParseNetSsdConfParameter(param, &ssd_bottoms, &dropped_layers, bottom_names[2]);
        // priorbox
        for (int j = 0; j < param.layer_size(); j++) {
          if (param.layer(j).top(0) == bottom_names[bottom_names.size() - 2]) {
            dropped_layers[j] = 1;
            for (int k = 0; k < param.layer(j).bottom_size(); k++) {
              int index = NetGetLayerIndexByTopName(param, param.layer(j).bottom(k));
              dropped_layers[index] = 1;
              ssdDetectionLayer.add_priorbox_params()
                  ->CopyFrom(param.layer(index).prior_box_param());
              ssd_bottoms.push_back(param.layer(index).bottom(0));
            }
          }
        }
        ssd_bottoms.push_back(data_name);
        for (int j = 0; j < ssd_bottoms.size(); j++) {
          ssdDetectionLayer.add_bottom(ssd_bottoms[j]);
        }

        for (int j = 0; j < param.layer_size(); j++) {
          if (!dropped_layers[j])
            param_optimized->add_layer()->CopyFrom(param.layer(j));
        }
        param_optimized->add_layer()->CopyFrom(ssdDetectionLayer);
      }
    }
  } else {
    LOG(INFO) << "[opt_level set] the model doesn't need ssd optimization";
  }
}


#endif  // USE_MLU

template <typename Dtype>
void Net<Dtype>::FilterNet(const NetParameter& param,
                           NetParameter* param_filtered) {
  NetState net_state(param.state());
  param_filtered->CopyFrom(param);
  param_filtered->clear_layer();
  for (int i = 0; i < param.layer_size(); ++i) {
    const LayerParameter& layer_param = param.layer(i);
    const string& layer_name = layer_param.name();
    CHECK(layer_param.include_size() == 0 || layer_param.exclude_size() == 0)
        << "Specify either include rules or exclude rules; not both.";
    // If no include rules are specified, the layer is included by default and
    // only excluded if it meets one of the exclude rules.
    bool layer_included = (layer_param.include_size() == 0);
    for (int j = 0; layer_included && j < layer_param.exclude_size(); ++j) {
      if (StateMeetsRule(net_state, layer_param.exclude(j), layer_name)) {
        layer_included = false;
      }
    }
    for (int j = 0; !layer_included && j < layer_param.include_size(); ++j) {
      if (StateMeetsRule(net_state, layer_param.include(j), layer_name)) {
        layer_included = true;
      }
    }
    if (layer_included) {
      param_filtered->add_layer()->CopyFrom(layer_param);
    }
  }
}

template <typename Dtype>
bool Net<Dtype>::StateMeetsRule(const NetState& state, const NetStateRule& rule,
                                const string& layer_name) {
  // Check whether the rule is broken due to phase.
  if (rule.has_phase()) {
    if (rule.phase() != state.phase()) {
      LOG_IF(INFO, Caffe::root_solver())
          << "The NetState phase (" << state.phase()
          << ") differed from the phase (" << rule.phase()
          << ") specified by a rule in layer " << layer_name;
      return false;
    }
  }
  // Check whether the rule is broken due to min level.
  if (rule.has_min_level()) {
    if (state.level() < rule.min_level()) {
      LOG_IF(INFO, Caffe::root_solver())
          << "The NetState level (" << state.level()
          << ") is above the min_level (" << rule.min_level()
          << ") specified by a rule in layer " << layer_name;
      return false;
    }
  }
  // Check whether the rule is broken due to max level.
  if (rule.has_max_level()) {
    if (state.level() > rule.max_level()) {
      LOG_IF(INFO, Caffe::root_solver())
          << "The NetState level (" << state.level()
          << ") is above the max_level (" << rule.max_level()
          << ") specified by a rule in layer " << layer_name;
      return false;
    }
  }
  // Check whether the rule is broken due to stage. The NetState must
  // contain ALL of the rule's stages to meet it.
  for (int i = 0; i < rule.stage_size(); ++i) {
    // Check that the NetState contains the rule's ith stage.
    bool has_stage = false;
    for (int j = 0; !has_stage && j < state.stage_size(); ++j) {
      if (rule.stage(i) == state.stage(j)) {
        has_stage = true;
      }
    }
    if (!has_stage) {
      LOG_IF(INFO, Caffe::root_solver())
          << "The NetState did not contain stage '" << rule.stage(i)
          << "' specified by a rule in layer " << layer_name;
      return false;
    }
  }
  // Check whether the rule is broken due to not_stage. The NetState must
  // contain NONE of the rule's not_stages to meet it.
  for (int i = 0; i < rule.not_stage_size(); ++i) {
    // Check that the NetState contains the rule's ith not_stage.
    bool has_stage = false;
    for (int j = 0; !has_stage && j < state.stage_size(); ++j) {
      if (rule.not_stage(i) == state.stage(j)) {
        has_stage = true;
      }
    }
    if (has_stage) {
      LOG_IF(INFO, Caffe::root_solver())
          << "The NetState contained a not_stage '" << rule.not_stage(i)
          << "' specified by a rule in layer " << layer_name;
      return false;
    }
  }
  return true;
}

// Helper for Net::Init: add a new top blob to the net.
template <typename Dtype>
void Net<Dtype>::AppendTop(const NetParameter& param, const int layer_id,
                           const int top_id, set<string>* available_blobs,
                           map<string, int>* blob_name_to_idx) {
  shared_ptr<LayerParameter> layer_param(
      new LayerParameter(param.layer(layer_id)));
  const string& blob_name = (layer_param->top_size() > top_id)
                                ? layer_param->top(top_id)
                                : "(automatic)";
  // Check if we are doing in-place computation
  if (blob_name_to_idx && layer_param->bottom_size() > top_id &&
      blob_name == layer_param->bottom(top_id)) {
    // In-place computation
    LOG_IF(INFO, Caffe::root_solver()) << layer_param->name() << " -> "
                                       << blob_name << " (in-place)";
    top_vecs_[layer_id].push_back(blobs_[(*blob_name_to_idx)[blob_name]].get());
    top_id_vecs_[layer_id].push_back((*blob_name_to_idx)[blob_name]);
  } else if (blob_name_to_idx &&
             blob_name_to_idx->find(blob_name) != blob_name_to_idx->end()) {
    // If we are not doing in-place computation but have duplicated blobs,
    // raise an error.
    LOG(FATAL) << "Top blob '" << blob_name
               << "' produced by multiple sources.";
  } else {
    // Normal output.
    if (Caffe::root_solver()) {
      LOG(INFO) << layer_param->name() << " -> " << blob_name;
    }
    shared_ptr<Blob<Dtype>> blob_pointer(new Blob<Dtype>());
    const int blob_id = blobs_.size();
    blobs_.push_back(blob_pointer);
    blob_names_.push_back(blob_name);
    blob_need_backward_.push_back(false);
    if (blob_name_to_idx) {
      (*blob_name_to_idx)[blob_name] = blob_id;
    }
    top_id_vecs_[layer_id].push_back(blob_id);
    top_vecs_[layer_id].push_back(blob_pointer.get());
  }
  if (available_blobs) {
    available_blobs->insert(blob_name);
  }
}

// Helper for Net::Init: add a new bottom blob to the net.
template <typename Dtype>
int Net<Dtype>::AppendBottom(const NetParameter& param, const int layer_id,
                             const int bottom_id, set<string>* available_blobs,
                             map<string, int>* blob_name_to_idx) {
  const LayerParameter& layer_param = param.layer(layer_id);
  const string& blob_name = layer_param.bottom(bottom_id);
  const int blob_id = (*blob_name_to_idx)[blob_name];
  LOG_IF(INFO, Caffe::root_solver()) << layer_names_[layer_id] << " <- "
                                     << blob_name;
  bottom_vecs_[layer_id].push_back(blobs_[blob_id].get());
  bottom_id_vecs_[layer_id].push_back(blob_id);
  bool need_backward = blob_need_backward_[blob_id];
  // Check if the backpropagation on bottom_id should be skipped
  if (layer_param.propagate_down_size() > 0) {
    need_backward = layer_param.propagate_down(bottom_id);
  }
  bottom_need_backward_[layer_id].push_back(need_backward);
  return blob_id;
}

template <typename Dtype>
void Net<Dtype>::AppendParam(const NetParameter& param, const int layer_id,
                             const int param_id) {
  const LayerParameter& layer_param = layers_[layer_id]->layer_param();
  const int param_size = layer_param.param_size();
  string param_name =
      (param_size > param_id) ? layer_param.param(param_id).name() : "";
  if (param_name.size()) {
    param_display_names_.push_back(param_name);
  } else {
    ostringstream param_display_name;
    param_display_name << param_id;
    param_display_names_.push_back(param_display_name.str());
  }
  const int net_param_id = params_.size();
  params_.push_back(layers_[layer_id]->blobs()[param_id]);
  param_id_vecs_[layer_id].push_back(net_param_id);
  param_layer_indices_.push_back(make_pair(layer_id, param_id));
  ParamSpec default_param_spec;
  const ParamSpec* param_spec = (layer_param.param_size() > param_id)
                                    ? &layer_param.param(param_id)
                                    : &default_param_spec;
  if (!param_size || !param_name.size() ||
      (param_name.size() &&
       param_names_index_.find(param_name) == param_names_index_.end())) {
    // This layer "owns" this parameter blob -- it is either anonymous
    // (i.e., not given a param_name) or explicitly given a name that we
    // haven't already seen.
    param_owners_.push_back(-1);
    if (param_name.size()) {
      param_names_index_[param_name] = net_param_id;
    }
    const int learnable_param_id = learnable_params_.size();
    learnable_params_.push_back(params_[net_param_id].get());
    learnable_param_ids_.push_back(learnable_param_id);
    has_params_lr_.push_back(param_spec->has_lr_mult());
    has_params_decay_.push_back(param_spec->has_decay_mult());
    params_lr_.push_back(param_spec->lr_mult());
    params_weight_decay_.push_back(param_spec->decay_mult());
  } else {
    // Named param blob with name we've seen before: share params
    const int owner_net_param_id = param_names_index_[param_name];
    param_owners_.push_back(owner_net_param_id);
    const pair<int, int>& owner_index =
        param_layer_indices_[owner_net_param_id];
    const int owner_layer_id = owner_index.first;
    const int owner_param_id = owner_index.second;
    LOG_IF(INFO, Caffe::root_solver())
        << "Sharing parameters '" << param_name << "' owned by "
        << "layer '" << layer_names_[owner_layer_id] << "', param "
        << "index " << owner_param_id;
    Blob<Dtype>* this_blob = layers_[layer_id]->blobs()[param_id].get();
    Blob<Dtype>* owner_blob =
        layers_[owner_layer_id]->blobs()[owner_param_id].get();
    const int param_size = layer_param.param_size();
    if (param_size > param_id && (layer_param.param(param_id).share_mode() ==
                                  ParamSpec_DimCheckMode_PERMISSIVE)) {
      // Permissive dimension checking -- only check counts are the same.
      CHECK_EQ(this_blob->count(), owner_blob->count())
          << "Cannot share param '" << param_name << "' owned by layer '"
          << layer_names_[owner_layer_id] << "' with layer '"
          << layer_names_[layer_id] << "'; count mismatch.  Owner layer param "
          << "shape is " << owner_blob->shape_string() << "; sharing layer "
          << "shape is " << this_blob->shape_string();
    } else {
      // Strict dimension checking -- all dims must be the same.
      CHECK(this_blob->shape() == owner_blob->shape())
          << "Cannot share param '" << param_name << "' owned by layer '"
          << layer_names_[owner_layer_id] << "' with layer '"
          << layer_names_[layer_id] << "'; shape mismatch.  Owner layer param "
          << "shape is " << owner_blob->shape_string() << "; sharing layer "
          << "expects shape " << this_blob->shape_string();
    }
    const int learnable_param_id = learnable_param_ids_[owner_net_param_id];
    learnable_param_ids_.push_back(learnable_param_id);
    if (param_spec->has_lr_mult()) {
      if (has_params_lr_[learnable_param_id]) {
        CHECK_EQ(param_spec->lr_mult(), params_lr_[learnable_param_id])
            << "Shared param '" << param_name << "' has mismatched lr_mult.";
      } else {
        has_params_lr_[learnable_param_id] = true;
        params_lr_[learnable_param_id] = param_spec->lr_mult();
      }
    }
    if (param_spec->has_decay_mult()) {
      if (has_params_decay_[learnable_param_id]) {
        CHECK_EQ(param_spec->decay_mult(),
                 params_weight_decay_[learnable_param_id])
            << "Shared param '" << param_name << "' has mismatched decay_mult.";
      } else {
        has_params_decay_[learnable_param_id] = true;
        params_weight_decay_[learnable_param_id] = param_spec->decay_mult();
      }
    }
  }
}

#ifdef USE_MLU

template <typename Dtype>
void Net<Dtype>::RecalculateWeightsInt8Info(shared_ptr<Blob<Dtype>> weights_blob,
    const LayerParameter& param) {
  BlobDataType blob_dtype = get_quantized_info(*weights_blob, param, "common", DT_INT8);
  weights_blob->set_mlu_position(blob_dtype.position(0));
  weights_blob->set_mlu_scale(blob_dtype.scale(0));
  LOG(INFO) << param.name() << " weights of int8 info:";
  LOG(INFO) << "position: " << blob_dtype.position(0);
  LOG(INFO) << "scale: " << blob_dtype.scale(0);
  LOG(INFO) << "------------------------------------------------------";
}

template <typename Dtype>
Dtype Net<Dtype>::ForwardFromTo(int start, int end) {
  // Reshape logic is pelt from Forward in layers. Now, it's handled
  // by Net here to reduce reshaping which could be unneeded.
  Reshape();

  switch (Caffe::mode()) {
    case Caffe::CPU:
    case Caffe::MLU:
      return ForwardFromTo_default(start, end);
      break;
    case Caffe::MFUS:
      return ForwardFromTo_mfus(start, end);
      break;
    default:
      NOT_IMPLEMENTED;
      break;
  }
  return 0;
}

// caffe's original implementation of ForwardFromTo
template <typename Dtype>
Dtype Net<Dtype>::ForwardFromTo_default(int start, int end) {
  CHECK_GE(start, 0);
  CHECK_LT(end, layers_.size());
  Dtype loss = 0;
  vector<int> debug_layer_ids;
  debug_layer_ids.clear();
  for (int i = start; i <= end; ++i) {
    for (int c = 0; c < before_forward_.size(); ++c) {
      before_forward_[c]->run(i);
    }
    Dtype layer_loss = layers_[i]->Forward(bottom_vecs_[i], top_vecs_[i]);
    loss += layer_loss;

    for (int c = 0; c < after_forward_.size(); ++c) {
      after_forward_[c]->run(i);
    }
  }
  if (debug_info_) {
    for (int idx = start; idx <= end; ++idx) {
      // ForwardDebugInfo(idx);
      // print out top[0] data for the current layer if necessary
      if (top_vecs_[idx].size() == 0)
         continue;
      LOG(INFO) << "write " << idx << "th layer output to file, "
                << top_vecs_[idx][0]->shape_string() << " "
                << top_vecs_[idx][0]->num_axes();
      std::ofstream fout;
      const Dtype* data = top_vecs_[idx][0]->cpu_data();
      std::stringstream fname;
      std::string name;
      fname << "tmp_layer_" << caffe::format_int(forward_iteration, 5) << "_"
            << caffe::format_int(idx, 5) << "_"
            << layers_[idx]->layer_param().name() << "_"
            << layers_[idx]->layer_param().type();
      fname >> name;
      // replace the char '/' in layer name in prototxt
      std::string::size_type pos;
      while ((pos = name.find("/")) != std::string::npos) {
        name.replace(pos, 1, "_");
      }
      fout.open(name.c_str(), std::ios::out);
      for (int j = 0; j < top_vecs_[idx][0]->count(); ++j) {
        fout << data[j] << '\n';
      }
      fout.close();
      fname.clear();
      name.clear();
    }
  }
  return loss;
}

template <typename Dtype>
Dtype Net<Dtype>::ForwardFromTo_mfus(int start, int end) {
  CHECK_GE(start, 0);
  CHECK_LT(end, layers_.size());

  for (auto subnet : subnets_) {
    if (Caffe::reshapeMode() == Caffe::ReshapeMode::ALWAYS) {
      subnet->Reshape();
    }
    subnet->Forward(start, end);
  }

  return 0;
}

#else  // USE_MLU (false)

template <typename Dtype>
Dtype Net<Dtype>::ForwardFromTo(int start, int end) {
  CHECK_GE(start, 0);
  CHECK_LT(end, layers_.size());
  Dtype loss = 0;
  for (int i = start; i <= end; ++i) {
    for (int c = 0; c < before_forward_.size(); ++c) {
      before_forward_[c]->run(i);
    }
    Dtype layer_loss = layers_[i]->Forward(bottom_vecs_[i], top_vecs_[i]);
    loss += layer_loss;
    if (debug_info_) {
      ForwardDebugInfo(i);
    }
    for (int c = 0; c < after_forward_.size(); ++c) {
      after_forward_[c]->run(i);
    }
  }
  return loss;
}

#endif  // USE_MLU

template <typename Dtype>
Dtype Net<Dtype>::ForwardFrom(int start) {
  return ForwardFromTo(start, layers_.size() - 1);
}

template <typename Dtype>
Dtype Net<Dtype>::ForwardTo(int end) {
  return ForwardFromTo(0, end);
}

template <typename Dtype>
const vector<Blob<Dtype>*>& Net<Dtype>::Forward(Dtype* loss) {
  if (loss != NULL) {
    *loss = ForwardFromTo(0, layers_.size() - 1);
  } else {
    ForwardFromTo(0, layers_.size() - 1);
  }
  return net_output_blobs_;
}

template <typename Dtype>
const vector<Blob<Dtype>*>& Net<Dtype>::Forward(
    const vector<Blob<Dtype>*>& bottom, Dtype* loss) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
  LOG_EVERY_N(WARNING, 1000)
      << "DEPRECATED: Forward(bottom, loss) "
      << "will be removed in a future version. Use Forward(loss).";
  // Copy bottom to net bottoms
  for (int i = 0; i < bottom.size(); ++i) {
    net_input_blobs_[i]->CopyFrom(*bottom[i]);
  }
  return Forward(loss);
#pragma GCC diagnostic pop
}

template <typename Dtype>
void Net<Dtype>::BackwardFromTo(int start, int end) {
  CHECK_GE(end, 0);
  CHECK_LT(start, layers_.size());
  for (int i = start; i >= end; --i) {
    for (int c = 0; c < before_backward_.size(); ++c) {
      before_backward_[c]->run(i);
    }
    if (layer_need_backward_[i]) {
      layers_[i]->Backward(top_vecs_[i], bottom_need_backward_[i],
                           bottom_vecs_[i]);
      if (debug_info_) {
        BackwardDebugInfo(i);
      }
    }
    for (int c = 0; c < after_backward_.size(); ++c) {
      after_backward_[c]->run(i);
    }
  }
}

template <typename Dtype>
void Net<Dtype>::ForwardDebugInfo(const int layer_id) {
  for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
    const Blob<Dtype>& blob = *top_vecs_[layer_id][top_id];
    const string& blob_name = blob_names_[top_id_vecs_[layer_id][top_id]];
    const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
    LOG_IF(INFO, Caffe::root_solver()) << "    [Forward] "
                                       << "Layer " << layer_names_[layer_id]
                                       << ", top blob " << blob_name
                                       << " data: " << data_abs_val_mean;
  }
  for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
       ++param_id) {
    const Blob<Dtype>& blob = *layers_[layer_id]->blobs()[param_id];
    const int net_param_id = param_id_vecs_[layer_id][param_id];
    const string& blob_name = param_display_names_[net_param_id];
    const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
    LOG_IF(INFO, Caffe::root_solver()) << "    [Forward] "
                                       << "Layer " << layer_names_[layer_id]
                                       << ", param blob " << blob_name
                                       << " data: " << data_abs_val_mean;
  }
}

template <typename Dtype>
void Net<Dtype>::BackwardDebugInfo(const int layer_id) {
  const vector<Blob<Dtype>*>& bottom_vec = bottom_vecs_[layer_id];
  for (int bottom_id = 0; bottom_id < bottom_vec.size(); ++bottom_id) {
    if (!bottom_need_backward_[layer_id][bottom_id]) {
      continue;
    }
    const Blob<Dtype>& blob = *bottom_vec[bottom_id];
    const string& blob_name = blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
    const Dtype diff_abs_val_mean = blob.asum_diff() / blob.count();
    LOG_IF(INFO, Caffe::root_solver()) << "    [Backward] "
                                       << "Layer " << layer_names_[layer_id]
                                       << ", bottom blob " << blob_name
                                       << " diff: " << diff_abs_val_mean;
  }
  for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
       ++param_id) {
    if (!layers_[layer_id]->param_propagate_down(param_id)) {
      continue;
    }
    const Blob<Dtype>& blob = *layers_[layer_id]->blobs()[param_id];
    const Dtype diff_abs_val_mean = blob.asum_diff() / blob.count();
    LOG_IF(INFO, Caffe::root_solver()) << "    [Backward] "
                                       << "Layer " << layer_names_[layer_id]
                                       << ", param blob " << param_id
                                       << " diff: " << diff_abs_val_mean;
  }
}

template <typename Dtype>
void Net<Dtype>::UpdateDebugInfo(const int param_id) {
  const Blob<Dtype>& blob = *params_[param_id];
  const int param_owner = param_owners_[param_id];
  const string& layer_name = layer_names_[param_layer_indices_[param_id].first];
  const string& param_display_name = param_display_names_[param_id];
  const Dtype diff_abs_val_mean = blob.asum_diff() / blob.count();
  if (param_owner < 0) {
    const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
    LOG_IF(INFO, Caffe::root_solver()) << "    [Update] Layer " << layer_name
                                       << ", param " << param_display_name
                                       << " data: " << data_abs_val_mean
                                       << "; diff: " << diff_abs_val_mean;
  } else {
    const string& owner_layer_name =
        layer_names_[param_layer_indices_[param_owner].first];
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Update] Layer " << layer_name << ", param blob "
        << param_display_name << " (owned by layer " << owner_layer_name << ", "
        << "param " << param_display_names_[param_owners_[param_id]] << ")"
        << " diff: " << diff_abs_val_mean;
  }
}
#ifdef USE_MLU
template <typename Dtype>
void Net<Dtype>::AppendCpuInfo(
    const string& file, const vector<vector<string>>& input_blob_array,
    const vector<vector<string>>& output_blob_array) {
  FILE* offline_fp = fopen(file.c_str(), "ab");
  fseek(offline_fp, 0, SEEK_END);

  int16_t framework_flag = 0;
  if (fwrite(&framework_flag, sizeof(int16_t), 1, offline_fp) != 1) {
    LOG(ERROR) << "Write framework flag failed!";
  }
  fclose(offline_fp);
  SegmentInfo* segment_info =
      GenSegmentInfo(input_blob_array, output_blob_array);
  segment_info->set_flag(0);  // caffe use flag 0
  std::fstream final_file(file.c_str(),
                          std::ios::out | std::ios::app | std::ios::binary);
  segment_info->SerializeToOstream(&final_file);
  final_file.close();
}

template <typename Dtype>
SegmentInfo* Net<Dtype>::GenSegmentInfo(
    const vector<vector<string>>& input_blob_array,
    const vector<vector<string>>& output_blob_array) {
  SegmentInfo* seginfo = new SegmentInfo;
  int start = 0;
  int end = 0;
  int subnet_count = 0;
  int mlu_subnet = 0;
  for (; start < this->layers().size(); start = end, subnet_count++) {
    vector<int> pair_vec;
    pair_vec.push_back(start);
    if (start == this->layers().size() - 1) pair_vec.push_back(start);
    for (end = start + 1; end < this->layers().size(); end++) {
      if (this->layers()[start]->mfus_supported() !=
          this->layers()[end]->mfus_supported()) {
        pair_vec.push_back(end - 1);
        break;
      }
      if (end == this->layers().size() - 1) {
        pair_vec.push_back(end);
      }
    }
    if (this->layers()[pair_vec[0]]->mfus_supported()) {
      SegmentInfoUnit* tmp_seg_unit_a = seginfo->add_unit();
      for (int i = 0; i < input_blob_array[subnet_count].size(); i++) {
        tmp_seg_unit_a->add_bottom(input_blob_array[subnet_count][i]);
      }
      for (int i = 0; i < output_blob_array[subnet_count].size(); i++) {
        tmp_seg_unit_a->add_top(output_blob_array[subnet_count][i]);
      }
      std::stringstream subnet_name;
      subnet_name << "subnet" << mlu_subnet++;
      tmp_seg_unit_a->set_type(SegmentInfoUnit_TYPE_MLU);
      tmp_seg_unit_a->set_name(subnet_name.str());
      tmp_seg_unit_a->set_start(pair_vec[0]);
      tmp_seg_unit_a->set_end(pair_vec[1]);
    } else {
      SegmentInfoUnit* tmp_seg_unit_b = seginfo->add_unit();
      LayerParameter tmp_layer_param_c =
          this->layers()[pair_vec[0]]->layer_param();
      for (int i = 0; i < tmp_layer_param_c.bottom_size(); i++) {
        tmp_seg_unit_b->add_bottom(tmp_layer_param_c.bottom(i));
      }
      LayerParameter tmp_layer_param_d =
          this->layers()[pair_vec[1]]->layer_param();
      for (int i = 0; i < tmp_layer_param_d.top_size(); i++) {
        tmp_seg_unit_b->add_top(tmp_layer_param_d.top(i));
      }
      tmp_seg_unit_b->set_name(tmp_layer_param_c.name());
      tmp_seg_unit_b->set_type(SegmentInfoUnit_TYPE_CPU);
      tmp_seg_unit_b->set_start(pair_vec[0]);
      tmp_seg_unit_b->set_end(pair_vec[1]);
    }
  }
  NetParameter* net_param_proto = new NetParameter;
  net_param_proto->Clear();
  *net_param_proto = this->net_param_without_weights();
  seginfo->set_allocated_net_proto(net_param_proto);
  NetParameter* net_param_weights = new NetParameter;
  net_param_weights->Clear();
  net_param_weights->set_name(this->name());
  for (int i = 0; i < this->layers().size(); ++i) {
    if (!(this->layers()[i]->mfus_supported())) {
      LayerParameter* layer_param = net_param_weights->add_layer();
      if (layer_param == NULL) LOG(INFO) << "layer_param is NULL";
      this->layers()[i]->ToProto(layer_param, false);
    }
  }
  seginfo->set_allocated_net_weights(net_param_weights);
  return seginfo;
}

template <typename Dtype>
void Net<Dtype>::OfflineRunInit(const SegmentInfo& seg_info) {
  cnrtInit(0);
  cnrtDev_t dev;
  cnrtGetDeviceHandle(&dev, 0);
  cnrtSetCurrentDevice(dev);
  for (int i = 0; i < seg_info.unit_size(); i++) {
    SegmentInfoUnit unit_info = seg_info.unit(i);
    const NetParameter& net_param = this->net_param_without_weights();
    for (int i = 0; i < net_param.layer_size(); i++) {
      LayerParameter layer_param = net_param.layer(i);
      const vector<Blob<Dtype>*>& tops = this->top_vecs()[i];
      const vector<Blob<Dtype>*>& bottoms = this->bottom_vecs()[i];
      for (int j = 0; j < layer_param.bottom_size(); j++) {
        if (name_to_data_.find(layer_param.bottom(j)) == name_to_data_.end()) {
          name_to_data_[layer_param.bottom(j)] = bottoms[j];
        }
      }
      for (int j = 0; j < layer_param.top_size(); j++) {
        if (name_to_data_.find(layer_param.top(j)) == name_to_data_.end()) {
          name_to_data_[layer_param.top(j)] = tops[j];
        }
      }
    }
  }
  offline_init_flag_ = true;
}

template <typename Dtype>
void Net<Dtype>::OfflineMluSubnetRun(const cnrtModel_t& model,
                                     const cnrtDataType_t& dtype,
                                     const SegmentInfoUnit& unit_info,
                                     void** cpuData) {
  void** inputMluPtrS;
  void** outputMluPtrS;
  void** inputCpuPtrS;
  void** outputCpuPtrS;
  int input_num, output_num;
  int64_t *inputSizeArray;
  int64_t *outputSizeArray;
  string func_name = unit_info.name();
  if (name_to_cnrt_func_.find(func_name) == name_to_cnrt_func_.end()) {
    cnrtFunction_t func;
    cnrtCreateFunction(&func);
    cnrtExtractFunction(&func, model, func_name.c_str());
    name_to_cnrt_func_[func_name] = func;
  }

  if (name_to_cnrt_queue_.find(func_name) == name_to_cnrt_queue_.end()) {
    cnrtQueue_t cnrt_queue;
    cnrtCreateQueue(&cnrt_queue);
    name_to_cnrt_queue_[func_name] = cnrt_queue;
  }

  inputCpuPtrS = reinterpret_cast<void**>(malloc(sizeof(void**) * input_num));
  outputCpuPtrS = reinterpret_cast<void**>(malloc(sizeof(void**) * output_num));

  CNRT_CHECK(cnrtGetInputDataSize(&inputSizeArray, &input_num,
                                  name_to_cnrt_func_[func_name]));
  CNRT_CHECK(cnrtGetOutputDataSize(&outputSizeArray, &output_num,
                                   name_to_cnrt_func_[func_name]));
  for (int i = 0; i < input_num; i++) {
    Blob<Dtype>* blob = name_to_data_.at(unit_info.bottom(i));
    if ("subnet0" == func_name) {
      caffe_copy(inputSizeArray[i], reinterpret_cast<Dtype*>(cpuData[0]),
                 blob->mutable_cpu_data());
    }
    inputCpuPtrS[i] = reinterpret_cast<void*>(blob->mutable_cpu_data());
  }

  for (int i = 0; i < output_num; i++) {
    Blob<Dtype>* blob = name_to_data_.at(unit_info.top(i));
    outputCpuPtrS[i] = reinterpret_cast<void*>(blob->mutable_cpu_data());
  }

  void** param = reinterpret_cast<void**>(
      malloc(sizeof(void**) * (input_num + output_num)));

  for (int i = 0; i < input_num; i++) {
    cnrtMalloc(&(inputMluPtrS[i]), inputSizeArray[i]);
    param[i] = inputMluPtrS[i];
  }
  for (int i = 0; i < output_num; i++) {
    cnrtMalloc(&(outputMluPtrS[i]), outputSizeArray[i]);
    param[input_num + i] = outputMluPtrS[i];
  }
  for (int i = 0; i < input_num; i++) {
    CNRT_CHECK(cnrtMemcpy(inputMluPtrS[i], inputCpuPtrS[i],
          inputSizeArray[i], CNRT_MEM_TRANS_DIR_HOST2DEV));
  }
  //  create start_event and end_event
  cnrtNotifier_t notifierBeginning, notifierEnd;
  cnrtCreateNotifier(&notifierBeginning);
  cnrtCreateNotifier(&notifierEnd);
  float event_time_used;
  cnrtPlaceNotifier(notifierBeginning, name_to_cnrt_queue_.at(func_name));
  cnrtRuntimeContext_t rt_ctx;
  if (cnrtCreateRuntimeContext(&rt_ctx, name_to_cnrt_func_[func_name],
                               nullptr) != CNRT_RET_SUCCESS) {
    LOG(FATAL)<< "Failed to create runtime context";
  }
  cnrtInitRuntimeContext(rt_ctx, NULL);
  CNRT_CHECK(cnrtInvokeRuntimeContext(rt_ctx, param,
                     name_to_cnrt_queue_.at(func_name), nullptr));
  cnrtPlaceNotifier(notifierEnd, name_to_cnrt_queue_.at(func_name));
  if (cnrtSyncQueue(name_to_cnrt_queue_.at(func_name)) == CNRT_RET_SUCCESS) {
    cnrtNotifierDuration(notifierBeginning, notifierEnd, &event_time_used);
    LOG(INFO) << " execution time: " << event_time_used << std::endl;
  } else {
    LOG(ERROR) << " SyncQueue error " << std::endl;
  }
  for (int i = 0; i < output_num; i++) {
    CNRT_CHECK(cnrtMemcpy(outputCpuPtrS[i], outputMluPtrS[i],
          outputSizeArray[i], CNRT_MEM_TRANS_DIR_DEV2HOST));
  }

  free(inputCpuPtrS);
  free(outputCpuPtrS);
  free(param);
  cnrtFreeArray(inputMluPtrS, input_num);
  cnrtFreeArray(outputMluPtrS, output_num);
}

template <typename Dtype>
void Net<Dtype>::OfflineCpuSubnetRun(const SegmentInfoUnit& unit_info) {
  int layer_start = unit_info.start();
  int layer_end = unit_info.end();
  this->ForwardFromTo(layer_start, layer_end);
}

template <typename Dtype>
void Net<Dtype>::OfflineNetRun(const SegmentInfo& seg_info,
                               const cnrtModel_t& model,
                               const cnrtDataType_t& dtype,
                               void** cpuData) {
  if (!offline_init_flag_) OfflineRunInit(seg_info);
  for (int i = 0; i < seg_info.unit_size(); i++) {
    const SegmentInfoUnit& seg_unit = seg_info.unit(i);
    switch (seg_unit.type()) {
      case SegmentInfoUnit_TYPE_CPU:
        OfflineCpuSubnetRun(seg_unit);
        break;
      case SegmentInfoUnit_TYPE_MLU:
        OfflineMluSubnetRun(model, dtype, seg_unit, cpuData);
        break;
    }
  }
}

template <typename Dtype>
void Net<Dtype>::OfflineDestroy() {
  for (auto& m : name_to_cnrt_func_) {
    cnrtDestroyFunction(m.second);
  }
  for (auto& m : name_to_cnrt_queue_) {
    cnrtDestroyQueue(m.second);
  }
  name_to_data_.clear();
  cnrtDestroy();
  offline_init_flag_ = false;
}

#endif

template <typename Dtype>
void Net<Dtype>::ShareTrainedLayersWith(const Net* other) {
  int num_source_layers = other->layers().size();
  for (int i = 0; i < num_source_layers; ++i) {
    Layer<Dtype>* source_layer = other->layers()[i].get();
    const string& source_layer_name = other->layer_names()[i];
    int target_layer_id = 0;
    while (target_layer_id != layer_names_.size() &&
           layer_names_[target_layer_id] != source_layer_name) {
      ++target_layer_id;
    }
    if (target_layer_id == layer_names_.size()) {
      LOG(INFO) << "Ignoring source layer " << source_layer_name;
      continue;
    }
    DLOG(INFO) << "Copying source layer " << source_layer_name;
    vector<shared_ptr<Blob<Dtype>>>& target_blobs =
        layers_[target_layer_id]->blobs();
    CHECK_EQ(target_blobs.size(), source_layer->blobs().size())
        << "Incompatible number of blobs for layer " << source_layer_name;
    for (int j = 0; j < target_blobs.size(); ++j) {
      Blob<Dtype>* source_blob = source_layer->blobs()[j].get();
      CHECK(target_blobs[j]->shape() == source_blob->shape())
          << "Cannot share param " << j << " weights from layer '"
          << source_layer_name << "'; shape mismatch.  Source param shape is "
          << source_blob->shape_string() << "; target param shape is "
          << target_blobs[j]->shape_string();
      target_blobs[j]->ShareData(*source_blob);
    }
  }
}

template <typename Dtype>
void Net<Dtype>::BackwardFrom(int start) {
  BackwardFromTo(start, 0);
}

template <typename Dtype>
void Net<Dtype>::BackwardTo(int end) {
  BackwardFromTo(layers_.size() - 1, end);
}

template <typename Dtype>
void Net<Dtype>::Backward() {
  BackwardFromTo(layers_.size() - 1, 0);
  if (debug_info_) {
    Dtype asum_data = 0, asum_diff = 0, sumsq_data = 0, sumsq_diff = 0;
    for (int i = 0; i < learnable_params_.size(); ++i) {
      asum_data += learnable_params_[i]->asum_data();
      asum_diff += learnable_params_[i]->asum_diff();
      sumsq_data += learnable_params_[i]->sumsq_data();
      sumsq_diff += learnable_params_[i]->sumsq_diff();
    }
    const Dtype l2norm_data = std::sqrt(sumsq_data);
    const Dtype l2norm_diff = std::sqrt(sumsq_diff);
    LOG(ERROR) << "    [Backward] All net params (data, diff): "
               << "L1 norm = (" << asum_data << ", " << asum_diff << "); "
               << "L2 norm = (" << l2norm_data << ", " << l2norm_diff << ")";
  }
}

template <typename Dtype>
void Net<Dtype>::Reshape() {
#ifdef USE_MLU
  if (reshape_helper_->needReshape()) {
    LOG(INFO) << "reshaping...";
    switch (Caffe::mode()) {
      case Caffe::CPU:
      case Caffe::MLU:
        for (int i = 0; i < layers_.size(); ++i) {
          layers_[i]->Reshape_dispatch(bottom_vecs_[i], top_vecs_[i]);
        }
        break;
      case Caffe::MFUS:
        for (auto subnet : subnets_) {
          subnet->Reshape();
        }
        break;
      default:
        NOT_IMPLEMENTED;
        break;
    }
  }
#else   // USE_MLU (false)
  for (int i = 0; i < layers_.size(); ++i) {
    layers_[i]->Reshape(bottom_vecs_[i], top_vecs_[i]);
  }
#endif  // USE_MLU
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFrom(const NetParameter& param) {
  int num_source_layers = param.layer_size();
  vector<string> tmp_layers;
  for (int i = 0; i < layer_names_.size(); i++) {
    if (layers_[i]->layer_param().type() == "Convolution" ||
        layers_[i]->layer_param().type() == "InnerProduct")
      tmp_layers.push_back(layers_[i]->layer_param().name());
  }
  for (int i = 0; i < num_source_layers; ++i) {
    const LayerParameter& source_layer = param.layer(i);
    const string& source_layer_name = source_layer.name();
    int target_layer_id = 0;
    while (target_layer_id != layer_names_.size() &&
           layer_names_[target_layer_id] != source_layer_name) {
      ++target_layer_id;
    }
    if (target_layer_id == layer_names_.size()) {
      LOG(INFO) << "Ignoring source layer " << source_layer_name;
      continue;
    }
    DLOG(INFO) << "Copying source layer " << source_layer_name;
    for (auto it = tmp_layers.begin(); it != tmp_layers.end(); it++) {
      if (*it == source_layer_name) {
        tmp_layers.erase(it);
        break;
      }
    }
    vector<shared_ptr<Blob<Dtype>>>& target_blobs =
        layers_[target_layer_id]->blobs();
    CHECK_EQ(target_blobs.size(), source_layer.blobs_size())
        << "Incompatible number of blobs for layer " << source_layer_name;
    for (int j = 0; j < target_blobs.size(); ++j) {
      if (!(target_blobs[j]->CountEquals(source_layer.blobs(j)) ||
          target_blobs[j]->CountGE3(source_layer.blobs(j)))) {
        Blob<Dtype> source_blob;
        const bool kReshape = true;
        source_blob.FromProto(source_layer.blobs(j), kReshape);
        LOG(FATAL)
            << "Cannot copy param " << j << " weights from layer '"
            << source_layer_name << "'; shape mismatch.  Source param shape is "
            << source_blob.shape_string() << "; target param shape is "
            << target_blobs[j]->shape_string() << ". "
            << "To learn this layer's parameters from scratch rather than "
            << "copying from a saved net, rename the layer.";
      }
      const bool kReshape = false;
      target_blobs[j]->FromProto(source_layer.blobs(j), kReshape);
    }
#ifdef USE_MLU
    // recalculate weights int8 info for int8 ConvBnScale optimization
    if (opt_level_ == -2 &&
        layers_[target_layer_id]->layer_param().blobs_dtype().size() &&
        target_blobs.size() > 0) {
      RecalculateWeightsInt8Info(target_blobs[0],
          layers_[target_layer_id]->layer_param());
    }
#endif
  }
  if (tmp_layers.size()) {
    for (auto it : tmp_layers) {
      LOG(ERROR) << "weights of " << it << " are not initialized from caffemodel";
    }
  }
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFrom(const string trained_filename) {
#ifdef USE_HDF5
  if (H5Fis_hdf5(trained_filename.c_str())) {
    CopyTrainedLayersFromHDF5(trained_filename);
  } else {
#else
  if (1) {
#endif
    CopyTrainedLayersFromBinaryProto(trained_filename);
  }
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFromBinaryProto(
    const string trained_filename) {
  NetParameter param;
  ReadNetParamsFromBinaryFileOrDie(trained_filename, &param
#ifdef USE_MLU
      , opt_level_, &net_param_without_weights_
#endif
      );  //NOLINT
  CopyTrainedLayersFrom(param);
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFrom(void* buffer, int buffer_size) {
  CopyTrainedLayersFromBinaryProto(buffer, buffer_size);
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFromBinaryProto(void* buffer,
                                                  int buffer_size) {
  NetParameter param;
  ReadNetParamsFromBinaryMemOrDie(buffer, buffer_size, &param);
  CopyTrainedLayersFrom(param);
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFromHDF5(const string trained_filename) {
#ifdef USE_HDF5
  hid_t file_hid =
      H5Fopen(trained_filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  CHECK_GE(file_hid, 0) << "Couldn't open " << trained_filename;
  hid_t data_hid = H5Gopen2(file_hid, "data", H5P_DEFAULT);
  CHECK_GE(data_hid, 0) << "Error reading weights from " << trained_filename;
  int num_layers = hdf5_get_num_links(data_hid);
  for (int i = 0; i < num_layers; ++i) {
    string source_layer_name = hdf5_get_name_by_idx(data_hid, i);
    if (!layer_names_index_.count(source_layer_name)) {
      LOG(INFO) << "Ignoring source layer " << source_layer_name;
      continue;
    }
    int target_layer_id = layer_names_index_[source_layer_name];
    DLOG(INFO) << "Copying source layer " << source_layer_name;
    vector<shared_ptr<Blob<Dtype>>>& target_blobs =
        layers_[target_layer_id]->blobs();
    hid_t layer_hid =
        H5Gopen2(data_hid, source_layer_name.c_str(), H5P_DEFAULT);
    CHECK_GE(layer_hid, 0) << "Error reading weights from " << trained_filename;
    // Check that source layer doesn't have more params than target layer
    int num_source_params = hdf5_get_num_links(layer_hid);
    CHECK_LE(num_source_params, target_blobs.size())
        << "Incompatible number of blobs for layer " << source_layer_name;
    for (int j = 0; j < target_blobs.size(); ++j) {
      ostringstream oss;
      oss << j;
      string dataset_name = oss.str();
      int target_net_param_id = param_id_vecs_[target_layer_id][j];
      if (!H5Lexists(layer_hid, dataset_name.c_str(), H5P_DEFAULT)) {
        // Target param doesn't exist in source weights...
        if (param_owners_[target_net_param_id] != -1) {
          // ...but it's weight-shared in target, so that's fine.
          continue;
        } else {
          LOG(FATAL) << "Incompatible number of blobs for layer "
                     << source_layer_name;
        }
      }
      hdf5_load_nd_dataset(layer_hid, dataset_name.c_str(), 0, kMaxBlobAxes,
                           target_blobs[j].get());
    }
    H5Gclose(layer_hid);
  }
  H5Gclose(data_hid);
  H5Fclose(file_hid);
#endif
}

template <typename Dtype>
void Net<Dtype>::ToProto(NetParameter* param, bool write_diff) const {
  param->Clear();
  param->set_name(name_);
  // Add bottom and top
  DLOG(INFO) << "Serializing " << layers_.size() << " layers";
  for (int i = 0; i < layers_.size(); ++i) {
    LayerParameter* layer_param = param->add_layer();
    layers_[i]->ToProto(layer_param, write_diff);
  }
}

template <typename Dtype>
void Net<Dtype>::ToHDF5(const string& filename, bool write_diff) const {
#ifdef USE_HDF5
  hid_t file_hid =
      H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  CHECK_GE(file_hid, 0) << "Couldn't open " << filename << " to save weights.";
  hid_t data_hid =
      H5Gcreate2(file_hid, "data", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  CHECK_GE(data_hid, 0) << "Error saving weights to " << filename << ".";
  hid_t diff_hid = -1;
  if (write_diff) {
    diff_hid =
        H5Gcreate2(file_hid, "diff", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    CHECK_GE(diff_hid, 0) << "Error saving weights to " << filename << ".";
  }
  for (int layer_id = 0; layer_id < layers_.size(); ++layer_id) {
    const LayerParameter& layer_param = layers_[layer_id]->layer_param();
    string layer_name = layer_param.name();
    hid_t layer_data_hid = H5Gcreate2(data_hid, layer_name.c_str(), H5P_DEFAULT,
                                      H5P_DEFAULT, H5P_DEFAULT);
    CHECK_GE(layer_data_hid, 0) << "Error saving weights to " << filename
                                << ".";
    hid_t layer_diff_hid = -1;
    if (write_diff) {
      layer_diff_hid = H5Gcreate2(diff_hid, layer_name.c_str(), H5P_DEFAULT,
                                  H5P_DEFAULT, H5P_DEFAULT);
      CHECK_GE(layer_diff_hid, 0) << "Error saving weights to " << filename
                                  << ".";
    }
    int num_params = layers_[layer_id]->blobs().size();
    for (int param_id = 0; param_id < num_params; ++param_id) {
      ostringstream dataset_name;
      dataset_name << param_id;
      const int net_param_id = param_id_vecs_[layer_id][param_id];
      if (param_owners_[net_param_id] == -1) {
        // Only save params that own themselves
        hdf5_save_nd_dataset<Dtype>(layer_data_hid, dataset_name.str(),
                                    *params_[net_param_id]);
      }
      if (write_diff) {
        // Write diffs regardless of weight-sharing
        hdf5_save_nd_dataset<Dtype>(layer_diff_hid, dataset_name.str(),
                                    *params_[net_param_id], true);
      }
    }
    H5Gclose(layer_data_hid);
    if (write_diff) {
      H5Gclose(layer_diff_hid);
    }
  }
  H5Gclose(data_hid);
  if (write_diff) {
    H5Gclose(diff_hid);
  }
  H5Fclose(file_hid);
#else
  LOG(FATAL) << "Need USE_HDF5 support";
#endif
}

template <typename Dtype>
void Net<Dtype>::Update() {
  for (int i = 0; i < learnable_params_.size(); ++i) {
    learnable_params_[i]->Update();
  }
}

template <typename Dtype>
void Net<Dtype>::ClearParamDiffs() {
  for (int i = 0; i < learnable_params_.size(); ++i) {
    Blob<Dtype>* blob = learnable_params_[i];
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_set(blob->count(), static_cast<Dtype>(0),
                blob->mutable_cpu_diff());
      break;
    case Caffe::GPU:
#ifdef USE_CUDA
      caffe_gpu_set(blob->count(), static_cast<Dtype>(0),
                    blob->mutable_gpu_diff());
#else
        NO_GPU;
#endif
        break;
      case Caffe::MLU:
      case Caffe::MFUS:
        NOT_IMPLEMENTED;
        break;
    }
  }
}

template <typename Dtype>
void Net<Dtype>::ShareWeights() {
  for (int i = 0; i < params_.size(); ++i) {
    if (param_owners_[i] < 0) {
      continue;
    }
    params_[i]->ShareData(*params_[param_owners_[i]]);
    params_[i]->ShareDiff(*params_[param_owners_[i]]);
  }
}

template <typename Dtype>
bool Net<Dtype>::has_blob(const string& blob_name) const {
  return blob_names_index_.find(blob_name) != blob_names_index_.end();
}

template <typename Dtype>
const shared_ptr<Blob<Dtype>> Net<Dtype>::blob_by_name(
    const string& blob_name) const {
  shared_ptr<Blob<Dtype>> blob_ptr;
  if (has_blob(blob_name)) {
    blob_ptr = blobs_[blob_names_index_.find(blob_name)->second];
  } else {
    blob_ptr.reset((Blob<Dtype>*)(NULL));
    LOG(WARNING) << "Unknown blob name " << blob_name;
  }
  return blob_ptr;
}

template <typename Dtype>
bool Net<Dtype>::has_layer(const string& layer_name) const {
  return layer_names_index_.find(layer_name) != layer_names_index_.end();
}

template <typename Dtype>
const shared_ptr<Layer<Dtype>> Net<Dtype>::layer_by_name(
    const string& layer_name) const {
  shared_ptr<Layer<Dtype>> layer_ptr;
  if (has_layer(layer_name)) {
    layer_ptr = layers_[layer_names_index_.find(layer_name)->second];
  } else {
    layer_ptr.reset((Layer<Dtype>*)(NULL));
    LOG(WARNING) << "Unknown layer name " << layer_name;
  }
  return layer_ptr;
}

template <typename Dtype>
void Net<Dtype>::blob_info() const {
  vector<vector<Blob<Dtype>*>> bottom_vecs = this->bottom_vecs();
  vector<vector<Blob<Dtype>*>> top_vecs = this->top_vecs();
  for (int i = 0; i < this->layers().size(); i++) {
    LayerParameter layer_param = this->layers()[i]->layer_param();
    LOG(INFO) << "Layer name: " << layer_param.name();
    for (int j = 0; j < bottom_vecs[i].size(); j++)
      LOG(INFO) << "bottom " << j << " shape: " << bottom_vecs[i][j]->shape_string();
    for (int j = 0; j < top_vecs[i].size(); j++)
      LOG(INFO) << "top " << j << " shape: " << top_vecs[i][j]->shape_string();
  }
}

template <typename Dtype>
void Net<Dtype>::ToquantizedPrototxt(map<string, Dtype>* max_value,
                                string output_file, string mode, BaseDataType type,
                                BaseDataType top_dtype, bool use_ini, bool write ) {
  NetParameter net_param;
  net_param.set_name("default");
  auto bottom_vecs = this->bottom_vecs();
#ifdef USE_MLU
  for (int i = 0; i < this->layers().size(); i++) {
    LayerParameter* layer_param = this->layers()[i]->mutable_layer_param();
    string layer_type = layer_param->type();
    if (layer_type == "Convolution" ||
        layer_type == "Convolution3D" ||
        layer_type == "Deconvolution" ||
        layer_type == "InnerProduct" || layer_type == "LRN" ||
        layer_type == "Reorg") {
      // bottom dtype
      BlobDataType blob_dtype;
      for (int j = 0; j < bottom_vecs[i].size(); j++) {
        blob_dtype = get_quantized_info(*bottom_vecs[i][j],
            *layer_param, mode, type, false, max_value);
        if (write) {
          if (layer_param->bottom_mlu_dtype_size() > j) {
            *(layer_param->mutable_bottom_mlu_dtype(j)) = blob_dtype;
          } else {
            layer_param->add_bottom_mlu_dtype()->CopyFrom(blob_dtype);
          }
        }
      }
      // blobs dtype
      auto blobs = this->layers()[i]->blobs();
      if (blobs.size() != 0) {
        blob_dtype  =
            get_quantized_info(*blobs[0].get(), *layer_param, mode, type, true);
        if (write) {
          if (layer_param->blobs_dtype_size()) {
            *(layer_param->mutable_blobs_dtype(0)) = blob_dtype;
          } else {
            layer_param->add_blobs_dtype()->CopyFrom(blob_dtype);
          }
        }
        if (top_dtype == DT_FLOAT32)
          layer_param->set_top_mlu_dtype(DT_FLOAT32);
      }
    } else if (layer_type == "Normalize") {
      int n = bottom_vecs[i][0]->num();
      int c = bottom_vecs[i][0]->channels();
      int h = bottom_vecs[i][0]->height();
      int w = bottom_vecs[i][0]->width();
      Blob<Dtype> div_blob(n, c, h, w);
      Blob<Dtype> gemv_blob(n, 1, h, w);
      Blob<Dtype> sqr_blob(n, c, h, w);

      Dtype max = 0;
      for (int n = 0; n < sqr_blob.count(); n++) {
        if (max < std::abs(bottom_vecs[i][0]->cpu_data()[n]))
          max = std::abs(bottom_vecs[i][0]->cpu_data()[n]);
      }
      for (int n = 0; n < sqr_blob.count(); n++) {
        div_blob.mutable_cpu_data()[n] = bottom_vecs[i][0]->cpu_data()[n] / max;
      }
      BlobDataType blob_dtype = get_quantized_info(div_blob,
          *layer_param, mode, type, false, (map<string, Dtype>* const)nullptr, true);
      if (write) {
        if (layer_param->bottom_mlu_dtype_size()) {
          *(layer_param->mutable_bottom_mlu_dtype(0)) = blob_dtype;
        } else {
          layer_param->add_bottom_mlu_dtype()->CopyFrom(blob_dtype);
        }
        if (top_dtype == DT_FLOAT32)
          layer_param->set_top_mlu_dtype(DT_FLOAT32);
      }
      NormalizeParameter norm_param = layer_param->norm_param();
      if (!norm_param.across_spatial()) {
        int n = bottom_vecs[i][0]->num();
        int c = bottom_vecs[i][0]->channels();
        int h = bottom_vecs[i][0]->height();
        int w = bottom_vecs[i][0]->width();

        Blob<Dtype> sqr_blob(n, c, h, w);
        for (int n = 0; n < sqr_blob.count(); n++) {
          sqr_blob.mutable_cpu_data()[n] = div_blob.cpu_data()[n]
              * div_blob.cpu_data()[n];
        }

        Blob<Dtype> gemv_blob(n, 1, h, w);
        for (int j = 0; j < n; j++) {
          for (int k = 0; k < h; k++) {
            for (int p = 0; p < w; p++) {
              for (int q = 0; q < c; q++) {
                gemv_blob.mutable_cpu_data()[gemv_blob.offset(j, 0, k, p)] +=
                    sqr_blob.cpu_data()[sqr_blob.offset(j, q, k, p)];
              }
            }
          }
        }
        for (int n = 0; n < gemv_blob.count(); n++) {
          gemv_blob.mutable_cpu_data()[n] = pow(gemv_blob.cpu_data()[n], 0.5);
        }
        blob_dtype =
            get_quantized_info(gemv_blob, *layer_param, mode, type);
        if (write) {
          if (layer_param->bottom_mlu_dtype_size() > 1) {
            *(layer_param->mutable_bottom_mlu_dtype(1)) = blob_dtype;
          } else {
            layer_param->add_bottom_mlu_dtype()->CopyFrom(blob_dtype);
          }
        }
      }
      if (write) {
        blob_dtype.Clear();
        blob_dtype.set_type(DT_INT8);
        if (mode == "common") {
          (norm_param.across_spatial()) ?
              blob_dtype.add_position(-6):
              blob_dtype.add_position(-16);
        } else {
          blob_dtype.add_position(0);
          blob_dtype.add_scale(pow(2, static_cast<int>(10)));
        }
        if (layer_param->blobs_dtype_size())
          *(layer_param->mutable_blobs_dtype(0)) = blob_dtype;
        else
          layer_param->add_blobs_dtype()->CopyFrom(blob_dtype);
      }
    }
    if (write) {
      layer_param->clear_include();
      layer_param->clear_phase();
      net_param.add_layer()->CopyFrom(*layer_param);
    }
  }
#endif
  if (write) {
    int n, c, h, w;
    string name;
    if (net_param.layer(0).type() == "ImageData") {
      n = net_param.layer(0).image_data_param().batch_size();
      c = 3;
      name = net_param.layer(0).top(0);
      if (!net_param.layer(0).image_data_param().is_color())
        c = 1;
      h = net_param.layer(0).image_data_param().new_height();
      w = net_param.layer(0).image_data_param().new_width();
    } else if (net_param.layer(0).type() == "Data") {
      n = 1;
      c = 3;
      h = w = net_param.layer(0).transform_param().crop_size();
      name = net_param.layer(0).top(0);
    }
    if (use_ini) {
      NetParameter net_param1;
      LayerParameter* input_layer_param = net_param1.add_layer();
      input_layer_param->Clear();
      input_layer_param->set_name("data");
      input_layer_param->set_type("Input");
      input_layer_param->add_top(name);
      InputParameter* input_param = new InputParameter();
      input_param->Clear();
      BlobShape* blobshape = input_param->add_shape();
      blobshape->add_dim(n);
      blobshape->add_dim(c);
      blobshape->add_dim(h);
      blobshape->add_dim(w);
      input_layer_param->set_allocated_input_param(input_param);
      if (top_dtype == DT_FLOAT32)
        input_layer_param->set_top_mlu_dtype(DT_FLOAT32);
      for (int i = 1; i < net_param.layer_size(); i++) {
        net_param1.add_layer()->CopyFrom(net_param.layer(i));
      }
      WriteProtoToTextFile(net_param1, output_file);
    } else {
      WriteProtoToTextFile(net_param, output_file);
    }
  }
}

INSTANTIATE_CLASS(Net);

}  // namespace caffe
