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

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/signal_handler.h"

using caffe::Blob;
using caffe::Caffe;
using caffe::Layer;
using caffe::Net;
using caffe::shared_ptr;
using caffe::Solver;
using caffe::string;
using caffe::Timer;
using caffe::vector;
using std::ostringstream;

DEFINE_string(model, "", "The model definition protocol buffer text file.");
DEFINE_string(weights, "", "The weights of the given model.");

// A simple registry for caffe commands.
typedef int (*BrewFunction)();
typedef std::map<caffe::string, BrewFunction> BrewMap;
BrewMap g_brew_map;

#define RegisterBrewFunction(func)         \
  namespace {                              \
  class __Registerer_##func {              \
   public: /* NOLINT */                    \
    __Registerer_##func() {                \
      g_brew_map[#func] = &func;           \
    }                                      \
  };                                       \
  __Registerer_##func g_registerer_##func; \
  }

static BrewFunction GetBrewFunction(const caffe::string& name) {
  if (g_brew_map.count(name)) {
    return g_brew_map[name];
  } else {
    LOG(ERROR) << "Available caffe actions:";
    for (BrewMap::iterator it = g_brew_map.begin(); it != g_brew_map.end(); ++it) {
      LOG(ERROR) << "\t" << it->first;
    }
    LOG(FATAL) << "Unknown action: " << name;
    return NULL;  // not reachable, just to suppress old compiler warnings.
  }
}

// : convert float/double caffemodel to hdf5 mlu model.
int hdf5() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition";
  CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to convert";

  Net<float> caffe_net(FLAGS_model, caffe::TEST);
  Net<float> mlu_net(FLAGS_model, caffe::TEST);

  vector<shared_ptr<Layer<float>>> caffe_layers = caffe_net.layers();
  vector<shared_ptr<Layer<float>>> mlu_layers = mlu_net.layers();

  for (size_t i = 0; i != caffe_net.layers().size(); i++) {
    caffe::LayerParameter* caffe_param = caffe_layers[i]->mutable_layer_param();
    caffe_param->clear_blobs_dtype();
  }

  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);

  for (size_t i = 0; i != caffe_layers.size(); i++) {
    caffe::LayerParameter* caffe_param =
      caffe_layers[i]->mutable_layer_param();
    caffe::LayerParameter mlu_param = mlu_layers[i]->layer_param();
    caffe_param->CopyFrom(mlu_param);
  }

  string store_name = FLAGS_weights;
  store_name =
      store_name.substr(0, store_name.find(".caffemodel")) + ".cnml.caffemodel.h5";

  caffe_net.ToHDF5(store_name, false);

  return 0;
}
RegisterBrewFunction(hdf5);

// : convert float/double caffemodel to proto mlu model.
int proto() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition";
  CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to convert";

  Net<float> caffe_net(FLAGS_model, caffe::TEST);
  Net<float> mlu_net(FLAGS_model, caffe::TEST);

  vector<shared_ptr<Layer<float>>> caffe_layers = caffe_net.layers();
  vector<shared_ptr<Layer<float>>> mlu_layers = mlu_net.layers();

  for (size_t i = 0; i != caffe_net.layers().size(); i++) {
    caffe::LayerParameter* caffe_param = caffe_layers[i]->mutable_layer_param();
    caffe_param->clear_blobs_dtype();
  }

  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);

  for (size_t i = 0; i != caffe_layers.size(); i++) {
    caffe::LayerParameter* caffe_param =
      caffe_layers[i]->mutable_layer_param();
    caffe::LayerParameter mlu_param = mlu_layers[i]->layer_param();
    caffe_param->CopyFrom(mlu_param);
  }

  string mlu_store_name = FLAGS_weights;
  mlu_store_name =
      mlu_store_name.substr(0, mlu_store_name.find(".caffemodel")) + ".cnml.caffemodel";

  caffe::NetParameter net_param;
  caffe_net.ToProto(&net_param, false);
  WriteProtoToBinaryFile(net_param, mlu_store_name);

  return 0;
}
RegisterBrewFunction(proto);

int main(int argc, char** argv) {
  gflags::SetUsageMessage(
      "Command line brew\n"
      "Usage: convert_caffemodel <command> <args>\n"
      "commands:\n"
      "  proto    convert a model to proto sparse model\n"
      "  hdf5     convert a model to hdf5 sparse model\n"
      "args:\n"
      "  -model   a prototxt file\n"
      "  -weights the corresponding caffemodel file\n");
  FLAGS_alsologtostderr = 1;
  // Google flags.
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  // Google logging.
  google::InitGoogleLogging(argv[0]);
  // Provide a backtrace on segfault.
  google::InstallFailureSignalHandler();

  if (argc == 2) {
    return GetBrewFunction(caffe::string(argv[1]))();
  } else {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_caffemodel");
  }
  return 0;
}
