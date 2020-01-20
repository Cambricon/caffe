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

#include <boost/property_tree/ini_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <algorithm>
#include <cmath>
#include <iosfwd>
#include <memory>
#include <numeric>
#include <utility>
#include <iostream>  // NOLINT(readability/streams)
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include "boost/algorithm/string.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

using caffe::Blob;
using std::string;
using std::vector;

DEFINE_string(ini_file, "", "The ini_file used to show quantized information");
DEFINE_string(model, "", "The model definition protocol buffer text file.");
DEFINE_string(outputmodel, "", "The output file name of protocol buffer text file.");
DEFINE_string(
    weights,
    "",
    "Optional; the pretrained weights to initialize finetuning, "
    "separated by ','. Cannot be set simultaneously with snapshot.");
DEFINE_string(
    mode,
    "common",
    "Optional; determine which mode to generate quantized pt, "
    "common: position + scale(default); scale: only scale;"
    "int8_channel: channel quantize + scales");
DEFINE_int32(iterations, 1,
    "Optional; determine to read images iterations ");
DEFINE_string(blobs_dtype, "INT8", "Set the quantized data type."
    "The optional data types include INT8 and INT16");
DEFINE_string(top_dtype, "FLOAT16", "Set the output data type."
    "The optional data types include FLOAT16 and FLOAT32");

void CreateInputLayer(
    LayerParameter* layer_param,
    int n,
    int c,
    int h,
    int w,
    string type,
    string file,
    int crop,
    string name,
    vector<float> mean_value,
    float scale,
    bool use_firstconv) {
  layer_param->Clear();
  layer_param->set_name("data");
  if (type == "Data") {
    layer_param->set_type("Data");
  } else {
    layer_param->set_type("ImageData");
  }
  layer_param->add_top(name);
  layer_param->add_top("label");
  NetStateRule* net_rule = layer_param->add_include();
  net_rule->set_phase(TEST);
  if (type == "Data") {
    TransformationParameter* trans_param = new TransformationParameter();
    if (crop != -1) {
      trans_param->set_crop_size(crop);
    }
    if (!use_firstconv) {
      for (auto mean : mean_value) {
        trans_param->add_mean_value(mean);
      }
      trans_param->set_scale(scale);
    }
    trans_param->set_mirror(false);
    layer_param->set_allocated_transform_param(trans_param);
    DataParameter* data_param = new DataParameter();
    data_param->set_source(file);
    data_param->set_batch_size(n);
    data_param->set_backend(DataParameter::LMDB);
    layer_param->set_allocated_data_param(data_param);
  } else {
    TransformationParameter* trans_param = new TransformationParameter();
    if (!use_firstconv) {
      for (auto mean : mean_value) {
        trans_param->add_mean_value(mean);
      }
      trans_param->set_scale(scale);
    }
    trans_param->set_mirror(false);
    layer_param->set_allocated_transform_param(trans_param);
    ImageDataParameter* data_param = new ImageDataParameter();
    data_param->set_source(file);
    data_param->set_batch_size(n);
    data_param->set_new_height(h);
    data_param->set_new_width(w);
    if (c == 1) {
      data_param->set_is_color(false);
    }
    layer_param->set_allocated_image_data_param(data_param);
  }
}

vector<float> parseData(string data) {
  stringstream ss(data);
  vector<float> values;
  string item;
  while (getline(ss, item, ',')) {
    values.push_back(stof(item));
  }
  return values;
}

int absMaxGenerator(caffe::NetParameter net_param_processed,
                    const string ori_weights_path, int iteration,
                    const string save_model_path, const string mode,
                    string data_type, string output_type, bool use_ini) {
  caffe::Caffe::set_mode(caffe::Caffe::CPU);
  caffe::Net<float>* net = new caffe::Net<float>(net_param_processed);
  net->CopyTrainedLayersFrom(ori_weights_path);
  std::map<string, float> max_value;
  BaseDataType dtype;
  if (boost::iequals(data_type, "INT8")) {
    dtype = DT_INT8;
  } else if (boost::iequals(data_type, "INT16")) {
    dtype = DT_INT16;
  } else {
    LOG(FATAL) << "blobs_dtype: The specified data type is not supported.";
  }
  BaseDataType top_dtype;
  if (boost::iequals(output_type, "FLOAT16")) {
    top_dtype = DT_FLOAT16;
  } else if (boost::iequals(output_type, "FLOAT32")) {
    top_dtype = DT_FLOAT32;
  } else {
    LOG(FATAL) << "top_dtype: The specified data type is not supported.";
  }
  for (int i = 0; i < iteration; i++) {
    net->Forward();
    if (i == iteration - 1) {
      net->ToquantizedPrototxt(&max_value, save_model_path,
          mode, dtype, top_dtype, use_ini, true);
    } else {
      net->ToquantizedPrototxt(&max_value, save_model_path,
          mode, dtype, top_dtype, use_ini);
    }
  }

  LOG(INFO) << "Output file is " << save_model_path << ", iteration: " << iteration;
  return 0;
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = 1;
  gflags::SetUsageMessage(
      "command line brew\n"
      "Usage: generate_quantized_pt -ini_file convert_quantized.ini [optional]\n"
      "  optional: if specified, covers the corresponding value setted in ini_file\n"
      "  -model net.prototxt"
      "  -weights net.caffemodel"
      "  -outputmodel new_net.prototxt");
  if (argc == 1) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/generate_quantized_pt");
    return 1;
  }
  // Google flags.
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  // Google logging.
  google::InitGoogleLogging(argv[0]);
  // Provide a backtrace on segfault.
  google::InstallFailureSignalHandler();

  string mode = FLAGS_mode;
  string ori_model_path, save_model_path, ori_weights_path;
  caffe::NetParameter net_param, net_param1;
  int iterations = FLAGS_iterations;
  bool use_ini = false;

  if (FLAGS_ini_file.size()) {
    LOG(INFO) << "Use CPU.";
    LOG(INFO) << "ini_file path : " << FLAGS_ini_file;
    use_ini = true;

    // first write ini_file, and then read ini file
    boost::property_tree::ptree pt;
    boost::property_tree::ini_parser::read_ini(FLAGS_ini_file, pt);

    if (FLAGS_model.size() > 0) {
      pt.put<string>("model.original_models_path", FLAGS_model);
    }
    if (FLAGS_weights.size() > 0) {
      pt.put<string>("weights.original_weights_path", FLAGS_weights);
    }
    if (FLAGS_outputmodel.size() > 0) {
      pt.put<string>("model.save_model_path", FLAGS_outputmodel);
    }
    ori_model_path = pt.get<string>("model.original_models_path");
    save_model_path = pt.get<string>("model.save_model_path");
    boost::optional<string> img_folder_path =
        pt.get_optional<string>("data.images_folder_path");
    boost::optional<string> img_db_path = pt.get_optional<string>("data.images_db_path");
    boost::optional<int> used_img_num = pt.get_optional<int>("data.used_images_num");
    if (used_img_num) {
      iterations = used_img_num.get();
    }
    ori_weights_path = pt.get<string>("weights.original_weights_path");
    boost::optional<string> mean = pt.get_optional<string>("preprocess.mean");
    vector<float> mean_value;
    if (!mean) {
      mean_value = parseData("0, 0, 0");
    } else {
      mean_value = parseData(mean.get());
    }
    boost::optional<float> std = pt.get_optional<float>("preprocess.std");
    float std_;
    if (!std)
      std_ = 1.0;
    else
      std_ = std.get();
    boost::optional<string> scale_value_tmp = pt.get_optional<string>("preprocess.scale");
    vector<float> scale_value;
    if (!scale_value_tmp) {
      scale_value = parseData("-1, -1");
    } else {
      scale_value = parseData(scale_value_tmp.get());
    }
    boost::optional<string> crop_value = pt.get_optional<string>("preprocess.crop");
    vector<float> crop;
    if (!crop_value) {
      crop = parseData("-1, -1");
    } else {
      crop = parseData(crop_value.get());
    }
    boost::optional<string> op_list = pt.get_optional<string>("config.op_list");
    string use_firstconv =
        pt.get<string>("config.use_firstconv");
    bool use_firstconv_ = true;
    std::istringstream(use_firstconv) >> use_firstconv_;

    // read model
    ReadProtoFromTextFile(ori_model_path, &net_param);
    int n = 1, c = 3, h = scale_value[0], w = scale_value[1];
    string name = "data";
    if (net_param.input_dim_size() > 0) {
      name = net_param.input(0);
      n = net_param.input_dim(0);
      c = net_param.input_dim(1);
      h = net_param.input_dim(2);
      w = net_param.input_dim(3);
    } else if (net_param.input_shape_size() > 0) {
      name = net_param.input(0);
      n = net_param.input_shape(0).dim(0);
      c = net_param.input_shape(0).dim(1);
      h = net_param.input_shape(0).dim(2);
      w = net_param.input_shape(0).dim(3);
    } else if (net_param.layer(0).type() == "Input") {
      name = net_param.layer(0).top(0);
      auto shape = net_param.layer(0).input_param().shape(0);
      n = shape.dim(0);
      c = shape.dim(1);
      h = shape.dim(2);
      w = shape.dim(3);
    } else if (net_param.layer(0).type() == "Data") {
      name = net_param.layer(0).top(0);
      h = w = net_param.layer(0).transform_param().crop_size();
    } else if (net_param.layer(0).type() == "ImageData") {
      name = net_param.layer(0).top(0);
      n = net_param.layer(0).image_data_param().batch_size();
      if (!net_param.layer(0).image_data_param().is_color()) {
        c = 1;
      }
      h = net_param.layer(0).image_data_param().new_height();
      w = net_param.layer(0).image_data_param().new_width();
    }
    if (h != scale_value[0] || w != scale_value[1]) {
      h = scale_value[0];
      w = scale_value[1];
    }
    if (crop.size() == 2) {
      if (crop[0] != crop[1]) {
        crop[0] = crop[1] = -1;
      }
    }

    // substitute net_param
    LayerParameter* layer_param = net_param1.add_layer();
    if (!img_db_path) {
      CreateInputLayer(
          layer_param,
          n,
          c,
          h,
          w,
          "ImageData",
          img_folder_path.get(),
          crop[0],
          name,
          mean_value,
          std_,
          use_firstconv_);
    } else {
      CreateInputLayer(layer_param,
          n,
          c,
          h,
          w,
          "Data",
          img_db_path.get(),
          crop[0],
          name,
          mean_value,
          std_,
          use_firstconv_);
    }
    int input_layer_num = 1;
    if (net_param.input_dim_size() > 0 || net_param.input_shape_size() > 0) {
      if (net_param.input_size() == 2) {
        LayerParameter* input_layer_param = net_param1.add_layer();
        input_layer_param->Clear();
        input_layer_param->set_name(net_param.input(1));
        input_layer_param->set_type("Input");
        input_layer_param->add_top(net_param.input(1));
        InputParameter* input_param1 = new InputParameter();
        input_param1->Clear();
        BlobShape* blobshape = input_param1->add_shape();
        blobshape->add_dim(net_param.input_shape(1).dim(0));
        blobshape->add_dim(net_param.input_shape(1).dim(1));
        input_layer_param->set_allocated_input_param(input_param1);
        input_layer_num++;
      }
      for (int i = 0; i < net_param.layer_size(); i++) {
        net_param1.add_layer()->CopyFrom(net_param.layer(i));
      }
    } else {
      for (int i = 1; i < net_param.layer_size(); i++) {
        net_param1.add_layer()->CopyFrom(net_param.layer(i));
      }
    }
    if (use_firstconv_) {
      auto param = net_param1.mutable_layer(input_layer_num)->mutable_convolution_param();
      auto param1 = net_param1.layer(input_layer_num).convolution_param();
      if (param1.has_mean_file()) {
        param->clear_mean_file();
      } else if (param1.mean_value_size() > 0) {
        param->clear_mean_value();
      }
      for (auto mean : mean_value) {
        param->add_mean_value(mean);
      }
      param->set_std(std_);
    }
  } else {
    ori_model_path = FLAGS_model;
    ori_weights_path = FLAGS_weights;
    save_model_path = FLAGS_outputmodel;
    ReadProtoFromTextFile(ori_model_path, &net_param1);
    if (net_param1.layer(0).type() == "ImageData") {
      iterations = net_param1.layer(0).image_data_param().iterations();
    }
  }
  int status = absMaxGenerator(net_param1, ori_weights_path, iterations, save_model_path,
        mode, FLAGS_blobs_dtype, FLAGS_top_dtype, use_ini);
  return status;
}
