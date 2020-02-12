/*
All modification made by Cambricon Corporation: © 2019 Cambricon Corporation
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

#include <iostream>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

using caffe::BaseDataType;
using caffe::BlobDataType;
using caffe::NetParameter;
using caffe::ReadProtoFromBinaryFile;
using std::set;
using std::string;

using namespace caffe;

void update_faster_rcnn(string model_file, NetParameter net_param) {
  for (int i = 0; i < net_param.input_size(); i++) {
    if (net_param.input(i) == "data"){
      net_param.mutable_input_shape(i)->set_dim(2, 600);
      net_param.mutable_input_shape(i)->set_dim(3, 800);
    }
  }

  for (int i = 0; i < net_param.layer_size(); i++) {
    if (net_param.layer(i).type() == "Python") {
      net_param.mutable_layer(i)->clear_python_param();
      net_param.mutable_layer(i)->set_type("Proposal");
      auto layer_param = net_param.mutable_layer(i)->mutable_proposal_param();
      layer_param->set_stride(16);
      layer_param->set_im_min_w(16);
      layer_param->set_im_min_h(16);
      layer_param->set_top_num(6000);
      layer_param->set_nms_thresh(0.7);
      layer_param->set_nms_num(300);
      layer_param->set_anchor_num(9);
    }
  }

  LayerParameter* image_detect_param = net_param.add_layer();
  image_detect_param->Clear();
  image_detect_param->set_name("image_detect_out");
  image_detect_param->set_type("ImageDetect");
  image_detect_param->add_bottom("bbox_pred");
  image_detect_param->add_bottom("cls_prob");
  image_detect_param->add_bottom("rois");
  image_detect_param->add_top("bbox_output");
  image_detect_param->mutable_image_detect_param()->set_num_class(21);
  image_detect_param->mutable_image_detect_param()->set_im_h(600);
  image_detect_param->mutable_image_detect_param()->set_im_w(800);
  image_detect_param->mutable_image_detect_param()->set_scale(1);
  image_detect_param->mutable_image_detect_param()->set_nms_thresh(0.3);
  image_detect_param->mutable_image_detect_param()->set_score_thresh(0.05);
  // Save new format prototxt.
  WriteProtoToTextFile(net_param, model_file + "_new");
  LOG(INFO) << "SUCCESS! Output file is " << model_file + "_new";
}

void update_rfcn(string model_file, NetParameter net_param) {
  for (int i = 0; i < net_param.input_size(); i++) {
    if (net_param.input(i) == "data"){
      net_param.mutable_input_shape(i)->set_dim(2, 600);
      net_param.mutable_input_shape(i)->set_dim(3, 800);
    }
  }
  for (int i = 0; i < net_param.layer_size(); i++) {
    if (net_param.layer(i).type() == "Python") {
      net_param.mutable_layer(i)->clear_python_param();
      net_param.mutable_layer(i)->set_type("Proposal");
      auto layer_param = net_param.mutable_layer(i)->mutable_proposal_param();
      layer_param->set_stride(16);
      layer_param->set_im_min_w(16);
      layer_param->set_im_min_h(16);
      layer_param->set_top_num(6000);
      layer_param->set_nms_thresh(0.7);
      layer_param->set_nms_num(304);
      layer_param->set_anchor_num(9);
    }
  }
  // Save new format prototxt.
  WriteProtoToTextFile(net_param, model_file + "_new");
  LOG(INFO) << "SUCCESS! Output file is " << model_file + "_new";
}

void update_yolov3(string model_file, NetParameter net_param) {
  for (int i = 0; i < net_param.input_size(); i++) {
    if (net_param.input(i) == "data"){
      net_param.mutable_input_shape(i)->set_dim(2, 416);
      net_param.mutable_input_shape(i)->set_dim(3, 416);
    }
  }
  string bottom_name[3];
  int index = 0;
  for (int i = net_param.layer_size()- 1; i >= 0; i--) {
    if (index >= 3) break;
    if (net_param.layer(i).type() == "Convolution") {
      if (net_param.layer(i).convolution_param().num_output() == 255) {
        bottom_name[index] = net_param.layer(i).top(0);
        index++;
      }
    }
  }
  LayerParameter* yolov3_param = net_param.add_layer();
  yolov3_param->Clear();
  yolov3_param->set_name("yolo-layer");
  yolov3_param->set_type("Yolov3Detection");
  yolov3_param->add_bottom(bottom_name[2]);
  yolov3_param->add_bottom(bottom_name[1]);
  yolov3_param->add_bottom(bottom_name[0]);
  yolov3_param->add_top("yolo_1");
  yolov3_param->mutable_yolov3_param()->set_im_w(416);
  yolov3_param->mutable_yolov3_param()->set_im_h(416);
  yolov3_param->mutable_yolov3_param()->set_num_box(1024);
  yolov3_param->mutable_yolov3_param()->set_confidence_threshold(0.005);
  yolov3_param->mutable_yolov3_param()->set_nms_threshold(0.45);
  float biases[] = {116, 90, 156, 198, 373, 326,
                    30, 61, 62, 45, 59, 119, 10,
                    13, 16, 30, 33, 23};
  for (int i = 0; i < 18; i++) {
    yolov3_param->mutable_yolov3_param()->add_biases(biases[i]);
  }
  // Save new format prototxt.
  WriteProtoToTextFile(net_param, model_file + "_new");
  LOG(INFO) << "SUCCESS! Output file is " << model_file + "_new";
}

void update_yolov2(string model_file, NetParameter net_param) {
  float biases[] = {1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892,
                      9.47112, 4.84053, 11.2364, 10.0071};
  if (net_param.layer(net_param.layer_size()-1).type() == "Region") {
    int anchor_size =
      net_param.layer(net_param.layer_size()-1).region_param().anchors_size();
    float region_biase[anchor_size];
    for(int i = 0; i < anchor_size; i++) {
      region_biase[i] =
        net_param.layer(net_param.layer_size()-1).region_param().anchors(i);
    }
    net_param.mutable_layer(net_param.layer_size()-1)->clear_region_param();
    net_param.mutable_layer(net_param.layer_size()-1)->set_type("DetectionOut");
    auto layer_param = net_param.mutable_layer(net_param.layer_size()-1)->mutable_detection_out_param();
    for (int i = 0; i < anchor_size; i++) {
      layer_param->add_biases(region_biase[i]);
    }
    layer_param->set_side(13);
    layer_param->set_num_classes(20);
    layer_param->set_num_box(5);
    layer_param->set_confidence_threshold(0.005);
    layer_param->set_nms_threshold(0.45);
  } else {
    string name = net_param.layer(net_param.layer_size()-1).top(0);
    LayerParameter* yolov2_param = net_param.add_layer();
    yolov2_param->Clear();
    yolov2_param->set_name("detection_out");
    yolov2_param->set_type("DetectionOut");
    yolov2_param->add_bottom(name);
    yolov2_param->add_top("detection_out");
    yolov2_param->mutable_detection_out_param()->set_confidence_threshold(0.005);
    yolov2_param->mutable_detection_out_param()->set_nms_threshold(0.45);
    yolov2_param->mutable_detection_out_param()->set_side(13);
    yolov2_param->mutable_detection_out_param()->set_num_classes(20);
    yolov2_param->mutable_detection_out_param()->set_num_box(5);
    for (int i = 0; i < 10; i++) {
      yolov2_param->mutable_detection_out_param()->add_biases(biases[i]);
    }
  }
  // Save new format prototxt.
  WriteProtoToTextFile(net_param, model_file + "_new");
  LOG(INFO) << "SUCCESS! Output file is " << model_file + "_new";
}
void update_mobilenet(string model_file, NetParameter net_param) {
  for (int i = 0; i < net_param.layer_size(); i++) {
    if (net_param.layer(i).type() == "Convolution") {
      auto layer_param = net_param.mutable_layer(i)->mutable_convolution_param();
      if (layer_param->has_engine()) {
        layer_param->clear_engine();
      }
    }
  }
  // Save new format prototxt.
  WriteProtoToTextFile(net_param, model_file + "_new");
  LOG(INFO) << "SUCCESS! Output file is " << model_file + "_new";
}

void update_c3d(string model_file, NetParameter net_param) {
  if (net_param.layer(0).type() != "VideoData")
    LOG(FATAL) << "C3D model transformation requires the input layer to be the videodata layer.";
  int n = 1, c = 3;
  int h, w, d;
  float mean;
  string name;
  h = w = net_param.layer(0).transform_param().crop_size();
  mean = net_param.layer(0).transform_param().mean_value(0);
  name = net_param.layer(0).top(0);
  d = net_param.layer(0).video_data_param().new_length();
  NetParameter new_param;
  LayerParameter* input_layer_param = new_param.add_layer();
  input_layer_param->Clear();
  input_layer_param->set_name("data");
  input_layer_param->set_type("Input");
  input_layer_param->add_top(name);
  InputParameter* input_param = new InputParameter();
  input_param->Clear();
  BlobShape* blobshape = input_param->add_shape();
  blobshape->add_dim(n);
  blobshape->add_dim(c);
  blobshape->add_dim(d);
  blobshape->add_dim(h);
  blobshape->add_dim(w);
  input_layer_param->set_allocated_input_param(input_param);
  for (int i = 1; i < net_param.layer_size(); i++) {
    if (net_param.layer(i).type() == "Accuracy" ||
        net_param.layer(i).type() == "SoftmaxWithLoss") continue;
    new_param.add_layer()->CopyFrom(net_param.layer(i));
  }
  auto param = new_param.mutable_layer(1)->mutable_convolution3d_param();
  if (param->has_mean_file()) {
    param->clear_mean_file();
  } else if (param->mean_value_size() > 0) {
    param->clear_mean_value();
  }
  for (int i = 0; i < c; i++) {  
    param->add_mean_value(mean);
  }
  
  WriteProtoToTextFile(new_param, model_file + "_new");
  LOG(INFO) << "SUCCESS! Output file is " << model_file + "_new";
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;
#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (argc != 3) {
    LOG(ERROR) << "Usage: " << argv[0] << " model_file net_option"
              << "    model_file: a prototxt file"
              << "    net_option: 1.faster_rcnn, "
              << "                2.rfcn"
              << "                3.yolov3"
              << "                4.mobilenet_v1/v2,ssd_mobilenet_v1/v2"
              << "                5.yolov2"
              << "                6.c3d";
    return 1;
  }

  std::stringstream ss;
  int net_option;
  ss << argv[2];
  ss >> net_option;
  string model_file = argv[1];
  NetParameter net_param;
  if (!ReadProtoFromTextFile(model_file, &net_param)) {
    LOG(ERROR) << "Failed to parse input text file as NetParameter!";
    return 1;
  }
  if (net_option == 1) {
    update_faster_rcnn(model_file, net_param);
  } else if (net_option == 2) {
    update_rfcn(model_file, net_param);
  } else if (net_option == 3) {
    update_yolov3(model_file, net_param);
  } else if (net_option == 4) {
    update_mobilenet(model_file, net_param);
  } else if (net_option == 5) {
    update_yolov2(model_file, net_param);
  } else if (net_option == 6) {
    update_c3d(model_file, net_param);
  }  else {
    LOG(ERROR) << "not support!";
  }
  return 0;
}
