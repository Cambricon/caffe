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

void set_sparse(string model_file, float sparsity) {
  NetParameter net_param;
  if (!ReadProtoFromTextFile(model_file, &net_param)) {
    LOG(ERROR) << "Failed to parse input text file as NetParameter!";
    return;
  }
  for (int i = 0; i < net_param.layer_size(); i++) {
    if (net_param.layer(i).type() == "Convolution") {
      net_param.mutable_layer(i)->mutable_convolution_param()->set_sparse_mode(true);
      net_param.mutable_layer(i)->set_sparsity(sparsity);
    } else if (net_param.layer(i).type() == "InnerProduct") {
      net_param.mutable_layer(i)->mutable_inner_product_param()->set_sparse_mode(true);
      net_param.mutable_layer(i)->set_sparsity(sparsity);
    }
  }
  // Save new format prototxt.
  WriteProtoToTextFile(net_param, model_file + ".sparse");
  LOG(INFO) << "Output file is " << model_file + ".sparse";
}

int main(int argc, char** argv) {
  if (argc != 3) {
    LOG(ERROR) << "Usage: " << argv[0] << " model_file sparsity"
              << "    model_file: a prototxt file"
              << "    sparsity: a float number, 0 <= sparsity < 1";
    return 1;
  }
  set_sparse(argv[1], atof(argv[2]));
  return 0;
}
