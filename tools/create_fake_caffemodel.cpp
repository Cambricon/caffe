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

#include <boost/shared_ptr.hpp>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>
#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"

using boost::shared_ptr;
using caffe::Blob;
using caffe::Layer;
using std::vector;

int main(int argc, char** argv) {
  if (argc != 3) {
    LOG(ERROR) << "Usage: " << argv[0] << " input_proto_file output_model_file";
    return 1;
  }
  std::string proto_file = argv[1];
  std::string model_file = argv[2];
  caffe::NetParameter src_param, dst_param;
  ReadProtoFromTextFileOrDie(proto_file, &src_param);
  UpgradeNetAsNeeded(proto_file, &src_param);
  src_param.mutable_state()->set_phase(caffe::TEST);
  caffe::Net<float> net(src_param);
  std::vector<shared_ptr<Layer<float>>> layers = net.layers();
  for (int i = 0; i < layers.size(); i++) {
    static unsigned int seed = 1;
    shared_ptr<Layer<float>> layer = layers[i];
    std::vector<shared_ptr<Blob<float>>> blobs = layer->blobs();
    for (int j = 0; j < blobs.size(); j++) {
      if (!blobs[j]->count()) {
        continue;
      }
      for (int k = 0; k < blobs[j]->count(); k++) {
        blobs[j]->mutable_cpu_data()[k] =
            1.0 * rand_r(&seed) / INT_MAX / blobs[j]->count();
      }
    }
  }
  net.ToProto(&dst_param, false);
  WriteProtoToBinaryFile(dst_param, model_file);
  return 0;
}
