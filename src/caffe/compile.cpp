
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
#ifdef USE_MLU
#include <gflags/gflags.h>
#include <assert.h>
#include <caffe/caffe.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <set>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <iostream>
#include"caffe/compile.hpp"

namespace caffe {

bool compile(int modelType,
  std::vector<std::string> *path,
  std::string *buildpath,
  cnmlCoreVersion_t buildType,
  std::string name) {
  // Use fake device, these need to be done before
  // any Caffe function is called
  Caffe::DeviceFlag = Caffe::FakeDevice;
  Caffe::set_rt_core(buildType);
  Caffe::set_mode(Caffe::MFUS);
  Caffe::setReshapeMode(Caffe::ReshapeMode::SETUPONLY);
  string model = (string)(*path)[0];
  string weights = (string)(*path)[1];
  string FLAGS_output_dir = *buildpath;
  Net<float>* net_ = NULL;

  // init Net
  net_ = new Net<float>(model, caffe::TEST);
  if (weights.empty()) {
    LOG(ERROR) << "Invalid weights file!";
    return false;
  }
  net_->CopyTrainedLayersFrom(weights);
  string model_name = FLAGS_output_dir + "/" + name;
  // generate offline model
  net_->genOfflineModel(model_name);
  if (net_) {
    delete net_;
  }
  return true;
}

bool compile(int modelType, std::vector<uint8_t*> buffer,
  std::vector<uint32_t> buffersize,
  uint8_t* buildbuffer, uint32_t buildbuffersize,
  uint32_t& modelsize, cnmlCoreVersion_t buildType // NOLINT
  ) {
  if (!buffer[0] || !buffersize[0]) {
    LOG(ERROR) << "Invalid Model!" << std::endl;
    return false;
  }
  if (!buffer[1] || !buffersize[1]) {
    LOG(ERROR) << "Invalid Weights!" << std::endl;
    return false;
  }
  Caffe::DeviceFlag = Caffe::FakeDevice;
  Caffe::set_rt_core(buildType);
  Caffe::set_mode(Caffe::MFUS);
  Caffe::setReshapeMode(Caffe::ReshapeMode::SETUPONLY);
  Net<float>* net_;
  net_ = new Net<float>(buffer[0], buffersize[0], caffe::TEST);
  net_->CopyTrainedLayersFrom(reinterpret_cast<void*>(buffer[1]),
                              buffersize[1]);
  uint64_t build_buffer_size = buildbuffersize;
  uint64_t model_size;
  if (!net_->genOfflineModelToMem(buildbuffer,
      &build_buffer_size,
      &model_size)) {
    delete net_;
    return false;
  }
  modelsize = model_size;
  delete net_;
  return true;
}

}  // namespace caffe

#endif  // USE_MLU
