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
  hardwareReshape_t hardwareReshape) {
    return compile(modelType, path, buildpath, buildType, hardwareReshape, 1);
}

bool compile(int modelType,
            std::vector<std::string> *path,
            std::string *buildpath,
            cnmlCoreVersion_t buildType,
            hardwareReshape_t hardwareReshape,
            int model_parallel) {
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
  string model_name = FLAGS_output_dir + "/" + "offline";
  // generate offline model
  net_->genOfflineModel(model_name,
                        hardwareReshape,
                        model_parallel);
  if (net_) {
    delete net_;
  }
  return true;
}

bool compile(int modelType, std::vector<uint8_t*> buffer,
  std::vector<uint32_t> buffersize,
  uint8_t* buildbuffer, uint32_t buildbuffersize,
  uint32_t& modelsize, cnmlCoreVersion_t buildType, // NOLINT
  hardwareReshape_t hardwareReshape) {
    return compile(modelType, buffer, buffersize, buildbuffer,
      buildbuffersize, modelsize, buildType, hardwareReshape, 1);
}

bool compile(int modelType, std::vector<uint8_t*> buffer,
             std::vector<uint32_t> buffersize,
             uint8_t* buildbuffer, uint32_t buildbuffersize,
             uint32_t& modelsize, // NOLINT
             cnmlCoreVersion_t buildType,
             hardwareReshape_t hardwareReshape,
             int model_parallel) {
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
      &model_size,
      hardwareReshape,
      model_parallel)) {
    delete net_;
    return false;
  }
  modelsize = model_size;
  delete net_;
  return true;
}

}  // namespace caffe

#endif  // USE_MLU
