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

#ifndef INCLUDE_CAFFE_COMMON_HPP_
#define INCLUDE_CAFFE_COMMON_HPP_

#include <boost/algorithm/string.hpp>
#include <boost/shared_ptr.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <climits>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>  // pair
#include <vector>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/device_alternate.hpp"

// Convert macro to string
#define STRINGIFY(m) #m
#define AS_STRING(m) STRINGIFY(m)

// gflags 2.1 issue: namespace google was changed to gflags without warning.
// Luckily we will be able to use GFLAGS_GFLAGS_H_ to detect if it is version
// 2.1. If yes, we will add a temporary solution to redirect the namespace.
// TODO(Yangqing): Once gflags solves the problem in a more elegant way, let's
// remove the following hack.
#ifndef GFLAGS_GFLAGS_H_
namespace gflags = google;
#endif  // GFLAGS_GFLAGS_H_

// Disable the copy and assignment operator for a class.
#define DISABLE_COPY_AND_ASSIGN(classname) \
 private:                                  \
  classname(const classname&);             \
  classname& operator=(const classname&)

// Disable the new and delete operator for a class.
#define DISABLE_NEW_AND_DELETE()             \
 private:                                    \
  void* operator new(const size_t) = delete; \
  void operator delete(void* ptr) = delete

// Instantiate a class with float and double specifications.
#define INSTANTIATE_CLASS(classname)   \
  char gInstantiationGuard##classname; \
  template class classname<float>;     \
  template class classname<double>;

#define INSTANTIATE_LAYER_GPU_FORWARD(classname) \
  template void classname<float>::Forward_gpu(   \
      const std::vector<Blob<float>*>& bottom,   \
      const std::vector<Blob<float>*>& top);     \
  template void classname<double>::Forward_gpu(  \
      const std::vector<Blob<double>*>& bottom,  \
      const std::vector<Blob<double>*>& top);

#define INSTANTIATE_LAYER_GPU_BACKWARD(classname) \
  template void classname<float>::Backward_gpu(   \
      const std::vector<Blob<float>*>& top,       \
      const std::vector<bool>& propagate_down,    \
      const std::vector<Blob<float>*>& bottom);   \
  template void classname<double>::Backward_gpu(  \
      const std::vector<Blob<double>*>& top,      \
      const std::vector<bool>& propagate_down,    \
      const std::vector<Blob<double>*>& bottom);

#define INSTANTIATE_LAYER_GPU_FUNCS(classname) \
  INSTANTIATE_LAYER_GPU_FORWARD(classname);    \
  INSTANTIATE_LAYER_GPU_BACKWARD(classname);

// A simple macro to mark codes that are not implemented, so that when the code
// is executed we will see a fatal log.
#define NOT_IMPLEMENTED LOG(FATAL) << "Not Implemented Yet"

// See PR #1236
namespace cv {
class Mat;
}

namespace caffe {

// We will use the boost shared_ptr instead of the new C++11 one mainly
// because cuda does not work (at least now) well with C++11 features.
using boost::shared_ptr;

// Common functions and classes from std that caffe often uses.
using std::fstream;
using std::ios;
using std::isnan;
using std::isinf;
using std::iterator;
using std::make_pair;
using std::map;
using std::ostringstream;
using std::pair;
using std::set;
using std::string;
using std::stringstream;
using std::unordered_map;
using std::unordered_set;
using std::vector;

// A global initialization function that you should call in your main function.
// Currently it initializes google flags and google logging.
void GlobalInit(int* pargc, char*** pargv);

// A singleton class to hold common caffe stuff, such as the handler that
// caffe is going to use for cublas, curand, etc.
class Caffe {
  public:
  ~Caffe();

  // Thread local context for Caffe. Moved to common.cpp instead of
  // including boost/thread.hpp to avoid a boost/NVCC issues (#1009, #1010)
  // on OSX. Also fails on Linux with CUDA 7.0.18.
  static Caffe& Get();

  enum Brew { CPU, GPU, MLU, MFUS };

#ifdef USE_MLU
  /**
   * @brief ReshapeMode provides flexible options for user to optimize
   *        performance of their network running on MLU devices.
   *
   * In Caffe's design, Blobs are reshaped each time the network forwards.
   * The reshaping penalty on MLU devices is significant as the tensor
   * descriptors and memory are re-allocated when reshaping by default.
   * The penalty slows down MLU operations, especially in Caffe::MFUS mode.
   *
   * To reduce the reshaping, we introduce *ReshapeMode* - three modes
   * with which users can leverage in different scenarios:
   *  1. ALWAYS    - always reshape, same as original caffe, default mode.
   *                 Blobs always reshape when setupping and forwarding.
   *  2. SETUPONLY - never reshape when forwading, set by user.
   *                 This is recomended for scenarios where the network
   *                 input shape never changes.
   *
   * For more detail, refer to logic related with class ReshapeHelper.
   */
  enum class ReshapeMode { ALWAYS, SETUPONLY, NONE };

  // Some tools don't utilize real device, so a fake device is enough
  // for them to do their work.
  // When the flag is true, fake device is used.
  // The flag is supposed to be initialied ONCE in the early possible stage.

  // 1: fake device.  Pass -1 as parameter to setDevice method
  // 0: true device

  static const int FakeDevice = 1;

  static int DeviceFlag;
#endif  // USE_MLU

  // This random number generator facade hides boost and CUDA rng
  // implementation from one another (for cross-platform compatibility).
  class RNG {
    public:
    RNG();
    explicit RNG(unsigned int seed);
    explicit RNG(const RNG&);
    RNG& operator=(const RNG&);
    void* generator();

    private:
    class Generator;
    shared_ptr<Generator> generator_;
  };

  // Getters for boost rng, curand, and cublas handles
  inline static RNG& rng_stream() {
    if (!Get().random_generator_) {
      Get().random_generator_.reset(new RNG());
    }
    return *(Get().random_generator_);
  }
#ifdef USE_CUDA
  inline static cublasHandle_t cublas_handle() { return Get().cublas_handle_; }
  inline static curandGenerator_t curand_generator() {
    return Get().curand_generator_;
  }
#endif

  // Returns the mode: running on CPU or GPU.
  inline static Brew mode() { return Get().mode_; }
  // The setters for the variables
  // Sets the mode. It is recommended that you don't change the mode halfway
  // into the program since that may cause allocation of pinned memory being
  // freed in a non-pinned way, which may cause problems - I haven't verified
  // it personally but better to note it here in the header file.
  inline static void set_mode(Brew mode) { Get().mode_ = mode; }
#ifdef USE_MLU
  inline static void set_mode(const string& mode) {
    if (mode.size() == 0 || boost::iequals(mode, "CPU")) {
      set_mode(CPU);
    } else if (boost::iequals(mode, "MLU")) {
      set_mode(MLU);
    } else if (boost::iequals(mode, "MFUS")) {
      set_mode(MFUS);
    } else {
      NOT_IMPLEMENTED << " Caffe::mode " << mode;
    }
  }
#endif  // USE_MLU
  // Sets the random seed of both boost and curand
  static void set_random_seed(const unsigned int seed);
  // Sets the device. Since we have cublas and curand stuff, set device also
  // requires us to reset those values.
  static void SetDevice(const int device_id);
  // Prints the current GPU status.
  static void DeviceQuery();
  // Check if specified device is available
  static bool CheckDevice(const int device_id);
  // Search from start_id to the highest possible device ordinal,
  // return the ordinal of the first available device.
  static int FindDevice(const int start_id = 0);
  // Parallel training
  inline static int solver_count() { return Get().solver_count_; }
  inline static void set_solver_count(int val) { Get().solver_count_ = val; }
  inline static int solver_rank() { return Get().solver_rank_; }
  inline static void set_solver_rank(int val) { Get().solver_rank_ = val; }
  inline static bool multiprocess() { return Get().multiprocess_; }
  inline static void set_multiprocess(bool val) { Get().multiprocess_ = val; }
  inline static bool root_solver() { return Get().solver_rank_ == 0; }
#ifdef USE_MLU
  inline static void set_mlu_device(int dev_id) { Get().setDevice(dev_id); }
  inline static void set_functype(string functype) {
    if (functype != "1H8" && functype != "1H16") {
      LOG(FATAL) << "The value of functype can only be set to 1H8 or 1H16";
    } else if (functype == "1H8") {
      Get().affinity_ = 0x02;
    }
  }
  inline static void setModelParallel(int model_parallel) {
    if (model_parallel * Get().data_parallel_ <= 32 && model_parallel >= 1) {
      Get().model_parallel_ = model_parallel;
    } else {
      LOG(FATAL) << "Invalid model_parallel,"
                 << " model_parallel*data_parallel range from 1 to 32";
    }
  }
  inline static int model_parallel() { return Get().model_parallel_; }
  inline static void setDataParallel(int data_parallel) {
    if (data_parallel * Get().model_parallel_ <= 32 && data_parallel >= 1) {
      Get().data_parallel_ = data_parallel;
    } else {
      LOG(FATAL) << "Invalid data_parallel,"
                 << " data_parallel*model_parallel range from 1 to 32";
    }
  }
  inline static int data_parallel() { return Get().data_parallel_; }

  inline static void setDataStrategy(const vector<int>& datastrategy) {
    Get().in_datastrategy_ = datastrategy[0];
    Get().out_datastrategy_ = datastrategy[1];
  }
  inline static int getInDataStrategy() {
    return Get().in_datastrategy_;
  }
  inline static int getOutDataStrategy() {
    return Get().out_datastrategy_;
  }
  inline static cnmlDataPreprocessStrategy_t in_datastrategy() {
    cnmlDataPreprocessStrategy_t data_strategy;
    int value = Get().in_datastrategy_;
    if (value != -1)
      data_strategy = static_cast<cnmlDataPreprocessStrategy_t>(value);
    return data_strategy;
  }
  inline static cnmlDataPreprocessStrategy_t out_datastrategy() {
    cnmlDataPreprocessStrategy_t data_strategy;
    int value = Get().out_datastrategy_;
    if (value != -1)
      data_strategy = static_cast<cnmlDataPreprocessStrategy_t>(value);
    return data_strategy;
  }
  inline static void setCpuDataOrder(const vector<int>& dataorder) {
    Get().in_dataorder_ = dataorder[0];
    Get().out_dataorder_ = dataorder[1];
  }
  inline static cnmlDataOrder_t in_dataorder() {
    cnmlDataOrder_t data_order = CNML_NCHW;
    int value = Get().in_dataorder_;
    if (1 == value)
      data_order = CNML_NHWC;
    return data_order;
  }
  inline static cnmlDataOrder_t out_dataorder() {
    cnmlDataOrder_t data_order = CNML_NCHW;
    int value = Get().out_dataorder_;
    if (1 == value)
      data_order = CNML_NHWC;
    return data_order;
  }

  inline static void setChannelId(int id) { Get().setDevChannelId(id); }
  inline static cnrtQueue_t queue() { return Get().devQueue(); }
  inline static void freeQueue() {
    cnrtQueue_t qptr = Get().queue_;
    if (nullptr != qptr) {
      cnrtDestroyQueue(qptr);
      Get().queue_ = nullptr;
    }
  }
  inline static cnrtInvokeFuncParam_t* forward_param() {
    return &(Get().compute_forw_param_);
  }
  inline static cnmlCoreVersion_t rt_core() { return Get().coreVersion(); }
  inline static void set_rt_core(cnmlCoreVersion_t rt_core) {
    Get().setCoreVersion(rt_core);
  }
  inline static void set_rt_core(const string& rt_core) {
    Get().setCoreVersion(rt_core);
  }
  inline static ReshapeMode reshapeMode() { return Get().reshape_mode_; }
  inline static void setReshapeMode(ReshapeMode mode) {
    Get().reshape_mode_ = mode;
  }
  inline static void setReshapeMode(const string& mode) {
    if (mode.size() == 0) {
      setReshapeMode(ReshapeMode::SETUPONLY);
    } else if (boost::iequals(mode, "ALWAYS")) {
      setReshapeMode(ReshapeMode::ALWAYS);
    } else if (boost::iequals(mode, "SETUPONLY")) {
      setReshapeMode(ReshapeMode::SETUPONLY);
    } else {
      NOT_IMPLEMENTED << " Caffe::ReshapeMode " << mode;
    }
  }
  void setDevice(int dev_id) {
    if (Caffe::FakeDevice != Caffe::DeviceFlag) {
      unsigned dev_count;
      CNRT_CHECK(cnrtGetDeviceCount(&dev_count));
      if (dev_id >= 0) {
        CHECK_NE(dev_count, 0) << "No device found";
        CHECK_LT(dev_id, dev_count) << "Valid device count: "<< dev_count;
      } else {
        LOG(FATAL) << "Invalid device number";
      }
      cnrtDev_t dev;
      LOG(INFO) << "Using MLU device " << dev_id;
      CNRT_CHECK(cnrtGetDeviceHandle(&dev, dev_id));
      CNRT_CHECK(cnrtSetCurrentDevice(dev));
    } else {
      LOG(WARNING) << "Should not call setDevice when it's FakeDevice";
    }
  }
  void setCoreVersion(cnmlCoreVersion_t v) { core_version_ = v; }
  void setCoreVersion(const std::string& v) {
    if (boost::iequals(v, "1H8")) {
      core_version_ = CNML_1H8;
    } else if (boost::iequals(v, "1H16")) {
      core_version_ = CNML_1H16;
    } else if (boost::iequals(v, "MLU100")) {
      core_version_ = CNML_C10;
    } else {
      LOG(FATAL) << "unknown core version: " << v;
    }
  }
  cnmlCoreVersion_t coreVersion() { return core_version_; }
  void setDevChannelId(int id) {
    CHECK_GE(id, 0);
    CHECK_LE(id, 3);
    LOG(WARNING) << "Using MLU Channel " << id;
    CNRT_CHECK(cnrtSetCurrentChannel((cnrtChannelType_t)id));
  }
  cnrtQueue_t devQueue() {
    if (queue_ == nullptr) {
      CNRT_CHECK(cnrtCreateQueue(&queue_));
    }
    return queue_;
  }
#endif

  protected:
#ifdef USE_CUDA
  cublasHandle_t cublas_handle_;
  curandGenerator_t curand_generator_;
#endif
  shared_ptr<RNG> random_generator_;

  Brew mode_;

  // Parallel training
  int solver_count_;
  int solver_rank_;
  bool multiprocess_;

  private:
  // The private constructor to avoid duplicate instantiation.
  Caffe();

#ifdef USE_MLU
  int data_parallel_;  // The number of cores used, each core processes independant frame.
  int model_parallel_;  // The number of cores used, cores process the same frame.
  int channel_id_;
  int in_datastrategy_;
  int out_datastrategy_;
  unsigned int affinity_;
  cnrtQueue_t queue_;
  cnrtInvokeFuncParam_t compute_forw_param_;
  cnmlCoreVersion_t core_version_;
  ReshapeMode reshape_mode_ = ReshapeMode::SETUPONLY;
  int in_dataorder_;
  int out_dataorder_;
#endif

  DISABLE_COPY_AND_ASSIGN(Caffe);
};

}  // namespace caffe
#endif  // INCLUDE_CAFFE_COMMON_HPP_
