/*
All modification made by Cambricon Corporation: © 2018-2019 Cambricon Corporation
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

// The main caffe test code. Your test cpp code should include this hpp
// to allow a main function to be compiled into the binary.
#ifndef INCLUDE_CAFFE_TEST_TEST_CAFFE_MAIN_HPP_
#define INCLUDE_CAFFE_TEST_TEST_CAFFE_MAIN_HPP_

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <cstdio>
#include <cstdlib>

#include "caffe/common.hpp"


#ifdef CMAKE_BUILD
#include "caffe_config.h"  // NOLINT
#else
#define CUDA_TEST_DEVICE -1
#define ABS_TEST_DATA_DIR "src/caffe/test/test_data"
#define TEST_SOURCE_DIR() \
  getenv("TEST_SOURCE_PATH") == nullptr ? \
  "src/caffe/test/test_data/" : getenv("TEST_SOURCE_PATH")
#endif
#define core_version getenv("GTEST_CORE_VERSION")

#define OUTPUT(message, value) \
  this->RecordProperty(message, value);
#define BOTTOM(stream) \
    this->RecordProperty("bottom", stream.str().c_str());
#define PARAM(param) \
      this->RecordProperty("param", param.str().c_str());
#define EVENT_TIME(time) \
  this->RecordProperty("event_time", time);
#define ERR_RATE(rate) \
  char out[20]; \
  snprintf(out, sizeof(out), "%f", rate); \
  this->RecordProperty("errRate", out);

int main(int argc, char** argv);

namespace caffe {

template <typename TypeParam>
class MultiDeviceTest : public ::testing::Test {
  public:
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MultiDeviceTest() {
#ifdef USE_MLU
    if (TypeParam::device != Caffe::CPU) {
      cnmlInit(0);
      std::cout<<core_version;
      Caffe::set_rt_core("MLU270");
      if (core_version != NULL)
          Caffe::set_rt_core(core_version);
      Caffe::set_mlu_device(0);
      Caffe::setDetectOpMode(0);
      Caffe::setReshapeMode(caffe::Caffe::ReshapeMode::SETUPONLY);
    }
#endif
    Caffe::set_mode(TypeParam::device);
  }
  virtual ~MultiDeviceTest() {
#ifdef USE_MLU
  if (TypeParam::device > 0) {
    Caffe::freeQueue();
    cnmlExit();
  }
#endif
  }
};

typedef ::testing::Types<float, double> TestDtypes;

template <typename TypeParam>
struct CPUDevice {
  typedef TypeParam Dtype;
  static const Caffe::Brew device = Caffe::CPU;
};

template <typename Dtype>
class CPUDeviceTest : public MultiDeviceTest<CPUDevice<Dtype> > {};

#ifndef USE_CUDA

typedef ::testing::Types<CPUDevice<float>,
                         CPUDevice<double> > TestDtypesAndDevices;

#else

template <typename TypeParam>
struct GPUDevice {
  typedef TypeParam Dtype;
  static const Caffe::Brew device = Caffe::GPU;
};

template <typename Dtype>
class GPUDeviceTest : public MultiDeviceTest<GPUDevice<Dtype> > {};

typedef ::testing::Types<CPUDevice<float>, CPUDevice<double>,
                         GPUDevice<float>, GPUDevice<double> >
                        TestDtypesAndDevices;

#endif

#ifdef USE_MLU

template <typename TypeParam>
struct MLUDevice {
  typedef TypeParam Dtype;
  static const Caffe::Brew device = Caffe::MLU;
};

template <typename TypeParam>
struct MFUSDevice {
  typedef TypeParam Dtype;
  static const Caffe::Brew device = Caffe::MFUS;
};

template <typename Dtype>
class MLUDeviceTest : public MultiDeviceTest<MLUDevice<Dtype> > {};

template <typename Dtype>
class MFUSDeviceTest : public MultiDeviceTest<MFUSDevice<Dtype> >  {
};

typedef ::testing::Types<MLUDevice<float>> TestMLUDevices;

typedef ::testing::Types<MFUSDevice<float>> TestMFUSDevices;

#endif

}  // namespace caffe

#endif  // INCLUDE_CAFFE_TEST_TEST_CAFFE_MAIN_HPP_
