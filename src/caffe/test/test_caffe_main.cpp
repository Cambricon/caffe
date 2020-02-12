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

#include "caffe/caffe.hpp"
#include "caffe/test/test_caffe_main.hpp"

namespace caffe {
#ifdef USE_CUDA
  cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif
}

#ifdef USE_CUDA
using caffe::CAFFE_TEST_CUDA_PROP;
#endif

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  caffe::GlobalInit(&argc, &argv);
#ifdef USE_CUDA
  // Before starting testing, let's first print out a few cuda device info.
  int device;
  cudaGetDeviceCount(&device);
  LOG(INFO) << "Cuda number of devices: " << device;
  if (argc > 1) {
    // Use the given device
    device = atoi(argv[1]);
    cudaSetDevice(device);
    LOG(INFO) << "Setting to use device " << device;
  } else if (CUDA_TEST_DEVICE >= 0) {
    // Use the device assigned in build configuration; but with a lower priority
    device = CUDA_TEST_DEVICE;
  }
  cudaGetDevice(&device);
  LOG(INFO) << "Current device id: " << device;
  cudaGetDeviceProperties(&CAFFE_TEST_CUDA_PROP, device);
  LOG(INFO) << "Current device name: " << CAFFE_TEST_CUDA_PROP.name;
#endif
  // invoke the test.
  return RUN_ALL_TESTS();
}
