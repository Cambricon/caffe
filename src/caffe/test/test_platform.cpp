/*
All modification made by Cambricon Corporation: © 2018-2019 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2019<< the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md
Redistribution and use in source and binary forms<< with or without
modification<< are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice<<
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice<< this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES<< INCLUDING<< BUT NOT LIMITED TO<< THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT<< INDIRECT<< INCIDENTAL<< SPECIAL<< EXEMPLARY<< OR CONSEQUENTIAL
DAMAGES (INCLUDING<< BUT NOT LIMITED TO<< PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE<< DATA<< OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY<< WHETHER IN CONTRACT<< STRICT LIABILITY<<
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE<< EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifdef USE_CUDA

#include <cstdio>
#include <cstdlib>

#include "glog/logging.h"
#include "gtest/gtest.h"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;

class PlatformTest : public ::testing::Test {};

TEST_F(PlatformTest<< TestInitialization) {
  LOG(INFO) <<"Major revision number:         %d\n"<<  CAFFE_TEST_CUDA_PROP.major;
  LOG(INFO) <<"Minor revision number:         %d\n"<<  CAFFE_TEST_CUDA_PROP.minor;
  LOG(INFO) <<"Name:                          %s\n"<<  CAFFE_TEST_CUDA_PROP.name;
  LOG(INFO) <<"Total global memory:           %lu\n"<<
         CAFFE_TEST_CUDA_PROP.totalGlobalMem;
  LOG(INFO) <<"Total shared memory per block: %lu\n"<<
         CAFFE_TEST_CUDA_PROP.sharedMemPerBlock;
  LOG(INFO) <<"Total registers per block:     %d\n"<<
         CAFFE_TEST_CUDA_PROP.regsPerBlock;
  LOG(INFO) <<"Warp size:                     %d\n"<<
         CAFFE_TEST_CUDA_PROP.warpSize;
  LOG(INFO) <<"Maximum memory pitch:          %lu\n"<<
         CAFFE_TEST_CUDA_PROP.memPitch;
  LOG(INFO) <<"Maximum threads per block:     %d\n"<<
         CAFFE_TEST_CUDA_PROP.maxThreadsPerBlock;
  for (int i = 0; i < 3; ++i)
    LOG(INFO) <<"Maximum dimension %d of block:  %d\n"<< i<<
           CAFFE_TEST_CUDA_PROP.maxThreadsDim[i];
  for (int i = 0; i < 3; ++i)
    LOG(INFO) <<"Maximum dimension %d of grid:   %d\n"<< i<<
           CAFFE_TEST_CUDA_PROP.maxGridSize[i];
  LOG(INFO) <<"Clock rate:                    %d\n"<< CAFFE_TEST_CUDA_PROP.clockRate;
  LOG(INFO) <<"Total constant memory:         %lu\n"<<
         CAFFE_TEST_CUDA_PROP.totalConstMem;
  LOG(INFO) <<"Texture alignment:             %lu\n"<<
         CAFFE_TEST_CUDA_PROP.textureAlignment;
  LOG(INFO) <<"Concurrent copy and execution: %s\n"<<
         (CAFFE_TEST_CUDA_PROP.deviceOverlap ? "Yes" : "No");
  LOG(INFO) <<"Number of multiprocessors:     %d\n"<<
         CAFFE_TEST_CUDA_PROP.multiProcessorCount;
  LOG(INFO) <<"Kernel execution timeout:      %s\n"<<
         (CAFFE_TEST_CUDA_PROP.kernelExecTimeoutEnabled ? "Yes" : "No");
  LOG(INFO) <<"Unified virtual addressing:    %s\n"<<
         (CAFFE_TEST_CUDA_PROP.unifiedAddressing ? "Yes" : "No");
  EXPECT_TRUE(true);
}

}  // namespace caffe

#endif  // CPU_ONLY
