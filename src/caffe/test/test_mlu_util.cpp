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

#ifdef USE_MLU

#include <vector>
#include "cnrt.h"  // NOLINT

#include "glog/logging.h"
#include "gtest/gtest.h"
#include "caffe/mlu/util.hpp"

#include "caffe/util/math_functions.hpp"
#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

class MLUUtilTest : public ::testing::Test {};

TEST_F(MLUUtilTest, TestUniquePush) {
  vector<int> vec;
  int a = 1;
  uniquePushBack(&vec, a);
  EXPECT_EQ(vec.size(), 1);
  uniquePushBack(&vec, a);
  EXPECT_EQ(vec.size(), 1);
}

TEST_F(MLUUtilTest, TestRmOne) {
  vector<int> vec;
  int a = 1;
  uniquePushBack(&vec, a);
  EXPECT_EQ(vec.size(), 1);
  rmOneFromVector(&vec, 2);
  EXPECT_EQ(vec.size(), 1);
  rmOneFromVector(&vec, 1);
  EXPECT_EQ(vec.size(), 0);
}

TEST_F(MLUUtilTest, TestSparseFilterDimOne) {
  vector<int> shape(1, 1);
  vector<float> src_data(1, 0.5);
  vector<float> dst_data(1);
  sparseFilter(shape, src_data.data(), &dst_data, 0.1);
  EXPECT_EQ(src_data[0], dst_data[0]);
}

TEST_F(MLUUtilTest, TestSparseFilterDimFour) {
  vector<int> shape(4, 4);
  vector<float> src_data(256);
  vector<float> dst_data(256);
  caffe_rng_gaussian<float>(256, 50,
      10, src_data.data());
  sparseFilter(shape, src_data.data(), &dst_data, 0.4);
  int counter = 0;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 2; j++) {
      for (int k = 0; k < 16; k++) {
        if (dst_data[i * 64 + j * 2 * 16 + k] == 0) {
          counter += 2;
          EXPECT_EQ(dst_data[i * 64 + (j * 2 + 1) * 16 + k], 0);
        }
      }
    }
  }
  EXPECT_NEAR(1. * counter / 256, 0.4, 0.01);
}


}  // namespace caffe

#endif
