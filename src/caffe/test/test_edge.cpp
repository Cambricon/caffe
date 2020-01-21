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

#include "gtest/gtest.h"

#include "caffe/common.hpp"
#include "caffe/mlu/edge.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class EdgeTest : public ::testing::Test {
  protected:
  EdgeTest() : cur_(2), neighbor_(6), blob_(new Blob<Dtype>()) {
    edge_ = new Edge<Dtype>(cur_, neighbor_, blob_);
  }
  virtual ~EdgeTest() { delete blob_;}
  int cur_;
  int neighbor_;
  Blob<Dtype>* blob_;
  Edge<Dtype>* edge_;
};

TYPED_TEST_CASE(EdgeTest, TestDtypes);

TYPED_TEST(EdgeTest, TestInitialization) {
  EXPECT_EQ(this->edge_->cur(), 2);
  EXPECT_EQ(this->edge_->neighbor(), 6);
  EXPECT_EQ(this->edge_->blob(), this->blob_);
  EXPECT_TRUE(this->edge_ == this->edge_);
}

}  // namespace caffe

#endif
