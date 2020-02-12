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
#include <iostream>
#include <fstream>
#include <cstdio>
#include <vector>
#include <cmath>
#include <string>
// #define float float

#ifndef CORE_INCLUDE_UTIL_PROPOSAL_GENERATE_ANCHORS_HPP_
#define CORE_INCLUDE_UTIL_PROPOSAL_GENERATE_ANCHORS_HPP_


float *whctrs(std::vector<float> anchor);

std::vector<std::vector<float> >
mk_anchors(std::vector<int> ws, std::vector<int> hs, float x_ctr, float y_ctr);

std::vector<std::vector<float> > ratio_enum(std::vector<float> anchor,
                                            std::vector<float> ratios);

std::vector<std::vector<float> > scale_enum(std::vector<float> anchor,
                                            std::vector<float> scales);

void generate_anchor_box(int H, int W, int feat_stride, int base_size,
                         std::vector<float> scales, std::vector<float> ratios,
                         bool pad_hw, float *anchors_cxcywh);
void generate_anchor_box_pvanet(int H, int W, int feat_stride, int base_size,
                                std::vector<float> scales, std::vector<float> ratios,
                                bool pad_hw, float *anchors_cxcywh);

void preprocess_anchors(int feat_stride, int H, int W,
                        std::vector<std::vector<float> > anchors, float *coords,
                        bool pad_hw);

std::vector<std::vector<float> > generate_anchors(std::vector<float> ratios,
                                                  std::vector<float> scales,
                                                  int base_size = 16);



#endif  // CORE_INCLUDE_UTIL_PROPOSAL_GENERATE_ANCHORS_HPP_
