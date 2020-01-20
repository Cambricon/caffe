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

#ifndef EXAMPLES_SSD_POST_PROCESS_SSD_PROCESSOR_HPP_
#define EXAMPLES_SSD_POST_PROCESS_SSD_PROCESSOR_HPP_
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "glog/logging.h"
#include "post_processor.hpp"

using std::map;
using std::max;
using std::min;
using std::queue;
using std::stringstream;

template<typename Dtype, template <typename> class Qtype>
class SsdProcessor : public PostProcessor<Dtype, Qtype> {
  public:
  SsdProcessor() {}
  virtual ~SsdProcessor() {}
  cv::Scalar HSV2RGB(const float h, const float s, const float v);
  vector<cv::Scalar> getColors(const int n);
  void WriteVisualizeBBox(const vector<cv::Mat>& images,
                   const vector<vector<vector<float>>>& detections,
                   const float threshold, const vector<cv::Scalar>& colors,
                   const map<int, string>& labelNameMap, const vector<string>& names,
                   const int& from, const int& to);
  void readLabels(map<int, string>* label_name_map);

  protected:
  map<int, string> labelNameMap;
};

#endif  // EXAMPLES_SSD_POST_PROCESS_SSD_PROCESSOR_HPP_
