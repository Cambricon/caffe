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

#ifndef EXAMPLES_YOLO_V2_POST_PROCESS_YOLOV2_ON_POST_HPP_
#define EXAMPLES_YOLO_V2_POST_PROCESS_YOLOV2_ON_POST_HPP_
#include "yolov2_processor.hpp"
#include "caffe/caffe.hpp"
#include "threadPool.h"
#include "simple_interface.hpp"

template<typename Dtype, template <typename> class Qtype>
class YoloV2OnPostProcessor: public YoloV2Processor<Dtype, Qtype> {
  public:
  YoloV2OnPostProcessor() : outCpuPtrs_(nullptr) {
    tp_ = new zl::ThreadPool(SimpleInterface::thread_num);
  }
  ~YoloV2OnPostProcessor() {
    if (outCpuPtrs_) delete [] outCpuPtrs_;
    delete tp_;
  }
  virtual void runParallel();
  virtual void runSerial();

  protected:
  vector<vector<float>> getResults(vector<cv::Mat> *imgs,
                                   vector<string> *img_names);

  private:
  Dtype* outCpuPtrs_;
  zl::ThreadPool *tp_;
  Dtype* resultDataPtr_;
};
#endif  // EXAMPLES_YOLO_V2_POST_PROCESS_YOLOV2_ON_POST_HPP_
