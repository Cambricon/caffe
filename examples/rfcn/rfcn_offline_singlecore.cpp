/*
All modification made by Cambricon Corporation: © 2018--2019 Cambricon Corporation
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

#if defined(USE_MLU) && defined(USE_OPENCV)
#include <sys/time.h>
#include <gflags/gflags.h>
#include <vector>
#include <queue>
#include <string>
#include <fstream>
#include "pipeline.hpp"
#include "off_data_provider.hpp"
#include "off_runner.hpp"
#include "rfcn_off_post.hpp"
#include "common_functions.hpp"

using std::vector;
using std::queue;
using std::string;
using std::thread;
using std::stringstream;

DEFINE_int32(dump, 1, "0 or 1, dump output images or not.");
DEFINE_string(outputdir, ".", "The directory used to save output images and txt.");

typedef OffDataProvider<void*, Queue> OffDataProviderT;
typedef OffRunner<void*, Queue> OffRunnerT;
typedef RfcnOffPostProcessor<void*, Queue> RfcnOffPostProcessorT;
typedef Pipeline<void*, Queue> PipelineT;

int main(int argc, char* argv[]) {
  {
    const char * env = getenv("log_prefix");
    if (!env || strcmp(env, "true") != 0)
      FLAGS_log_prefix = false;
  }
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;
#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif
  gflags::SetUsageMessage("Do detection using RFCN offline mode.\n"
        "Usage:\n"
        "    rfcn_offline [FLAGS] model_file list_file\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (argc == 0) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "examples/rfcn/rfcn_offline");
    return 1;
  }

  if (FLAGS_logdir != "") {
    FLAGS_log_dir = FLAGS_logdir;
  } else {
    //  log to terminal's stderr if no log path specified
    FLAGS_alsologtostderr = 1;
  }

  std::stringstream sdevice(FLAGS_mludevice);
  vector<int> deviceIds_;
  std::string item;
  while (getline(sdevice, item, ',')) {
    int device = std::atoi(item.c_str());
    deviceIds_.push_back(device);
  }

  cnrtInit(0);

  unsigned devNum;
  cnrtGetDeviceCount(&devNum);
  if (deviceIds_[0] >= 0) {
    CHECK_NE(devNum, 0) << "No device found";
    CHECK_LT(deviceIds_[0], devNum) << "valid device count: " <<devNum;
  } else {
    LOG(FATAL) << "Invalid device number";
  }

  ImageReader img_reader(FLAGS_images, FLAGS_threads);
  auto&& imageList = img_reader.getImageList();
  int imageNum = img_reader.getImageNum();

  auto provider = new OffDataProviderT(FLAGS_meanfile, FLAGS_meanvalue,
                                       imageList[0]);

  auto runner = new OffRunnerT(FLAGS_offlinemodel, 0, 1,
                               deviceIds_[0], deviceIds_.size());

  auto postprocessor = new RfcnOffPostProcessorT();

  auto pipeline = new PipelineT(provider, runner, postprocessor);

  Timer timer;
  pipeline->runSerial();
  timer.log("Total execution time");
  float execTime = timer.getDuration();
  float mluTime = pipeline->runner()->runTime();
  int batchsize = pipeline->runner()->n();
  printPerf(imageNum, execTime, mluTime, 1, batchsize);
  saveResult(imageNum, (-1), (-1), (-1), mluTime, execTime, 1, batchsize);

  delete pipeline;
  cnrtDestroy();
}

#else
#include <glog/logging.h>
int main(int argc, char* argv[]) {
  LOG(FATAL) << "This program should be compiled with the defintion"
             <<" of both USE_MLU and USE_OPENCV!";
  return 0;
}
#endif  // defined(USE_MLU) && defined(USE_OPENCV)
