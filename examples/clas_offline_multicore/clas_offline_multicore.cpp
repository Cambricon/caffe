/*
All modification made by Cambricon Corporation: © 2018-2019 Cambricon Corporation
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

#if defined(USE_MLU) && defined(USE_OPENCV)
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <sys/time.h>
#include <gflags/gflags.h>
#include <vector>
#include <queue>
#include <string>
#include <fstream>
#include "pipeline.hpp"
#include "off_data_provider.hpp"
#include "off_runner.hpp"
#include "clas_off_post.hpp"
#include "common_functions.hpp"

using std::vector;
using std::queue;
using std::string;
using std::thread;
using std::stringstream;
#define PRE_READ

typedef OffDataProvider<void*, BlockingQueue> OffDataProviderT;
typedef OffRunner<void*, BlockingQueue> OffRunnerT;
typedef ClassOffPostProcessor<void*, BlockingQueue> ClassOffPostProcessorT;
typedef Pipeline<void*, BlockingQueue> PipelineT;

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
  gflags::SetUsageMessage("Do offline multicore classification.\n"
        "Usage:\n"
        "    clas_offline_multicore [FLAGS] modelfile listfile\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (argc == 0) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "examples/clas_offline_multicore/");
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
  int totalThreads = FLAGS_threads * deviceIds_.size();

  cnrtInit(0);

  ImageReader img_reader(FLAGS_images, totalThreads);
  auto&& imageList = img_reader.getImageList();
  int imageNum = img_reader.getImageNum();

  vector<thread*> stageThreads;
  vector<PipelineT* > pipelines;
  for (int i = 0; i < totalThreads; i++) {
    int deviceId = deviceIds_[i % deviceIds_.size()];
    auto provider = new OffDataProviderT(FLAGS_meanfile, FLAGS_meanvalue,
                                         imageList[i]);

    auto runner = new OffRunnerT(FLAGS_offlinemodel, i,
                                 FLAGS_dataparallel, deviceId, deviceIds_.size());

    auto postprocessor = new ClassOffPostProcessorT();

    auto pipeline = new PipelineT(provider, runner, postprocessor);

    stageThreads.push_back(new thread(&PipelineT::runParallel, pipeline));
    pipelines.push_back(pipeline);
  }

  for (int i = 0; i < stageThreads.size(); i++) {
    pipelines[i]->notifyAll();
  }

  Timer timer;
  for (int i = 0; i < stageThreads.size(); i++) {
    stageThreads[i]->join();
    delete stageThreads[i];
  }
  timer.log("Total execution time");
  float execTime = timer.getDuration();

  int acc1 = 0;
  int acc5 = 0;
  float mluTime = 0;
  for (int i = 0; i < pipelines.size(); i++) {
    acc1 += pipelines[i]->postProcessor()->top1();
    acc5 += pipelines[i]->postProcessor()->top5();
    mluTime += pipelines[i]->runner()->runTime();
  }
  printfAccuracy(imageNum, acc1, acc5);
  printPerf(imageNum, execTime, mluTime, totalThreads);

  for (auto iter : pipelines)
    delete iter;

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
