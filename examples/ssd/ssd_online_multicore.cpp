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
#include "caffe/caffe.hpp"
#include "pipeline.hpp"
#include "on_data_provider.hpp"
#include "on_runner.hpp"
#include "ssd_on_post.hpp"
#include "common_functions.hpp"
#include "simple_interface.hpp"

using std::vector;
using std::queue;
using std::string;
using std::thread;
using std::stringstream;

DEFINE_int32(dump, 1, "0 or 1, dump output images or not.");
DEFINE_double(confidencethreshold, 0.5,  "Only keep detections with score equal "
                                          "to or higher than the threshold.");
DEFINE_string(outputdir, ".", "The directory used to save output images and txt.");

typedef DataProvider<float, BlockingQueue> DataProviderT;
typedef OnDataProvider<float, BlockingQueue> OnDataProviderT;
typedef OnRunner<float, BlockingQueue> OnRunnerT;
typedef SsdOnPostProcessor<float, BlockingQueue> SsdOnPostProcessorT;
typedef Pipeline<float, BlockingQueue> PipelineT;

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
  gflags::SetUsageMessage("Do online multicore ssd.\n"
        "Usage:\n"
        "    ssd_online_multicore [FLAGS] modelfile listfile\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (argc == 0) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "examples/ssd_online_multicore/");
    return 1;
  }
  CHECK(FLAGS_mmode != std::string("CPU")) << "CPU mode is not supported!";

  SimpleInterface& simpleInterface = SimpleInterface::getInstance();
  // if simple_compile option has been specified to 1 by user, simple compile
  int provider_num = 1;
  simpleInterface.setFlag(true);
  provider_num = SimpleInterface::data_provider_num_;
  std::stringstream sdevice(FLAGS_mludevice);
  vector<int> deviceIds_;
  std::string item;
  while (getline(sdevice, item, ',')) {
    int device = std::atoi(item.c_str());
    deviceIds_.push_back(device);
  }
  int totalThreads = FLAGS_threads * deviceIds_.size();

  cnmlInit(0);

  ImageReader img_reader(FLAGS_images, totalThreads * provider_num);
  auto&& imageList = img_reader.getImageList();
  int imageNum = img_reader.getImageNum();

  vector<thread*> stageThreads;
  vector<PipelineT*> pipelines;
  vector<DataProviderT*> providers;
  for (int i = 0; i < totalThreads; i++) {
    int devideId = deviceIds_[i % deviceIds_.size()];
    DataProviderT* provider;
    PipelineT* pipeline;

    providers.clear();
    // provider_num is 1 for flexible compile.
    for (int j = 0; j < provider_num; j++) {
      provider = new OnDataProviderT(FLAGS_meanfile, FLAGS_meanvalue,
                                        imageList[provider_num * i + j]);
      providers.push_back(provider);
    }

    auto runner = new OnRunnerT(FLAGS_model, FLAGS_weights,
                                i, devideId, deviceIds_.size());

    auto postprocessor = new SsdOnPostProcessorT();

    pipeline = new PipelineT(providers, runner, postprocessor);
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
  float mluTime = 0;
  for (int i = 0; i < pipelines.size(); i++) {
    mluTime += pipelines[i]->runner()->runTime();
  }
  int batchsize = pipelines[0]->runner()->n();
  printPerf(imageNum, execTime, mluTime, 1, batchsize);
  saveResult(imageNum, (-1), (-1), (-1), mluTime, execTime, 1, batchsize);

  caffe::Caffe::freeQueue();
  for (auto iter : pipelines)
    delete iter;

  cnmlExit();
}

#else
#include "caffe/common.hpp"
int main(int argc, char* argv[]) {
  LOG(FATAL)  << "This program should be compiled with the defintion"
              <<" of both USE_MLU and USE_OPENCV!";
  return 0;
}
#endif  // defined(USE_MLU) && defined(USE_OPENCV)
