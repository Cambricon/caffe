/*
All modification made by Cambricon Corporation: © 2018 Cambricon Corporation
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

#include <sys/time.h>
#include <algorithm>
#include <atomic>
#include <condition_variable> // NOLINT
#include <thread> // NOLINT
#include <utility>
#include <string>
#include <vector>

#include "include/blocking_queue.hpp"
#include "include/queue.hpp"
#include "include/pipeline.hpp"

using std::queue;
using std::thread;

template <typename Dtype, template <typename> class Qtype>
int Pipeline<Dtype, Qtype>::imageNum = 0;
template <typename Dtype, template <typename> class Qtype>
vector<queue<string>> Pipeline<Dtype, Qtype>::imageList;
template <typename Dtype, template <typename> class Qtype>
std::condition_variable Pipeline<Dtype, Qtype>::condition;
template <typename Dtype, template <typename> class Qtype>
std::mutex Pipeline<Dtype, Qtype>::condition_m;
template <typename Dtype, template <typename> class Qtype>
int Pipeline<Dtype, Qtype>::start;
template <typename Dtype, template <typename> class Qtype>
vector<thread*> Pipeline<Dtype, Qtype>::stageThreads;
template <typename Dtype, template <typename> class Qtype>
vector<Pipeline<Dtype, Qtype>*> Pipeline<Dtype, Qtype>::pipelines;

template <typename Dtype, template <typename> class Qtype>
Pipeline<Dtype, Qtype>::Pipeline(DataProvider<Dtype, Qtype> *provider,
                      Runner<Dtype, Qtype> *runner,
                      PostProcessor<Dtype, Qtype> *postprocessor) {
  data_provider_ = provider;
  runner_ = runner;
  postProcessor_ = postprocessor;

  data_provider_->setRunner(runner_);
  postProcessor_->setRunner(runner_);
  runner_->setPostProcessor(postProcessor_);

  data_provider_->setThreadId(runner_->threadId());
  postProcessor_->setThreadId(runner_->threadId());

#ifdef PRE_READ
  data_provider_->preRead();
#endif
}

template <typename Dtype, template <typename> class Qtype>
Pipeline<Dtype, Qtype>::~Pipeline() {
  delete runner_;
  delete data_provider_;
  delete postProcessor_;
}

template <typename Dtype, template <typename> class Qtype>
void Pipeline<Dtype, Qtype>::runParallel() {
  vector<thread*> threads(3, nullptr);
  threads[0] = new thread(&DataProvider<Dtype, Qtype>::runParallel, data_provider_);
  threads[1] = new thread(&Runner<Dtype, Qtype>::runParallel, runner_);
  threads[2] = new thread(&PostProcessor<Dtype, Qtype>::runParallel, postProcessor_);
  for (auto th : threads) {
    th->join();
    delete th;
  }
}

template <typename Dtype, template <typename> class Qtype>
void Pipeline<Dtype, Qtype>::runSerial() {
  while (!data_provider_->imageIsEmpty()) {
    data_provider_->runSerial();
    runner_->runSerial();
    postProcessor_->runSerial();
  }
}

template <typename Dtype, template <typename> class Qtype>
void Pipeline<Dtype, Qtype>::notifyAll() {
  {
    std::lock_guard<std::mutex> lk(condition_m);
    start = 1;
    LOG(INFO) << "Notify to start ...";
  }
  condition.notify_all();
}

template <typename Dtype, template <typename> class Qtype>
void Pipeline<Dtype, Qtype>::waitForNotification() {
  std::unique_lock<std::mutex> lk(condition_m);
  LOG(INFO) << "Waiting ...";
  condition.wait(lk, [](){return start;});
  lk.unlock();
}

INSTANTIATE_ALL_CLASS(Pipeline);
