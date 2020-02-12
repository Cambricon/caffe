
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

#include "runner_strategy.hpp"
#include <gflags/gflags.h>
#include <vector>
#include <string>
#include <queue>
#include <sstream>
#include <thread> // NOLINT
#include <utility>
#include <iomanip>
#include <map>
#include <fstream>
#ifdef USE_MLU
#include "cnrt.h" // NOLINT
#endif
#include "blocking_queue.hpp"
#include "queue.hpp"
#include "post_processor.hpp"
#include "off_runner.hpp"
#include "common_functions.hpp"
#include "command_option.hpp"

template<typename Dtype, template <typename> class Qtype>
void SimpleStrategy<Dtype, Qtype>::runParallel(OffRunner<Dtype, Qtype>* runner) {
// #define PINGPONG
#ifdef PINGPONG
#define RES_SIZE 2
#else
#define RES_SIZE 1
#endif

  // set device to runtime context binded device
  cnrtSetCurrentContextDevice(runner->runtimeContext());
  cnrtQueue_t queue[RES_SIZE];
  cnrtNotifier_t notifierBeginning[RES_SIZE];
  cnrtNotifier_t notifierEnd[RES_SIZE];

  for (int i = 0; i < RES_SIZE; i++) {
    CHECK(cnrtCreateQueue(&queue[i]) == CNRT_RET_SUCCESS)
          << "CNRT create queue error, thread_id " << runner->threadId();
    cnrtCreateNotifier(&notifierBeginning[i]);
    cnrtCreateNotifier(&notifierEnd[i]);
  }
  float eventInterval[RES_SIZE] = {0};
  Dtype* mluInData[RES_SIZE];
  Dtype* mluOutData[RES_SIZE];

  auto do_pop = [&](int index, void **param) {
    mluInData[index] = runner->popValidInputData();
    if ( mluInData[index] == nullptr )
      return false;
    mluOutData[index] = runner->popFreeOutputData();
    for (int i = 0; i < runner->inBlobNum(); i++) {
      param[i] = mluInData[index][i];
    }
    for (int i = 0; i < runner->outBlobNum(); i++) {
      param[runner->inBlobNum() + i] = mluOutData[index][i];
    }

    return true;
  };

  auto do_invoke = [&](int index, void** param) {
    cnrtPlaceNotifier(notifierBeginning[index], queue[index]);

    CNRT_CHECK(cnrtInvokeRuntimeContext(runner->runtimeContext(),
               param, queue[index], nullptr));
  };

  auto do_sync = [&](int index) {
    cnrtPlaceNotifier(notifierEnd[index], queue[index]);
    if (cnrtSyncQueue(queue[index]) == CNRT_RET_SUCCESS) {
      cnrtNotifierDuration(notifierBeginning[index],
             notifierEnd[index], &eventInterval[index]);
      runner->setRunTime(runner->runTime() + eventInterval[index]);
      printfMluTime(eventInterval[index]);
    } else {
      LOG(ERROR) << " SyncQueue error";
    }
    runner->pushValidOutputData(mluOutData[index]);
    runner->pushFreeInputData(mluInData[index]);
  };

#ifdef PINGPONG
  bool pong_valid = false;
  while (true) {
    void* param[runner->inBlobNum() + runner->outBlobNum()];

    // pop - ping
    if (do_pop(0, static_cast<void **>(param)) == false) {
      if (pong_valid)
        do_sync(1);
      break;
    }
    // invoke - ping
    do_invoke(0, static_cast<void **>(param));

    // sync - pong
    if (pong_valid)
      do_sync(1);

    // pop - pong
    if (do_pop(1, static_cast<void **>(param)) == false) {
      do_sync(0);
      break;
    }

    // invoke - pong
    do_invoke(1, static_cast<void **>(param));
    pong_valid = true;

    // sync - ping
    do_sync(0);
  }
#else
  while (true) {
    void* param[runner->inBlobNum() + runner->outBlobNum()];
    // pop - ping
    if (do_pop(0, static_cast<void **>(param)) == false) {
      break;
    }
    // invoke - ping
    do_invoke(0, static_cast<void **>(param));

    // sync - ping
    do_sync(0);
  }
#endif

  for (int i = 0; i < RES_SIZE; i++) {
    cnrtDestroyNotifier(&notifierBeginning[i]);
    cnrtDestroyNotifier(&notifierEnd[i]);
    cnrtDestroyQueue(queue[i]);
  }

  // tell postprocessor to exit
  runner->pushValidOutputData(static_cast<Dtype*>(nullptr));
}

template<typename Dtype, template <typename> class Qtype>
void FlexibleStrategy<Dtype, Qtype>::runParallel(OffRunner<Dtype, Qtype>* runner) {
  setDeviceId(runner->deviceId());
  cnrtQueue_t queue = runner->queue();
  CHECK(cnrtCreateQueue(&queue) == CNRT_RET_SUCCESS)
        << "CNRT create queue error, thread_id " << runner->threadId();
  runner->setQueue(queue);
  // initialize function memory
  cnrtNotifier_t notifierBeginning, notifierEnd;
  cnrtCreateNotifier(&notifierBeginning);
  cnrtCreateNotifier(&notifierEnd);
  float eventInterval = 0;

  while (true) {
    Dtype* mluInData = runner->popValidInputData();
    if ( mluInData == nullptr ) break;  // no more images

    Dtype* mluOutData = runner->popFreeOutputData();
    void* param[runner->inBlobNum() + runner->outBlobNum()];
    for (int i = 0; i < runner->inBlobNum(); i++) {
      param[i] = mluInData[i];
    }
    for (int i = 0; i < runner->outBlobNum(); i++) {
      param[runner->inBlobNum() + i] = mluOutData[i];
    }

    cnrtPlaceNotifier(notifierBeginning, runner->queue());
    CNRT_CHECK(cnrtInvokeRuntimeContext(runner->runtimeContext(),
               param, runner->queue(), nullptr));
    cnrtPlaceNotifier(notifierEnd, runner->queue());

    if (cnrtSyncQueue(runner->queue()) == CNRT_RET_SUCCESS) {
      cnrtNotifierDuration(notifierBeginning, notifierEnd, &eventInterval);
      runner->setRunTime(runner->runTime() + eventInterval);
      printfMluTime(eventInterval);
    } else {
      LOG(ERROR) << " SyncQueue error";
    }
    runner->pushValidOutputData(mluOutData);
    runner->pushFreeInputData(mluInData);
  }

  cnrtDestroyNotifier(&notifierBeginning);
  cnrtDestroyNotifier(&notifierEnd);
  // tell postprocessor to exit
  runner->pushValidOutputData(static_cast<Dtype*>(nullptr));
}
INSTANTIATE_OFF_CLASS(RunnerStrategy);
INSTANTIATE_OFF_CLASS(SimpleStrategy);
INSTANTIATE_OFF_CLASS(FlexibleStrategy);
