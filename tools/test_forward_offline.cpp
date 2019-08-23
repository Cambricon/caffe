/*
All modification made by Cambricon Corporation: © 2018--2019 Cambricon Corporation
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
#include "glog/logging.h"
#ifdef USE_MLU
#include <cnrt.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using std::string;
using std::vector;

DEFINE_int32(mludevice, 0, "set using mlu device number, default: 0");

void rand1(float* data, int length) {
  unsigned int seed = 1024;
  for (int i = 0; i < length; ++i) {
    if (i % 5 == 4) {
      data[i] = rand_r(&seed) % 100 / 100. + 0.0625;
    } else if (i % 5 >= 2) {
      data[i] = data[i - 2] + (rand_r(&seed) % 100) / 100.0 + 0.0625;
    } else {
      data[i] = (rand_r(&seed) % 100) / 100. + 0.0625;
    }
  }
}

void rand2(float* data, int length) {
  unsigned int seed = 1024;
  for (int i = 0; i < length; ++i) {
    if (i % 5 == 0) {
      data[i] = rand_r(&seed) % 100 / 100. + 0.0625;
    } else if (i % 5 > 2) {
      data[i] = data[i - 2] + (rand_r(&seed) % 100) / 100.0 + 0.0625;
    } else {
      data[i] = (rand_r(&seed) % 100) / 100. + 0.0625;
    }
  }
}

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;
#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (argc < 4) {
    LOG(INFO) << "USAGE: " << argv[0] << ": <cambricon_file>"
              << " <output_file> <function_name0> <function_name1> ...";
    return 1;
  }
  cnrtInit(0);
  unsigned devNum;
  cnrtGetDeviceCount(&devNum);
  if (FLAGS_mludevice >= 0) {
    CHECK_NE(devNum, 0) << "No device found";
    CHECK_LE(FLAGS_mludevice, devNum) << "valid device count: " << devNum;
  } else {
    LOG(FATAL) << "Invalid device number";
  }

  cnrtDev_t dev;
  cnrtGetDeviceHandle(&dev, FLAGS_mludevice);
  cnrtSetCurrentDevice(dev);
  // 2. load model and get function
  cnrtModel_t model;
  string fname = (string)argv[1];
  LOG(INFO) << "load file: " << fname;
  cnrtLoadModel(&model, fname.c_str());
  cnrtFunction_t function;
  unsigned int in_n, in_c, in_h, in_w;
  unsigned int out_n, out_c, out_h, out_w;
  const unsigned int BATCH_1 = 1;
  unsigned int affinity = 0x01;
  int data_parallel = 1;
  cnrtInitFuncParam_t initFuncParam;
  bool muta = false;
  int dp = data_parallel;
  initFuncParam.muta = &muta;
  initFuncParam.data_parallelism = &dp;
  initFuncParam.affinity = &affinity;
  initFuncParam.end = CNRT_PARAM_END;

  cnrtDim3_t dim = {1, 1, 1};
  cnrtInvokeFuncParam_t invokeFuncParam;
  invokeFuncParam.data_parallelism = &dp;
  invokeFuncParam.affinity = &affinity;
  invokeFuncParam.end = CNRT_PARAM_END;

  struct timeval tpend, tpstart;
  gettimeofday(&tpstart, NULL);

  for (int n = 3; n < argc; n++) {
    string name = (string)argv[n];
    cnrtCreateFunction(&function);
    cnrtExtractFunction(&function, model, name.c_str());
    // initialize function memory
    cnrtInitFunctionMemory_V2(function, &initFuncParam);
    // 3. get function's I/O DataDesc
    int inputNum, outputNum;
    cnrtDataDescArray_t inputDescS, outputDescS;
    cnrtGetInputDataDesc(&inputDescS, &inputNum, function);
    cnrtGetOutputDataDesc(&outputDescS, &outputNum, function);
#if !defined(CROSS_COMPILE) && !defined(CROSS_COMPILE_ARM64)
    uint64_t stack_size;
    cnrtQueryModelStackSize(model, &stack_size);
    unsigned int current_device_size;
    cnrtGetStackMem(&current_device_size);
    if (stack_size > current_device_size) {
      cnrtSetStackMem(stack_size + 50);
    }
#endif  // CROSS_COMPILE && CROSS_COMPILE_ARM64
    // 4. allocate I/O data space on CPU memory and prepare Input data
    void** inputCpuPtrS = reinterpret_cast<void**>(malloc(sizeof(void*) * inputNum));
    void** outputCpuPtrS = reinterpret_cast<void**>(malloc(sizeof(void*) * outputNum));
    vector<float*> output_cpu;
    vector<int> in_count;
    vector<int> out_count;
    void** param =
        reinterpret_cast<void**>(malloc(sizeof(void*) * (inputNum + outputNum)));
    srand(10);
    for (int i = 0; i < inputNum; i++) {
      int ip;
      float* databuf;
      cnrtDataDesc_t inputDesc = inputDescS[i];
      cnrtSetHostDataLayout(inputDesc, CNRT_FLOAT32, CNRT_NCHW);
      cnrtGetHostDataCount(inputDesc, &ip);
      databuf = reinterpret_cast<float*>(malloc(sizeof(float) * ip));
      if (i == 0) {
        rand1(databuf, ip);
      } else {
        rand2(databuf, ip);
      }
      in_count.push_back(ip);
      inputCpuPtrS[i] = reinterpret_cast<void*>(databuf);
      cnrtGetDataShape(inputDesc, &in_n, &in_c, &in_h, &in_w);
    }
    for (int i = 0; i < outputNum; i++) {
      int op;
      float* outcpu;
      cnrtDataDesc_t outputDesc = outputDescS[i];
      cnrtSetHostDataLayout(outputDesc, CNRT_FLOAT32, CNRT_NCHW);
      cnrtGetHostDataCount(outputDesc, &op);
      outcpu = reinterpret_cast<float*>(malloc(op * sizeof(float)));
      out_count.push_back(op);
      output_cpu.push_back(outcpu);
      outputCpuPtrS[i] = reinterpret_cast<void*>(outcpu);
      cnrtGetDataShape(outputDesc, &out_n, &out_c, &out_h, &out_w);
      LOG(INFO) << "out_n " << out_n << " out_c " << out_c << " out_h " << out_h
                << " out_w " << out_w;
    }
    // 5. allocate I/O data space on MLU memory and copy Input data
    // Only 1 batch so far
    void** inputMluPtrS;
    void** outputMluPtrS;
    cnrtMallocBatchByDescArray(&inputMluPtrS, inputDescS, inputNum, BATCH_1);
    cnrtMallocBatchByDescArray(&outputMluPtrS, outputDescS, outputNum, BATCH_1);
    for (int i = 0; i < inputNum; i++) {
      param[i] = inputMluPtrS[i];
    }
    for (int i = 0; i < outputNum; i++) {
      param[inputNum + i] = outputMluPtrS[i];
    }
    // 6. create queue and run function
    cnrtQueue_t cnrt_queue;
    cnrtCreateQueue(&cnrt_queue);
    // initialize function memory, should be called
    // once before cnrtInvokeFunction
    cnrtInitFunctionMemory_V2(function, &initFuncParam);
    cnrtMemcpyBatchByDescArray(
        inputMluPtrS,
        inputCpuPtrS,
        inputDescS,
        inputNum,
        BATCH_1,
        CNRT_MEM_TRANS_DIR_HOST2DEV);
    // create start_event and end_event
    cnrtNotifier_t notifierBeginning, notifierEnd;
    cnrtCreateNotifier(&notifierBeginning);
    cnrtCreateNotifier(&notifierEnd);
    float event_time_use;
    // run MLU
    // place start_event to queue 
    cnrtPlaceNotifier(notifierBeginning, cnrt_queue);
    CNRT_CHECK(cnrtInvokeFunction_V2(function, dim,
          param, (cnrtFunctionType_t)0, cnrt_queue, &invokeFuncParam));
    // place end_event to cnrt_queue 
    cnrtPlaceNotifier(notifierEnd, cnrt_queue);
    if (cnrtSyncQueue(cnrt_queue) == CNRT_RET_SUCCESS) {
      // get start_event and end_event elapsed time
      cnrtNotifierDuration(notifierBeginning, notifierEnd, &event_time_use);
#if !defined(CROSS_COMPILE) && !defined(CROSS_COMPILE_ARM64)
      LOG(INFO) << " hardware time: " << event_time_use;
#endif
    } else {
      LOG(INFO) << " SyncQueue Error ";
    }
    cnrtMemcpyBatchByDescArray(
        outputCpuPtrS,
        outputMluPtrS,
        outputDescS,
        outputNum,
        BATCH_1,
        CNRT_MEM_TRANS_DIR_DEV2HOST);
    for (int i = 0; i < outputNum; i++) {
      LOG(INFO) << "copying output data of " << i << "th" << " function: " << argv[n];
      std::stringstream ss;
      if (outputNum > 1) {
        ss << argv[2] << "_" << argv[n] << i;
      } else {
        ss << argv[2] << "_" << argv[n];
      }
      string output_name = ss.str();
      LOG(INFO) << "writing output file of segment " << argv[n] << " output: "
                << i << "th" << " output file name: " << output_name;
      std::ofstream fout(output_name, std::ios::out);
      fout << std::flush;
      for (int j = 0; j < out_count[i]; ++j) {
        fout << output_cpu[i][j] << std::endl;
      }
      fout << std::flush;
      fout.close();
    }
    for (auto flo : output_cpu) {
      free(flo);
    }
    output_cpu.clear();
    // 8. free memory space
    free(inputCpuPtrS);
    free(outputCpuPtrS);
    cnrtFreeArray(inputMluPtrS, inputNum);
    cnrtFreeArray(outputMluPtrS, outputNum);
    cnrtDestroyQueue(cnrt_queue);
    cnrtDestroyFunction(function);
  }
  cnrtUnloadModel(model);
  gettimeofday(&tpend, NULL);
  float execTime = 1000000 * (tpend.tv_sec - tpstart.tv_sec) +
    tpend.tv_usec - tpstart.tv_usec;
  LOG(INFO) << " execution time: " << execTime << " us";
  cnrtDestroy();
  return 0;
}
#else
int main() {
  LOG(FATAL) << "This program should be compiled with USE_MLU!";
  return 0;
}
#endif  // USE_MLU
