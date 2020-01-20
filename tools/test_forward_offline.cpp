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

#include <sys/time.h>
#include "glog/logging.h"
#ifdef USE_MLU
#include <cnrt.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "caffe/mlu/data_trans.hpp"

using std::string;
using std::vector;

DEFINE_int32(mludevice, 0, "set using mlu device number, default: 0");
DEFINE_int32(apiversion, 2, "specify the version of CNRT to run.");

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
  CHECK_LE(FLAGS_apiversion, 2) << "The version number should be 1 or 2";
  CHECK_GE(FLAGS_apiversion, 1) << "The version number should be 1 or 2";

  cnrtDev_t dev;
  cnrtGetDeviceHandle(&dev, FLAGS_mludevice);
  cnrtSetCurrentDevice(dev);
  // 2. load model and get function
  cnrtModel_t model;
  string fname = (string)argv[1];
  LOG(INFO) << "load file: " << fname;
  cnrtLoadModel(&model, fname.c_str());
  cnrtFunction_t function;

  struct timeval tpend, tpstart;
  gettimeofday(&tpstart, NULL);

  for (int n = 3; n < argc; n++) {
    string name = (string)argv[n];
    cnrtCreateFunction(&function);
    cnrtExtractFunction(&function, model, name.c_str());
    // 3. get function's I/O DataDesc
    int inputNum, outputNum;
    int64_t* inputSizeArray;
    int64_t* outputSizeArray;
    cnrtDataType_t* inputDataTypeArray;
    cnrtDataType_t* outputDataTypeArray;
    CNRT_CHECK(cnrtGetInputDataSize(&inputSizeArray, &inputNum, function));
    CNRT_CHECK(cnrtGetOutputDataSize(&outputSizeArray, &outputNum, function));
    CNRT_CHECK(cnrtGetInputDataType(&inputDataTypeArray, &inputNum, function));
    CNRT_CHECK(cnrtGetOutputDataType(&outputDataTypeArray, &outputNum, function));

    // 4. allocate I/O data space on CPU memory and prepare Input data
    void** inputCpuPtrS = reinterpret_cast<void**>(malloc(sizeof(void*) * inputNum));
    void** inputSyncPtrS = reinterpret_cast<void**>(malloc(sizeof(void*) * inputNum));
    void** inputSyncTmpPtrS = reinterpret_cast<void**>(malloc(sizeof(void*) * inputNum));
    void** outputCpuPtrS = reinterpret_cast<void**>(malloc(sizeof(void*) * outputNum));
    void** outputSyncPtrS = reinterpret_cast<void**>(malloc(sizeof(void*) * outputNum));
    vector<float*> output_cpu;
    vector<int> in_count;
    vector<int> out_count;
    vector<vector<int>> inputShape;
    vector<vector<int>> outputShape;
    void** param =
        reinterpret_cast<void**>(malloc(sizeof(void*) * (inputNum + outputNum)));
    srand(10);
    float* outcpu;
    float* databuf;
    vector<int*> inputDimValues(inputNum, 0);
    vector<int> inputDimNumS(inputNum, 0);
    vector<int*> outputDimValues(outputNum, 0);
    vector<int> outputDimNumS(outputNum, 0);
    for (int i = 0; i < inputNum; i++) {
      databuf = reinterpret_cast<float*>(malloc(sizeof(float) * inputSizeArray[i]));
      if (i == 0) {
        rand1(databuf, inputSizeArray[i]);
      } else {
        rand2(databuf, inputSizeArray[i]);
      }
      in_count.push_back(inputSizeArray[i]);
      inputCpuPtrS[i] = reinterpret_cast<void*>(databuf);
      inputSyncPtrS[i] = reinterpret_cast<void*>(malloc(inputSizeArray[i]));
      inputSyncTmpPtrS[i] = reinterpret_cast<void*>(malloc(inputSizeArray[i] / 4 * 3));
      cnrtGetInputDataShape(&(inputDimValues[i]), &(inputDimNumS[i]), i, function);
      vector<int> in_shape;
      for (int idx = 0; idx < inputDimNumS[i]; idx++) {
        in_shape.push_back(inputDimValues[i][idx]);
      }
      inputShape.push_back(in_shape);
    }
    for (int i = 0; i < outputNum; i++) {
      int outDataCount = 1;
      cnrtGetOutputDataShape(&(outputDimValues[i]), &(outputDimNumS[i]), i, function);
      vector<int> out_shape;
      for (int idx = 0; idx < outputDimNumS[i]; idx++) {
        outDataCount = outDataCount * outputDimValues[i][idx];
        out_shape.push_back(outputDimValues[i][idx]);
      }
      outputShape.push_back(out_shape);
      outcpu = reinterpret_cast<float*>(malloc(outDataCount * sizeof(float)));
      out_count.push_back(outDataCount);
      output_cpu.push_back(outcpu);
      outputCpuPtrS[i] = reinterpret_cast<void*>(outcpu);
      outputSyncPtrS[i] =
          reinterpret_cast<void*>(malloc(outputSizeArray[i]));
    }
    // 5. allocate I/O data space on MLU memory and copy Input data
    // Only 1 batch so far
    void** inputMluPtrS = reinterpret_cast<void**>(malloc(sizeof(void*) * inputNum));
    void** outputMluPtrS = reinterpret_cast<void**>(malloc(sizeof(void*) * outputNum));
    for (int i = 0; i < inputNum; i++) {
      cnrtMalloc(&(inputMluPtrS[i]), inputSizeArray[i]);
      param[i] = inputMluPtrS[i];
    }
    for (int i = 0; i < outputNum; i++) {
      cnrtMalloc(&(outputMluPtrS[i]), outputSizeArray[i]);
      param[inputNum + i] = outputMluPtrS[i];
    }
    // 6. create cnrt_queue and run function
    cnrtQueue_t cnrt_queue;
    cnrtCreateQueue(&cnrt_queue);
    cnrtRuntimeContext_t rt_ctx;
    if (cnrtCreateRuntimeContext(&rt_ctx, function, nullptr) != CNRT_RET_SUCCESS) {
      LOG(FATAL)<< "Failed to create runtime context";
    }
    cnrtSetRuntimeContextDeviceId(rt_ctx, FLAGS_mludevice);
    cnrtInitRuntimeContext(rt_ctx, NULL);
    cnrtDataType_t inputCpuDtype = CNRT_FLOAT32;
    cnrtDataType_t inputMluDtype = inputDataTypeArray[0];

    for (int i = 0; i < inputNum; i++) {
      bool useFirstConv = inputMluDtype == CNRT_UINT8 && inputShape[i][3] == 4;
      caffe::transAndCast(reinterpret_cast<void*>(inputCpuPtrS[i]), inputCpuDtype,
                   reinterpret_cast<void*>(inputSyncPtrS[i]), inputMluDtype,
                   reinterpret_cast<void*>(inputSyncTmpPtrS[i]),
                   caffe::to_cpu_shape(inputShape[i]),
                   useFirstConv, "CPU2MLU");

      CNRT_CHECK(cnrtMemcpy(inputMluPtrS[i], inputSyncPtrS[i],
            inputSizeArray[i], CNRT_MEM_TRANS_DIR_HOST2DEV));
    }
    // create start_event and end_event
    cnrtNotifier_t notifierBeginning, notifierEnd;
    cnrtCreateNotifier(&notifierBeginning);
    cnrtCreateNotifier(&notifierEnd);
    float event_time_use;
    // run MLU
    // place start_event to cnrt_queue
    cnrtPlaceNotifier(notifierBeginning, cnrt_queue);
    CNRT_CHECK(cnrtInvokeRuntimeContext(rt_ctx, param, cnrt_queue, nullptr));
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
    cnrtDataType_t outputCpuDtype = CNRT_FLOAT32;
    cnrtDataType_t outputMluDtype = outputDataTypeArray[0];
    for (int i = 0; i < outputNum; i++) {
      CNRT_CHECK(cnrtMemcpy(outputSyncPtrS[i], outputMluPtrS[i],
            outputSizeArray[i], CNRT_MEM_TRANS_DIR_DEV2HOST));
      caffe::transAndCast(reinterpret_cast<void*>(outputSyncPtrS[i]),
                   outputMluDtype,
                   reinterpret_cast<void*>(outputCpuPtrS[i]),
                   outputCpuDtype,
                   nullptr,
                   outputShape[i],
                   false,
                   "MLU2CPU");
    }
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
    free(inputSyncPtrS);
    free(inputSyncTmpPtrS);
    free(outputSyncPtrS);
    cnrtFreeArray(inputMluPtrS, inputNum);
    cnrtFreeArray(outputMluPtrS, outputNum);
    cnrtDestroyQueue(cnrt_queue);
    cnrtDestroyFunction(function);
    cnrtDestroyRuntimeContext(rt_ctx);
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
