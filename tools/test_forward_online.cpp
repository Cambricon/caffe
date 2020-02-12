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

#include <caffe/caffe.hpp>
#include <iostream>
#include <set>
#include <string>

DEFINE_int32(mludevice, 0,
    "set using mlu device number, default: 0");
DEFINE_int32(batchsize, -1, "Read images size every batch for inference");
DEFINE_int32(core_number, 1, "The number of cores are used for inference");
DEFINE_string(output_dtype, "FLOAT16", "output_dtype");

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

void release_resource(int mlu_option) {
#ifdef USE_MLU
  if (mlu_option > 0) {
    caffe::Caffe::freeQueue();
    cnmlExit();
  }
#endif
}

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;
#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (argc != 6 && argc != 7 && argc != 8) {
    LOG(ERROR) << "Usage1: " << argv[0] << " proto_file model_file"
              << " mlu_option mcore output_file";
    LOG(ERROR) << "Usage2: " << argv[0] << " proto_file model_file"
              << " mlu_option mcore output_file outputaddr_file"
              << " outputsize_file";
    exit(1);
  }
  std::string proto_file = argv[1];
  std::string model_file = argv[2];

  std::stringstream ss;
  int mlu_option;
  ss << argv[3];
  ss >> mlu_option;
  if (mlu_option < 0 || mlu_option > 2) {
    LOG(ERROR) << "Unrecognized mlu option: " << mlu_option
              << "Available options: 0(cpu), 1(mlu), 2(mfus: mlu with fusion).";
    exit(1);
  }

  std::string mcore = argv[4];
  if ((mcore != std::string("MLU100")) &&
      (mcore != std::string("1H16")) &&
      (mcore != std::string("1H8")) &&
      (mcore != std::string("MLU220")) &&
      (mcore != std::string("MLU270"))) {
    LOG(ERROR) << "Unrecognized mcore option: " << mcore
              << "Available options: MLU100, 1H16, 1H8, MLU270，MLU220.";
    exit(1);
  }

#ifdef USE_MLU
  LOG(INFO) << ">>> test forward >>> mlu option ???" << mlu_option;
  if (mlu_option > 0) {
    cnmlInit(0);
    caffe::Caffe::set_rt_core(argv[4]);
    caffe::Caffe::set_mlu_device(FLAGS_mludevice);
    caffe::Caffe::set_mode(caffe::Caffe::MLU);
    caffe::Caffe::setTopDataType(FLAGS_output_dtype);
    if (mlu_option == 2) {
      caffe::Caffe::set_mode(caffe::Caffe::MFUS);
    }
    caffe::Caffe::setReshapeMode(caffe::Caffe::ReshapeMode::SETUPONLY);
  } else {
    LOG(INFO) << "Use CPU.";
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
  }
  if (FLAGS_batchsize > 0) {
    caffe::Caffe::setBatchsize(FLAGS_batchsize);
    caffe::Caffe::setCoreNumber(FLAGS_core_number);
    caffe::Caffe::setSimpleFlag(true);
  }
#else
  LOG(INFO) << "Use CPU.";
  caffe::Caffe::set_mode(caffe::Caffe::CPU);
#endif

  caffe::Net<float>* net = new caffe::Net<float>(proto_file, caffe::TEST);
  if (model_file != std::string("NULL")) {
    net->CopyTrainedLayersFrom(model_file);
  }

  for (int k = 0; k < net->input_blobs().size(); k++) {
    if (k == 0) {
      rand1(net->input_blobs()[k]->mutable_cpu_data(), net->input_blobs()[k]->count());
    } else {
      rand2(net->input_blobs()[k]->mutable_cpu_data(), net->input_blobs()[k]->count());
    }
  }

  if (argc == 7) {
    LOG(WARNING) << "This usage is not supported any more. Use other appropriate ones.";
    release_resource(mlu_option);
    exit(2);
  } else {
    net->Forward();
  }
  LOG(INFO) << "Forward finished.";
  if (std::string(argv[5]) != std::string("NULL") && argc != 8) {
    std::ofstream fout(argv[5], std::ios::out);
    for (auto blob : net->output_blobs()) {
      for (int i = 0; i < blob->count(); i++) {
        fout << blob->cpu_data()[i] << std::endl;
      }
    }
    fout << std::flush;
    fout.close();
  } else {
    if (std::string(argv[6]) != std::string("NULL")) {
      if (std::string(argv[7]) != std::string("NULL")) {
        std::ofstream fout1(argv[6], std::ios::out);
        std::ofstream fout2(argv[7], std::ios::out);
        for (int k = 0; k < net->output_blobs().size(); k++) {
          fout1 << net->output_blobs()[k]->cpu_data() << std::endl;
          fout2 << net->output_blobs()[k]->count() << std::endl;
        }
        fout1 << std::flush;
        fout2 << std::flush;
        fout1.close();
        fout2.close();
      } else {
        LOG(INFO) << "Usage: " << argv[0] << " proto_file model_file mlu_option mcore"
                  << " output_file outputaddr_file outputsize_file";
        release_resource(mlu_option);
        exit(1);
      }
    }
  }

  delete net;
  release_resource(mlu_option);
  return 0;
}
