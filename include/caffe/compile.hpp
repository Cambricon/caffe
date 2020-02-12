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

#ifndef INCLUDE_CAFFE_COMPILE_HPP_
#define INCLUDE_CAFFE_COMPILE_HPP_
#ifdef USE_MLU
#include <string>
#include <vector>
#include "cnml.h"  // NOLINT

namespace caffe {

/**
 * @brief Save offline model into a specific file.
 *
 * @param modelType deprecated
 * @param path input prototxt file path and caffe weights file path
 * @param buildpath target file path
 * @param buildType core version
 * @param hardwareReshape if true, use MLU to reshape data
 *
*/
bool compile(int modelType, std::vector<std::string> *path,
      std::string *buildpath, cnmlCoreVersion_t buildType,
      std::string name);
/**
 * brief Save offline model into a specific buffer.
 *
 * @param modelType deprecated
 * @param buffer input prototxt and caffe weights buffers
 * @param buffersize the size input buffers
 * @param buildbuffer the output buffer
 * @param buildbuffersize the size of output buffer
 * @param modelsize The size of gennerated offline model
 * @param buildType core version
 * @param hardwareReshape if true, use MLU to reshape data
 * @param name generated offline model name
 *
 */
  bool compile(int modelType, std::vector<uint8_t*> buffer,
              std::vector<uint32_t> buffersize,
              uint8_t* buildbuffer,
              uint32_t buildbuffersize,
              uint32_t& modelsize,  //NOLINT
              cnmlCoreVersion_t buildType);
}  // namespace caffe
#endif  // USE_MLU
#endif  // INCLUDE_CAFFE_COMPILE_HPP_
