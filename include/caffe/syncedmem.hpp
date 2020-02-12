/*
All modification made by Cambricon Corporation: © 2018-2019 Cambricon Corporation
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

#ifndef INCLUDE_CAFFE_SYNCEDMEM_HPP_
#define INCLUDE_CAFFE_SYNCEDMEM_HPP_

#include <cstdlib>
#include <vector>

#ifdef USE_MKL
  #include "mkl.h"
#endif

#include "caffe/common.hpp"
#include "caffe/mlu/tensor.hpp"

namespace caffe {

// If CUDA is available and in GPU mode, host memory will be allocated pinned,
// using cudaMallocHost. It avoids dynamic pinning for transfers (DMA).
// The improvement in performance seems negligible in the single GPU case,
// but might be more significant for parallel training. Most importantly,
// it improved stability for large models on many GPUs.
inline void CaffeMallocHost(void** ptr, size_t size, bool* use_cuda) {
#ifdef USE_CUDA
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaMallocHost(ptr, size));
    *use_cuda = true;
    return;
  }
#endif
#ifdef USE_MKL
  *ptr = mkl_malloc(size ? size:1, 64);
#else
  *ptr = malloc(size);
#endif
  *use_cuda = false;
  CHECK(*ptr) << "host allocation of size " << size << " failed";
}

inline void CaffeFreeHost(void* ptr, bool use_cuda) {
#ifdef USE_CUDA
  if (use_cuda) {
    CUDA_CHECK(cudaFreeHost(ptr));
    return;
  }
#endif
#ifdef USE_MKL
  mkl_free(ptr);
#else
  free(ptr);
#endif
}


/**
 * @brief Manages memory allocation and synchronization between the host (CPU)
 *        and device (GPU) (MLU).
 *
 * We use a finite state machine to manage the cpu,mlu,gpu memory in caffe.
 * different states are enumerated in SyncedHead.
 *
 * Six functions can trigger these states:cpu_data(),gpu_data(),mlu_data(),
 * mutable_cpu_data(),mutable_gpu_data(),mutable_mlu_data().
 *
 * Six transfer functions can transfer a state to another:to_cpu(),to_gpu(),
 * to_mlu(),mutable_cpu(),mutable_gpu(),mutable_mlu().
 *
 * To get a thorough understanding, see the FSM figure on cambricon wiki or
 * on the internet.
 */

class SyncedMemory {
  public:
  SyncedMemory();
  explicit SyncedMemory(size_t size);

  ~SyncedMemory();
  const void* cpu_data();
  void set_cpu_data(void* data);
  const void* gpu_data();
  void set_gpu_data(void* data);
  void* mutable_cpu_data();
  void* mutable_gpu_data();
#ifdef USE_MLU
  void* mutable_cpu_data(const MLUTensorDesc& mlu_tensor_desc);
  const void* cpu_data(const MLUTensorDesc& mlu_tensor_desc);
  void set_mlu_data(void* data);
  void* mutable_mlu_data(const MLUTensorDesc& mlu_tensor_desc);
  const void* mlu_data(const MLUTensorDesc& mlu_tensor_desc);
  void* mutable_sync_data(const MLUTensorDesc& mlu_tensor_desc);
#endif
  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, HEAD_AT_MLU, SYNCED };
  SyncedHead head() { return head_; }
  size_t size() { return size_; }

#ifdef USE_CUDA
  void async_gpu_push(const cudaStream_t& stream);
#endif

  private:
  void check_device();

  void to_cpu();
  void to_gpu();
  void* cpu_ptr_;
  void* gpu_ptr_;
  size_t size_;
  SyncedHead head_;

#ifdef USE_MLU
  void* mlu_ptr_;
  void* sync_ptr_; // sync mlu data to cpu, mlu data format: float16, int8, uint8 etc
  void to_cpu(const MLUTensorDesc& mlu_tensor_desc);
  void to_mlu(const MLUTensorDesc& mlu_tensor_desc);
  void to_sync(const MLUTensorDesc& mlu_tensor_desc);
#endif

  bool own_cpu_data_;
  bool cpu_malloc_use_cuda_;
  bool own_gpu_data_;
#ifdef USE_MLU
  bool own_mlu_data_;
#endif
  int device_;

  DISABLE_COPY_AND_ASSIGN(SyncedMemory);
};  // class SyncedMemory

}  // namespace caffe

#endif  // INCLUDE_CAFFE_SYNCEDMEM_HPP_
