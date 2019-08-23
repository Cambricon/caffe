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

#include <vector>

#include "caffe/common.hpp"
#include "caffe/mlu/tensor.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

SyncedMemory::SyncedMemory()
  : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(0), head_(UNINITIALIZED),
#ifdef USE_MLU
    mlu_ptr_(nullptr),
#endif
    own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false)
#ifdef USE_MLU
    , own_mlu_data_(false) {
#else
    {
#endif
#ifdef USE_CUDA
#ifdef DEBUG
  CUDA_CHECK(cudaGetDevice(&device_));
#endif
#endif
}

SyncedMemory::SyncedMemory(size_t size)
  : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
#ifdef USE_MLU
    mlu_ptr_(nullptr),
#endif
    own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false)
#ifdef USE_MLU
    , own_mlu_data_(false) {
#else
    {
#endif
#ifdef USE_CUDA
#ifdef DEBUG
  CUDA_CHECK(cudaGetDevice(&device_));
#endif
#endif
}

SyncedMemory::~SyncedMemory() {
  check_device();
  if (cpu_ptr_ && own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
  }

#ifdef USE_CUDA
  if (gpu_ptr_ && own_gpu_data_) {
    CUDA_CHECK(cudaFree(gpu_ptr_));
  }
#endif  // CPU_ONLY

#ifdef USE_MLU
  if (mlu_ptr_ && own_mlu_data_) {
    MLU_CHECK(cnmlFreeBuffer(mlu_ptr_));
  }
#endif
}

inline void SyncedMemory::to_cpu() {
  check_device();
  switch (head_) {
  case UNINITIALIZED:
    CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
    caffe_memset(size_, 0, cpu_ptr_);
    head_ = HEAD_AT_CPU;
    own_cpu_data_ = true;
    break;
  case HEAD_AT_GPU:
#ifdef USE_CUDA
    if (cpu_ptr_ == NULL) {
      CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
      own_cpu_data_ = true;
    }
    caffe_gpu_memcpy(size_, gpu_ptr_, cpu_ptr_);
    head_ = SYNCED;
#else
    NO_GPU;
#endif
    break;
#ifdef USE_MLU
  case HEAD_AT_MLU:
    LOG(FATAL) << "Head is AT_MLU, tensor descriptor is needed to copy data to CPU!";
    break;
#endif
  case HEAD_AT_CPU:
  case SYNCED:
    break;
  default:
    NOT_IMPLEMENTED;
  }
}

inline void SyncedMemory::to_gpu() {
  check_device();
#ifdef USE_CUDA
  switch (head_) {
  case UNINITIALIZED:
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    caffe_gpu_memset(size_, 0, gpu_ptr_);
    head_ = HEAD_AT_GPU;
    own_gpu_data_ = true;
    break;
  case HEAD_AT_CPU:
    if (gpu_ptr_ == NULL) {
      CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
      own_gpu_data_ = true;
    }
    caffe_gpu_memcpy(size_, cpu_ptr_, gpu_ptr_);
    head_ = SYNCED;
    break;
  case HEAD_AT_MLU:
    LOG(FATAL) << "Head is AT_MLU, data can not be copied to GPU!";
    break;
  case HEAD_AT_GPU:
  case SYNCED:
    break;
  default:
    NOT_IMPLEMENTED;
  }
#else
  NO_GPU;
#endif
}

const void* SyncedMemory::cpu_data() {
  check_device();
  to_cpu();
  return (const void*)cpu_ptr_;
}

void SyncedMemory::set_cpu_data(void* data) {
  check_device();
  CHECK(data);
  if (own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
  }
  cpu_ptr_ = data;
  head_ = HEAD_AT_CPU;
  own_cpu_data_ = false;
}

const void* SyncedMemory::gpu_data() {
  check_device();
#ifdef USE_CUDA
  to_gpu();
  return (const void*)gpu_ptr_;
#else
  NO_GPU;
  return NULL;
#endif
}

void SyncedMemory::set_gpu_data(void* data) {
  check_device();
#ifdef USE_CUDA
  CHECK(data);
  if (own_gpu_data_) {
    CUDA_CHECK(cudaFree(gpu_ptr_));
  }
  gpu_ptr_ = data;
  head_ = HEAD_AT_GPU;
  own_gpu_data_ = false;
#else
  NO_GPU;
#endif
}

void* SyncedMemory::mutable_cpu_data() {
  check_device();
  to_cpu();
  head_ = HEAD_AT_CPU;
  return cpu_ptr_;
}

void* SyncedMemory::mutable_gpu_data() {
  check_device();
#ifdef USE_CUDA
  to_gpu();
  head_ = HEAD_AT_GPU;
  return gpu_ptr_;
#else
  NO_GPU;
  return NULL;
#endif
}

#ifdef USE_MLU

void SyncedMemory::set_mlu_data(void* data) {
  CHECK(data);
  if (own_mlu_data_) {
    MLU_CHECK(cnmlFreeBuffer(mlu_ptr_));
  }
  mlu_ptr_ = data;
  head_ = HEAD_AT_MLU;
  own_mlu_data_ = false;
}

void* SyncedMemory::mutable_cpu_data(const MLUTensorDesc& mlu_tensor_desc) {
  check_device();
  to_cpu(mlu_tensor_desc);
  head_ = HEAD_AT_CPU;
  return cpu_ptr_;
}

const void* SyncedMemory::cpu_data(const MLUTensorDesc& mlu_tensor_desc) {
  check_device();
  to_cpu(mlu_tensor_desc);
  return (const void*)cpu_ptr_;
}
inline void SyncedMemory::to_cpu(const MLUTensorDesc& mlu_tensor_desc) {
  check_device();
  switch (head_) {
  case UNINITIALIZED:
    CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
    caffe_memset(size_, 0, cpu_ptr_);
    head_ = HEAD_AT_CPU;
    own_cpu_data_ = true;
    break;
  case HEAD_AT_GPU:
#ifdef USE_CUDA
    if (cpu_ptr_ == NULL) {
      CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
      own_cpu_data_ = true;
    }
    caffe_gpu_memcpy(size_, gpu_ptr_, cpu_ptr_);
    head_ = SYNCED;
#else
    NO_GPU;
#endif
    break;
#ifdef USE_MLU
  case HEAD_AT_MLU:
    if (cpu_ptr_ == NULL) {
      CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
      own_cpu_data_ = true;
    }
    MLU_CHECK(cnmlMemcpyBatchTensorToHost(mlu_tensor_desc.mlu(),
         mlu_ptr_, mlu_tensor_desc.cpu(), cpu_ptr_, Caffe::data_parallel()));
    head_ = SYNCED;
    break;
#endif
  case HEAD_AT_CPU:
  case SYNCED:
    break;
  default:
    NOT_IMPLEMENTED;
  }
}

inline void SyncedMemory::to_mlu(const MLUTensorDesc& mlu_tensor_desc) {
  switch (head_) {
  case UNINITIALIZED:
    CHECK(mlu_ptr_ == nullptr);
    mlu_ptr_ = cnmlMallocBatchBuffer(
        mlu_tensor_desc.mlu(), Caffe::data_parallel());
    CHECK_NOTNULL(mlu_ptr_);
    head_ = HEAD_AT_MLU;
    own_mlu_data_ = true;
    break;
  case HEAD_AT_GPU:
    LOG(FATAL) << "Head is AT_GPU, data can not be copied from GPU to MLU!";
    break;
  case HEAD_AT_CPU:
    CHECK_NOTNULL(cpu_ptr_);
    if (mlu_ptr_ == nullptr) {
      mlu_ptr_ = cnmlMallocBatchBuffer(
          mlu_tensor_desc.mlu(), Caffe::data_parallel());
      CHECK_NOTNULL(mlu_ptr_);
    }
    MLU_CHECK(cnmlMemcpyBatchTensorToDevice(mlu_tensor_desc.cpu(),
        cpu_ptr_, mlu_tensor_desc.mlu(), mlu_ptr_, Caffe::data_parallel()));
    head_ = SYNCED;
    own_mlu_data_ = true;
    break;
  case HEAD_AT_MLU:
  case SYNCED:
    break;
  default:
    NOT_IMPLEMENTED;
  }
}

void* SyncedMemory::mutable_mlu_data(const MLUTensorDesc& mlu_tensor_desc) {
  to_mlu(mlu_tensor_desc);
  head_ = HEAD_AT_MLU;
  return mlu_ptr_;
}

const void* SyncedMemory::mlu_data(const MLUTensorDesc& mlu_tensor_desc) {
  to_mlu(mlu_tensor_desc);
  return (const void*)mlu_ptr_;
}

#endif


#ifdef USE_CUDA
void SyncedMemory::async_gpu_push(const cudaStream_t& stream) {
  check_device();
  CHECK(head_ == HEAD_AT_CPU);
  if (gpu_ptr_ == NULL) {
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    own_gpu_data_ = true;
  }
  const cudaMemcpyKind put = cudaMemcpyHostToDevice;
  CUDA_CHECK(cudaMemcpyAsync(gpu_ptr_, cpu_ptr_, size_, put, stream));
  // Assume caller will synchronize on the stream before use
  head_ = SYNCED;
}
#endif

void SyncedMemory::check_device() {
#ifdef USE_CUDA
#ifdef DEBUG
  int device;
  cudaGetDevice(&device);
  CHECK(device == device_);
  if (gpu_ptr_ && own_gpu_data_) {
    cudaPointerAttributes attributes;
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, gpu_ptr_));
    CHECK(attributes.device == device_);
  }
#endif
#endif
}

}  // namespace caffe

