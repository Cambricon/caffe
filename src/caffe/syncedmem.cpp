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

#include <vector>

#include "caffe/common.hpp"
#include "caffe/mlu/tensor.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

static inline void cast_data_type(void* src_addr, void* dst_addr,
   cnrtMemTransDir_t dir, const MLUTensorDesc& mlu_tensor_desc) {
  cnrtDataType_t src_data_type, dst_data_type;
  BaseDataType mlu_type = mlu_tensor_desc.mlu_type();
  cnmlTensorType_t type = mlu_tensor_desc.type();
  int shape_dim = mlu_tensor_desc.shape_dim();
  vector<int> dim_values;
  vector<int> dim_order(shape_dim, 0);
  size_t size;
  MLU_CHECK(cnmlGetTensorSize_V2(mlu_tensor_desc.mlu(), &size));
  cnrtDataType_t cpu_dtype = to_cnrt_dtype(mlu_tensor_desc.cpu_type());
  cnrtDataType_t mlu_dtype = to_cnrt_dtype(mlu_tensor_desc.mlu_type());

  // init
  if (dir == CNRT_MEM_TRANS_DIR_HOST2DEV) {
    src_data_type = cpu_dtype;
    dst_data_type = mlu_dtype;
    dim_values = mlu_tensor_desc.cpu_shape();
    dim_order[0] = 0;
    dim_order[shape_dim - 1] = 1;
    for (int i = 1; i < shape_dim - 1; i++) {
      dim_order[i] = i + 1;
    }
  } else if (dir == CNRT_MEM_TRANS_DIR_DEV2HOST) {
    if (mlu_tensor_desc.has_dim_strides())
       LOG(WARNING) << "The data is supplemented with the stride,"
         << " and the data is synchronized from the mlu device.";
    src_data_type = mlu_dtype;
    dst_data_type = cpu_dtype;
    dim_values = mlu_tensor_desc.mlu_shape();
    dim_order[0] = 0;
    dim_order[1] = shape_dim - 1;
    for (int i = 2 ; i < shape_dim; i++) {
      dim_order[i] = i - 1;
    }
  } else {
    LOG(FATAL) << "Unsurported cast type: " << dir;
  }

  // quantized param
  cnrtQuantizedParam_t param = nullptr;
  if ((mlu_type == DT_INT8 || mlu_type == DT_INT16) &&
      (type == CNML_FILTER || type == CNML_CONST)) {
    if (!mlu_tensor_desc.has_position()) {
      LOG(FATAL) << "Quantize tensor should have position";
    }
    CNRT_CHECK(cnrtCreateQuantizedParam(&param, mlu_tensor_desc.position(),
          mlu_tensor_desc.scale(), 0));
  }

  // castDataType
  int cast_size = mlu_tensor_desc.data_num() * cnrtDataTypeSize(dst_data_type);
  void* cast_ptr = malloc(cast_size);
  if (src_data_type != dst_data_type) {
     cnrtCastDataType(src_addr, src_data_type, cast_ptr, dst_data_type,
         mlu_tensor_desc.data_num(), param);
  } else {
     memcpy(cast_ptr, src_addr, cast_size);
  }

  // addStrideData
  void* stride_ptr = nullptr;
  vector<int> dim_value_for_trans = dim_values;
  bool add_stride = mlu_tensor_desc.has_dim_strides() &&
    dir == CNRT_MEM_TRANS_DIR_HOST2DEV;
  if (add_stride) {
     stride_ptr = malloc(size);
     cnrtAddDataStride(cast_ptr, dst_data_type, stride_ptr, shape_dim,
        mlu_tensor_desc.cpu_shape().data(), mlu_tensor_desc.dim_strides().data());
     for (int i = 0; i < mlu_tensor_desc.shape_dim(); i++) {
        dim_value_for_trans[i] += mlu_tensor_desc.dim_strides().data()[i];
     }
  }

  // transDataOrder
  if (!mlu_tensor_desc.is_preprocess()) {
    memcpy(dst_addr, cast_ptr, cast_size);
    return;
  }
  cnrtTransDataOrder(add_stride? stride_ptr: cast_ptr,
                     dst_data_type, dst_addr, shape_dim,
                     dim_value_for_trans.data(), dim_order.data());

  // free
  if (param) {
    CNRT_CHECK(cnrtDestroyQuantizedParam(param));
  }
  if (cast_ptr != nullptr) free(cast_ptr);
  if (stride_ptr != nullptr) free(stride_ptr);
}

SyncedMemory::SyncedMemory()
  : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(0), head_(UNINITIALIZED),
#ifdef USE_MLU
    mlu_ptr_(nullptr), sync_ptr_(NULL),
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
    mlu_ptr_(nullptr), sync_ptr_(NULL),
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
    CNRT_CHECK(cnrtFree(mlu_ptr_));
  }
  if (sync_ptr_) {
    CaffeFreeHost(sync_ptr_, cpu_malloc_use_cuda_);  // free sync memmory
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

void* SyncedMemory::mutable_sync_data(const MLUTensorDesc& mlu_tensor_desc) {
  check_device();
  to_sync(mlu_tensor_desc);
  return sync_ptr_;
}

void SyncedMemory::to_sync(const MLUTensorDesc& mlu_tensor_desc) {
  // bind const data
  switch (head_) {
  case HEAD_AT_CPU:
    CHECK_NOTNULL(cpu_ptr_);
    size_t sync_cpu_size;
    MLU_CHECK(cnmlGetTensorSize_V2(mlu_tensor_desc.mlu(), &sync_cpu_size));
    if (sync_ptr_ == NULL) {
      CaffeMallocHost(&sync_ptr_, sync_cpu_size, &cpu_malloc_use_cuda_);
    }
    cast_data_type(cpu_ptr_, sync_ptr_, CNRT_MEM_TRANS_DIR_HOST2DEV, mlu_tensor_desc);
    break;
  case SYNCED:
    break;
  case UNINITIALIZED:
    if (cpu_ptr_ == NULL) {
      CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
      caffe_memset(size_, 0, cpu_ptr_);
      head_ = HEAD_AT_CPU;
    }
    own_cpu_data_ = true;
    CHECK_NOTNULL(cpu_ptr_);
    size_t sync_uninitalized_size;
    MLU_CHECK(cnmlGetTensorSize_V2(mlu_tensor_desc.mlu(), &sync_uninitalized_size));
    if (sync_ptr_ == NULL) {
      CaffeMallocHost(&sync_ptr_, sync_uninitalized_size, &cpu_malloc_use_cuda_);
    }
    break;
  case HEAD_AT_GPU:
    break;
  case HEAD_AT_MLU:
    break;
  default:
    NOT_IMPLEMENTED;
  }
}

void SyncedMemory::set_mlu_data(void* data) {
  CHECK(data);
  if (own_mlu_data_) {
    CNRT_CHECK(cnrtFree(mlu_ptr_));
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
    size_t cpu_sync_size;
    MLU_CHECK(cnmlGetTensorSize_V2(mlu_tensor_desc.mlu(), &cpu_sync_size));
    if (cpu_ptr_ == NULL) {
      CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
      own_cpu_data_ = true;
    }
    if (sync_ptr_ == NULL) {
      CaffeMallocHost(&sync_ptr_, cpu_sync_size, &cpu_malloc_use_cuda_);
    }
    CNRT_CHECK(cnrtMemcpy(sync_ptr_, mlu_ptr_, cpu_sync_size,
          CNRT_MEM_TRANS_DIR_DEV2HOST));
    cast_data_type(sync_ptr_, cpu_ptr_, CNRT_MEM_TRANS_DIR_DEV2HOST, mlu_tensor_desc);
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
    size_t mlu_uninitalized_size;
    MLU_CHECK(cnmlGetTensorSize_V2(mlu_tensor_desc.mlu(), &mlu_uninitalized_size));
    CNRT_CHECK(cnrtMalloc(&mlu_ptr_, mlu_uninitalized_size));
    CHECK_NOTNULL(mlu_ptr_);
    head_ = HEAD_AT_MLU;
    own_mlu_data_ = true;
    break;
  case HEAD_AT_GPU:
    LOG(FATAL) << "Head is AT_GPU, data can not be copied from GPU to MLU!";
    break;
  case HEAD_AT_CPU:
    CHECK_NOTNULL(cpu_ptr_);
    size_t mlu_cpu_size;
    MLU_CHECK(cnmlGetTensorSize_V2(mlu_tensor_desc.mlu(), &mlu_cpu_size));
    if (mlu_ptr_ == nullptr) {
      CNRT_CHECK(cnrtMalloc(&mlu_ptr_, mlu_cpu_size));
      CHECK_NOTNULL(mlu_ptr_);
    }
    if (sync_ptr_ == NULL) {
      CaffeMallocHost(&sync_ptr_, mlu_cpu_size, &cpu_malloc_use_cuda_);
    }
    cast_data_type(cpu_ptr_, sync_ptr_, CNRT_MEM_TRANS_DIR_HOST2DEV, mlu_tensor_desc);
    CNRT_CHECK(cnrtMemcpy(mlu_ptr_, sync_ptr_, mlu_cpu_size,
          CNRT_MEM_TRANS_DIR_HOST2DEV));
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
