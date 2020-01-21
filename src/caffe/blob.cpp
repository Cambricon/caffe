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

#include <climits>
#include <vector>

#include "algorithm"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/mlu/util.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

#ifdef USE_MLU

template <typename Dtype>
Blob<Dtype>::Blob(const int num, const int channels, const int height,
                  const int width, BaseDataType cpu_type, BaseDataType mlu_type,
                  cnmlTensorType_t tensor_type)
    : capacity_(0),
      shape_order_(CNML_NCHW) {
  Reshape(num, channels, height, width, cpu_type, mlu_type, tensor_type);
}

template <typename Dtype>
Blob<Dtype>::Blob(const vector<int>& shape, BaseDataType cpu_type,
                  BaseDataType mlu_type, cnmlTensorType_t tensor_type,
                  cnmlDataOrder_t shape_order,
                  vector<int>* dim_strides)
    : capacity_(0),
      shape_order_(shape_order) {
  Reshape(shape, cpu_type, mlu_type, tensor_type, shape_order, dim_strides);
}

template <typename Dtype>
void Blob<Dtype>::Reshape(const int num, const int channels, const int height,
                          const int width, BaseDataType cpu_type,
                          BaseDataType mlu_type, cnmlTensorType_t tensor_type) {
  vector<int> shape(4);
  shape[0] = num;
  shape[1] = channels;
  shape[2] = height;
  shape[3] = width;
  Reshape(shape, cpu_type, mlu_type, tensor_type, CNML_NCHW);
}

template <typename Dtype>
void Blob<Dtype>::Reshape(const vector<int>& shape, BaseDataType cpu_type,
                          BaseDataType mlu_type, cnmlTensorType_t tensor_type,
                          cnmlDataOrder_t shape_order,
                          vector<int>* dim_strides) {
  Reshape(shape);
  tensor_desc_.remember(shape, tensor_type, cpu_type, mlu_type,
                            shape_order, dim_strides);
}

template <typename Dtype>
void Blob<Dtype>::Reshape(const BlobShape& shape, BaseDataType cpu_type,
                          BaseDataType mlu_type, cnmlTensorType_t tensor_type,
                          cnmlDataOrder_t shape_order,
                          vector<int>* dim_strides) {
  CHECK_LE(shape.dim_size(), kMaxBlobAxes);
  vector<int> shape_vec(shape.dim_size());
  for (int i = 0; i < shape.dim_size(); ++i) {
    shape_vec[i] = shape.dim(i);
  }
  Reshape(shape_vec, cpu_type, mlu_type, tensor_type, shape_order, dim_strides);
}
#endif

template <typename Dtype>
void Blob<Dtype>::Reshape(const int num, const int channels, const int height,
                          const int width) {
  vector<int> shape(4);
  shape[0] = num;
  shape[1] = channels;
  shape[2] = height;
  shape[3] = width;
  Reshape(shape);
}

template <typename Dtype>
void Blob<Dtype>::Reshape(const vector<int>& shape) {
  CHECK_LE(shape.size(), kMaxBlobAxes);
  count_ = 1;
  shape_.resize(shape.size());
  if (!shape_data_ || shape_data_->size() < shape.size() * sizeof(int)) {
    shape_data_.reset(new SyncedMemory(shape.size() * sizeof(int)));
  }
  int* shape_data = static_cast<int*>(shape_data_->mutable_cpu_data());
  for (int i = 0; i < shape.size(); ++i) {
    CHECK_GE(shape[i], 0);
    if (count_ != 0) {
      CHECK_LE(shape[i], INT_MAX / count_) << "blob size exceeds INT_MAX";
    }
    count_ *= shape[i];
    shape_[i] = shape[i];
    shape_data[i] = shape[i];
  }
  if (count_ > capacity_) {
    capacity_ = count_;
    data_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
    diff_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
  }

#ifdef USE_MLU
  // mlu tensor desc
  BaseDataType dtype = sizeof(Dtype) == 8 ? DT_DOUBLE : DT_FLOAT32;
  tensor_desc_.remember(shape, CNML_TENSOR, dtype, DT_FLOAT16, CNML_NCHW);
#endif
}

template <typename Dtype>
void Blob<Dtype>::Reshape(const BlobShape& shape) {
  CHECK_LE(shape.dim_size(), kMaxBlobAxes);
  vector<int> shape_vec(shape.dim_size());
  for (int i = 0; i < shape.dim_size(); ++i) {
    shape_vec[i] = shape.dim(i);
  }
  Reshape(shape_vec);
}

template <typename Dtype>
void Blob<Dtype>::ReshapeLike(const Blob<Dtype>& other) {
  Reshape(other.shape());
}

template <typename Dtype>
Blob<Dtype>::Blob(const int num, const int channels, const int height,
                  const int width)
    : capacity_(0)
#ifdef USE_MLU
          ,
      shape_order_(CNML_NCHW)
#endif
{
  Reshape(num, channels, height, width);
}

template <typename Dtype>
Blob<Dtype>::Blob(const vector<int>& shape)
    : capacity_(0)
#ifdef USE_MLU
          ,
      shape_order_(CNML_NCHW)
#endif
{
  Reshape(shape);
}

template <typename Dtype>
const int* Blob<Dtype>::gpu_shape() const {
  CHECK(shape_data_);
  return (const int*)shape_data_->gpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_data() const {
  CHECK(data_);
#ifdef USE_MLU
  return (const Dtype*)data_->cpu_data(tensor_desc_);
#else
  return (const Dtype*)data_->cpu_data();
#endif
}

template <typename Dtype>
void Blob<Dtype>::set_cpu_data(Dtype* data) {
  CHECK(data);
  size_t size = count_ * sizeof(Dtype);
  if (data_->size() != size) {
    data_.reset(new SyncedMemory(size));
    diff_.reset(new SyncedMemory(size));
  }
  data_->set_cpu_data(data);
}

template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_data() const {
  CHECK(data_);
  return (const Dtype*)data_->gpu_data();
}

#ifdef USE_MLU

template <typename Dtype>
void Blob<Dtype>::set_mlu_data(Dtype* data) {
  CHECK(data);
  size_t size = count_ * sizeof(Dtype);
  if (data_->size() != size) {
    data_.reset(new SyncedMemory(size));
    diff_.reset(new SyncedMemory(size));
  }
  data_->set_mlu_data(data);
}

template <typename Dtype>
const Dtype* Blob<Dtype>::mlu_data() {
  CHECK(data_);
  tensor_desc_.Create();
  return (const Dtype*)data_->mlu_data(tensor_desc_);
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_mlu_data() {
  CHECK(data_);
  tensor_desc_.Create();
  return static_cast<Dtype*>(data_->mutable_mlu_data(tensor_desc_));
}

template <typename Dtype>
Dtype* Blob<Dtype>::sync_data() {
  CHECK(data_);
  tensor_desc_.Create();
  return static_cast<Dtype*>(data_->mutable_sync_data(tensor_desc_));
}
#endif

template <typename Dtype>
void Blob<Dtype>::set_gpu_data(Dtype* data) {
  CHECK(data);
  // Make sure CPU and GPU sizes remain equal
  size_t size = count_ * sizeof(Dtype);
  if (data_->size() != size) {
    data_.reset(new SyncedMemory(size));
    diff_.reset(new SyncedMemory(size));
  }
  data_->set_gpu_data(data);
}

template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_diff() const {
  CHECK(diff_);
  return (const Dtype*)diff_->cpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_diff() const {
  CHECK(diff_);
  return (const Dtype*)diff_->gpu_data();
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_data() {
  CHECK(data_);
#ifdef USE_MLU
  return static_cast<Dtype*>(data_->mutable_cpu_data(tensor_desc_));
#else
  return static_cast<Dtype*>(data_->mutable_cpu_data());
#endif
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_data() {
  CHECK(data_);
  return static_cast<Dtype*>(data_->mutable_gpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_diff() {
  CHECK(diff_);
  return static_cast<Dtype*>(diff_->mutable_cpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_diff() {
  CHECK(diff_);
  return static_cast<Dtype*>(diff_->mutable_gpu_data());
}

template <typename Dtype>
void Blob<Dtype>::ShareData(const Blob& other) {
  CHECK_EQ(count_, other.count());
  data_ = other.data();
}

template <typename Dtype>
void Blob<Dtype>::ShareDiff(const Blob& other) {
  CHECK_EQ(count_, other.count());
  diff_ = other.diff();
}

// The "update" method is used for parameter blobs in a Net, which are stored
// as Blob<float> or Blob<double> -- hence we do not define it for
// Blob<int> or Blob<unsigned int>.
template <>
void Blob<unsigned int>::Update() {
  NOT_IMPLEMENTED;
}
template <>
void Blob<int>::Update() {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void Blob<Dtype>::Update() {
  // We will perform update based on where the data is located.
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    // perform computation on CPU
    caffe_axpy<Dtype>(count_, Dtype(-1),
        static_cast<const Dtype*>(diff_->cpu_data()),
        static_cast<Dtype*>(data_->mutable_cpu_data()));
    break;
  case SyncedMemory::HEAD_AT_GPU:
    break;
  case SyncedMemory::SYNCED:
    // perform computation on CPU
    caffe_axpy<Dtype>(count_, Dtype(-1),
        static_cast<const Dtype*>(diff_->cpu_data()),
        static_cast<Dtype*>(data_->mutable_cpu_data()));
    break;
  default:
    LOG(FATAL) << "Syncedmem is not initialized.";
  }
}

template <>
unsigned int Blob<unsigned int>::asum_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <>
int Blob<int>::asum_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype>
Dtype Blob<Dtype>::asum_data() const {
  if (!data_) {
    return 0;
  }
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    return caffe_cpu_asum(count_, cpu_data());
  case SyncedMemory::HEAD_AT_GPU:
    break;
  case SyncedMemory::HEAD_AT_MLU:
    break;
  case SyncedMemory::SYNCED:
    return caffe_cpu_asum(count_, cpu_data());
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
  return 0;
}

template <>
unsigned int Blob<unsigned int>::asum_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <>
int Blob<int>::asum_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype>
Dtype Blob<Dtype>::asum_diff() const {
  if (!diff_) {
    return 0;
  }
  switch (diff_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    return caffe_cpu_asum(count_, cpu_diff());
  case SyncedMemory::HEAD_AT_GPU:
    break;
  case SyncedMemory::HEAD_AT_MLU:
    break;
  case SyncedMemory::SYNCED:
    return caffe_cpu_asum(count_, cpu_diff());
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << diff_->head();
  }
  return 0;
}

template <>
unsigned int Blob<unsigned int>::sumsq_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <>
int Blob<int>::sumsq_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype>
Dtype Blob<Dtype>::sumsq_data() const {
  Dtype sumsq;
  const Dtype* data;
  if (!data_) {
    return 0;
  }
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    data = cpu_data();
    sumsq = caffe_cpu_dot(count_, data, data);
    break;
  case SyncedMemory::HEAD_AT_GPU:
    break;
  case SyncedMemory::HEAD_AT_MLU:
    break;
  case SyncedMemory::SYNCED:
    data = cpu_data();
    sumsq = caffe_cpu_dot(count_, data, data);
    break;
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
  return sumsq;
}

template <>
unsigned int Blob<unsigned int>::sumsq_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <>
int Blob<int>::sumsq_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype>
Dtype Blob<Dtype>::sumsq_diff() const {
  Dtype sumsq;
  const Dtype* diff;
  if (!diff_) {
    return 0;
  }
  switch (diff_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    diff = cpu_diff();
    sumsq = caffe_cpu_dot(count_, diff, diff);
    break;
  case SyncedMemory::HEAD_AT_GPU:
    break;
  case SyncedMemory::HEAD_AT_MLU:
    break;
  case SyncedMemory::SYNCED:
    diff = cpu_diff();
    sumsq = caffe_cpu_dot(count_, diff, diff);
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
  return sumsq;
}

template <>
void Blob<unsigned int>::scale_data(unsigned int scale_factor) {
  NOT_IMPLEMENTED;
}

template <>
void Blob<int>::scale_data(int scale_factor) {
  NOT_IMPLEMENTED;
}
/**
 * @brief scale the data in blobs
 * @param scale_factor the factor of scale operation, data *= scale_factor
 */
template <typename Dtype>
void Blob<Dtype>::scale_data(Dtype scale_factor) {
  Dtype* data;
  if (!data_) { return; }
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    data = mutable_cpu_data();
    caffe_scal(count_, scale_factor, data);
    return;
  case SyncedMemory::HEAD_AT_GPU:
    break;
  case SyncedMemory::HEAD_AT_MLU:
    break;
  case SyncedMemory::SYNCED:
    data = mutable_cpu_data();
    caffe_scal(count_, scale_factor, data);
    return;
  case SyncedMemory::UNINITIALIZED:
    return;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
}

template <>
void Blob<unsigned int>::scale_diff(unsigned int scale_factor) {
  NOT_IMPLEMENTED;
}

template <>
void Blob<int>::scale_diff(int scale_factor) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void Blob<Dtype>::scale_diff(Dtype scale_factor) {
  Dtype* diff;
  if (!diff_) { return; }
  switch (diff_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    diff = mutable_cpu_diff();
    caffe_scal(count_, scale_factor, diff);
    return;
  case SyncedMemory::HEAD_AT_GPU:
    break;
  case SyncedMemory::HEAD_AT_MLU:
    break;
  case SyncedMemory::SYNCED:
    diff = mutable_cpu_diff();
    caffe_scal(count_, scale_factor, diff);
    return;
  case SyncedMemory::UNINITIALIZED:
    return;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << diff_->head();
  }
}

template <typename Dtype>
bool Blob<Dtype>::ShapeEquals(const BlobProto& other) {
  if (other.has_num() || other.has_channels() || other.has_height() ||
      other.has_width()) {
    // Using deprecated 4D Blob dimensions --
    // shape is (num, channels, height, width).
    // Note: we do not use the normal Blob::num(), Blob::channels(), etc.
    // methods as these index from the beginning of the blob shape, where legacy
    // parameter blobs were indexed from the end of the blob shape (e.g., bias
    // Blob shape (1 x 1 x 1 x N), IP layer weight Blob shape (1 x 1 x M x N)).
    return shape_.size() <= 4 && LegacyShape(-4) == other.num() &&
           LegacyShape(-3) == other.channels() &&
           LegacyShape(-2) == other.height() &&
           LegacyShape(-1) == other.width();
  }
  vector<int> other_shape(other.shape().dim_size());
  for (int i = 0; i < other.shape().dim_size(); ++i) {
    other_shape[i] = other.shape().dim(i);
  }
  return shape_ == other_shape;
}

// CountEquals is a compromise of ShapeEqual.
// CNML tensor is encapsulated in Blob for convenience.
// In InnerProduct layer, the shape of weights blob is not same as
// that in old Caffe because of initializing CNML tensor.
template <typename Dtype>
bool Blob<Dtype>::CountEquals(const BlobProto& other) {
  if (other.has_num() || other.has_channels() || other.has_height() ||
      other.has_width()) {
    // Using deprecated 4D Blob dimensions --
    // shape is (num, channels, height, width).
    // Note: we do not use the normal Blob::num(), Blob::channels(), etc.
    // methods as these index from the beginning of the blob shape, where legacy
    // parameter blobs were indexed from the end of the blob shape (e.g., bias
    // Blob shape (1 x 1 x 1 x N), IP layer weight Blob shape (1 x 1 x M x N)).
    int count = other.num() * other.channels() * other.height() * other.width();
    return count_ == count;
  }
  int count = 1;
  for (int i = 0; i < other.shape().dim_size(); ++i) {
    count *= other.shape().dim(i);
  }
  return count_ == count;
}

// The output number of channels of YUV2RGB layer is 4
// which is not equal to that in traditional caffemodel.
template <typename Dtype>
bool Blob<Dtype>::CountGE3(const BlobProto& other) {
  if (other.has_num() || other.has_channels() || other.has_height() ||
      other.has_width()) {
    // Using deprecated 4D Blob dimensions --
    // shape is (num, channels, height, width).
    // Note: we do not use the normal Blob::num(), Blob::channels(), etc.
    // methods as these index from the beginning of the blob shape, where legacy
    // parameter blobs were indexed from the end of the blob shape (e.g., bias
    // Blob shape (1 x 1 x 1 x N), IP layer weight Blob shape (1 x 1 x M x N)).
    int count = other.num() * other.channels() * other.height() * other.width();
    if (other.channels() == 3)
      return count_ / 4 * 3 == count;
    else
      return false;
  }
  int count = 1;
  for (int i = 0; i < other.shape().dim_size(); ++i) {
    count *= other.shape().dim(i);
  }
  if (other.shape().dim_size() > 1 && other.shape().dim(1) == 3)
    return count_ / 4 * 3 == count;
  else
    return false;
}
template <typename Dtype>
void Blob<Dtype>::CopyFrom(const Blob& source, bool copy_diff, bool reshape) {
  if (source.count() != count_ || source.shape() != shape_) {
    if (reshape) {
      ReshapeLike(source);
    } else {
      LOG(FATAL) << "Trying to copy blobs of different sizes.";
    }
  }
  switch (Caffe::mode()) {
    case Caffe::GPU:
      break;
      if (copy_diff) {
        caffe_copy(count_, source.gpu_diff(),
                   static_cast<Dtype*>(diff_->mutable_gpu_data()));
      } else {
        caffe_copy(count_, source.gpu_data(),
                   static_cast<Dtype*>(data_->mutable_gpu_data()));
      }
      break;
    case Caffe::CPU:
    case Caffe::MLU:
    case Caffe::MFUS:
      if (copy_diff) {
        caffe_copy(count_, source.cpu_diff(),
                   static_cast<Dtype*>(diff_->mutable_cpu_data()));
      } else {
        caffe_copy(count_, source.cpu_data(),
                   static_cast<Dtype*>(data_->mutable_cpu_data()));
      }
      break;
    default:
      LOG(FATAL) << "Unknown caffe mode.";
  }
}

template <typename Dtype>
void Blob<Dtype>::FromProto(const BlobProto& proto, bool reshape) {
  if (reshape) {
    vector<int> shape;
    if (proto.has_num() || proto.has_channels() || proto.has_height() ||
        proto.has_width()) {
      // Using deprecated 4D Blob dimensions --
      // shape is (num, channels, height, width).
      shape.resize(4);
      shape[0] = proto.num();
      shape[1] = proto.channels();
      shape[2] = proto.height();
      shape[3] = proto.width();
    } else {
      shape.resize(proto.shape().dim_size());
      for (int i = 0; i < proto.shape().dim_size(); ++i) {
        shape[i] = proto.shape().dim(i);
      }
    }
    Reshape(shape);
  } else {
    // shape mismatch
    // CHECK(ShapeEquals(proto)) << "shape mismatch (reshape not set)";
    if (CountEquals(proto) || CountGE3(proto)) {
      if (!CountEquals(proto)) {
        LOG(WARNING) << "the count of two blobs don't mismatch";
      }
    } else {
      LOG(FATAL) << "the count of two blobs don't mismatch";
    }
  }

  // copy data
  Dtype* data_vec = mutable_cpu_data();
  if (proto.double_data_size() > 0) {
    for (int i = 0; i < proto.double_data_size(); ++i) {
      data_vec[i] = proto.double_data(i);
    }
  } else {
    for (int i = 0; i < proto.data_size(); ++i) {
      data_vec[i] = proto.data(i);
    }
  }
  if (proto.double_diff_size() > 0) {
    Dtype* diff_vec = mutable_cpu_diff();
    for (int i = 0; i < proto.double_diff_size(); ++i) {
      diff_vec[i] = proto.double_diff(i);
    }
  } else if (proto.diff_size() > 0) {
    Dtype* diff_vec = mutable_cpu_diff();
    for (int i = 0; i < proto.diff_size(); ++i) {
      diff_vec[i] = proto.diff(i);
    }
  }
}

template <>
void Blob<double>::ToProto(BlobProto* proto, bool write_diff) const {
  proto->clear_shape();
  for (int i = 0; i < shape_.size(); ++i) {
    proto->mutable_shape()->add_dim(shape_[i]);
  }
  proto->clear_double_data();
  proto->clear_double_diff();
  const double* data_vec = cpu_data();
  for (int i = 0; i < count_; ++i) {
    proto->add_double_data(data_vec[i]);
  }
  if (write_diff) {
    const double* diff_vec = cpu_diff();
    for (int i = 0; i < count_; ++i) {
      proto->add_double_diff(diff_vec[i]);
    }
  }
}

template <>
void Blob<float>::ToProto(BlobProto* proto, bool write_diff) const {
  proto->clear_shape();
  for (int i = 0; i < shape_.size(); ++i) {
    proto->mutable_shape()->add_dim(shape_[i]);
  }
  proto->clear_data();
  proto->clear_diff();
  const float* data_vec = cpu_data();
  for (int i = 0; i < count_; ++i) {
    proto->add_data(data_vec[i]);
  }
  if (write_diff) {
    const float* diff_vec = cpu_diff();
    for (int i = 0; i < count_; ++i) {
      proto->add_diff(diff_vec[i]);
    }
  }
}

#ifdef USE_MLU
template <>
void Blob<float>::ToProto(BlobProto* proto, bool write_diff, float sparsity) {
  proto->clear_shape();
  for (int i = 0; i < shape_.size(); i++) {
    proto->mutable_shape()->add_dim(shape_[i]);
  }
  proto->clear_data();
  proto->clear_diff();

  const float* data_vec = cpu_data();
  vector<float> data(count_);
  sparseFilter<float>(shape_, data_vec, &data, sparsity);
  for (int i = 0; i < count_; ++i) {
    proto->add_data(data[i]);
  }
  if (write_diff) {
    const float* diff_vec = cpu_diff();
    for (int i = 0; i < count_; ++i) {
      proto->add_diff(diff_vec[i]);
    }
  }
}

template <>
void Blob<double>::ToProto(BlobProto* proto, bool write_diff, float sparsity) {
  proto->clear_shape();
  for (int i = 0; i < shape_.size(); i++) {
    proto->mutable_shape()->add_dim(shape_[i]);
  }
  proto->clear_data();
  proto->clear_diff();

  const double* data_vec = cpu_data();
  vector<double> data(count_);
  sparseFilter<double>(shape_, data_vec, &data, sparsity);
  for (int i = 0; i < count_; ++i) {
    proto->add_data(data[i]);
  }
  if (write_diff) {
    const double* diff_vec = cpu_diff();
    for (int i = 0; i < count_; ++i) {
      proto->add_diff(diff_vec[i]);
    }
  }
}
#endif

INSTANTIATE_CLASS(Blob);
template class Blob<int>;
template class Blob<unsigned int>;

}  // namespace caffe
