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

#ifndef INCLUDE_CAFFE_BLOB_HPP_
#define INCLUDE_CAFFE_BLOB_HPP_

#include <algorithm>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/syncedmem.hpp"

const int kMaxBlobAxes = 4;  ///< max dimensions of a blob, limited by CNML

namespace caffe {

/**
 * @brief A wrapper around SyncedMemory holders serving as the basic
 *        computational unit through which Layer%s, Net%s, and Solver%s
 *        interact.
 *
 * Caffe stores and communicates data using blobs.
 *
 * Blobs provides a unified memory interface holding data;
 * e.g., batches of images,model parameters,and derivatives for optimization.
 *
 * The conventional blob dimensions for batches of image data
 * are N*C*H*W, i.e. number,channel,height,width.
 *
 * number is the batch size of the data,
 * if you don't know it, check out how batch gradient descent work
 * in a simple fully-connect neural network.
 *
 * channel is the feature dimension,take an 256*256 RGB image for instance,
 * it has the Red,Green,Blue feature maps, each feature map is arithmetically
 * a 256*256 matrix,whose value ranges from 0 - 1 to infer the intensity of
 * some specific color.
 *
 * height and weight are used to uniquely identify a point in a feature map.
 *
 * Blob memory is row-major(memory are physically linear bytes) in layout,
 * so the rightmost dimension changes fastest
 * (in the NCHW case, W changes fastest).
 *
 * e.g. in a 4D blob,the value at index(n,c,h,w)
 * is located at index ((n*K+k)*H+h)*W+w physically.
 * see the offset function,draw an example,and figure out how it works.
 * this is very important.
 *
 * Note that not all blobs in Caffe are 4D
 * recall the simplest fully-connected neural network
 * all the data in it are 2D,i.e. batch size and neuron number in each layer.
 * you can pass h=1 and w=1 to the Blob constructor to get this.
 *
 * Parameter blob dimensions vary according to the type and configuration of
 * the layer. For a convolution layer with 96 filters(of kernels) of 11 * 11
 * spatial dimension and 3 input channels, the blob shape is 96 * 3 * 11 * 11.
 * For an inner product/fully-connected layer with 1000 output channels and
 * 1024 input channels,the parameter blob is 1000 * 1024.
 */
template <typename Dtype>
class Blob {
  public:
  Blob()
      : data_(),
        diff_(),
        count_(0),
        capacity_(0)
#ifdef USE_MLU
        ,
        shape_order_(CNML_NCHW)
#endif
  {
  }

  /// @brief Deprecated; use <code>Blob(const vector<int>& shape)</code>.
  explicit Blob(const int num, const int channels, const int height,
                const int width);
  explicit Blob(const vector<int>& shape);
#ifdef USE_MLU
  explicit Blob(const int num, const int channels, const int height,
                const int width, BaseDataType cpu_type, BaseDataType mlu_type,
                cnmlTensorType_t tensor_type);
  Blob(const vector<int>& shape, BaseDataType cpu_type, BaseDataType mlu_type,
       cnmlTensorType_t tensor_type, cnmlDataOrder_t shape_order = CNML_NCHW);
#endif

  /// @brief Deprecated; use <code>Reshape(const vector<int>& shape)</code>.
  void Reshape(const int num, const int channels, const int height,
               const int width);
#ifdef USE_MLU
  void Reshape(const int num, const int channels, const int height,
               const int width, BaseDataType cpu_type, BaseDataType mlu_type,
               cnmlTensorType_t tensor_type);
#endif

  /**
   * @brief Change the dimensions of the blob, allocating new memory if
   *        necessary.
   *
   * This function can be called both to create an initial allocation
   * of memory, and to adjust the dimensions of a top blob during Layer::Reshape
   * or Layer::Forward. When changing the size of blob, memory will only be
   * reallocated if sufficient memory does not already exist, and excess memory
   * will never be freed.
   *
   * Note that reshaping an input blob and immediately calling Net::Backward is
   * an error; either Net::Forward or Net::Reshape need to be called to
   * propagate the new input shape to higher layers.
   */
  void Reshape(const vector<int>& shape);
#ifdef USE_MLU
  void Reshape(const vector<int>& shape, BaseDataType cpu_type,
               BaseDataType mlu_type, cnmlTensorType_t tensor_type,
               cnmlDataOrder_t shape_order = CNML_NCHW);
#endif

  void Reshape(const BlobShape& shape);
#ifdef USE_MLU
  void Reshape(const BlobShape& shape, BaseDataType cpu_type,
               BaseDataType mlu_type, cnmlTensorType_t tensor_type,
               cnmlDataOrder_t shape_order = CNML_NCHW);
#endif
  void ReshapeLike(const Blob& other);
  /**
   * @brief Returns the shape string of a blob
   * e.g. 2 2 1 1(4)
   */
  inline string shape_string() const {
    ostringstream stream;
    for (int i = 0; i < shape_.size(); ++i) {
      stream << shape_[i] << " ";
    }
    stream << "(" << count_ << ")";
    return stream.str();
  }
  inline const vector<int>& shape() const { return shape_; }
  /**
   * @brief Returns the dimension of the index-th axis (or the negative index-th
   *        axis from the end, if index is negative).
   *
   * @param index the axis index, which may be negative as it will be
   *        "canonicalized" using CanonicalAxisIndex.
   *        Dies on out of range index.
   */
  inline int shape(int index) const {
    return shape_[CanonicalAxisIndex(index)];
  }
  inline int num_axes() const { return shape_.size(); }
  inline int count() const { return count_; }

  /**
   * @brief Compute the volume of a slice; i.e., the product of dimensions
   *        among a range of axes.
   *
   * @param start_axis The first axis to include in the slice.
   *
   * @param end_axis The first axis to exclude from the slice.
   */
  inline int count(int start_axis, int end_axis) const {
    CHECK_LE(start_axis, end_axis);
    CHECK_GE(start_axis, 0);
    CHECK_GE(end_axis, 0);
    CHECK_LE(start_axis, num_axes());
    CHECK_LE(end_axis, num_axes());
    int count = 1;
    for (int i = start_axis; i < end_axis; ++i) {
      count *= shape(i);
    }
    return count;
  }
  /**
   * @brief Compute the volume of a slice spanning from a particular first
   *        axis to the final axis.
   *
   * @param start_axis The first axis to include in the slice.
   */
  inline int count(int start_axis) const {
    return count(start_axis, num_axes());
  }

  /**
   * @brief Returns the 'canonical' version of a (usually) user-specified axis,
   *        allowing for negative indexing (e.g., -1 for the last axis).
   *
   * @param axis_index the axis index.
   *        If 0 <= index < num_axes(), return index.
   *        If -num_axes <= index <= -1, return (num_axes() - (-index)),
   *        e.g., the last axis index (num_axes() - 1) if index == -1,
   *        the second to last if index == -2, etc.
   *        Dies on out of range index.
   */
  inline int CanonicalAxisIndex(int axis_index) const {
    CHECK_GE(axis_index, -num_axes())
        << "axis " << axis_index << " out of range for " << num_axes()
        << "-D Blob with shape " << shape_string();
    CHECK_LT(axis_index, num_axes()) << "axis " << axis_index
                                     << " out of range for " << num_axes()
                                     << "-D Blob with shape " << shape_string();
    if (axis_index < 0) {
      return axis_index + num_axes();
    }
    return axis_index;
  }

  /// @brief Deprecated legacy shape accessor num: use shape(0) instead.
  inline int num() const { return LegacyShape(0); }
  /// @brief Deprecated legacy shape accessor channels: use shape(1) instead.
  inline int channels() const { return LegacyShape(1); }
  /// @brief Deprecated legacy shape accessor height: use shape(2) instead.
  inline int height() const { return LegacyShape(2); }
  /// @brief Deprecated legacy shape accessor width: use shape(3) instead.
  inline int width() const { return LegacyShape(3); }
  inline int LegacyShape(int index) const {
    CHECK_LE(num_axes(), 4)
        << "Cannot use legacy accessors on Blobs with > 4 axes.";
    CHECK_LT(index, 4);
    CHECK_GE(index, -4);
    if (index >= num_axes() || index < -num_axes()) {
      // Axis is out of range, but still in [0, 3] (or [-4, -1] for reverse
      // indexing) -- this special case simulates the one-padding used to fill
      // extraneous axes of legacy blobs.
      return 1;
    }
    return shape(index);
  }
  /**
   * @brief Return the position of a specfic data
   *        e.g. if you have batches of pictures, and you want to get the memory
   * position
   *        of the pixel in batch 3,channel 2,height 256 and width 256.
   *        you can use your_blob->cpu_data()[offset(3,2,256,256)] to get its
   * value,or use
   *        the data_at function,which is a simple wrapper for offset.
   */
  inline int offset(const int n, const int c = 0, const int h = 0,
                    const int w = 0) const {
    CHECK_GE(n, 0);
    CHECK_LE(n, num());
    CHECK_GE(channels(), 0);
    CHECK_LE(c, channels());
    CHECK_GE(height(), 0);
    CHECK_LE(h, height());
    CHECK_GE(width(), 0);
    CHECK_LE(w, width());
    return ((n * channels() + c) * height() + h) * width() + w;
  }

  inline int offset(const vector<int>& indices) const {
    CHECK_LE(indices.size(), num_axes());
    int offset = 0;
    for (int i = 0; i < num_axes(); ++i) {
      offset *= shape(i);
      if (indices.size() > i) {
        CHECK_GE(indices[i], 0);
        CHECK_LT(indices[i], shape(i));
        offset += indices[i];
      }
    }
    return offset;
  }
  /**
   * @brief Copy from a source Blob.
   *
   * @param source the Blob to copy from
   * @param copy_diff if false, copy the data; if true, copy the diff
   * @param reshape if false, require this Blob to be pre-shaped to the shape
   *        of other (and die otherwise); if true, Reshape this Blob to other's
   *        shape if necessary
   */
  void CopyFrom(const Blob<Dtype>& source, bool copy_diff = false,
                bool reshape = false);
  /**
   * @brief A simple wrapper for offset
   */
  inline Dtype data_at(const int n, const int c, const int h,
                       const int w) const {
    return cpu_data()[offset(n, c, h, w)];
  }

  inline Dtype diff_at(const int n, const int c, const int h,
                       const int w) const {
    return cpu_diff()[offset(n, c, h, w)];
  }

  inline Dtype data_at(const vector<int>& index) const {
    return cpu_data()[offset(index)];
  }

  inline Dtype diff_at(const vector<int>& index) const {
    return cpu_diff()[offset(index)];
  }

  inline const shared_ptr<SyncedMemory>& data() const {
    CHECK(data_);
    return data_;
  }

  inline const shared_ptr<SyncedMemory>& diff() const {
    CHECK(diff_);
    return diff_;
  }

  const Dtype* cpu_data() const;
  void set_cpu_data(Dtype* data);
  const int* gpu_shape() const;
  const Dtype* gpu_data() const;
  void set_gpu_data(Dtype* data);
  const Dtype* cpu_diff() const;
  const Dtype* gpu_diff() const;

#ifdef USE_MLU
  const Dtype* mlu_data();
  void set_mlu_data(Dtype* data);
  Dtype* mutable_mlu_data();

  cnmlTensor_t mlu_tensor() {
    mlu_tensor_desc_.mluCreate();
    return mlu_tensor_desc_.mlu();
  }
  cnmlCpuTensor_t cpu_tensor() {
    mlu_tensor_desc_.cpuCreate();
    return mlu_tensor_desc_.cpu();
  }
  void set_mlu_position(int position) {
    mlu_tensor_desc_.set_position(position);
  }
  void set_mlu_positions(const vector<int>& positions) {
    mlu_tensor_desc_.set_positions(positions);
  }
  const int mlu_position() const { return mlu_tensor_desc_.position(); }
  const vector<int>& mlu_positions() const {
      return mlu_tensor_desc_.positions();
  }
  void set_mlu_scale(float scale) { mlu_tensor_desc_.set_scale(scale); }
  void set_mlu_scales(const vector<float>& scales) {
     mlu_tensor_desc_.set_scales(scales);
  }
  const float mlu_scale() const { return mlu_tensor_desc_.scale(); }
  const vector<float>& mlu_scales() const {
      return mlu_tensor_desc_.scales();
  }
  void set_cpu_type(BaseDataType cpu_dtype) {
    mlu_tensor_desc_.set_cpu_type(cpu_dtype);
  }
  const BaseDataType cpu_type() const { return mlu_tensor_desc_.cpu_type(); }
  void set_mlu_type(BaseDataType mlu_dtype) {
    mlu_tensor_desc_.set_mlu_type(mlu_dtype);
  }
  const BaseDataType mlu_type() const { return mlu_tensor_desc_.mlu_type(); }
  cnmlTensorType_t tensor_type() { return mlu_tensor_desc_.type(); }
  bool has_mlu_position() const { return mlu_tensor_desc_.has_position(); }
  bool has_mlu_scale() const { return mlu_tensor_desc_.has_scale(); }
  void setCpuTensorOrder(cnmlDataOrder_t order) {
    mlu_tensor_desc_.setCpuTensorOrder(order);
  }

#endif

  Dtype* mutable_cpu_data();
  Dtype* mutable_gpu_data();
  Dtype* mutable_cpu_diff();
  Dtype* mutable_gpu_diff();
  void Update();
  /// @brief Construct a blob from prototxt
  void FromProto(const BlobProto& proto, bool reshape = true);
  /// @brief Save a blob's information to prototxt
  void ToProto(BlobProto* proto, bool write_diff = false) const;
#ifdef USE_MLU
  void ToProto(BlobProto* proto, bool write_diff, float sparsity);
#endif

  /// @brief Compute the sum of absolute values (L1 norm) of the data.
  Dtype asum_data() const;
  /// @brief Compute the sum of absolute values (L1 norm) of the diff.
  Dtype asum_diff() const;
  /// @brief Compute the sum of squares (L2 norm squared) of the data.
  Dtype sumsq_data() const;
  /// @brief Compute the sum of squares (L2 norm squared) of the diff.
  Dtype sumsq_diff() const;

  /// @brief Scale the blob data by a constant factor.
  void scale_data(Dtype scale_factor);
  /// @brief Scale the blob diff by a constant factor.
  void scale_diff(Dtype scale_factor);

  /**
   * @brief Set the data_ shared_ptr to point to the SyncedMemory holding the
   *        data_ of Blob other -- useful in Layer%s which simply perform a copy
   *        in their Forward pass.
   *
   * This deallocates the SyncedMemory holding this Blob's data_, as
   * shared_ptr calls its destructor when reset with the "=" operator.
   */
  void ShareData(const Blob& other);
  /**
   * @brief Set the diff_ shared_ptr to point to the SyncedMemory holding the
   *        diff_ of Blob other -- useful in Layer%s which simply perform a copy
   *        in their Forward pass.
   *
   * This deallocates the SyncedMemory holding this Blob's diff_, as
   * shared_ptr calls its destructor when reset with the "=" operator.
   */
  void ShareDiff(const Blob& other);
  /**
   * @brief Judge whether the blob's shape is identical to another.
   */
  bool ShapeEquals(const BlobProto& other);

  /**
   * @brief Judge whether the blob's count is identical to another.
   */
  bool CountEquals(const BlobProto& other);

  /**
   * @brief Judge whether the blob's count / 4 * 3 is identical to another.
   */
  bool CountGE3(const BlobProto& other);

  protected:
  shared_ptr<SyncedMemory> data_;
  shared_ptr<SyncedMemory> diff_;
  shared_ptr<SyncedMemory> shape_data_;
  vector<int> shape_;
  int count_;     ///< the product of all a blob's dimensions.
  int capacity_;  ///< use to check whether to ask for more memory.
#ifdef USE_MLU
  cnmlDataOrder_t shape_order_;
  MLUTensorDesc mlu_tensor_desc_;
#endif
  DISABLE_COPY_AND_ASSIGN(Blob);
};  // class Blob

}  // namespace caffe

#endif  // INCLUDE_CAFFE_BLOB_HPP_
