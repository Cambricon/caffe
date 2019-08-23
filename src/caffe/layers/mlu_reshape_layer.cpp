/*
All modification made by Cambricon Corporation: © 2018 Cambricon Corporation
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

#ifdef USE_MLU
#include <vector>
#include "caffe/layers/mlu_reshape_layer.hpp"

namespace caffe {

template <typename Dtype>
void MLUReshapeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
  ReshapeLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void MLUReshapeLayer<Dtype>::Reshape_tensor
    (const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top) {
  const int input_start_axis = this->layer_param_.reshape_param().axis();
  const int start_axis = (input_start_axis >= 0) ? input_start_axis :
      bottom[0]->num_axes() + input_start_axis + 1;
  CHECK_GE(start_axis, 0) << "axis " << input_start_axis << " out of range";
  CHECK_LE(start_axis, bottom[0]->num_axes()) << "axis " << input_start_axis
      << " out of range for " << bottom[0]->num_axes() << "-D input blob";
  const int num_axes = this->layer_param_.reshape_param().num_axes();
  CHECK_GE(num_axes, -1) << "num_axes must be >= 0, or -1 for all";
  const int end_axis =
      (num_axes == -1) ? bottom[0]->num_axes() : (start_axis + num_axes);
  CHECK_LE(end_axis, bottom[0]->num_axes())
      << "end_axis = axis + num_axes is out of range";
  const int num_axes_replaced = end_axis - start_axis;
  const int num_axes_retained =
    bottom[0]->num_axes() - num_axes_replaced;
  const BlobShape& top_blob_shape =
    this->layer_param_.reshape_param().shape();
  const int num_new_axes = top_blob_shape.dim_size();
  vector<int> top_shape(num_axes_retained + num_new_axes);
  int top_shape_index = 0;
  for (int i = 0; i < start_axis; ++i) {
    top_shape[top_shape_index++] = bottom[0]->shape(i);
  }
  for (int i = 0; i < num_new_axes; ++i) {
    top_shape[top_shape_index++] = top_blob_shape.dim(i);
  }
  for (int i = end_axis; i < bottom[0]->num_axes(); ++i) {
    top_shape[top_shape_index++] = bottom[0]->shape(i);
  }
  CHECK_EQ(top_shape_index, top_shape.size());
  for (int i = 0; i < this->copy_axes_.size(); ++i) {
    const int copy_axis_index = this->copy_axes_[i];
    CHECK_GT(bottom[0]->num_axes(), start_axis + copy_axis_index)
        << "new shape contains a 0, but there was no corresponding bottom axis "
        << "to copy";
    top_shape[start_axis + copy_axis_index] =
        bottom[0]->shape(start_axis + copy_axis_index);
  }
  if (this->inferred_axis_ >= 0) {
    // A -1 dim was specified; infer the correct dimension by computing the
    // product of the other dimensions.
    int explicit_count = this->constant_count_;
    explicit_count *= bottom[0]->count(0, start_axis);
    explicit_count *= bottom[0]->count(end_axis);
    for (int i = 0; i < this->copy_axes_.size(); ++i) {
      const int copy_axis_index = this->copy_axes_[i];
      explicit_count *= top_shape[start_axis + copy_axis_index];
    }
    CHECK_EQ(0, bottom[0]->count() % explicit_count) << "bottom count ("
        << bottom[0]->count() << ") must be divisible by the product of "
        << "the specified dimensions (" << explicit_count << ")";
    const int inferred_dim = bottom[0]->count() / explicit_count;
    top_shape[start_axis + this->inferred_axis_] = inferred_dim;
  }

  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
  BaseDataType mlu_dtype = DT_FLOAT16;
  top[0]->Reshape(top_shape, cpu_dtype, mlu_dtype, CNML_TENSOR);
  CHECK_EQ(top[0]->count(), bottom[0]->count())
      << "output count must match input count";
  top[0]->ShareDiff(*bottom[0]);
}

template <typename Dtype>
void MLUReshapeLayer<Dtype>::MLUCreateOpBindData(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  MLU_CHECK(cnmlCreateReshapeOpParam(&reshape_param_,
      top[0]->shape().size() > 0 ? top[0]->shape(0) / Caffe::data_parallel(): 1,
      top[0]->shape().size() > 1 ? top[0]->shape(1) : 1,
      top[0]->shape().size() > 2 ? top[0]->shape(2) : 1,
      top[0]->shape().size() > 3 ? top[0]->shape(3) : 1,
      CNML_NCHW));
  MLU_CHECK(cnmlCreateReshapeOp(&reshape_op_ptr_,
                               reshape_param_,
                               bottom[0]->mlu_tensor(),
                               top[0]->mlu_tensor()));
}

template <typename Dtype>
void MLUReshapeLayer<Dtype>::MLUCompileOp() {
  MLU_CHECK(cnmlCompileBaseOp(reshape_op_ptr_,
                              Caffe::rt_core(),
                              Caffe::model_parallel()));
}

template <typename Dtype>
void MLUReshapeLayer<Dtype>::MLUDestroyOp() {
  if (reshape_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&reshape_op_ptr_));
    reshape_op_ptr_ = nullptr;
  }

  if (reshape_param_ != nullptr) {
    MLU_CHECK(cnmlDestroyReshapeOpParam(&reshape_param_));
    reshape_param_ = nullptr;
  }
}

template <typename Dtype>
MLUReshapeLayer<Dtype>::~MLUReshapeLayer() {
  MLUDestroyOp();
}

template <typename Dtype>
void MLUReshapeLayer<Dtype>::Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  MLU_CHECK(cnmlComputeReshapeOpForward_V3(reshape_op_ptr_,
                           bottom[0]->mutable_mlu_data(),
                           top[0]->mutable_mlu_data(),
                           Caffe::forward_param(), Caffe::queue()));
}

INSTANTIATE_CLASS(MLUReshapeLayer);
}  // namespace caffe
#endif  // USE_MLU
