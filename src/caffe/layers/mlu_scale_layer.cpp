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

#ifdef USE_MLU
#include <memory>
#include <vector>
#include "caffe/filler.hpp"
#include "caffe/layers/mlu_scale_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MLUScaleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
  // The shape of alpha is 1*C*1*1, which C is the count of dimension
  // selected by either bottom[1] or layer parameter.
  // bottom[0] is reshaped to N*C*H*1, N is the ourter dim, H is inner dim
  // If the layer has 1 input, alpha is in blob[0], and ScaleOp is used to
  // do the real work. If it has two inputs, bottom[1] is alpha and its shape
  // determines the one of alpha, and CycleMult and CycleAdd is used.

  // The core idea here is to reshape the input to N*C*H*1, and alpha/beta
  // to 1*C*1*1. The reshape operator has its price. If the scale shape
  // matches a certain criteria, reshape can be avoided.
  // The criteria is alpha is a scalar(1,1,1,1) or axis/num_axes is 1/1 for
  // layer with only one bottom.
  const ScaleParameter& param = this->layer_param_.scale_param();
  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
  BaseDataType mlu_dtype = bottom[0]->mlu_type();

  const int num_axes = param.num_axes();
  this->axis_ = bottom[0]->CanonicalAxisIndex(param.axis());

  vector <int> bias_shape(bottom[0]->num_axes(), 1);
  const vector<int>::const_iterator& shape_start =
    bottom[0]->shape().begin() + this->axis_;
  const vector<int>::const_iterator& shape_end =
    (bottom.size() > 1) ? shape_start + bottom[1]->num_axes():
    (num_axes == -1) ? bottom[0]->shape().end() : (shape_start + num_axes);

  alpha_shape = vector<int>(shape_start, shape_end);
  for (int i = 0; i < alpha_shape.size(); i++)
    bias_shape[1] *= alpha_shape[i];

  if (bottom.size() == 1 && this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else if (bottom.size() == 1) {
    // scale is a learned parameter; initialize it
    CHECK_GE(num_axes, -1) << "num_axes must be non-negative, "
                           << "or -1 to extend to the end of bottom[0]";
    if (num_axes >= 0) {
      CHECK_GE(bottom[0]->num_axes(), this->axis_ + num_axes)
          << "scale blob's shape extends past bottom[0]'s shape when applied "
          << "starting with bottom[0] axis = " << this->axis_;
    }
    this->blobs_.resize(1);
    this->blobs_[0].reset(
        new Blob<Dtype>(bias_shape, cpu_dtype, mlu_dtype, CNML_CONST));

    FillerParameter filler_param(param.filler());
    if (!param.has_filler()) {
      // Default to unit (1) filler for identity operation.
      filler_param.set_type("constant");
      filler_param.set_value(1);
    }
    shared_ptr<Filler<Dtype> > filler(GetFiller<Dtype>(filler_param));
    filler->Fill(this->blobs_[0].get());
  }

  if (param.bias_term()) {
    if (this->blobs_.size() + bottom.size() < 3) {
      // case: blobs.size == 1 && bottom.size == 1
      // or blobs.size == 0 && bottom.size == 2
      bias_param_id_ = this->blobs_.size();
      this->blobs_.resize(bias_param_id_ + 1);
    } else {
      // bias param already initialized
      bias_param_id_ = this->blobs_.size() - 1;
    }

    this->blobs_[bias_param_id_].reset(
        new Blob<Dtype>(bias_shape, cpu_dtype, mlu_dtype, CNML_CONST));

    FillerParameter filler_param(param.bias_filler());
    if (!param.has_bias_filler()) {
      // Default to 0 for no bias.
      filler_param.set_type("constant");
      filler_param.set_value(0);
    }
    shared_ptr<Filler<Dtype> > filler(GetFiller<Dtype>(filler_param));
    filler->Fill(this->blobs_[bias_param_id_].get());
  }
  true_bias_data_.Reshape(bias_shape, cpu_dtype, mlu_dtype, CNML_CONST);
  caffe_set(this->true_bias_data_.count(), Dtype(0),
            this->true_bias_data_.mutable_cpu_data());

  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void MLUScaleLayer<Dtype>::Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  BaseDataType cpu_dtype = sizeof(Dtype) ==4 ? DT_FLOAT32: DT_DOUBLE;
  BaseDataType mlu_dtype = bottom[0]->mlu_type();
  const ScaleParameter& param = this->layer_param_.scale_param();
  Blob<Dtype>* scale = (bottom.size() > 1) ? bottom[1] : this->blobs_[0].get();

  this->axis_ = (scale->num_axes() == 0) ?
      0 : bottom[0]->CanonicalAxisIndex(param.axis());
  if (bottom.size() > 1) {
    CHECK_GE(bottom[0]->num_axes(), this->axis_ + scale->num_axes())
      << "scale blob's shape extends past bottom[0]'s shape when applied "
      << "starting with bottom[0] axis = " << this->axis_;
    for (int i = 0; i < scale->num_axes(); ++i) {
      CHECK_EQ(bottom[0]->shape(this->axis_ + i), scale->shape(i))
        << "dimension mismatch between bottom[0]->shape(" << this->axis_ + i
        << ") and scale->shape(" << i << ")";
     }
  }
  this->outer_dim_ = bottom[0]->count(0, this->axis_);
  this->scale_dim_ = scale->count();
  this->inner_dim_ = bottom[0]->count() / (this->outer_dim_ * this->scale_dim_);

  inter_bottom_shape[0] = this->outer_dim_;
  inter_bottom_shape[1] = this->scale_dim_;
  inter_bottom_shape[2] = this->inner_dim_;
  real_bottom_shape = bottom[0]->shape();
  inter_scale_shape[1] = this->scale_dim_;

  // if scale shape is 1*1*1*1, or axis and num_axes are both 1 when input,
  // number of layer is 1,shape requirement of ScaleOp and Cycle*Op is met,
  // no reshape is needed.
  if ((bottom.size() == 1) &&
      (this->scale_dim_ == 1 || (this->axis_ == 1 && alpha_shape.size() == 1)))
    need_reshape_ = false;

  op_bottom0_blob_.Reshape(inter_bottom_shape, cpu_dtype, mlu_dtype, CNML_TENSOR);
  op_bottom1_blob_.Reshape(inter_scale_shape, cpu_dtype, mlu_dtype, CNML_TENSOR);
  op_top0_blob_.Reshape(inter_bottom_shape, cpu_dtype, mlu_dtype, CNML_TENSOR);
  op_top1_blob_.Reshape(inter_bottom_shape, cpu_dtype, mlu_dtype, CNML_TENSOR);

  top[0]->Reshape(bottom[0]->shape(), cpu_dtype, mlu_dtype, CNML_TENSOR);
  true_bias_data_.Reshape(inter_scale_shape, cpu_dtype, mlu_dtype, CNML_CONST);

  transpose_bottom_d2h_blob_.Reshape(bottom[0]->shape(),
                                     cpu_dtype, mlu_dtype,
                                     CNML_TENSOR, CNML_NHWC);
  transpose_bottom_h2d_blob_.Reshape(op_bottom0_blob_.shape(),
                                     cpu_dtype, mlu_dtype,
                                     CNML_TENSOR, CNML_NHWC);
  if (bottom.size() > 1) {
    transpose_alpha_d2h_blob_.Reshape(bottom[1]->shape(), cpu_dtype,
        mlu_dtype, CNML_TENSOR, CNML_NHWC);
    transpose_alpha_h2d_blob_.Reshape(op_bottom1_blob_.shape(), cpu_dtype,
        mlu_dtype, CNML_TENSOR, CNML_NHWC);
  } else {
    transpose_alpha_d2h_blob_.Reshape(op_top0_blob_.shape(), cpu_dtype,
        mlu_dtype, CNML_TENSOR, CNML_NHWC);
    transpose_alpha_h2d_blob_.Reshape(top[0]->shape(), cpu_dtype,
        mlu_dtype, CNML_TENSOR, CNML_NHWC);
  }

  transpose_top_d2h_blob_.Reshape(op_top0_blob_.shape(), cpu_dtype,
      mlu_dtype, CNML_TENSOR, CNML_NHWC);
  transpose_top_h2d_blob_.Reshape(top[0]->shape(), cpu_dtype,
      mlu_dtype, CNML_TENSOR, CNML_NHWC);
}

template <typename Dtype>
void MLUScaleLayer<Dtype>::MLUDestroyOp() {
  for (int i = 0; i < op_ptrs_.size(); i++) {
    if (*op_ptrs_[i] != nullptr) {
      MLU_CHECK(cnmlDestroyBaseOp(op_ptrs_[i]));
      *op_ptrs_[i] = nullptr;
    }
  }
  if (reshape_param_ != nullptr) {
    cnmlDestroyReshapeOpParam(&reshape_param_);
    reshape_param_ = nullptr;
  }
  if (reshape1_param_ != nullptr) {
    cnmlDestroyReshapeOpParam(&reshape1_param_);
    reshape1_param_ = nullptr;
  }
  if (reshape2_param_ != nullptr) {
    cnmlDestroyReshapeOpParam(&reshape2_param_);
    reshape2_param_ = nullptr;
  }
  if (transpose_bottom_d2h_param_ != nullptr) {
    cnmlDestroyNdTransposeOpParam(&transpose_bottom_d2h_param_);
    transpose_bottom_d2h_param_ = nullptr;
  }
  if (transpose_bottom_h2d_param_ != nullptr) {
    cnmlDestroyNdTransposeOpParam(&transpose_bottom_h2d_param_);
    transpose_bottom_h2d_param_ = nullptr;
  }
  if (transpose_alpha_d2h_param_ != nullptr) {
    cnmlDestroyNdTransposeOpParam(&transpose_alpha_d2h_param_);
    transpose_alpha_d2h_param_ = nullptr;
  }
  if (transpose_alpha_h2d_param_ != nullptr) {
    cnmlDestroyNdTransposeOpParam(&transpose_alpha_h2d_param_);
    transpose_alpha_h2d_param_ = nullptr;
  }
  if (transpose_top_d2h_param_ != nullptr) {
    cnmlDestroyNdTransposeOpParam(&transpose_top_d2h_param_);
    transpose_top_d2h_param_ = nullptr;
  }
  if (transpose_top_h2d_param_ != nullptr) {
    cnmlDestroyNdTransposeOpParam(&transpose_top_h2d_param_);
    transpose_top_h2d_param_ = nullptr;
  }
}

template <typename Dtype>
void MLUScaleLayer<Dtype>::MLUCreateOpBindData(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  if (bottom.size() > 1) {  // use cycleMul and cycleadd op
    need_reshape_ ? MLUCreateOpBindCycleMult(bottom, top)
                 : MLUCreateOpBindCycleMult_(bottom, top);
  } else {
    need_reshape_ ? MLUCreateOpBindScale(bottom, top)
                 : MLUCreateOpBindScale_(bottom, top);
  }
}

template <typename Dtype>
void MLUScaleLayer<Dtype>::MLUCreateOpBindCycleMult(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Blob<Dtype>* out_tensor = & op_top0_blob_;

  int bottom_axes = bottom[0]->mlu_shape().size();
  vector<int> trans_dim_order_d2h(bottom_axes, 0);
  trans_dim_order_d2h[1] = bottom_axes - 1;
  for (int i = 2; i < bottom_axes; i++) {
    trans_dim_order_d2h[i] = i-1;
  }
  MLU_CHECK(cnmlCreateNdTransposeOpParam(&transpose_bottom_d2h_param_,
            trans_dim_order_d2h.data(),
            bottom_axes));
  MLU_CHECK(cnmlCreateNdTransposeProOp(&transpose_bottom_d2h_op_ptr_,
            bottom[0]->mlu_tensor(),
            transpose_bottom_d2h_blob_.mlu_tensor(),
            transpose_bottom_d2h_param_));
  int output_axes = op_bottom0_blob_.mlu_shape().size();
  vector<int> transpose_bottom_d2h_shape =
              transpose_bottom_h2d_blob_.mlu_shape();
  MLU_CHECK(cnmlCreateNdReshapeOpParam(&reshape_param_,
                                     transpose_bottom_d2h_shape.data(),
                                     output_axes));

  MLU_CHECK(cnmlCreateReshapeOp(&reshape_op0_ptr_,
                              reshape_param_,
                              transpose_bottom_d2h_blob_.mlu_tensor(),
                              transpose_bottom_h2d_blob_.mlu_tensor()));
  vector<int> trans_dim_order_h2d(output_axes, 0);
  trans_dim_order_h2d[output_axes - 1] = 1;
  for (int i = 1; i < output_axes - 1; i++) {
    trans_dim_order_h2d[i] = i + 1;
  }
  MLU_CHECK(cnmlCreateNdTransposeOpParam(&transpose_bottom_h2d_param_,
            trans_dim_order_h2d.data(),
            output_axes));
  /* TransposeProOp */
  MLU_CHECK(cnmlCreateNdTransposeProOp(&transpose_bottom_h2d_op_ptr_,
            transpose_bottom_h2d_blob_.mlu_tensor(),
            op_bottom0_blob_.mlu_tensor(),
            transpose_bottom_h2d_param_));

  MLU_CHECK(cnmlCreateNdTransposeOpParam(&transpose_alpha_d2h_param_,
            trans_dim_order_d2h.data(),
            bottom_axes));
  /* TransposeProOp */
  MLU_CHECK(cnmlCreateNdTransposeProOp(&transpose_alpha_d2h_op_ptr_,
            bottom[1]->mlu_tensor(),
            transpose_alpha_d2h_blob_.mlu_tensor(),
            transpose_alpha_d2h_param_));

  vector<int> transpose_alpha_h2d_shape =
              transpose_alpha_h2d_blob_.mlu_shape();
  MLU_CHECK(cnmlCreateNdReshapeOpParam(&reshape1_param_,
                                     transpose_alpha_h2d_shape.data(),
                                     output_axes));
  MLU_CHECK(cnmlCreateReshapeOp(&reshape_op1_ptr_,
                              reshape1_param_,
                              transpose_alpha_d2h_blob_.mlu_tensor(),
                              transpose_alpha_h2d_blob_.mlu_tensor()));
  MLU_CHECK(cnmlCreateNdTransposeOpParam(&transpose_alpha_h2d_param_,
            trans_dim_order_h2d.data(),
            output_axes));
  /* TransposeProOp */
  MLU_CHECK(cnmlCreateNdTransposeProOp(&transpose_alpha_h2d_op_ptr_,
            transpose_alpha_h2d_blob_.mlu_tensor(),
            op_bottom1_blob_.mlu_tensor(),
            transpose_alpha_h2d_param_));
  MLU_CHECK(cnmlCreateCycleMultOp(&mlu_cmul_op_ptr_,
                              op_bottom0_blob_.mlu_tensor(),
                              op_bottom1_blob_.mlu_tensor(),
                              op_top0_blob_.mlu_tensor()));

  if (bias_param_id_ != -1) {
    MLU_CHECK(cnmlCreateCycleAddOp(&mlu_cadd_op_ptr_,
                              op_top0_blob_.mlu_tensor(),
                              this->blobs_[bias_param_id_]->mlu_tensor(),
                              op_top1_blob_.mlu_tensor()));
    out_tensor = & op_top1_blob_;
    MLU_CHECK(cnmlBindConstData_V2(this->blobs_[bias_param_id_]->mlu_tensor(),
                            this->blobs_[bias_param_id_]->sync_data(),
                            false));
  }
  MLU_CHECK(cnmlCreateNdTransposeOpParam(&transpose_top_d2h_param_,
            trans_dim_order_d2h.data(),
            bottom_axes));
  /* TransposeProOp */
  MLU_CHECK(cnmlCreateNdTransposeProOp(&transpose_top_d2h_op_ptr_,
            out_tensor->mlu_tensor(),
            transpose_top_d2h_blob_.mlu_tensor(),
            transpose_top_d2h_param_));
  vector<int> transpose_top_h2d_shape = transpose_top_h2d_blob_.mlu_shape();
  MLU_CHECK(cnmlCreateNdReshapeOpParam(&reshape2_param_,
                                     transpose_top_h2d_shape.data(),
                                     output_axes));
  MLU_CHECK(cnmlCreateReshapeOp(&reshape_op2_ptr_,
                              reshape2_param_,
                              transpose_top_d2h_blob_.mlu_tensor(),
                              transpose_top_h2d_blob_.mlu_tensor()));
  MLU_CHECK(cnmlCreateNdTransposeOpParam(&transpose_top_h2d_param_,
            trans_dim_order_h2d.data(),
            output_axes));
  /* TransposeProOp */
  MLU_CHECK(cnmlCreateNdTransposeProOp(&transpose_top_h2d_op_ptr_,
            transpose_top_h2d_blob_.mlu_tensor(),
            top[0]->mlu_tensor(),
            transpose_top_h2d_param_));
}

template <typename Dtype>
void MLUScaleLayer<Dtype>::MLUCreateOpBindCycleMult_(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Blob<Dtype>* out_tensor = (bias_param_id_ == -1) ? top[0] : &op_top0_blob_;

  MLU_CHECK(cnmlCreateCycleMultOp(&mlu_cmul_op_ptr_,
                              bottom[0]->mlu_tensor(),
                              bottom[1]->mlu_tensor(),
                              out_tensor->mlu_tensor()));
  if (bias_param_id_ != -1) {
    MLU_CHECK(cnmlCreateCycleAddOp(&mlu_cadd_op_ptr_,
                              out_tensor->mlu_tensor(),
                              this->blobs_[bias_param_id_]->mlu_tensor(),
                              top[0]->mlu_tensor()));
    MLU_CHECK(cnmlBindConstData_V2(this->blobs_[bias_param_id_]->mlu_tensor(),
                            this->blobs_[bias_param_id_]->sync_data(),
                            false));
  }
}

template <typename Dtype>
void MLUScaleLayer<Dtype>::MLUCompileOp() {
  const int p = Caffe::core_number();
  if (reshape_op2_ptr_ != nullptr) {  // scale by cycleMult and cycleadd
    if (need_reshape_) {
      MLU_CHECK(cnmlCompileBaseOp(transpose_bottom_d2h_op_ptr_, Caffe::rt_core(), p));
      MLU_CHECK(cnmlCompileBaseOp(reshape_op0_ptr_, Caffe::rt_core(), p));
      MLU_CHECK(cnmlCompileBaseOp(transpose_bottom_h2d_op_ptr_, Caffe::rt_core(), p));
      MLU_CHECK(cnmlCompileBaseOp(transpose_alpha_d2h_op_ptr_, Caffe::rt_core(), p));
      MLU_CHECK(cnmlCompileBaseOp(reshape_op1_ptr_, Caffe::rt_core(), p));
      MLU_CHECK(cnmlCompileBaseOp(transpose_alpha_h2d_op_ptr_, Caffe::rt_core(), p));
    }
    MLU_CHECK(cnmlCompileBaseOp(mlu_cmul_op_ptr_, Caffe::rt_core(), p));
    if (bias_param_id_ != -1)
      MLU_CHECK(cnmlCompileBaseOp(mlu_cadd_op_ptr_, Caffe::rt_core(), p));
    if (need_reshape_) {
      MLU_CHECK(cnmlCompileBaseOp(transpose_top_d2h_op_ptr_, Caffe::rt_core(), p));
      MLU_CHECK(cnmlCompileBaseOp(reshape_op2_ptr_, Caffe::rt_core(), p));
      MLU_CHECK(cnmlCompileBaseOp(transpose_top_h2d_op_ptr_, Caffe::rt_core(), p));
    }
  } else {
    if (need_reshape_) {
      MLU_CHECK(cnmlCompileBaseOp(transpose_bottom_d2h_op_ptr_, Caffe::rt_core(), p));
      MLU_CHECK(cnmlCompileBaseOp(reshape_op0_ptr_, Caffe::rt_core(), p));
      MLU_CHECK(cnmlCompileBaseOp(transpose_alpha_d2h_op_ptr_, Caffe::rt_core(), p));
    }
    MLU_CHECK(cnmlCompileBaseOp(mlu_scale_op_ptr_, Caffe::rt_core(), p));
    if (need_reshape_) {
      MLU_CHECK(cnmlCompileBaseOp(transpose_alpha_d2h_op_ptr_, Caffe::rt_core(), p));
      MLU_CHECK(cnmlCompileBaseOp(reshape_op1_ptr_, Caffe::rt_core(), p));
      MLU_CHECK(cnmlCompileBaseOp(transpose_alpha_h2d_op_ptr_, Caffe::rt_core(), p));
    }
  }
}
template <typename Dtype>
void MLUScaleLayer<Dtype>::fuse(MFusion<Dtype>* fuser) {
  if (this->layer_param_.scale_param().bias_term()) {
    caffe_copy(true_bias_data_.count(),
               this->blobs_[bias_param_id_]->cpu_data(),
               true_bias_data_.mutable_cpu_data());
  }
  if (reshape_op2_ptr_ != nullptr) {  // use cyclemul/cycleadd
    if (need_reshape_) {
      fuser->fuse(transpose_bottom_d2h_op_ptr_);
      fuser->fuse(reshape_op0_ptr_);
      fuser->fuse(transpose_bottom_h2d_op_ptr_);
      fuser->fuse(transpose_alpha_d2h_op_ptr_);
      fuser->fuse(reshape_op1_ptr_);
      fuser->fuse(transpose_alpha_h2d_op_ptr_);
    }
    fuser->fuse(mlu_cmul_op_ptr_);
    if (bias_param_id_ != -1) fuser->fuse(mlu_cadd_op_ptr_);
    if (need_reshape_) {
      fuser->fuse(transpose_top_d2h_op_ptr_);
      fuser->fuse(reshape_op2_ptr_);
      fuser->fuse(transpose_top_h2d_op_ptr_);
    }
  } else {
    if (need_reshape_) {
      fuser->fuse(transpose_bottom_d2h_op_ptr_);
      fuser->fuse(reshape_op0_ptr_);
      fuser->fuse(transpose_bottom_h2d_op_ptr_);
    }
    fuser->fuse(mlu_scale_op_ptr_);
    if (need_reshape_) {
      fuser->fuse(transpose_alpha_d2h_op_ptr_);
      fuser->fuse(reshape_op1_ptr_);
      fuser->fuse(transpose_alpha_h2d_op_ptr_);
  }
  }
}

template <typename Dtype>
void MLUScaleLayer<Dtype>::MLUCreateOpBindScale(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  if (bias_param_id_ != -1)
    caffe_copy(true_bias_data_.count(),
               this->blobs_[bias_param_id_]->cpu_data(),
               true_bias_data_.mutable_cpu_data());

  int bottom_axes = bottom[0]->mlu_shape().size();
  vector<int> trans_dim_order_d2h(bottom_axes, 0);
  trans_dim_order_d2h[1] = bottom_axes - 1;
  for (int i = 2; i < bottom_axes; i++) {
    trans_dim_order_d2h[i] = i-1;
  }
  MLU_CHECK(cnmlCreateNdTransposeOpParam(&transpose_bottom_d2h_param_,
            trans_dim_order_d2h.data(),
            bottom_axes));
  /* TransposeProOp */
  MLU_CHECK(cnmlCreateNdTransposeProOp(&transpose_bottom_d2h_op_ptr_,
            bottom[0]->mlu_tensor(),
            transpose_bottom_d2h_blob_.mlu_tensor(),
            transpose_bottom_d2h_param_));

  vector<int> transpose_bottom_h2d_shape = transpose_bottom_h2d_blob_.mlu_shape();
  int output_axes = op_top0_blob_.mlu_shape().size();
  MLU_CHECK(cnmlCreateNdReshapeOpParam(&reshape_param_,
                                     transpose_bottom_h2d_shape.data(),
                                     output_axes));
  MLU_CHECK(cnmlCreateReshapeOp(&reshape_op0_ptr_,
                              reshape_param_,
                              transpose_bottom_d2h_blob_.mlu_tensor(),
                              transpose_bottom_h2d_blob_.mlu_tensor()));
  vector<int> trans_dim_order_h2d(output_axes, 0);
  trans_dim_order_h2d[output_axes - 1] = 1;
  for (int i = 1; i < output_axes - 1; i++) {
    trans_dim_order_h2d[i] = i + 1;
  }
  MLU_CHECK(cnmlCreateNdTransposeOpParam(&transpose_bottom_h2d_param_,
            trans_dim_order_h2d.data(),
            output_axes));
  /* TransposeProOp */
  MLU_CHECK(cnmlCreateNdTransposeProOp(&transpose_bottom_h2d_op_ptr_,
            transpose_bottom_h2d_blob_.mlu_tensor(),
            op_bottom0_blob_.mlu_tensor(),
            transpose_bottom_h2d_param_));
  MLU_CHECK(cnmlCreateScaleOp(&mlu_scale_op_ptr_,
                              op_bottom0_blob_.mlu_tensor(),
                              op_top0_blob_.mlu_tensor(),
                              this->blobs_[0]->mlu_tensor(),
                              true_bias_data_.mlu_tensor()));

  MLU_CHECK(cnmlBindConstData_V2(this->blobs_[0]->mlu_tensor(),
                              this->blobs_[0]->sync_data(),
                              false));
  MLU_CHECK(cnmlBindConstData_V2(true_bias_data_.mlu_tensor(),
                              true_bias_data_.sync_data(),
                              false));
  MLU_CHECK(cnmlCreateNdTransposeOpParam(&transpose_alpha_d2h_param_,
            trans_dim_order_d2h.data(),
            bottom_axes));
  /* TransposeProOp */
  MLU_CHECK(cnmlCreateNdTransposeProOp(&transpose_alpha_d2h_op_ptr_,
            op_top0_blob_.mlu_tensor(),
            transpose_alpha_d2h_blob_.mlu_tensor(),
            transpose_alpha_d2h_param_));
  vector<int> transpose_alpha_h2d_shape =
              transpose_alpha_h2d_blob_.mlu_shape();
  MLU_CHECK(cnmlCreateNdReshapeOpParam(&reshape1_param_,
                                     transpose_alpha_h2d_shape.data(),
                                     output_axes));

  MLU_CHECK(cnmlCreateReshapeOp(&reshape_op1_ptr_,
                              reshape1_param_,
                              transpose_alpha_d2h_blob_.mlu_tensor(),
                              transpose_alpha_h2d_blob_.mlu_tensor()));
  MLU_CHECK(cnmlCreateNdTransposeOpParam(&transpose_alpha_h2d_param_,
            trans_dim_order_h2d.data(),
            output_axes));
  /* TransposeProOp */
  MLU_CHECK(cnmlCreateNdTransposeProOp(&transpose_alpha_h2d_op_ptr_,
            transpose_alpha_h2d_blob_.mlu_tensor(),
            top[0]->mlu_tensor(),
            transpose_alpha_h2d_param_));
}

template <typename Dtype>
void MLUScaleLayer<Dtype>::MLUCreateOpBindScale_(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  if (bias_param_id_ != -1)
    caffe_copy(true_bias_data_.count(),
               this->blobs_[bias_param_id_]->cpu_data(),
               true_bias_data_.mutable_cpu_data());

  MLU_CHECK(cnmlCreateNdScaleOp(&mlu_scale_op_ptr_,
                              bottom[0]->mlu_shape().size() -1,
                              bottom[0]->mlu_tensor(),
                              top[0]->mlu_tensor(),
                              this->blobs_[0]->mlu_tensor(),
                              true_bias_data_.mlu_tensor()));
  MLU_CHECK(cnmlBindConstData_V2(this->blobs_[0]->mlu_tensor(),
                              this->blobs_[0]->sync_data(),
                              false));
  MLU_CHECK(cnmlBindConstData_V2(true_bias_data_.mlu_tensor(),
                              true_bias_data_.sync_data(),
                              false));
}

template <typename Dtype>
void MLUScaleLayer<Dtype>::Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
  if (need_reshape_)
    ForwardMLU(bottom, top);
  else
    ForwardMLU_(bottom, top);
}

template <typename Dtype>
void MLUScaleLayer<Dtype>::ForwardMLU(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
  Blob<Dtype>* out_tensor = &op_top0_blob_;
  if (bottom.size() > 1) {
    MLU_CHECK(cnmlComputeNdTransposeProOpForward(transpose_bottom_d2h_op_ptr_,
              bottom[0]->mutable_mlu_data(),
              transpose_bottom_d2h_blob_.mutable_mlu_data(),
              Caffe::forward_param(), Caffe::queue()));
    MLU_CHECK(cnmlComputeReshapeOpForward_V3(reshape_op0_ptr_,
              transpose_bottom_d2h_blob_.mutable_mlu_data(),
              transpose_bottom_h2d_blob_.mutable_mlu_data(),
              Caffe::forward_param(), Caffe::queue()));
    MLU_CHECK(cnmlComputeNdTransposeProOpForward(transpose_bottom_h2d_op_ptr_,
              transpose_bottom_h2d_blob_.mutable_mlu_data(),
              op_bottom0_blob_.mutable_mlu_data(),
              Caffe::forward_param(), Caffe::queue()));
    MLU_CHECK(cnmlComputeNdTransposeProOpForward(transpose_alpha_d2h_op_ptr_,
              bottom[1]->mutable_mlu_data(),
              transpose_alpha_d2h_blob_.mutable_mlu_data(),
              Caffe::forward_param(), Caffe::queue()));
    MLU_CHECK(cnmlComputeReshapeOpForward_V3(reshape_op1_ptr_,
              transpose_alpha_d2h_blob_.mutable_mlu_data(),
              transpose_alpha_h2d_blob_.mutable_mlu_data(),
              Caffe::forward_param(), Caffe::queue()));
    MLU_CHECK(cnmlComputeNdTransposeProOpForward(transpose_alpha_h2d_op_ptr_,
              transpose_alpha_h2d_blob_.mutable_mlu_data(),
              op_bottom1_blob_.mutable_mlu_data(),
              Caffe::forward_param(), Caffe::queue()));
    MLU_CHECK(cnmlComputeCycleMultOpForward_V3(mlu_cmul_op_ptr_,
              op_bottom0_blob_.mutable_mlu_data(),
              op_bottom1_blob_.mutable_mlu_data(),
              op_top0_blob_.mutable_mlu_data(),
              Caffe::forward_param(),
              Caffe::queue()));
    if (bias_param_id_ != -1) {
      MLU_CHECK(cnmlComputeCycleAddOpForward_V3(mlu_cadd_op_ptr_,
              op_top0_blob_.mutable_mlu_data(),
              nullptr,
              op_top1_blob_.mutable_mlu_data(),
              Caffe::forward_param(),
              Caffe::queue()));
      out_tensor = &op_top1_blob_;
    }
    MLU_CHECK(cnmlComputeNdTransposeProOpForward(transpose_top_d2h_op_ptr_,
              out_tensor->mutable_mlu_data(),
              transpose_top_d2h_blob_.mutable_mlu_data(),
              Caffe::forward_param(), Caffe::queue()));
    MLU_CHECK(cnmlComputeReshapeOpForward_V3(reshape_op2_ptr_,
              transpose_top_d2h_blob_.mutable_mlu_data(),
              transpose_top_h2d_blob_.mutable_mlu_data(),
              Caffe::forward_param(), Caffe::queue()));
    MLU_CHECK(cnmlComputeNdTransposeProOpForward(transpose_top_h2d_op_ptr_,
              transpose_top_h2d_blob_.mutable_mlu_data(),
              top[0]->mutable_mlu_data(),
              Caffe::forward_param(), Caffe::queue()));
  } else {
    MLU_CHECK(cnmlComputeNdTransposeProOpForward(transpose_bottom_d2h_op_ptr_,
              bottom[0]->mutable_mlu_data(),
              transpose_bottom_d2h_blob_.mutable_mlu_data(),
              Caffe::forward_param(), Caffe::queue()));
    MLU_CHECK(cnmlComputeReshapeOpForward_V3(reshape_op0_ptr_,
              transpose_bottom_d2h_blob_.mutable_mlu_data(),
              transpose_bottom_h2d_blob_.mutable_mlu_data(),
              Caffe::forward_param(), Caffe::queue()));
    MLU_CHECK(cnmlComputeNdTransposeProOpForward(transpose_alpha_d2h_op_ptr_,
              transpose_bottom_h2d_blob_.mutable_mlu_data(),
              op_bottom0_blob_.mutable_mlu_data(),
              Caffe::forward_param(), Caffe::queue()));
    MLU_CHECK(cnmlComputeScaleOpForward_V3(mlu_scale_op_ptr_,
              op_bottom0_blob_.mutable_mlu_data(),
              op_top0_blob_.mutable_mlu_data(),
              Caffe::forward_param(),
              Caffe::queue()));
    MLU_CHECK(cnmlComputeNdTransposeProOpForward(transpose_alpha_d2h_op_ptr_,
              op_top0_blob_.mutable_mlu_data(),
              transpose_alpha_d2h_blob_.mutable_mlu_data(),
              Caffe::forward_param(), Caffe::queue()));
    MLU_CHECK(cnmlComputeReshapeOpForward_V3(reshape_op1_ptr_,
              transpose_alpha_d2h_blob_.mutable_mlu_data(),
              transpose_alpha_h2d_blob_.mutable_mlu_data(),
              Caffe::forward_param(), Caffe::queue()));
    MLU_CHECK(cnmlComputeNdTransposeProOpForward(transpose_alpha_h2d_op_ptr_,
              transpose_alpha_h2d_blob_.mutable_mlu_data(),
              top[0]->mutable_mlu_data(),
              Caffe::forward_param(), Caffe::queue()));
  }
}

template <typename Dtype>
void MLUScaleLayer<Dtype>::ForwardMLU_(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
  if (bottom.size() > 1) {
    Blob<Dtype>* out_tensor = (bias_param_id_ == -1) ? top[0] : &op_top0_blob_;
    MLU_CHECK(cnmlComputeCycleMultOpForward_V3(mlu_cmul_op_ptr_,
              bottom[0]->mutable_mlu_data(),
              bottom[1]->mutable_mlu_data(),
              out_tensor->mutable_mlu_data(),
              Caffe::forward_param(),
              Caffe::queue()));
    if (bias_param_id_ != -1)
      MLU_CHECK(cnmlComputeCycleAddOpForward_V3(mlu_cadd_op_ptr_,
              out_tensor->mutable_mlu_data(),
              nullptr,
              top[0]->mutable_mlu_data(),
              Caffe::forward_param(),
              Caffe::queue()));
  } else {
    MLU_CHECK(cnmlComputeNdScaleOpForward(mlu_scale_op_ptr_,
              NULL,
              bottom[0]->mutable_mlu_data(),
              NULL,
              top[0]->mutable_mlu_data(),
              Caffe::queue(),
              NULL));
  }
}

template <typename Dtype>
MLUScaleLayer<Dtype>::~MLUScaleLayer() {
  MLUDestroyOp();
}

INSTANTIATE_CLASS(MLUScaleLayer);

}  // namespace caffe
#endif  // USE_MLU
