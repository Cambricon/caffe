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

#ifdef USE_MLU

#include <cfloat>
#include <vector>

#include "caffe/layers/mlu_eltwise_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MLUEltwiseLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  EltwiseLayer<Dtype>::LayerSetUp(bottom, top);
  switch (this->op_) {
    case EltwiseParameter_EltwiseOp_SUM:
      vec_sum_op_ptr_.resize(bottom.size()-1, nullptr);
      temp_.resize(bottom.size()-2, nullptr);
      break;
    case EltwiseParameter_EltwiseOp_PROD:
      vec_mult_op_ptr_.resize(bottom.size()-1, nullptr);
      temp_.resize(bottom.size()-2, nullptr);
      break;
    case EltwiseParameter_EltwiseOp_MAX:
      vec_max_op_ptr_.resize(bottom.size()-1, nullptr);
      temp_.resize(bottom.size()-2, nullptr);
      break;
    default:
      LOG(FATAL) << "NOT IMPLEMENT";
      break;
  }
  for (int i = 0; i < alpha_.size(); i++) {
    alpha_[i] = new Blob<Dtype>();
  }
  for (int i = 0; i < temp_.size(); i++) {
    temp_[i] = new Blob<Dtype>();
  }
}

template <typename Dtype>
void MLUEltwiseLayer<Dtype>::Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
  for (int i = 1; i < bottom.size(); ++i) {
    CHECK(bottom[0]->channels() == bottom[i]->channels() &&
          bottom[0]->num() == bottom[i]->num() &&
          bottom[0]->count() == bottom[i]->count())
        << "bottom[0]: " << bottom[0]->shape_string()
        << ", bottom[" << i << "]: " << bottom[i]->shape_string();
  }

  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
  BaseDataType mlu_dtype = DT_FLOAT16;
  top[0]->Reshape(bottom[0]->shape(), cpu_dtype, mlu_dtype, CNML_TENSOR);
  for (int i = 0; i < temp_.size(); i++) {
    temp_[i]->Reshape(bottom[0]->shape(), cpu_dtype, mlu_dtype, CNML_TENSOR);
  }
}

template <typename Dtype>
void MLUEltwiseLayer<Dtype>::MLUDestroyOp() {
  for (int i = 0; i < vec_max_op_ptr_.size(); i++) {
    if (vec_max_op_ptr_[i] != nullptr) {
      cnmlDestroyBaseOp(&vec_max_op_ptr_[i]);
      vec_max_op_ptr_[i] = nullptr;
    }
  }
  for (int i = 0; i < vec_sum_op_ptr_.size(); i++) {
    if (vec_sum_op_ptr_[i] != nullptr) {
      cnmlDestroyBaseOp(&vec_sum_op_ptr_[i]);
      vec_sum_op_ptr_[i] = nullptr;
    }
  }
  for (int i = 0; i < vec_mult_op_ptr_.size(); i++) {
    if (vec_mult_op_ptr_[i] != nullptr) {
      cnmlDestroyBaseOp(&vec_mult_op_ptr_[i]);
      vec_mult_op_ptr_[i] = nullptr;
    }
  }
}

template <typename Dtype>
void MLUEltwiseLayer<Dtype>::MLUCreateOpBindData(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  switch (this->op_) {
    case EltwiseParameter_EltwiseOp_PROD:
      if (bottom.size() == 2) {
        MLU_CHECK(cnmlCreateMultOp(&vec_mult_op_ptr_[0],
                                  bottom[0]->mlu_tensor(),
                                  bottom[1]->mlu_tensor(),
                                  top[0]->mlu_tensor()));
      } else {
        MLU_CHECK(cnmlCreateMultOp(&vec_mult_op_ptr_[0],
                                  bottom[0]->mlu_tensor(),
                                  bottom[1]->mlu_tensor(),
                                  temp_[0]->mlu_tensor()));

        for (size_t i = 2; i < bottom.size()-1; i++) {
          MLU_CHECK(cnmlCreateMultOp(&vec_mult_op_ptr_[i-1],
                                    bottom[i]->mlu_tensor(),
                                    temp_[i-2]->mlu_tensor(),
                                    temp_[i-1]->mlu_tensor()));
        }
        MLU_CHECK(cnmlCreateMultOp(&vec_mult_op_ptr_[vec_mult_op_ptr_.size()-1],
                                  temp_[temp_.size()-1]->mlu_tensor(),
                                  bottom[bottom.size()-1]->mlu_tensor(),
                                  top[0]->mlu_tensor()));
      }
      break;
    case EltwiseParameter_EltwiseOp_SUM:
       if (bottom.size() == 2) {
          MLU_CHECK(cnmlCreateCoeffAddOp(&vec_sum_op_ptr_[0],
                      this->coeffs_[0], this->coeffs_[1],
                      bottom[0]->mlu_tensor(),
                      bottom[1]->mlu_tensor(),
                      top[0]->mlu_tensor()));
       } else {
          MLU_CHECK(cnmlCreateCoeffAddOp(&vec_sum_op_ptr_[0],
                      this->coeffs_[0], this->coeffs_[1],
                      bottom[0]->mlu_tensor(),
                      bottom[1]->mlu_tensor(),
                      temp_[0]->mlu_tensor()));
          for (int i = 2; i < bottom.size()-1; i++) {
              MLU_CHECK(cnmlCreateCoeffAddOp(&vec_sum_op_ptr_[i-1],
                      1., this->coeffs_[i],
                      temp_[i-2]->mlu_tensor(),
                      bottom[i]->mlu_tensor(),
                      temp_[i-1]->mlu_tensor()));
          }
          MLU_CHECK(cnmlCreateCoeffAddOp(
                    &vec_sum_op_ptr_[vec_sum_op_ptr_.size()-1],
                    1., this->coeffs_[this->coeffs_.size()-1],
                    temp_[temp_.size()-1]->mlu_tensor(),
                    bottom[bottom.size()-1]->mlu_tensor(),
                    top[0]->mlu_tensor()));
       }
      break;
    case EltwiseParameter_EltwiseOp_MAX:
      if (bottom.size() == 2) {
        MLU_CHECK(cnmlCreateMaxTTOp(&vec_max_op_ptr_[0],
                                   bottom[0]->mlu_tensor(),
                                   bottom[1]->mlu_tensor(),
                                   top[0]->mlu_tensor()));
      } else {
        MLU_CHECK(cnmlCreateMaxTTOp(&vec_max_op_ptr_[0],
                                   bottom[0]->mlu_tensor(),
                                   bottom[1]->mlu_tensor(),
                                   temp_[0]->mlu_tensor()));
        for (int i = 2; i < bottom.size()-1; i++) {
          MLU_CHECK(cnmlCreateMaxTTOp(&vec_max_op_ptr_[i-1],
                                     bottom[i]->mlu_tensor(),
                                     temp_[i-2]->mlu_tensor(),
                                     temp_[i-1]->mlu_tensor()));
        }

        MLU_CHECK(cnmlCreateMaxTTOp(&vec_max_op_ptr_[vec_max_op_ptr_.size()-1],
                                   temp_[temp_.size()-1]->mlu_tensor(),
                                   bottom[bottom.size()-1]->mlu_tensor(),
                                   top[0]->mlu_tensor()));
      }
      break;
  }
}

template <typename Dtype>
void MLUEltwiseLayer<Dtype>::MLUCompileOp() {
  switch (this->op_) {
    case EltwiseParameter_EltwiseOp_PROD:
      for (int i = 0; i < vec_mult_op_ptr_.size(); i++) {
        MLU_CHECK(cnmlCompileBaseOp(vec_mult_op_ptr_[i],
                                    Caffe::rt_core(),
                                    Caffe::model_parallel()));
      }
      break;
    case EltwiseParameter_EltwiseOp_SUM:
      for (int i = 0; i < vec_sum_op_ptr_.size(); i++) {
        MLU_CHECK(cnmlCompileBaseOp(vec_sum_op_ptr_[i],
                                    Caffe::rt_core(),
                                    Caffe::model_parallel()));
      }
      break;
    case EltwiseParameter_EltwiseOp_MAX:
      for (int i = 0; i < vec_max_op_ptr_.size(); i++) {
        MLU_CHECK(cnmlCompileBaseOp(vec_max_op_ptr_[i],
                                    Caffe::rt_core(),
                                    Caffe::model_parallel()));
      }
      break;
  }
}

template <typename Dtype>
void MLUEltwiseLayer<Dtype>::fuse(MFusion<Dtype>* fuser) {
  switch (this->op_) {
    case EltwiseParameter_EltwiseOp_PROD:
      for (int i = 0; i < vec_mult_op_ptr_.size(); i++) {
        fuser->fuse(vec_mult_op_ptr_[i]);
      }
      break;
    case EltwiseParameter_EltwiseOp_SUM:
      for (int i = 0; i < vec_sum_op_ptr_.size(); i++) {
        fuser->fuse(vec_sum_op_ptr_[i]);
      }
      break;
    case EltwiseParameter_EltwiseOp_MAX:
      for (int i = 0; i < vec_max_op_ptr_.size(); i++) {
        fuser->fuse(vec_max_op_ptr_[i]);
      }
      break;
  }
}

template <typename Dtype>
MLUEltwiseLayer<Dtype>::~MLUEltwiseLayer() {
  MLUDestroyOp();
  for (int i = 0; i < alpha_.size(); i++) { delete alpha_[i]; }
  for (int i = 0; i < temp_.size(); i++)  { delete temp_[i]; }
}

template <typename Dtype>
void MLUEltwiseLayer<Dtype>::Forward_mlu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  switch (this->op_) {
    case EltwiseParameter_EltwiseOp_PROD:
      if (bottom.size() == 2) {
        MLU_CHECK(cnmlComputeMultOpForward_V3(vec_mult_op_ptr_[0],
                                 bottom[0]->mutable_mlu_data(),
                                 bottom[1]->mutable_mlu_data(),
                                 top[0]->mutable_mlu_data(),
                                 Caffe::forward_param(),
                                 Caffe::queue()));
      } else {
        MLU_CHECK(cnmlComputeMultOpForward_V3(vec_mult_op_ptr_[0],
                                 bottom[0]->mutable_mlu_data(),
                                 bottom[1]->mutable_mlu_data(),
                                 temp_[0]->mutable_mlu_data(),
                                 Caffe::forward_param(),
                                 Caffe::queue()));
        for (size_t i = 2; i < bottom.size()-1; i++) {
          MLU_CHECK(cnmlComputeMultOpForward_V3(vec_mult_op_ptr_[i-1],
                                   bottom[i]->mutable_mlu_data(),
                                   temp_[i-2]->mutable_mlu_data(),
                                   temp_[i-1]->mutable_mlu_data(),
                                   Caffe::forward_param(),
                                   Caffe::queue()));
        }
        MLU_CHECK(cnmlComputeMultOpForward_V3(
                                 vec_mult_op_ptr_[vec_mult_op_ptr_.size()-1],
                                 temp_[temp_.size()-1]->mutable_mlu_data(),
                                 bottom[bottom.size()-1]->mutable_mlu_data(),
                                 top[0]->mutable_mlu_data(),
                                 Caffe::forward_param(),
                                 Caffe::queue()));
      }
      break;
    case EltwiseParameter_EltwiseOp_SUM:
      if (bottom.size() == 2) {
         MLU_CHECK(cnmlComputeCoeffAddOpForward_V3(vec_sum_op_ptr_[0],
                                bottom[0]->mutable_mlu_data(),
                                bottom[1]->mutable_mlu_data(),
                                top[0]->mutable_mlu_data(),
                                Caffe::forward_param(), Caffe::queue()));
      } else {
        MLU_CHECK(cnmlComputeCoeffAddOpForward_V3(vec_sum_op_ptr_[0],
                                bottom[0]->mutable_mlu_data(),
                                bottom[1]->mutable_mlu_data(),
                                temp_[0]->mutable_mlu_data(),
                                Caffe::forward_param(), Caffe::queue()));
        for (int i = 2; i < bottom.size()-1; i++) {
            MLU_CHECK(cnmlComputeCoeffAddOpForward_V3(vec_sum_op_ptr_[i-1],
                                temp_[i-2]->mutable_mlu_data(),
                                bottom[i]->mutable_mlu_data(),
                                temp_[i-1]->mutable_mlu_data(),
                                Caffe::forward_param(), Caffe::queue()));
        }
        MLU_CHECK(cnmlComputeCoeffAddOpForward_V3(
                                vec_sum_op_ptr_[vec_sum_op_ptr_.size()-1],
                                temp_[temp_.size()-1]->mutable_mlu_data(),
                                bottom[bottom.size()-1]->mutable_mlu_data(),
                                top[0]->mutable_mlu_data(),
                                Caffe::forward_param(),
                                Caffe::queue()));
      }
    break;
    case EltwiseParameter_EltwiseOp_MAX:
      if (bottom.size() == 2) {
        MLU_CHECK(cnmlComputeMaxTTOpForward_V3(vec_max_op_ptr_[0],
                                 bottom[0]->mutable_mlu_data(),
                                 bottom[1]->mutable_mlu_data(),
                                 top[0]->mutable_mlu_data(),
                                 Caffe::forward_param(),
                                 Caffe::queue()));
      } else {
        MLU_CHECK(cnmlComputeMaxTTOpForward_V3(vec_max_op_ptr_[0],
                                 bottom[0]->mutable_mlu_data(),
                                 bottom[1]->mutable_mlu_data(),
                                 temp_[0]->mutable_mlu_data(),
                                 Caffe::forward_param(),
                                 Caffe::queue()));
        for (int i = 2; i < bottom.size()-1; i++) {
          MLU_CHECK(cnmlComputeMaxTTOpForward_V3(vec_max_op_ptr_[i-1],
                                   bottom[i]->mutable_mlu_data(),
                                   temp_[i-2]->mutable_mlu_data(),
                                   temp_[i-1]->mutable_mlu_data(),
                                   Caffe::forward_param(),
                                   Caffe::queue()));
        }
        MLU_CHECK(cnmlComputeMaxTTOpForward_V3(
                                 vec_max_op_ptr_[vec_max_op_ptr_.size()-1],
                                 temp_[temp_.size()-1]->mutable_mlu_data(),
                                 bottom[bottom.size()-1]->mutable_mlu_data(),
                                 top[0]->mutable_mlu_data(),
                                 Caffe::forward_param(),
                                 Caffe::queue()));
      }
      break;
  }  // switch
}

INSTANTIATE_CLASS(MLUEltwiseLayer);

}  // namespace caffe
#endif
