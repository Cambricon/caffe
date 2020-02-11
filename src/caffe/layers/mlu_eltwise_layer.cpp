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
      if (alpha_.size() == 0)
        alpha_.resize(bottom.size(), nullptr);
      beta_ =  new Blob<Dtype>();
      vec_scale_op_ptr_.resize(bottom.size(), nullptr);
      vec_add_op_ptr_.resize(bottom.size()-1, nullptr);
      temp_.resize(bottom.size(), nullptr);
      if (bottom.size() > 2)
        temp2_.resize(bottom.size() - 2, nullptr);
      break;
    case EltwiseParameter_EltwiseOp_PROD:
      vec_mult_op_ptr_.resize(bottom.size()-1, nullptr);
      temp_.resize(bottom.size()-2, nullptr);
      break;
    case EltwiseParameter_EltwiseOp_MAX:
      LOG(FATAL) << "NOT IMPLEMENT";
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
  for (int i = 0; i < temp2_.size(); i++) {
    temp2_[i] = new Blob<Dtype>();
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
  BaseDataType mlu_dtype = bottom[0]->mlu_type();
  top[0]->Reshape(bottom[0]->shape(), cpu_dtype, mlu_dtype, CNML_TENSOR);
  for (int i = 0; i < temp_.size(); i++) {
     temp_[i]->Reshape(bottom[0]->shape(), cpu_dtype, mlu_dtype, CNML_TENSOR);
  }
  for (int i = 0; i < temp2_.size(); i++) {
     temp2_[i]->Reshape(bottom[0]->shape(), cpu_dtype, mlu_dtype, CNML_TENSOR);
  }
}

template <typename Dtype>
void MLUEltwiseLayer<Dtype>::MLUDestroyOp() {
  for (int i = 0; i < vec_mult_op_ptr_.size(); i++) {
    if (vec_mult_op_ptr_[i] != nullptr) {
      cnmlDestroyBaseOp(&vec_mult_op_ptr_[i]);
      vec_mult_op_ptr_[i] = nullptr;
    }
  }
  for (int i = 0; i < vec_scale_op_ptr_.size(); i++) {
    if (vec_scale_op_ptr_[i] != nullptr) {
      cnmlDestroyBaseOp(&vec_scale_op_ptr_[i]);
      vec_scale_op_ptr_[i] = nullptr;
    }
  }
  for (int i = 0; i < vec_add_op_ptr_.size(); i++) {
    if (vec_add_op_ptr_[i] != nullptr) {
      cnmlDestroyBaseOp(&vec_add_op_ptr_[i]);
      vec_add_op_ptr_[i] = nullptr;
    }
  }
  for (int i = 0; i < vec_max_op_ptr_.size(); i++) {
    if (vec_max_op_ptr_[i] != nullptr) {
      cnmlDestroyBaseOp(&vec_max_op_ptr_[i]);
      vec_max_op_ptr_[i] = nullptr;
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
    case EltwiseParameter_EltwiseOp_SUM: {
      // reshape alpha and beta
      BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
      vector<int> param_shape(bottom[0]->num_axes(), 1);
      param_shape[0] = 1;
      param_shape[1] = bottom[0]->channels();
      for (int i = 0; i < alpha_.size(); i++) {
          alpha_[i]->Reshape(param_shape, cpu_dtype, DT_FLOAT16, CNML_CONST);
          for (int j = 0; j < alpha_[i]->count(); j++)
              alpha_[i]->mutable_cpu_data()[j] = this->coeffs_[i];
      }
      beta_->Reshape(param_shape, cpu_dtype, DT_FLOAT16, CNML_CONST);
      for (int j = 0; j < beta_->count(); j++)
          beta_->mutable_cpu_data()[j] = 0;
      if (bottom.size() == 1) {
        MLU_CHECK(cnmlCreateNdScaleOp(&vec_scale_op_ptr_[0],
                                      bottom[0]->num_axes() - 1,
                                      bottom[0]->mlu_tensor(),
                                      top[0]->mlu_tensor(),
                                      alpha_[0]->mlu_tensor(),
                                      beta_->mlu_tensor()));
        MLU_CHECK(cnmlBindConstData_V2(alpha_[0]->mlu_tensor(),
                                       alpha_[0]->sync_data(),
                                       false));
        MLU_CHECK(cnmlBindConstData_V2(beta_->mlu_tensor(),
                                       beta_->sync_data(),
                                       false));
      } else {
        for (int i = 0; i < bottom.size(); i++) {
          if (this->coeffs_[i] != 1) {
              MLU_CHECK(cnmlCreateNdScaleOp(&vec_scale_op_ptr_[i],
                                            bottom[0]->num_axes() - 1,
                                            bottom[i]->mlu_tensor(),
                                            temp_[i]->mlu_tensor(),
                                            alpha_[i]->mlu_tensor(),
                                            beta_->mlu_tensor()));
              MLU_CHECK(cnmlBindConstData_V2(alpha_[i]->mlu_tensor(),
                                             alpha_[i]->sync_data(),
                                             false));
              MLU_CHECK(cnmlBindConstData_V2(beta_->mlu_tensor(),
                                              beta_->sync_data(),
                                              false));
          }
        }
        if (bottom.size() == 2) {
            MLU_CHECK(cnmlCreateAddOp(&vec_add_op_ptr_[0],
                        this->coeffs_[0] != 1 ? temp_[0]->mlu_tensor()
                        : bottom[0]->mlu_tensor(),
                        this->coeffs_[1] != 1 ? temp_[1]->mlu_tensor()
                        : bottom[1]->mlu_tensor(),
                        top[0]->mlu_tensor()));
        } else {
            MLU_CHECK(cnmlCreateAddOp(&vec_add_op_ptr_[0],
                        this->coeffs_[0] != 1 ? temp_[0]->mlu_tensor()
                        : bottom[0]->mlu_tensor(),
                        this->coeffs_[1] != 1 ? temp_[1]->mlu_tensor()
                        : bottom[1]->mlu_tensor(),
                        temp2_[0]->mlu_tensor()));
            for (int i = 2; i < bottom.size()-1; i++) {
                MLU_CHECK(cnmlCreateAddOp(&vec_add_op_ptr_[i-1],
                        this->coeffs_[i] != 1 ? temp_[i]->mlu_tensor()
                        : bottom[i]->mlu_tensor(),
                        temp2_[i-2]->mlu_tensor(),
                        temp2_[i-1]->mlu_tensor()));
            }
            MLU_CHECK(cnmlCreateAddOp(&vec_add_op_ptr_[bottom.size()-2],
                        this->coeffs_[bottom.size()-1] != 1
                        ? temp_[bottom.size()-1]->mlu_tensor()
                        : bottom[bottom.size()-1]->mlu_tensor(),
                        temp2_[temp2_.size()-1]->mlu_tensor(),
                        top[0]->mlu_tensor()));
        }
      }
      break;
  }
    case EltwiseParameter_EltwiseOp_MAX:
      LOG(FATAL) << "NOT IMPLEMENT";
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
                                    Caffe::core_number()));
      }
      break;
    case EltwiseParameter_EltwiseOp_SUM:
      for (int i = 0; i < vec_scale_op_ptr_.size(); i++) {
          if (this->coeffs_[i] != 1)
              MLU_CHECK(cnmlCompileBaseOp(vec_scale_op_ptr_[i],
                                          Caffe::rt_core(),
                                          Caffe::core_number()));
      }
      for (int i = 0; i < vec_add_op_ptr_.size(); i++) {
          MLU_CHECK(cnmlCompileBaseOp(vec_add_op_ptr_[i],
                                      Caffe::rt_core(),
                                      Caffe::core_number()));
      }
      break;
    case EltwiseParameter_EltwiseOp_MAX:
      LOG(FATAL) << "NOT IMPLEMENT";
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
      for (int i = 0; i < vec_scale_op_ptr_.size(); i++) {
          if (this->coeffs_[i] != 1)
              fuser->fuse(vec_scale_op_ptr_[i]);
      }
      for (int i = 0; i < vec_add_op_ptr_.size(); i++) {
          fuser->fuse(vec_add_op_ptr_[i]);
      }
      break;
    case EltwiseParameter_EltwiseOp_MAX:
      LOG(FATAL) << "NOT IMPLEMENT";
      break;
  }
}

template <typename Dtype>
MLUEltwiseLayer<Dtype>::~MLUEltwiseLayer() {
  MLUDestroyOp();
  for (int i = 0; i < alpha_.size(); i++) {
    delete alpha_[i];
    alpha_[i] = nullptr;
  }
  for (int i = 0; i < temp_.size(); i++)  {
    delete temp_[i];
    temp_[i] = nullptr;
  }
  for (int i = 0; i < temp2_.size(); i++) {
    delete temp2_[i];
    temp2_[i] = nullptr;
  }
  if (beta_) {
    delete beta_;
    beta_ = nullptr;
  }
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
      if (bottom.size() == 1) {
        MLU_CHECK(cnmlComputeNdScaleOpForward(vec_scale_op_ptr_[0],
                                              NULL,
                                              bottom[0]->mutable_mlu_data(),
                                              NULL,
                                              top[0]->mutable_mlu_data(),
                                              Caffe::queue(),
                                              NULL));
      } else {
        for (int i = 0; i < bottom.size(); i++) {
            if (this->coeffs_[i] != 1)
                MLU_CHECK(cnmlComputeNdScaleOpForward(vec_scale_op_ptr_[i],
                            NULL,
                            bottom[i]->mutable_mlu_data(),
                            NULL,
                            temp_[i]->mutable_mlu_data(),
                            Caffe::queue(),
                            NULL));
        }
        if (bottom.size() == 2) {
            MLU_CHECK(cnmlComputeAddOpForward_V3(vec_add_op_ptr_[0],
                this->coeffs_[0] != 1 ? temp_[0]->mutable_mlu_data() :
                bottom[0]->mutable_mlu_data(),
                this->coeffs_[1] != 1 ? temp_[1]->mutable_mlu_data() :
                bottom[1]->mutable_mlu_data(),
                top[0]->mutable_mlu_data(),
                Caffe::forward_param(),
                Caffe::queue()));
        } else {
            MLU_CHECK(cnmlComputeAddOpForward_V3(vec_add_op_ptr_[0],
                this->coeffs_[0] != 1 ? temp_[0]->mutable_mlu_data() :
                bottom[0]->mutable_mlu_data(),
                this->coeffs_[1] != 1 ? temp_[1]->mutable_mlu_data() :
                bottom[1]->mutable_mlu_data(),
                temp2_[0]->mutable_mlu_data(),
                Caffe::forward_param(),
                Caffe::queue()));
            for (int i = 2; i < bottom.size()-1; i++) {
                MLU_CHECK(cnmlComputeAddOpForward_V3(vec_add_op_ptr_[i-1],
                            this->coeffs_[i] != 1 ?
                            temp_[i]->mutable_mlu_data() :
                            bottom[i]->mutable_mlu_data(),
                            temp2_[i-2]->mutable_mlu_data(),
                            temp2_[i-1]->mutable_mlu_data(),
                            Caffe::forward_param(),
                            Caffe::queue()));
            }
            MLU_CHECK(cnmlComputeAddOpForward_V3(vec_add_op_ptr_[bottom.size()-2],
                        this->coeffs_[bottom.size()-1] != 1 ?
                        temp_[bottom.size()-1]->mutable_mlu_data() :
                        bottom[bottom.size()-1]->mutable_mlu_data(),
                        temp2_[temp2_.size()-1]->mutable_mlu_data(),
                        top[0]->mutable_mlu_data(),
                        Caffe::forward_param(),
                        Caffe::queue()));
        }
      }
      break;
    case EltwiseParameter_EltwiseOp_MAX:
      LOG(FATAL) << "NOT IMPLEMENT";
      break;
  }  // switch
}

INSTANTIATE_CLASS(MLUEltwiseLayer);

}  // namespace caffe
#endif
