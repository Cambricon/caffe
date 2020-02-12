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

#include <algorithm>
#include <memory>
#include <vector>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/mlu_psroi_pooling_layer.hpp"
#include "caffe/layers/psroi_pooling_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "gtest/gtest.h"

using size_t = std::size_t;

namespace caffe {
// perpare for ng op data
template <typename Dtype>
struct NParray {
  NParray(Dtype* d, std::vector<int> s) {
    data = d;
    int sz = 1;
    for (size_t i = 0; i < s.size(); ++i) {
      sz *= s[i];
    }
    size = sz;
    shape = s;
  }
  Dtype& operator[](std::vector<int> idxs) {
    assert(idxs.size() == shape.size());
    int tmp = 1;
    int id = 0;
    for (int i = idxs.size()-1; i >= 0; --i) {
      id += idxs[i] * tmp;
      tmp *= shape[i];
    }
    return data[id];
  }
  void print() {
    int jj = shape.back();
    int ii = size / jj;
    int cnt = 0;
    for (int i = 0; i < ii; ++i) {
      for (int j = 0; j < jj; ++j) {
        LOG(INFO) << data[cnt++] << " ";
      }
    }
  }
  Dtype* data;
  int size;
  std::vector<int> shape;
};

template <typename Dtype>
void np_reshape(NParray<Dtype>* np_array, std::vector<int> shape) {
  int sz = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    sz *= shape[i];
  }
  assert(sz == (*np_array).size);
  (*np_array).shape = shape;
}

template <typename Dtype>
void np_transpose(NParray<Dtype>* np_array, std::vector<int> trans) {
  assert(trans.size() == (*np_array).shape.size());
  Dtype* tmp_data = new Dtype[(*np_array).size];
  std::vector<int> idx((*np_array).shape.size(), -1);
  std::vector<int> nshape((*np_array).shape.size(), 0);
  auto& shape = (*np_array).shape;
  for (int i = 0; i < trans.size(); i++) {
    nshape[i] = shape[trans[i]];
  }

  auto get_id = [&trans, &shape, &nshape]
    (std::vector<int> idx, bool do_trans = false) ->int {
    std::vector<int> s;
    if (do_trans) {
      std::vector<int> nidx(trans.size(), 0);
      for (int i = 0; i < trans.size(); i++) {
        nidx[i] = idx[trans[i]];
      }
      idx = nidx;
      s = nshape;
    } else {
      s = shape;
    }
    int tmp = 1;
    int id = 0;
    for (int i = idx.size()-1; i >= 0; --i) {
      id += idx[i] * tmp;
      tmp *= s[i];
    }
    return id;
  };

  int j = 0;
  int total_cnt = 0;
  while (j >= 0) {
    if (j == shape.size()) {
      int k1 = get_id(idx, true);
      int k2 = get_id(idx);
      tmp_data[k1] = (*np_array).data[k2];
      total_cnt++;
      j--;
    } else if (shape[j]-1 > idx[j]) {
      idx[j] += 1;
      j++;
    } else {
      idx[j] = -1;
      j--;
    }
  }
  // PLOG << total_cnt << " " << np_array.size;
  assert(total_cnt == (*np_array).size);

  for (int i = 0; i < (*np_array).size; ++i)
    (*np_array).data[i] = tmp_data[i];
  (*np_array).shape = nshape;

  delete [] tmp_data;
}
// Reference psroipool for check results:
template <typename Dtype>
void caffe_psroipool(const Blob<Dtype>* bottom_data_, const Blob<Dtype>* bottom_rois_,
                   PSROIPoolingParameter* psroi_param,
                   Blob<Dtype>* blob_top, const Dtype* top_data, int mode_option) {
  Dtype* output_data = blob_top->mutable_cpu_data();
  const Dtype* bottom_data = bottom_data_->cpu_data();
  const Dtype* bottom_rois = bottom_rois_->cpu_data();
  const int rois_num = bottom_rois_->channels();
  const int channels = bottom_data_->channels();
  const int height = bottom_data_->height();
  const int width = bottom_data_->width();
  const int pooled_height = psroi_param->group_size();
  const int pooled_width = psroi_param->group_size();
  int group_size = psroi_param->group_size();
  float spatial_scale = psroi_param->spatial_scale();
  int output_dim = psroi_param->output_dim();
  int count = blob_top->count();
  caffe_set(count, Dtype(0), output_data);
  Dtype roi_start_w;
  Dtype roi_start_h;
  Dtype roi_end_w;
  Dtype roi_end_h;
  for (int n = 0; n < rois_num; ++n) {
    int roi_add = n*5;
    int roi_batch_ind = 0;
    // [start, end) interval for spatial sampling
    if (mode_option == 0) {
      roi_batch_ind = bottom_rois[roi_add];
      roi_start_w = static_cast<Dtype>(round(bottom_rois[roi_add + 1])) * spatial_scale;
      roi_start_h = static_cast<Dtype>(round(bottom_rois[roi_add + 2])) * spatial_scale;
      roi_end_w = static_cast<Dtype>(round(bottom_rois[roi_add + 3])
                             + 1.) * spatial_scale;
      roi_end_h = static_cast<Dtype>(round(bottom_rois[roi_add + 4])
                             + 1.) * spatial_scale;
    } else {
      roi_batch_ind = bottom_rois[roi_add + 4];
      roi_start_w = static_cast<Dtype>(round(bottom_rois[roi_add + 0])) * spatial_scale;
      roi_start_h = static_cast<Dtype>(round(bottom_rois[roi_add + 1])) * spatial_scale;
      roi_end_w = static_cast<Dtype>(round(bottom_rois[roi_add + 2])
                             + 1.) * spatial_scale;
      roi_end_h = static_cast<Dtype>(round(bottom_rois[roi_add + 3])
                             + 1.) * spatial_scale;
    }

    // Force too small ROIs to be 1x1
    Dtype roi_width = std::max<Dtype>(roi_end_w - roi_start_w, 0.1);  // avoid 0
    Dtype roi_height = std::max<Dtype>(roi_end_h - roi_start_h, 0.1);
    // Compute w and h at bottom
    Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);
    Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);

    // Suppose rois anchor is feathure map, pooling in the rois feathure map
    for (int ctop = 0; ctop < output_dim; ++ctop) {
      for (int ph = 0; ph < pooled_height; ++ph) {
        for (int pw = 0; pw < pooled_width; ++pw) {
          int index = n*output_dim*pooled_height*pooled_width
                      + ctop*pooled_height*pooled_width + ph*pooled_width + pw;
          // The output is in order (n, ctop, ph, pw)
          int hstart = floor(static_cast<Dtype>(ph) * bin_size_h
                          + roi_start_h);
          int wstart = floor(static_cast<Dtype>(pw)* bin_size_w
                          + roi_start_w);
          int hend = ceil(static_cast<Dtype>(ph + 1) * bin_size_h
                          + roi_start_h);
          int wend = ceil(static_cast<Dtype>(pw + 1) * bin_size_w
                      + roi_start_w);
          // Add roi offsets and clip to input boundaries
          hstart = std::min(std::max(hstart, 0), height);
          hend = std::min(std::max(hend, 0), height);
          wstart = std::min(std::max(wstart, 0), width);
          wend = std::min(std::max(wend, 0), width);
          bool is_empty = (hend <= hstart) || (wend <= wstart);
          int gw = pw;
          int gh = ph;
          int c = (ctop*group_size + gh)*group_size + gw;

          // sum the data in the pooling group and get average pooling
          Dtype out_sum = 0;
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              int bottom_index = h*width + w;
              out_sum += bottom_data[(roi_batch_ind * channels + c)
                                      * height * width + bottom_index];
            }
          }
          Dtype bin_area = (hend - hstart)*(wend - wstart);
          if (is_empty) {
            output_data[index] = 0;
          } else {
            output_data[index] = out_sum/bin_area;
          }
        }
      }
    }
  }
  for (int i = 0; i < count; i++) {
    EXPECT_NEAR(top_data[i], output_data[i], 0.1);
  }
}

// prepare input data
void pad_uniform_distribution_data(float* data, int count,
                                               int seed, float min, float max) {
  assert(data);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> uniform(min, max);
  for (int i = 0; i < count; i++) {
    data[i] = uniform(gen);
  }
}

template <typename Dtype>
void pad_normal_distribution_data(Dtype* data, int count, int seed,
                                  Dtype mean, Dtype var) {
  assert(data);
  std::mt19937 gen(seed);
  std::normal_distribution<Dtype> normal(mean, var);
  for (int i = 0; i < count; i++) {
    data[i] = normal(gen);
  }
}

template <typename Dtype>
void setInputData(Blob<Dtype>* bottom_data, Blob<Dtype>* bottom_rois,
                  int mode_option, int group_size, int output_dim, bool op_option) {
  int num_rois = bottom_rois->count() / 5;
  int raw_count = bottom_data->count();
  Dtype* data_raw = static_cast<Dtype*>(malloc(raw_count*sizeof(Dtype)));
  pad_normal_distribution_data(data_raw, raw_count, 1000, Dtype(0), Dtype(1));
  for (int i = 0; i < raw_count; i++) {
    data_raw[i] = data_raw[i] * 3 + 5;
  }
  float* cx = static_cast<float*>(malloc(num_rois * sizeof(float)));
  float* cy = static_cast<float*>(malloc(num_rois * sizeof(float)));
  float* w = static_cast<float*>(malloc(num_rois * sizeof(float)));
  float* h = static_cast<float*>(malloc(num_rois * sizeof(float)));
  float* idx = static_cast<float*>(malloc(num_rois * sizeof(float)));
  memset(idx, 0, num_rois * sizeof(float));  // NOLINT
  pad_uniform_distribution_data(cx, num_rois, 1000, 5, 32);
  pad_uniform_distribution_data(cy, num_rois, 1000, 5, 32);
  pad_uniform_distribution_data(w, num_rois, 1000, 0, 10);
  pad_uniform_distribution_data(h, num_rois, 1000, 0, 10);

  float* x1 = static_cast<float*>(malloc(num_rois * sizeof(float)));
  float* y1 = static_cast<float*>(malloc(num_rois * sizeof(float)));
  float* x2 = static_cast<float*>(malloc(num_rois * sizeof(float)));
  float* y2 = static_cast<float*>(malloc(num_rois * sizeof(float)));
  for (int i = 0; i < num_rois; i++) {
    x1[i] = cx[i] - w[i] / 2;
    x1[i] = std::min(x1[i], static_cast<float>(32));
    y1[i] = cy[i] - h[i] / 2;
    y1[i] = std::min(y1[i], static_cast<float>(32));
    x2[i] = cx[i] + w[i] / 2;
    x2[i] = std::min(x2[i], static_cast<float>(32));
    y2[i] = cy[i] + h[i] / 2;
    y2[i] = std::min(y2[i], static_cast<float>(32));
  }
  free(cx);
  free(cy);
  free(w);
  free(h);

  int unit_num = 5;
  if (mode_option == 0) {
    float* concat_data =
        static_cast<float*>(malloc(num_rois * unit_num * sizeof(float)));
    for (int i = 0; i < num_rois; i++) {
      concat_data[i * unit_num] = idx[i];
      concat_data[i * unit_num + 1] = x1[i];
      concat_data[i * unit_num + 2] = y1[i];
      concat_data[i * unit_num + 3] = x2[i];
      concat_data[i * unit_num + 4] = y2[i];
    }
    for (int i = 0; i < bottom_data->count(); i++) {
      bottom_data->mutable_cpu_data()[i] = data_raw[i];
    }
    for (int i = 0; i < bottom_rois->count(); i++) {
      bottom_rois->mutable_cpu_data()[i] = concat_data[i];
    }
  } else {
    float* rois_conc_data =
        static_cast<float*>(malloc(num_rois * unit_num * sizeof(float)));
    for (int i = 0; i < num_rois; i++) {
      rois_conc_data[i * unit_num] = x1[i];
      rois_conc_data[i * unit_num + 1] = y1[i];
      rois_conc_data[i * unit_num + 2] = x2[i];
      rois_conc_data[i * unit_num + 3] = y2[i];
      rois_conc_data[i * unit_num + 4] = idx[i];
    }

    if (op_option) {  // for NG OP data
      std::vector<int> data_shape = bottom_data->shape();
      NParray<Dtype>* np_data = new NParray<Dtype>(data_raw, data_shape);
      int h_new = data_shape[2];
      int w_new = data_shape[3];
      int c_new = group_size * group_size;
      std::vector<int> data_shape_new = {1, output_dim, c_new, h_new, w_new};
      std::vector<int> data_trans = {0, 2, 1, 3, 4};
      np_reshape(np_data, data_shape_new);
      np_transpose(np_data, data_trans);
      np_reshape(np_data, data_shape);
      bottom_data->set_cpu_data(np_data->data);
    } else {  // for BANG OP data
      for (int i = 0; i < bottom_data->count(); i++) {
        bottom_data->mutable_cpu_data()[i] = data_raw[i];
      }
    }
    for (int i = 0; i < bottom_rois->count(); i++) {
      bottom_rois->mutable_cpu_data()[i] = rois_conc_data[i];
    }
  }
  free(x1);
  free(y1);
  free(x2);
  free(y2);
  free(idx);
}

//  CPU test
template <typename TypeParam>
class PSROIPoolingLayerTest : public CPUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  PSROIPoolingLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(1, 392, 14, 14)),
        blob_bottom_rois_(new Blob<Dtype>(304, 5, 1, 1)),
        blob_top_(new Blob<Dtype>()) {}

  void SetUp() {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    filler.Fill(this->blob_bottom_rois_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_rois_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~PSROIPoolingLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_rois_;
    delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_rois_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(PSROIPoolingLayerTest, TestDtypesAndDevices);

TYPED_TEST(PSROIPoolingLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PSROIPoolingParameter* psroi_param = layer_param.mutable_psroi_pooling_param();
  psroi_param->set_spatial_scale(0.0625);
  psroi_param->set_output_dim(8);
  psroi_param->set_group_size(7);
  PSROIPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_vec_[1]->num());
  EXPECT_EQ(this->blob_top_->channels(), psroi_param->output_dim());
  EXPECT_EQ(this->blob_top_->height(), psroi_param->group_size());
  EXPECT_EQ(this->blob_top_->width(), psroi_param->group_size());
}

TYPED_TEST(PSROIPoolingLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PSROIPoolingParameter* psroi_param = layer_param.mutable_psroi_pooling_param();
  psroi_param->set_spatial_scale(0.0625);
  psroi_param->set_output_dim(8);
  psroi_param->set_group_size(7);
  setInputData(this->blob_bottom_data_, this->blob_bottom_rois_, 0,
               psroi_param->group_size(), psroi_param->output_dim(), 0);
  PSROIPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* top_data = this->blob_top_->cpu_data();
  caffe_psroipool(this->blob_bottom_data_, this->blob_bottom_rois_, psroi_param,
                  this->blob_top_, top_data, 0);
}

//  MLU BangOp Test
#ifdef USE_MLU
template <typename TypeParam>
class MLUPSROIPoolingLayerTest : public MLUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MLUPSROIPoolingLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(1, 392, 14, 14)),
        blob_bottom_rois_(new Blob<Dtype>(304, 5, 1, 1)),
        blob_top_(new Blob<Dtype>()) {}

  void SetUp() {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    filler.Fill(this->blob_bottom_rois_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_rois_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~MLUPSROIPoolingLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_rois_;
    delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_rois_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MLUPSROIPoolingLayerTest, TestMLUDevices);

TYPED_TEST(MLUPSROIPoolingLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PSROIPoolingParameter* psroi_param = layer_param.mutable_psroi_pooling_param();
  psroi_param->set_spatial_scale(0.0625);
  psroi_param->set_output_dim(8);
  psroi_param->set_group_size(7);
  MLUPSROIPoolingLayer<Dtype> layer(layer_param);
  Caffe::setDetectOpMode(1);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_vec_[1]->channels());
  EXPECT_EQ(this->blob_top_->channels(), psroi_param->output_dim());
  EXPECT_EQ(this->blob_top_->height(), psroi_param->group_size());
  EXPECT_EQ(this->blob_top_->width(), psroi_param->group_size());
}

TYPED_TEST(MLUPSROIPoolingLayerTest, TestForwardBangOp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PSROIPoolingParameter* psroi_param = layer_param.mutable_psroi_pooling_param();
  psroi_param->set_spatial_scale(0.0625);
  psroi_param->set_output_dim(8);
  psroi_param->set_group_size(7);
  setInputData(this->blob_bottom_data_, this->blob_bottom_rois_, 1,
               psroi_param->group_size(), psroi_param->output_dim(), 0);
  Caffe::setDetectOpMode(1);
  MLUPSROIPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* top_data = this->blob_top_->cpu_data();
  caffe_psroipool(this->blob_bottom_data_, this->blob_bottom_rois_, psroi_param,
                  this->blob_top_, top_data, 1);
}

//  MFUS Test
template <typename TypeParam>
class MFUSPSROIPoolingLayerTest : public MFUSDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MFUSPSROIPoolingLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(1, 392, 14, 14)),
        blob_bottom_rois_(new Blob<Dtype>(304, 5, 1, 1)),
        blob_top_(new Blob<Dtype>()) {}

  void SetUp() {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    filler.Fill(this->blob_bottom_rois_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_rois_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~MFUSPSROIPoolingLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_rois_;
    delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_rois_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MFUSPSROIPoolingLayerTest, TestMLUDevices);

TYPED_TEST(MFUSPSROIPoolingLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PSROIPoolingParameter* psroi_param = layer_param.mutable_psroi_pooling_param();
  psroi_param->set_spatial_scale(0.0625);
  psroi_param->set_output_dim(8);
  psroi_param->set_group_size(7);
  Caffe::setDetectOpMode(1);
  MLUPSROIPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_vec_[1]->channels());
  EXPECT_EQ(this->blob_top_->channels(), psroi_param->output_dim());
  EXPECT_EQ(this->blob_top_->height(), psroi_param->group_size());
  EXPECT_EQ(this->blob_top_->width(), psroi_param->group_size());
}

TYPED_TEST(MFUSPSROIPoolingLayerTest, TestForwardBangOp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PSROIPoolingParameter* psroi_param = layer_param.mutable_psroi_pooling_param();
  psroi_param->set_spatial_scale(0.0625);
  psroi_param->set_output_dim(8);
  psroi_param->set_group_size(7);
  setInputData(this->blob_bottom_data_, this->blob_bottom_rois_, 1,
               psroi_param->group_size(), psroi_param->output_dim(), 0);
  Caffe::setDetectOpMode(1);
  MLUPSROIPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_TRUE(layer.mfus_supported());
  MFusion<Dtype> fuser;
  fuser.reset();
  fuser.addInputs(this->blob_bottom_vec_);
  fuser.addOutputs(this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.fuse(&fuser);
  fuser.compile();
  fuser.forward();
  const Dtype* top_data = this->blob_top_->cpu_data();
  caffe_psroipool(this->blob_bottom_data_, this->blob_bottom_rois_, psroi_param,
                  this->blob_top_, top_data, 1);
}

#endif
}
