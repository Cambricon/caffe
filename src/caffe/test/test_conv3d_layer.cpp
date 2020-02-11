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

#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/mlu/util.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/vol2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/layers/mlu_conv3d_layer.hpp"
#ifdef USE_CUDNN
#endif

#ifdef USE_MLU
#include "caffe/mlu/reshape_helper.hpp"
#include "caffe/mlu/spliter.hpp"
#include "caffe/mlu/subnet.hpp"

#endif

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
namespace caffe {
// Reference convolution for checking results:
// accumulate through explicit loops over input, output, and filters.
template <typename Dtype>
void caffe_conv(const Blob<Dtype>* in, Convolution3DParameter* conv3d_param,
                const vector<shared_ptr<Blob<Dtype> > >& weights,
                Blob<Dtype>* out) {
  const bool has_depth = (out->num_axes() == 5);
  if (!has_depth) {
    CHECK_EQ(4, out->num_axes());
  }
  // Kernel size, stride, and pad
  int kernel_h, kernel_w;
  if (conv3d_param->has_kernel_size()) {
    kernel_h = kernel_w = conv3d_param->kernel_size();
  }
  int pad_h, pad_w;
  if (conv3d_param->has_pad()) {
    pad_h = pad_w = conv3d_param->pad();
  }
  int stride_h, stride_w;
  if (conv3d_param->has_stride()) {
    stride_h = stride_w = conv3d_param->stride();
  }
  int dilation_h, dilation_w;
  dilation_h = dilation_w = 1;
  int kernel_d, pad_d, stride_d, dilation_d;
  if (has_depth) {
    kernel_d = kernel_h;
    stride_d = conv3d_param->temporal_stride();
    pad_d = pad_h;
    dilation_d = dilation_h;
  } else {
    kernel_d = stride_d = dilation_d = 1;
    pad_d = 0;
  }
  // Groups
  int groups = conv3d_param->group();
  int o_g = out->shape(1) / groups;
  int k_g = in->shape(1) / groups;
  int o_head, k_head;
  // Convolution
  vector<int> weight_offset(4 + has_depth);
  vector<int> in_offset(4 + has_depth);
  vector<int> out_offset(4 + has_depth);
  Dtype* out_data = out->mutable_cpu_data();
  for (int n = 0; n < out->shape(0); n++) {
    for (int g = 0; g < groups; g++) {
      o_head = o_g * g;
      k_head = k_g * g;
      for (int o = 0; o < o_g; o++) {
        for (int k = 0; k < k_g; k++) {
          for (int z = 0; z < (has_depth ? out->shape(2) : 1); z++) {
            for (int y = 0; y < out->shape(2 + has_depth); y++) {
              for (int x = 0; x < out->shape(3 + has_depth); x++) {
                for (int r = 0; r < kernel_d; r++) {
                  for (int p = 0; p < kernel_h; p++) {
                    for (int q = 0; q < kernel_w; q++) {
                      int in_z = z * stride_d - pad_d + r * dilation_d;
                      int in_y = y * stride_h - pad_h + p * dilation_h;
                      int in_x = x * stride_w - pad_w + q * dilation_w;
                      if (in_z >= 0 && in_z < (has_depth ? in->shape(2) : 1) &&
                          in_y >= 0 && in_y < in->shape(2 + has_depth) &&
                          in_x >= 0 && in_x < in->shape(3 + has_depth)) {
                        weight_offset[0] = o + o_head;
                        weight_offset[1] = k;
                        if (has_depth) {
                          weight_offset[2] = r;
                        }
                        weight_offset[2 + has_depth] = p;
                        weight_offset[3 + has_depth] = q;
                        in_offset[0] = n;
                        in_offset[1] = k + k_head;
                        if (has_depth) {
                          in_offset[2] = in_z;
                        }
                        in_offset[2 + has_depth] = in_y;
                        in_offset[3 + has_depth] = in_x;
                        out_offset[0] = n;
                        out_offset[1] = o + o_head;
                        if (has_depth) {
                          out_offset[2] = z;
                        }
                        out_offset[2 + has_depth] = y;
                        out_offset[3 + has_depth] = x;
                        out_data[out->offset(out_offset)] +=
                            in->data_at(in_offset) *
                            weights[0]->data_at(weight_offset);
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  // Bias
  if (conv3d_param->bias_term()) {
    const Dtype* bias_data = weights[1]->cpu_data();
    for (int n = 0; n < out->shape(0); n++) {
      for (int o = 0; o < out->shape(1); o++) {
        for (int z = 0; z < (has_depth ? out->shape(2) : 1); z++) {
          for (int y = 0; y < out->shape(2 + has_depth); y++) {
            for (int x = 0; x < out->shape(3 + has_depth); x++) {
              out_offset[0] = n;
              out_offset[1] = o;
              if (has_depth) {
                out_offset[2] = z;
              }
              out_offset[2 + has_depth] = y;
              out_offset[3 + has_depth] = x;
              out_data[out->offset(out_offset)] += bias_data[o];
            }
          }
        }
      }
    }
  }
}

template void caffe_conv(const Blob<float>* in,
                         Convolution3DParameter* conv3d_param,
                         const vector<shared_ptr<Blob<float> > >& weights,
                         Blob<float>* out);
template void caffe_conv(const Blob<double>* in,
                         Convolution3DParameter* conv3d_param,
                         const vector<shared_ptr<Blob<double> > >& weights,
                         Blob<double>* out);


#ifdef USE_MLU

template <typename TypeParam>
class MLUConvolution3DLayerTest : public MLUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MLUConvolution3DLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(1.);
    ConstantFiller<Dtype> filler(filler_param);
    static const int shape_values[] = {1, 64, 16, 4, 4};
    vector<int> shape_size(shape_values, shape_values + 5);
    blob_bottom_->Reshape(shape_size);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MLUConvolution3DLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  virtual Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
    this->ref_blob_top_.reset(new Blob<Dtype>());
    this->ref_blob_top_->ReshapeLike(*top);
    return this->ref_blob_top_.get();
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  shared_ptr<Blob<Dtype> > ref_blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MLUConvolution3DLayerTest, TestMLUDevices);

TYPED_TEST(MLUConvolution3DLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  Convolution3DParameter* convolution3d_param =
      layer_param.mutable_convolution3d_param();
  convolution3d_param->set_kernel_size(3);
  convolution3d_param->set_kernel_depth(3);
  convolution3d_param->set_stride(1);
  convolution3d_param->set_temporal_stride(1);
  convolution3d_param->set_num_output(128);
  convolution3d_param->set_pad(1);
  shared_ptr<Layer<Dtype> > layer(new MLUConvolution3DLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->shape(0), 1);
  EXPECT_EQ(this->blob_top_->shape(1), 128);
  EXPECT_EQ(this->blob_top_->shape(2), 14);
  EXPECT_EQ(this->blob_top_->shape(3), 4);
  EXPECT_EQ(this->blob_top_->shape(4), 4);
}
TYPED_TEST(MLUConvolution3DLayerTest, TestSimpleConvolution3d) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  Convolution3DParameter* convolution3d_param =
      layer_param.mutable_convolution3d_param();

  convolution3d_param->set_kernel_size(3);
  convolution3d_param->set_kernel_depth(3);
  convolution3d_param->set_stride(1);
  convolution3d_param->set_temporal_stride(1);
  convolution3d_param->set_num_output(128);
  convolution3d_param->set_temporal_pad(1);
  convolution3d_param->set_pad(1);
  convolution3d_param->mutable_weight_filler()->set_type("constant");
  convolution3d_param->mutable_weight_filler()->set_value(0.01);
  convolution3d_param->mutable_bias_filler()->set_type("constant");
  convolution3d_param->mutable_bias_filler()->set_value(0.1);
  BlobDataType blob_dtype;  // set position
  blob_dtype = get_quantized_info(*this->blob_bottom_, layer_param, "common", DT_INT8);
  layer_param.add_bottom_mlu_dtype()->CopyFrom(blob_dtype);
  int position = -13;  // set weight position
  int scale = 1.55029;
  BlobDataType blobs_dtype;
  blobs_dtype.set_type(DT_INT8);
  blobs_dtype.add_position(position);
  blobs_dtype.add_scale(scale);
  layer_param.add_blobs_dtype()->CopyFrom(blobs_dtype);
  shared_ptr<Layer<Dtype> > layer(new MLUConvolution3DLayer<Dtype>(layer_param));

  Caffe::setCoreNumber(4);
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  top_data = this->blob_top_->cpu_data();
  const Dtype* ref_top_data;
  caffe_conv(this->blob_bottom_, convolution3d_param, layer->blobs(),
             this->MakeReferenceTop(this->blob_top_));
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 0.26);
  }
  OUTPUT("bottom1", this->blob_bottom_->shape_string().c_str());
  EVENT_TIME(layer->get_event_time());
}
template <typename TypeParam>
class MFUSConvolution3DLayerTest : public MFUSDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MFUSConvolution3DLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(1.);
    ConstantFiller<Dtype> filler(filler_param);
    static const int shape_values[] = {1, 64, 16, 4, 4};
    vector<int> shape_size(shape_values, shape_values + 5);
    blob_bottom_->Reshape(shape_size);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~MFUSConvolution3DLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  virtual Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
    this->ref_blob_top_.reset(new Blob<Dtype>());
    this->ref_blob_top_->ReshapeLike(*top);
    return this->ref_blob_top_.get();
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  shared_ptr<Blob<Dtype> > ref_blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MFUSConvolution3DLayerTest, TestMLUDevices);

TYPED_TEST(MFUSConvolution3DLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  Convolution3DParameter* convolution3d_param =
      layer_param.mutable_convolution3d_param();
  convolution3d_param->set_kernel_size(3);
  convolution3d_param->set_kernel_depth(3);
  convolution3d_param->set_stride(1);
  convolution3d_param->set_temporal_stride(1);
  convolution3d_param->set_num_output(128);
  convolution3d_param->set_pad(1);
  shared_ptr<Layer<Dtype> > layer(new MLUConvolution3DLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->shape(0), 1);
  EXPECT_EQ(this->blob_top_->shape(1), 128);
  EXPECT_EQ(this->blob_top_->shape(2), 14);
  EXPECT_EQ(this->blob_top_->shape(3), 4);
  EXPECT_EQ(this->blob_top_->shape(4), 4);
}

TYPED_TEST(MFUSConvolution3DLayerTest, TestSimpleConvolution3D) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  Convolution3DParameter* convolution3d_param =
      layer_param.mutable_convolution3d_param();
  convolution3d_param->set_kernel_size(3);
  convolution3d_param->set_kernel_depth(3);
  convolution3d_param->set_stride(1);
  convolution3d_param->set_temporal_stride(1);
  convolution3d_param->set_num_output(128);
  convolution3d_param->set_temporal_pad(1);
  convolution3d_param->set_pad(1);
  convolution3d_param->mutable_weight_filler()->set_type("constant");
  convolution3d_param->mutable_weight_filler()->set_value(0.01);
  convolution3d_param->mutable_bias_filler()->set_type("constant");
  convolution3d_param->mutable_bias_filler()->set_value(0.1);
  BlobDataType blob_dtype;  // set position
  blob_dtype = get_quantized_info(*this->blob_bottom_, layer_param, "common", DT_INT8);
  layer_param.add_bottom_mlu_dtype()->CopyFrom(blob_dtype);
  int position = -13;  // set weight position
  int scale = 1.55029;
  BlobDataType blobs_dtype;
  blobs_dtype.set_type(DT_INT8);
  blobs_dtype.add_position(position);
  blobs_dtype.add_scale(scale);
  layer_param.add_blobs_dtype()->CopyFrom(blobs_dtype);
  shared_ptr<Layer<Dtype> > layer(new MLUConvolution3DLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  MFusion<Dtype> fuser;
  fuser.reset();
  fuser.addInputs(this->blob_bottom_vec_);
  fuser.addOutputs(this->blob_top_vec_);
  layer->Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->fuse(&fuser);
  fuser.compile();
  fuser.forward();
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_conv(this->blob_bottom_, convolution3d_param, layer->blobs(),
             this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 0.36);
  }
  OUTPUT("bottom1", this->blob_bottom_->shape_string().c_str());
  EVENT_TIME(fuser.get_event_time());
}

#endif
}  // namespace caffe
