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

// Make sure we include Python.h before any system header
// to avoid _POSIX_C_SOURCE redefinition
#ifdef WITH_PYTHON_LAYER
#include <boost/python.hpp>
#endif
#include <memory>
#include <string>
#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/layers/absval_layer.hpp"
#include "caffe/layers/add_layer.hpp"
#include "caffe/layers/argmax_layer.hpp"
#include "caffe/layers/axpy_layer.hpp"
#include "caffe/layers/batch_norm_layer.hpp"
#include "caffe/layers/bn_layer.hpp"
#include "caffe/layers/concat_layer.hpp"
#include "caffe/layers/conv_depthwise_layer.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/convolution3d_layer.hpp"
#include "caffe/layers/crelu_layer.hpp"
#include "caffe/layers/crop_layer.hpp"
#include "caffe/layers/cycleadd_layer.hpp"
#include "caffe/layers/cyclemult_layer.hpp"
#include "caffe/layers/cyclesub_layer.hpp"
#include "caffe/layers/deconv_layer.hpp"
#include "caffe/layers/detection_out_layer.hpp"
#include "caffe/layers/detection_output_layer.hpp"
#include "caffe/layers/detection_pose_output_layer.hpp"
#include "caffe/layers/dropout_layer.hpp"
#include "caffe/layers/eltwise_layer.hpp"
#include "caffe/layers/elu_layer.hpp"
#include "caffe/layers/exp_layer.hpp"
#include "caffe/layers/flatten_layer.hpp"
#include "caffe/layers/image_detect_layer.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/layers/interp_layer.hpp"
#include "caffe/layers/log_layer.hpp"
#include "caffe/layers/lrn_layer.hpp"
#include "caffe/layers/mult_layer.hpp"
#include "caffe/layers/normalize_layer.hpp"
#include "caffe/layers/permute_layer.hpp"
#include "caffe/layers/pooling_layer.hpp"
#include "caffe/layers/pool3d_layer.hpp"
#include "caffe/layers/power_layer.hpp"
#include "caffe/layers/prelu_layer.hpp"
#include "caffe/layers/prior_box_layer.hpp"
#include "caffe/layers/proposal_layer.hpp"
#include "caffe/layers/psroi_pooling_layer.hpp"
#include "caffe/layers/relu_layer.hpp"
#include "caffe/layers/reorg_layer.hpp"
#include "caffe/layers/reshape_layer.hpp"
#include "caffe/layers/reverse_layer.hpp"
#include "caffe/layers/rnn_layer.hpp"
#include "caffe/layers/roi_align_layer.hpp"
#include "caffe/layers/roi_pooling_layer.hpp"
#include "caffe/layers/scale_layer.hpp"
#include "caffe/layers/shufflechannel_layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"
#include "caffe/layers/silence_layer.hpp"
#include "caffe/layers/slice_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"
#include "caffe/layers/sqrt_layer.hpp"
#include "caffe/layers/sub_layer.hpp"
#include "caffe/layers/tanh_layer.hpp"
#include "caffe/layers/unpooling_layer.hpp"
#include "caffe/layers/upsample_layer.hpp"
#include "caffe/layers/threshold_layer.hpp"
#include "caffe/layers/yolov3_detection_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#ifdef USE_CUDNN
#include "caffe/layers/cudnn_conv_layer.hpp"
#include "caffe/layers/cudnn_deconv_layer.hpp"
#include "caffe/layers/cudnn_lcn_layer.hpp"
#include "caffe/layers/cudnn_lrn_layer.hpp"
#include "caffe/layers/cudnn_pooling_layer.hpp"
#include "caffe/layers/cudnn_relu_layer.hpp"
#include "caffe/layers/cudnn_sigmoid_layer.hpp"
#include "caffe/layers/cudnn_softmax_layer.hpp"
#include "caffe/layers/cudnn_tanh_layer.hpp"
#endif
#ifdef USE_MLU
#include "caffe/layers/mlu_absval_layer.hpp"
#include "caffe/layers/mlu_add_layer.hpp"
#include "caffe/layers/mlu_argmax_layer.hpp"
#include "caffe/layers/mlu_axpy_layer.hpp"
#include "caffe/layers/mlu_batch_norm_layer.hpp"
#include "caffe/layers/mlu_bn_layer.hpp"
#include "caffe/layers/mlu_concat_layer.hpp"
#include "caffe/layers/mlu_conv_depthwise_layer.hpp"
#include "caffe/layers/mlu_conv_layer.hpp"
#include "caffe/layers/mlu_conv3d_layer.hpp"
#include "caffe/layers/mlu_crelu_layer.hpp"
#include "caffe/layers/mlu_crop_layer.hpp"
#include "caffe/layers/mlu_cycleadd_layer.hpp"
#include "caffe/layers/mlu_cyclemult_layer.hpp"
#include "caffe/layers/mlu_cyclesub_layer.hpp"
#include "caffe/layers/mlu_deconv_layer.hpp"
#include "caffe/layers/mlu_detection_out_layer.hpp"
#include "caffe/layers/mlu_detection_output_layer.hpp"
#include "caffe/layers/mlu_dropout_layer.hpp"
#include "caffe/layers/mlu_eltwise_layer.hpp"
#include "caffe/layers/mlu_elu_layer.hpp"
#include "caffe/layers/mlu_exp_layer.hpp"
#include "caffe/layers/mlu_flatten_layer.hpp"
#include "caffe/layers/mlu_image_detect_layer.hpp"
#include "caffe/layers/mlu_inner_product_layer.hpp"
#include "caffe/layers/mlu_interp_layer.hpp"
#include "caffe/layers/mlu_log_layer.hpp"
#include "caffe/layers/mlu_lrn_layer.hpp"
#include "caffe/layers/mlu_mult_layer.hpp"
#include "caffe/layers/mlu_normalize_layer.hpp"
#include "caffe/layers/mlu_permute_layer.hpp"
#include "caffe/layers/mlu_pooling_layer.hpp"
#include "caffe/layers/mlu_pool3d_layer.hpp"
#include "caffe/layers/mlu_power_layer.hpp"
#include "caffe/layers/mlu_prelu_layer.hpp"
#include "caffe/layers/mlu_prior_box_layer.hpp"
#include "caffe/layers/mlu_proposal_layer.hpp"
#include "caffe/layers/mlu_psroi_pooling_layer.hpp"
#include "caffe/layers/mlu_relu_layer.hpp"
#include "caffe/layers/mlu_reorg_layer.hpp"
#include "caffe/layers/mlu_reshape_layer.hpp"
#include "caffe/layers/mlu_resizecrop_layer.hpp"
#include "caffe/layers/mlu_reverse_layer.hpp"
#include "caffe/layers/mlu_rnn_layer.hpp"
#include "caffe/layers/mlu_roi_pooling_layer.hpp"
#include "caffe/layers/mlu_scale_layer.hpp"
#include "caffe/layers/mlu_shufflechannel_layer.hpp"
#include "caffe/layers/mlu_sigmoid_layer.hpp"
#include "caffe/layers/mlu_silence_layer.hpp"
#include "caffe/layers/mlu_slice_layer.hpp"
#include "caffe/layers/mlu_softmax_layer.hpp"
#include "caffe/layers/mlu_sqrt_layer.hpp"
#include "caffe/layers/mlu_ssd_detection_layer.hpp"
#include "caffe/layers/mlu_sub_layer.hpp"
#include "caffe/layers/mlu_tanh_layer.hpp"
#include "caffe/layers/mlu_unpooling_layer.hpp"
#include "caffe/layers/mlu_upsample_layer.hpp"
#include "caffe/layers/mlu_threshold_layer.hpp"
#include "caffe/layers/mlu_yolov3_detection_layer.hpp"
#include "caffe/layers/mlu_addpad_layer.hpp"
#endif
#ifdef WITH_PYTHON_LAYER
#include "caffe/layers/python_layer.hpp"
#endif
namespace caffe {

/* Get UnPoolingLayer */
template <typename Dtype>
shared_ptr<Layer<Dtype>> GetUnPoolingLayer(const LayerParameter& param) {
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new UnPoolingLayer<Dtype>(param));
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS)
      return shared_ptr<Layer<Dtype>>(new MLUUnPoolingLayer<Dtype>(param));
    else
      return shared_ptr<Layer<Dtype>>(new UnPoolingLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
    throw;  // Avoids missing return warning
  }
}

REGISTER_LAYER_CREATOR(UnPooling, GetUnPoolingLayer);

/*Get CReLULayer*/
template <typename Dtype>
shared_ptr<Layer<Dtype>> GetCReLULayer(const LayerParameter& param) {
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new CReLULayer<Dtype>(param));
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS) {
      return shared_ptr<Layer<Dtype>>(new MLUCReLULayer<Dtype>(param));
    } else {
      return shared_ptr<Layer<Dtype>>(new CReLULayer<Dtype>(param));
    }
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
  }
}
REGISTER_LAYER_CREATOR(CReLU, GetCReLULayer);

/*Get UpsampleLayer*/
template <typename Dtype>
shared_ptr<Layer<Dtype>> GetUpsampleLayer(const LayerParameter& param) {
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new UpsampleLayer<Dtype>(param));
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS) {
      return shared_ptr<Layer<Dtype>>(new MLUUpsampleLayer<Dtype>(param));
    } else {
      return shared_ptr<Layer<Dtype>>(new UpsampleLayer<Dtype>(param));
    }
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
  }
}
REGISTER_LAYER_CREATOR(Upsample, GetUpsampleLayer);

/*Get SilenceLayer*/
template <typename Dtype>
shared_ptr<Layer<Dtype>> GetSilenceLayer(const LayerParameter& param) {
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new SilenceLayer<Dtype>(param));
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS) {
      return shared_ptr<Layer<Dtype>>(new MLUSilenceLayer<Dtype>(param));
    } else {
      return shared_ptr<Layer<Dtype>>(new SilenceLayer<Dtype>(param));
    }
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
  }
}
REGISTER_LAYER_CREATOR(Silence, GetSilenceLayer);

/*Get SqrtLayer*/
template <typename Dtype>
shared_ptr<Layer<Dtype>> GetSqrtLayer(const LayerParameter& param) {
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new SqrtLayer<Dtype>(param));
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS) {
      return shared_ptr<Layer<Dtype>>(new MLUSqrtLayer<Dtype>(param));
    } else {
      return shared_ptr<Layer<Dtype>>(new SqrtLayer<Dtype>(param));
    }
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
  }
}
REGISTER_LAYER_CREATOR(Sqrt, GetSqrtLayer);

/*Get ShuffleChannelLayer*/
template <typename Dtype>
shared_ptr<Layer<Dtype>> GetShuffleChannelLayer(const LayerParameter& param) {
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new ShuffleChannelLayer<Dtype>(param));
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS) {
      return shared_ptr<Layer<Dtype>>(new MLUShuffleChannelLayer<Dtype>(param));
    } else {
      return shared_ptr<Layer<Dtype>>(new ShuffleChannelLayer<Dtype>(param));
    }
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
  }
}
REGISTER_LAYER_CREATOR(ShuffleChannel, GetShuffleChannelLayer);

/* crop layer */
template <typename Dtype>
shared_ptr<Layer<Dtype>> GetCropLayer(const LayerParameter& param) {
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new CropLayer<Dtype>(param));
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS) {
      return shared_ptr<Layer<Dtype>>(new MLUCropLayer<Dtype>(param));
    } else {
      return shared_ptr<Layer<Dtype>>(new CropLayer<Dtype>(param));
    }
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
  }
}
REGISTER_LAYER_CREATOR(Crop, GetCropLayer);

template <typename Dtype>
shared_ptr<Layer<Dtype>> GetDropoutLayer(const LayerParameter& param) {
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new DropoutLayer<Dtype>(param));
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS) {
      return shared_ptr<Layer<Dtype>>(new MLUDropoutLayer<Dtype>(param));
    } else {
      return shared_ptr<Layer<Dtype>>(new DropoutLayer<Dtype>(param));
    }
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
  }
}
REGISTER_LAYER_CREATOR(Dropout, GetDropoutLayer);

template <typename Dtype>
shared_ptr<Layer<Dtype>> GetReorgLayer(const LayerParameter& param) {
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new ReorgLayer<Dtype>(param));
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS)
      return shared_ptr<Layer<Dtype>>(new MLUReorgLayer<Dtype>(param));
    else
      return shared_ptr<Layer<Dtype>>(new ReorgLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
  }
}
REGISTER_LAYER_CREATOR(Reorg, GetReorgLayer);

// Get Exp layer according to engine
template <typename Dtype>
shared_ptr<Layer<Dtype>> GetExpLayer(const LayerParameter& param) {
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new ExpLayer<Dtype>(param));
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS) {
      return shared_ptr<Layer<Dtype>>(new MLUEXPLayer<Dtype>(param));
    } else {
      return shared_ptr<Layer<Dtype>>(new ExpLayer<Dtype>(param));
    }
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << "has unknown engine.";
    throw;  // Avoids missing return warning
  }
}
REGISTER_LAYER_CREATOR(Exp, GetExpLayer);
// Get Add layer according to engine
template <typename Dtype>
shared_ptr<Layer<Dtype>> GetAddLayer(const LayerParameter& param) {
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new AddLayer<Dtype>(param));
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS)
      return shared_ptr<Layer<Dtype>>(new MLUAddLayer<Dtype>(param));
    else
      return shared_ptr<Layer<Dtype>>(new AddLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << "has unknown engine.";
    throw;  // Avoids missing return warning
  }
}
REGISTER_LAYER_CREATOR(Add, GetAddLayer);
// Get Axpy layer according to engine
template <typename Dtype>
shared_ptr<Layer<Dtype>> GetAxpyLayer(const LayerParameter& param) {
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new AxpyLayer<Dtype>(param));
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS)
      return shared_ptr<Layer<Dtype>>(new MLUAxpyLayer<Dtype>(param));
    else
      return shared_ptr<Layer<Dtype>>(new AxpyLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << "has unknown engine.";
    throw;  // Avoids missing return warning
  }
}
REGISTER_LAYER_CREATOR(Axpy, GetAxpyLayer);

// Get Power layer according to engine
template <typename Dtype>
shared_ptr<Layer<Dtype>> GetPowerLayer(const LayerParameter& param) {
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new PowerLayer<Dtype>(param));
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS)
      return shared_ptr<Layer<Dtype>>(new MLUPowerLayer<Dtype>(param));
    else
      return shared_ptr<Layer<Dtype>>(new PowerLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << "has unknown engine.";
    throw;  // Avoids missing return warning
  }
}
REGISTER_LAYER_CREATOR(Power, GetPowerLayer);

// Get Eltwise layer according to engine
template <typename Dtype>
shared_ptr<Layer<Dtype>> GetEltwiseLayer(const LayerParameter& param) {
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new EltwiseLayer<Dtype>(param));
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS)
      return shared_ptr<Layer<Dtype>>(new MLUEltwiseLayer<Dtype>(param));
    else
      return shared_ptr<Layer<Dtype>>(new EltwiseLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << "has unknown engine.";
    throw;  // Avoids missing return warning
  }
}
REGISTER_LAYER_CREATOR(Eltwise, GetEltwiseLayer);
// Get normalize layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype>> GetNormalizeLayer(const LayerParameter& param) {
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new NormalizeLayer<Dtype>(param));
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS)
      return shared_ptr<Layer<Dtype>>(new MLUNormalizeLayer<Dtype>(param));
    else
      return shared_ptr<Layer<Dtype>>(new NormalizeLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << "has unknown engine.";
    throw;  // Avoids missing return warning
  }
}
REGISTER_LAYER_CREATOR(Normalize, GetNormalizeLayer);
// Get Sub layer according to engine
template <typename Dtype>
shared_ptr<Layer<Dtype>> GetSubLayer(const LayerParameter& param) {
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new SubLayer<Dtype>(param));
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS)
      return shared_ptr<Layer<Dtype>>(new MLUSubLayer<Dtype>(param));
    else
      return shared_ptr<Layer<Dtype>>(new SubLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << "has unknown engine.";
    throw;  // Avoids missing return warning
  }
}
REGISTER_LAYER_CREATOR(Sub, GetSubLayer);
// Get Mult layer according to engine
template <typename Dtype>
shared_ptr<Layer<Dtype>> GetMultLayer(const LayerParameter& param) {
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new MultLayer<Dtype>(param));
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS)
      return shared_ptr<Layer<Dtype>>(new MLUMultLayer<Dtype>(param));
    else
      return shared_ptr<Layer<Dtype>>(new MultLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << "has unknown engine.";
    throw;  // Avoids missing return warning
  }
}
REGISTER_LAYER_CREATOR(Mult, GetMultLayer);
// Get CycleAdd layer according to engine
template <typename Dtype>
shared_ptr<Layer<Dtype>> GetCycleAddLayer(const LayerParameter& param) {
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new CycleAddLayer<Dtype>(param));
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS)
      return shared_ptr<Layer<Dtype>>(new MLUCycleAddLayer<Dtype>(param));
    else
      return shared_ptr<Layer<Dtype>>(new CycleAddLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << "has unknown engine.";
    throw;  // Avoids missing return warning
  }
}
REGISTER_LAYER_CREATOR(CycleAdd, GetCycleAddLayer);
// Get CycleSub layer according to engine
template <typename Dtype>
shared_ptr<Layer<Dtype>> GetCycleSubLayer(const LayerParameter& param) {
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new CycleSubLayer<Dtype>(param));
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS)
      return shared_ptr<Layer<Dtype>>(new MLUCycleSubLayer<Dtype>(param));
    else
      return shared_ptr<Layer<Dtype>>(new CycleSubLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << "has unknown engine.";
    throw;  // Avoids missing return warning
  }
}
REGISTER_LAYER_CREATOR(CycleSub, GetCycleSubLayer);
// Get CycleMult layer according to engine
template <typename Dtype>
shared_ptr<Layer<Dtype>> GetCycleMultLayer(const LayerParameter& param) {
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new CycleMultLayer<Dtype>(param));
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS)
      return shared_ptr<Layer<Dtype>>(new MLUCycleMultLayer<Dtype>(param));
    else
      return shared_ptr<Layer<Dtype>>(new CycleMultLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << "has unknown engine.";
    throw;  // Avoids missing return warning
  }
}
REGISTER_LAYER_CREATOR(CycleMult, GetCycleMultLayer);
// Get Reverse layer according to engine
template <typename Dtype>
shared_ptr<Layer<Dtype>> GetReverseLayer(const LayerParameter& param) {
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new ReverseLayer<Dtype>(param));
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS)
      return shared_ptr<Layer<Dtype>>(new MLUReverseLayer<Dtype>(param));
    else
      return shared_ptr<Layer<Dtype>>(new ReverseLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << "has unknown engine.";
    throw;  // Avoids missing return warning
  }
}
REGISTER_LAYER_CREATOR(Reverse, GetReverseLayer);
// Get Proposal layer according to engine
template <typename Dtype>
shared_ptr<Layer<Dtype>> GetProposalLayer(const LayerParameter& param) {
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new ProposalLayer<Dtype>(param));
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS)
      return shared_ptr<Layer<Dtype>>(new MLUProposalLayer<Dtype>(param));
    else
      return shared_ptr<Layer<Dtype>>(new ProposalLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << "has unknown engine.";
    throw;  // Avoids missing return warning
  }
}
REGISTER_LAYER_CREATOR(Proposal, GetProposalLayer);
// Get ROIPooling layer according to engine
template <typename Dtype>
shared_ptr<Layer<Dtype>> GetROIPoolingLayer(const LayerParameter& param) {
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new ROIPoolingLayer<Dtype>(param));
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS)
      return shared_ptr<Layer<Dtype>>(new MLUROIPoolingLayer<Dtype>(param));
    else
      return shared_ptr<Layer<Dtype>>(new ROIPoolingLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << "has unknown engine.";
    throw;  // Avoids missing return warning
  }
}
REGISTER_LAYER_CREATOR(ROIPooling, GetROIPoolingLayer);
// Get ROIAlign layer according to engine
template <typename Dtype>
shared_ptr<Layer<Dtype>> GetROIAlignLayer(const LayerParameter& param) {
  return shared_ptr<Layer<Dtype>>(new ROIAlignLayer<Dtype>(param));
}
REGISTER_LAYER_CREATOR(ROIAlign, GetROIAlignLayer);
// Get Interp layer according to engine
template <typename Dtype>
shared_ptr<Layer<Dtype>> GetInterpLayer(const LayerParameter& param) {
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new InterpLayer<Dtype>(param));
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS)
      return shared_ptr<Layer<Dtype>>(new MLUInterpLayer<Dtype>(param));
    else
      return shared_ptr<Layer<Dtype>>(new InterpLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << "has unknown engine.";
    throw;  // Avoids missing return warning
  }
}
REGISTER_LAYER_CREATOR(Interp, GetInterpLayer);
// Get convolution layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype>> GetConvolutionLayer(const LayerParameter& param) {
  ConvolutionParameter conv_param = param.convolution_param();
  Engine engine = param.engine();
#ifdef USE_CUDNN
  bool use_dilation = false;
  for (int i = 0; i < conv_param.dilation_size(); ++i) {
    if (conv_param.dilation(i) > 1) {
      use_dilation = true;
    }
  }
#endif
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_CUDNN
    if (!use_dilation) {
      engine = Engine::CUDNN;
    }
#endif
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new ConvolutionLayer<Dtype>(param));
#ifdef USE_CUDNN
  } else if (engine == Engine::CUDNN) {
    if (use_dilation) {
      LOG(FATAL) << "CuDNN doesn't support the dilated convolution at Layer "
                 << param.name();
    }
    return shared_ptr<Layer<Dtype>>(new CuDNNConvolutionLayer<Dtype>(param));
#endif
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS)
      return shared_ptr<Layer<Dtype>>(new MLUConvolutionLayer<Dtype>(param));
    else
      return shared_ptr<Layer<Dtype>>(new ConvolutionLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
    throw;  // Avoids missing return warning
  }
}
REGISTER_LAYER_CREATOR(Convolution, GetConvolutionLayer);

// Conv3D
template <typename Dtype>
shared_ptr<Layer<Dtype>> GetConvolution3DLayer(const LayerParameter& param) {
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new Convolution3DLayer<Dtype>(param));
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS)
      return shared_ptr<Layer<Dtype>>(new MLUConvolution3DLayer<Dtype>(param));
    else
      return shared_ptr<Layer<Dtype>>(new Convolution3DLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
    throw;  // Avoids missing return warning
  }
}
REGISTER_LAYER_CREATOR(Convolution3D, GetConvolution3DLayer);

// Get convolution_depthwise layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype>> GetConvolutionDepthwiseLayer(
    const LayerParameter& param) {
  ConvolutionParameter convolution_param = param.convolution_param();
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(
        new ConvolutionDepthwiseLayer<Dtype>(param));
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS)
      return shared_ptr<Layer<Dtype>>(
          new MLUConvolutionDepthwiseLayer<Dtype>(param));
    else
      return shared_ptr<Layer<Dtype>>(
          new ConvolutionDepthwiseLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
    throw;  // Avoids missing return warning
  }
}
REGISTER_LAYER_CREATOR(ConvolutionDepthwise, GetConvolutionDepthwiseLayer);

// deconvolution
template <typename Dtype>
shared_ptr<Layer<Dtype>> GetDeconvolutionLayer(const LayerParameter& param) {
  ConvolutionParameter conv_param = param.convolution_param();
  Engine engine = param.engine();
#ifdef USE_CUDNN
  bool use_dilation = false;
  for (int i = 0; i < conv_param.dilation_size(); ++i) {
    if (conv_param.dilation(i) > 1) {
      use_dilation = true;
    }
  }
#endif
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_CUDNN
    if (!use_dilation) {
      engine = Engine::CUDNN;
    }
#endif
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new DeconvolutionLayer<Dtype>(param));
#ifdef USE_CUDNN
  } else if (engine == Engine::CUDNN) {
    if (use_dilation) {
      LOG(FATAL) << "CuDNN doesn't support the dilated convolution at Layer "
                 << param.name();
    }
    return shared_ptr<Layer<Dtype>>(new CuDNNDeconvolutionLayer<Dtype>(param));
#endif
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS)
      return shared_ptr<Layer<Dtype>>(new MLUDeconvolutionLayer<Dtype>(param));
    else
      return shared_ptr<Layer<Dtype>>(new DeconvolutionLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
    throw;  // Avoids missing return warning
  }
}
REGISTER_LAYER_CREATOR(Deconvolution, GetDeconvolutionLayer);

// Get pooling layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype>> GetPoolingLayer(const LayerParameter& param) {
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_CUDNN
    engine = Engine::CUDNN;
#endif
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new PoolingLayer<Dtype>(param));
#ifdef USE_CUDNN
  } else if (engine == Engine::CUDNN) {
    if (param.top_size() > 1) {
      LOG(INFO) << "cuDNN does not support multiple tops. "
                << "Using Caffe's own pooling layer.";
      return shared_ptr<Layer<Dtype>>(new PoolingLayer<Dtype>(param));
    }
    // CuDNN assumes layers are not being modified in place, thus
    // breaking our index tracking for updates in some cases in Caffe.
    // Until there is a workaround in Caffe (index management) or
    // cuDNN, use Caffe layer to max pooling, or don't use in place
    // layers after max pooling layers
    if (param.pooling_param().pool() == PoolingParameter_PoolMethod_MAX) {
      return shared_ptr<Layer<Dtype>>(new PoolingLayer<Dtype>(param));
    } else {
      return shared_ptr<Layer<Dtype>>(new CuDNNPoolingLayer<Dtype>(param));
    }
#endif
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS)
      return shared_ptr<Layer<Dtype>>(new MLUPoolingLayer<Dtype>(param));
    else
      return shared_ptr<Layer<Dtype>>(new PoolingLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
    throw;  // Avoids missing return warning
  }
}
REGISTER_LAYER_CREATOR(Pooling, GetPoolingLayer);

// Get pooling3D layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype>> GetPooling3DLayer(const LayerParameter& param) {
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new Pooling3DLayer<Dtype>(param));
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS)
      return shared_ptr<Layer<Dtype>>(new MLUPooling3DLayer<Dtype>(param));
    else
      return shared_ptr<Layer<Dtype>>(new Pooling3DLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
    throw;  // Avoids missing return warning
  }
}
REGISTER_LAYER_CREATOR(Pooling3D, GetPooling3DLayer);

// Get AbsVal
template <typename Dtype>
shared_ptr<Layer<Dtype>> GetAbsValLayer(const LayerParameter& param) {
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new AbsValLayer<Dtype>(param));
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS)
      return shared_ptr<Layer<Dtype>>(new MLUAbsValLayer<Dtype>(param));
    else
      return shared_ptr<Layer<Dtype>>(new AbsValLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
    throw;  // Avoids missing return warning
  }
}
#ifdef USE_MLU
REGISTER_LAYER_CREATOR(AbsVal, GetAbsValLayer);
#endif
// Get Concat
template <typename Dtype>
shared_ptr<Layer<Dtype>> GetConcatLayer(const LayerParameter& param) {
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new ConcatLayer<Dtype>(param));
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS)
      return shared_ptr<Layer<Dtype>>(new MLUConcatLayer<Dtype>(param));
    else
      return shared_ptr<Layer<Dtype>>(new ConcatLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
    throw;  // Avoids missing return warning
  }
}
REGISTER_LAYER_CREATOR(Concat, GetConcatLayer);
// Get LRN layer according to engine
template <typename Dtype>
shared_ptr<Layer<Dtype>> GetLRNLayer(const LayerParameter& param) {
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_CUDNN
    engine = Engine::CUDNN;
#endif
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new LRNLayer<Dtype>(param));
#ifdef USE_CUDNN
  } else if (engine == Engine::CUDNN) {
    LRNParameter lrn_param = param.lrn_param();
    if (lrn_param.norm_region() == LRNParameter_NormRegion_WITHIN_CHANNEL) {
      return shared_ptr<Layer<Dtype>>(new CuDNNLCNLayer<Dtype>(param));
    } else {
      // local size is too big to be handled through cuDNN
      if (param.lrn_param().local_size() > CUDNN_LRN_MAX_N) {
        return shared_ptr<Layer<Dtype>>(new LRNLayer<Dtype>(param));
      } else {
        return shared_ptr<Layer<Dtype>>(new CuDNNLRNLayer<Dtype>(param));
      }
    }
#endif
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS)
      return shared_ptr<Layer<Dtype>>(new MLULRNLayer<Dtype>(param));
    else
      return shared_ptr<Layer<Dtype>>(new LRNLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
    throw;  // Avoids missing return warning
  }
}
REGISTER_LAYER_CREATOR(LRN, GetLRNLayer);
// Get relu layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype>> GetReLULayer(const LayerParameter& param) {
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_CUDNN
    engine = Engine::CUDNN;
#endif
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new ReLULayer<Dtype>(param));
#ifdef USE_CUDNN
  } else if (engine == Engine::CUDNN) {
    return shared_ptr<Layer<Dtype>>(new CuDNNReLULayer<Dtype>(param));
#endif
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS)
      return shared_ptr<Layer<Dtype>>(new MLUReLULayer<Dtype>(param));
    else
      return shared_ptr<Layer<Dtype>>(new ReLULayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
    throw;  // Avoids missing return warning
  }
}
REGISTER_LAYER_CREATOR(ReLU, GetReLULayer);
// get log layer
template <typename Dtype>
shared_ptr<Layer<Dtype>> GetLogLayer(const LayerParameter& param) {
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new LogLayer<Dtype>(param));
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS)
      return shared_ptr<Layer<Dtype>>(new MLULogLayer<Dtype>(param));
    else
      return shared_ptr<Layer<Dtype>>(new LogLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer" << param.name() << " has unknown engine.";
    throw;
  }
}
#ifdef USE_MLU
REGISTER_LAYER_CREATOR(Log, GetLogLayer);
#endif
// Get prelu layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype>> GetPReLULayer(const LayerParameter& param) {
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new PReLULayer<Dtype>(param));
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS)
      return shared_ptr<Layer<Dtype>>(new MLUPReLULayer<Dtype>(param));
    else
      return shared_ptr<Layer<Dtype>>(new PReLULayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
    throw;  // Avoids missing return warning
  }
}
REGISTER_LAYER_CREATOR(PReLU, GetPReLULayer);
// Get sigmoid layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype>> GetSigmoidLayer(const LayerParameter& param) {
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_CUDNN
    engine = Engine::CUDNN;
#endif
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new SigmoidLayer<Dtype>(param));
#ifdef USE_CUDNN
  } else if (engine == Engine::CUDNN) {
    return shared_ptr<Layer<Dtype>>(new CuDNNSigmoidLayer<Dtype>(param));
#endif
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS) {
      return shared_ptr<Layer<Dtype>>(new MLUSigmoidLayer<Dtype>(param));
    } else {
      return shared_ptr<Layer<Dtype>>(new SigmoidLayer<Dtype>(param));
    }
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
    throw;  // Avoids missing return warning
  }
}
REGISTER_LAYER_CREATOR(Sigmoid, GetSigmoidLayer);
template <typename Dtype>
shared_ptr<Layer<Dtype>> GetELULayer(const LayerParameter& param) {
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new ELULayer<Dtype>(param));
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS) {
      return shared_ptr<Layer<Dtype>>(new MLUELULayer<Dtype>(param));
    } else {
      return shared_ptr<Layer<Dtype>>(new ELULayer<Dtype>(param));
    }
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
    throw;  // Avoids missing return warning
  }
}
REGISTER_LAYER_CREATOR(ELU, GetELULayer);

/* cropresize layer */
template <typename Dtype>
shared_ptr<Layer<Dtype>> GetResizecropLayer(const LayerParameter& param) {
  ResizecropParameter resize_param = param.resize_crop_param();
  return shared_ptr<Layer<Dtype>>(new MLUResizecropLayer<Dtype>(param));
}
REGISTER_LAYER_CREATOR(MLUResizecrop, GetResizecropLayer);

// Argmax Register
template <typename Dtype>
shared_ptr<Layer<Dtype>> GetArgMaxLayer(const LayerParameter& param) {
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new ArgMaxLayer<Dtype>(param));
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS) {
      return shared_ptr<Layer<Dtype>>(new MLUArgMaxLayer<Dtype>(param));
    } else {
      return shared_ptr<Layer<Dtype>>(new ArgMaxLayer<Dtype>(param));
    }
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
    throw;  // Avoids missing return warning
  }
}
REGISTER_LAYER_CREATOR(ArgMax, GetArgMaxLayer);

// Get softmax layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype>> GetSoftmaxLayer(const LayerParameter& param) {
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_CUDNN
    engine = Engine::CUDNN;
#endif
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new SoftmaxLayer<Dtype>(param));
#ifdef USE_CUDNN
  } else if (engine == Engine::CUDNN) {
    return shared_ptr<Layer<Dtype>>(new CuDNNSoftmaxLayer<Dtype>(param));
#endif
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS)
      return shared_ptr<Layer<Dtype>>(new MLUSoftmaxLayer<Dtype>(param));
    else
      return shared_ptr<Layer<Dtype>>(new SoftmaxLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
    throw;  // Avoids missing return warning
  }
}
REGISTER_LAYER_CREATOR(Softmax, GetSoftmaxLayer);

// Get tanh layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype>> GetTanHLayer(const LayerParameter& param) {
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_CUDNN
    engine = Engine::CUDNN;
#endif
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new TanHLayer<Dtype>(param));
#ifdef USE_CUDNN
  } else if (engine == Engine::CUDNN) {
    return shared_ptr<Layer<Dtype>>(new CuDNNTanHLayer<Dtype>(param));
#endif
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS)
      return shared_ptr<Layer<Dtype>>(new MLUTanHLayer<Dtype>(param));
    else
      return shared_ptr<Layer<Dtype>>(new TanHLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
    throw;  // Avoids missing return warning
  }
}
#ifdef USE_MLU
REGISTER_LAYER_CREATOR(TanH, GetTanHLayer);
#endif
template <typename Dtype>
shared_ptr<Layer<Dtype>> GetInnerProductLayer(const LayerParameter& param) {
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new InnerProductLayer<Dtype>(param));
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS)
      return shared_ptr<Layer<Dtype>>(new MLUInnerProductLayer<Dtype>(param));
    else
      return shared_ptr<Layer<Dtype>>(new InnerProductLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
    throw;  // Avoids missing return warning
  }
}
REGISTER_LAYER_CREATOR(InnerProduct, GetInnerProductLayer);

template <typename Dtype>
shared_ptr<Layer<Dtype>> GetScaleLayer(const LayerParameter& param) {
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new ScaleLayer<Dtype>(param));
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS)
      return shared_ptr<Layer<Dtype>>(new MLUScaleLayer<Dtype>(param));
    else
      return shared_ptr<Layer<Dtype>>(new ScaleLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
    throw;
  }
}
REGISTER_LAYER_CREATOR(Scale, GetScaleLayer);

template <typename Dtype>
shared_ptr<Layer<Dtype>> GetBNLayer(const LayerParameter& param) {
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new BNLayer<Dtype>(param));
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS)
      return shared_ptr<Layer<Dtype>>(new MLUBNLayer<Dtype>(param));
    else
      return shared_ptr<Layer<Dtype>>(new BNLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
  }
}
REGISTER_LAYER_CREATOR(BN, GetBNLayer);

template <typename Dtype>
shared_ptr<Layer<Dtype>> GetBatchNormLayer(const LayerParameter& param) {
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new BatchNormLayer<Dtype>(param));
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS)
      return shared_ptr<Layer<Dtype>>(new MLUBatchNormLayer<Dtype>(param));
    else
      return shared_ptr<Layer<Dtype>>(new BatchNormLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
  }
}
REGISTER_LAYER_CREATOR(BatchNorm, GetBatchNormLayer);

template <typename Dtype>
shared_ptr<Layer<Dtype>> GetSliceLayer(const LayerParameter& param) {
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new SliceLayer<Dtype>(param));
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS)
      return shared_ptr<Layer<Dtype>>(new MLUSliceLayer<Dtype>(param));
    else
      return shared_ptr<Layer<Dtype>>(new SliceLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
    throw;  // Avoids missing return warning
  }
}
REGISTER_LAYER_CREATOR(Slice, GetSliceLayer);

template <typename Dtype>
shared_ptr<Layer<Dtype>> GetReshapeLayer(const LayerParameter& param) {
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new ReshapeLayer<Dtype>(param));
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS)
      return shared_ptr<Layer<Dtype>>(new MLUReshapeLayer<Dtype>(param));
    else
      return shared_ptr<Layer<Dtype>>(new ReshapeLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
    throw;  // Avoids missing return warning
  }
}

REGISTER_LAYER_CREATOR(Reshape, GetReshapeLayer);

template <typename Dtype>
shared_ptr<Layer<Dtype>> GetFlattenLayer(const LayerParameter& param) {
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new FlattenLayer<Dtype>(param));
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS)
      return shared_ptr<Layer<Dtype>>(new MLUFlattenLayer<Dtype>(param));
    else
      return shared_ptr<Layer<Dtype>>(new FlattenLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
    throw;  // Avoids missing return warning
  }
}

REGISTER_LAYER_CREATOR(Flatten, GetFlattenLayer);

template <typename Dtype>
shared_ptr<Layer<Dtype>> GetPermuteLayer(const LayerParameter& param) {
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new PermuteLayer<Dtype>(param));
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS)
      return shared_ptr<Layer<Dtype>>(new MLUPermuteLayer<Dtype>(param));
    else
      return shared_ptr<Layer<Dtype>>(new PermuteLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
    throw;  // Avoids missing return warning
  }
}

REGISTER_LAYER_CREATOR(Permute, GetPermuteLayer);

template <typename Dtype>
shared_ptr<Layer<Dtype>> GetRNNLayer(const LayerParameter& param) {
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new RNNLayer<Dtype>(param));
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS)
      return shared_ptr<Layer<Dtype>>(new MLURNNLayer<Dtype>(param));
    else
      return shared_ptr<Layer<Dtype>>(new RNNLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
    throw;  // Avoids missing return warning
  }
}
REGISTER_LAYER_CREATOR(RNN, GetRNNLayer);

template <typename Dtype>
shared_ptr<Layer<Dtype>> GetDetectionOutputLayer(const LayerParameter& param) {
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new DetectionOutputLayer<Dtype>(param));
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS)
      return shared_ptr<Layer<Dtype>>(
          new MLUDetectionOutputLayer<Dtype>(param));
    else
      return shared_ptr<Layer<Dtype>>(new DetectionOutputLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
    throw;  // Avoids missing return warning
  }
}

REGISTER_LAYER_CREATOR(DetectionOutput, GetDetectionOutputLayer);

template <typename Dtype>
shared_ptr<Layer<Dtype>> GetDetectionPoseOutputLayer(
    const LayerParameter& param) {
  return shared_ptr<Layer<Dtype>>(new DetectionPoseOutputLayer<Dtype>(param));
}

REGISTER_LAYER_CREATOR(DetectionPoseOutput, GetDetectionPoseOutputLayer);

template <typename Dtype>
shared_ptr<Layer<Dtype>> GetPriorBoxLayer(const LayerParameter& param) {
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new PriorBoxLayer<Dtype>(param));
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS)
      return shared_ptr<Layer<Dtype>>(new MLUPriorBoxLayer<Dtype>(param));
    else
      return shared_ptr<Layer<Dtype>>(new PriorBoxLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
    throw;  // Avoids missing return warning
  }
}

REGISTER_LAYER_CREATOR(PriorBox, GetPriorBoxLayer);

template <typename Dtype>
shared_ptr<Layer<Dtype>> GetPSROIPoolingLayer(const LayerParameter& param) {
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new PSROIPoolingLayer<Dtype>(param));
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS)
      return shared_ptr<Layer<Dtype>>(new MLUPSROIPoolingLayer<Dtype>(param));
    else
      return shared_ptr<Layer<Dtype>>(new PSROIPoolingLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
    throw;  // Avoids missing return warning
  }
}

REGISTER_LAYER_CREATOR(PSROIPooling, GetPSROIPoolingLayer);

#ifdef USE_MLU
template <typename Dtype>
shared_ptr<Layer<Dtype>> GetSsdDetectionLayer(const LayerParameter& param) {
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::MLU;
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new MLUSsdDetectionLayer<Dtype>(param));
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS)
      return shared_ptr<Layer<Dtype>>(new MLUSsdDetectionLayer<Dtype>(param));
    else
      return shared_ptr<Layer<Dtype>>(new MLUSsdDetectionLayer<Dtype>(param));
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
    throw;  // Avoids missing return warning
  }
}

REGISTER_LAYER_CREATOR(SsdDetection, GetSsdDetectionLayer);

#endif

template <typename Dtype>
shared_ptr<Layer<Dtype>> GetImageDetectLayer(const LayerParameter& param) {
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new ImageDetectLayer<Dtype>(param));
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS)
      return shared_ptr<Layer<Dtype>>(new MLUImageDetectLayer<Dtype>(param));
    else
      return shared_ptr<Layer<Dtype>>(new MLUImageDetectLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
    throw;  // Avoids missing return warning
  }
}

REGISTER_LAYER_CREATOR(ImageDetect, GetImageDetectLayer);

// Get Detection_out Layer according to engine
template <typename Dtype>
shared_ptr<Layer<Dtype>> GetDetectionOutLayer(const LayerParameter& param) {
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new DetectionOutLayer<Dtype>(param));
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::getDetectOpMode() == true)
      return shared_ptr<Layer<Dtype>>(new MLUDetectionOutLayer<Dtype>(param));
    else
      return shared_ptr<Layer<Dtype>>(new DetectionOutLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << "has unknown engine.";
    throw;
  }
}
REGISTER_LAYER_CREATOR(DetectionOut, GetDetectionOutLayer);

// Get Threshold Layer according to engine
template <typename Dtype>
shared_ptr<Layer<Dtype>> GetThresholdLayer(const LayerParameter& param) {
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new ThresholdLayer<Dtype>(param));
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS)
      return shared_ptr<Layer<Dtype>>(new MLUThresholdLayer<Dtype>(param));
    else
      return shared_ptr<Layer<Dtype>>(new ThresholdLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << "has unknown engine.";
    throw;
  }
}
REGISTER_LAYER_CREATOR(Threshold, GetThresholdLayer);

#ifdef USE_MLU
// Get AddPad Layer according to engine
template <typename Dtype>
shared_ptr<Layer<Dtype>> GetAddPadLayer(const LayerParameter& param) {
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::MLU;
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new MLUAddPadLayer<Dtype>(param));
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS)
      return shared_ptr<Layer<Dtype>>(new MLUAddPadLayer<Dtype>(param));
    else
      return shared_ptr<Layer<Dtype>>(new MLUAddPadLayer<Dtype>(param));
  } else {
    LOG(FATAL) << "Layer " << param.name() << "has unknown engine.";
    throw;
  }
}
REGISTER_LAYER_CREATOR(AddPad, GetAddPadLayer);
#endif

// Get yolov3 Layer according to engine
template <typename Dtype>
shared_ptr<Layer<Dtype>> GetYolov3DetectionLayer(const LayerParameter& param) {
  Engine engine = param.engine();
  if (engine == Engine::DEFAULT) {
    engine = Engine::CAFFE;
#ifdef USE_MLU
    engine = Engine::MLU;
#endif
  }
  if (engine == Engine::CAFFE) {
    return shared_ptr<Layer<Dtype>>(new Yolov3DetectionLayer<Dtype>(param));
#ifdef USE_MLU
  } else if (engine == Engine::MLU) {
    if (Caffe::mode() == Caffe::MLU || Caffe::mode() == Caffe::MFUS)
      return shared_ptr<Layer<Dtype>>(new MLUYolov3DetectionLayer<Dtype>(param));
    else
      return shared_ptr<Layer<Dtype>>(new MLUYolov3DetectionLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << "has unknown engine.";
    throw;
  }
}
REGISTER_LAYER_CREATOR(Yolov3Detection, GetYolov3DetectionLayer);

#ifdef WITH_PYTHON_LAYER
template <typename Dtype>
shared_ptr<Layer<Dtype>> GetPythonLayer(const LayerParameter& param) {
  Py_Initialize();
  try {
    bp::object module = bp::import(param.python_param().module().c_str());
    bp::object layer = module.attr(param.python_param().layer().c_str())(param);
    return bp::extract<shared_ptr<PythonLayer<Dtype>>>(layer)();
  } catch (bp::error_already_set) {
    PyErr_Print();
    throw;
  }
}
REGISTER_LAYER_CREATOR(Python, GetPythonLayer);
#endif
// Layers that use their constructor as their default creator should be
// registered in their corresponding cpp files. Do not register them here.
}  // namespace caffe
