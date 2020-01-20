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

#ifndef INCLUDE_CAFFE_LAYER_HPP_
#define INCLUDE_CAFFE_LAYER_HPP_

#include <algorithm>
#include <chrono> // NOLINT
#include <map>
#include <string>
#include <vector>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/mlu/fusion.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/math_functions.hpp"
#include "cnplugin.h"

/**
 Forward declare boost::thread instead of including boost/thread.hpp
 to avoid a boost/NVCC issues (#1009, #1010) on OSX.
 */
namespace boost {
class mutex;
}

namespace caffe {

/**
 * @brief An interface for the units of computation which can be composed into a
 *        Net.
 *
 * Layer%s must implement a Forward function, in which they take their input
 * (bottom) Blob%s (if any) and compute their output Blob%s (if any).
 * They may also implement a Backward function, in which they compute the error
 * gradients with respect to their input Blob%s, given the error gradients with
 * their output Blob%s.
 */
template <typename Dtype>
class Layer {
  public:
  /**
   * You should not implement your own constructor. Any set up code should go
   * to SetUp(), where the dimensions of the bottom blobs are provided to the
   * layer.
   */
  explicit Layer(const LayerParameter& param) : layer_param_(param) {
    // Set phase and copy blobs (if there are any).
    phase_ = param.phase();
    external_output_ = param.external_output();
    if (layer_param_.blobs_size() > 0) {
      blobs_.resize(layer_param_.blobs_size());
      for (int i = 0; i < layer_param_.blobs_size(); ++i) {
        blobs_[i].reset(new Blob<Dtype>());
        blobs_[i]->FromProto(layer_param_.blobs(i));
      }
    }
  }
  virtual ~Layer() {}

  /**
   * @brief Implements common layer setup functionality.
   *
   * @param bottom the preshaped input blobs
   * @param top
   *     the allocated but unshaped output blobs, to be shaped by Reshape
   *
   * Checks that the number of bottom and top blobs is correct.
   * Calls LayerSetUp to do special layer setup for individual layer types,
   * followed by Reshape to set up sizes of top blobs and internal buffers.
   * Sets up the loss weight multiplier blobs for any non-zero loss weights.
   * This method may not be overridden.
   */
  void SetUp(const vector<Blob<Dtype>*>& bottom,
             const vector<Blob<Dtype>*>& top) {
    CheckBlobCounts(bottom, top);
    LayerSetUp(bottom, top);
#ifdef USE_MLU
    Reshape_tensor(bottom, top);
#else
    Reshape(bottom, top);
#endif
    SetLossWeights(top);
  }

  /**
   * @brief Does layer-specific setup: your layer should implement this function
   *        as well as Reshape.
   *
   * @param bottom
   *     the preshaped input blobs, whose data fields store the input data for
   *     this layer
   * @param top
   *     the allocated but unshaped output blobs
   *
   * This method should do one-time layer specific setup. This includes reading
   * and processing relevent parameters from the <code>layer_param_</code>.
   * Setting up the shapes of top blobs and internal buffers should be done in
   * <code>Reshape</code>, which will be called before the forward pass to
   * adjust the top blob sizes.
   */

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top) {}

  /**
   * @brief Adjust the shapes of top blobs and internal buffers to accommodate
   *        the shapes of the bottom blobs.
   *
   * @param bottom the input blobs, with the requested input shapes
   * @param top the top blobs, which should be reshaped as needed
   *
   * This method should reshape top blobs as needed according to the shapes
   * of the bottom (input) blobs, as well as reshaping any internal buffers
   * and making any other necessary adjustments so that the layer can
   * accommodate the bottom blobs.
   */
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top) = 0;

#ifdef USE_MLU
  /**
   * @brief Adjust the shapes of top blobs and internal buffers to accommodate
   *        the shapes of the bottom blobs.
   *
   * @param bottom the input blobs, with the requested input shapes
   * @param top the top blobs, which should be reshaped as needed
   *
   * This method can be called any times unitl CreateOpBindData method is called.
   *
   */
  virtual void Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                              const vector<Blob<Dtype>*>& top) {
    CHECK(!mfus_supported());
    Reshape(bottom, top);
  }

  virtual void debug_dtype_info(const vector<Blob<Dtype>*>& bottom,
                        const vector<Blob<Dtype>*>& top) {
    LOG(INFO) << "------------------------------------";
    LOG(INFO) << "reshape_mlu: " << layer_param_.name();
    LOG(INFO) << "Layer Type: " << layer_param_.type();
    if (layer_param_.type() != "Input") {
      for (int i = 0; i < bottom.size(); i++) {
        LOG(INFO) << "bottom dtype: "
          << to_str_dtype(bottom[0]->mlu_type());
        LOG(INFO) << "bottom shape: " << bottom[0]->shape_string();
      }

    }
    for (int i = 0; i < top.size(); i++) {
      LOG(INFO) << "top dtype: " << to_str_dtype(top[0]->mlu_type());
      LOG(INFO) << "top shape: " << top[0]->shape_string();
    }
    for (int i = 0; i < blobs_.size(); i++) {
      LOG(INFO) << "blobs[" << i << "] dtype: "
        << to_str_dtype(blobs_[i]->mlu_type());
      LOG(INFO) << "blobs shape: " << blobs_[i]->shape_string();
    }
  }

  /**
   * @brief Reshape blobs in MLU mode.
   *
   */

  virtual void Reshape_mlu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top) {
    MLUDestroyOp();
    Reshape_tensor(bottom, top);
    MLUCreateOpBindData(bottom, top);
    if (layer_param_.debug_dtype())
      debug_dtype_info(bottom, top);
    MLUCompileOp();
  }

  /**
   * @brief Reshape blobs in Fusion mode.
   */
  virtual void Reshape_mfus(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top) {
    if (mfus_supported()) {
      MLUDestroyOp();
      Reshape_tensor(bottom, top);
      MLUCreateOpBindData(bottom, top);
      if (layer_param_.debug_dtype())
              debug_dtype_info(bottom, top);
    } else {
      Reshape_mlu(bottom, top);
    }
  }
  /**
   * @brief Dispatch reshape task to different reshape routines as
   *        MLU has specific job for different mode.
   *
   * This is similar with Forward();
  */
  inline void Reshape_dispatch(const vector<Blob<Dtype>*>& bottom,
                               const vector<Blob<Dtype>*>& top);

  /**
   * @brief Fuse current layer to subnet for Fusion and offline mode.
   *
  */
  virtual void fuse(MFusion<Dtype>* fuser) {
    CHECK(!mfus_supported());
    CHECK_NOTNULL(fuser);
    CHECK(Caffe::mode() == Caffe::MFUS);
  }
#endif

  /**
   * @brief Given the bottom blobs, compute the top blobs and the loss.
   *
   * @param bottom
   *     the input blobs, whose data fields store the input data for this layer
   * @param top
   *     the preshaped output blobs, whose data fields will store this layers'
   *     outputs
   * \return The total loss from the layer.
   *
   * The Forward wrapper calls the relevant device wrapper function
   * (Forward_cpu or Forward_gpu) to compute the top blob values given the
   * bottom blobs.  If the layer has any non-zero loss_weights, the wrapper
   * then computes and returns the loss.
   *
   * Your layer should implement Forward_cpu and (optionally) Forward_gpu.
  */
  inline Dtype Forward(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  /**
   * @brief Given the top blob error gradients, compute the bottom blob error
   *        gradients.
   *
   * @param top
   *     the output blobs, whose diff fields store the gradient of the error
   *     with respect to themselves
   * @param propagate_down
   *     a vector with equal length to bottom, with each index indicating
   *     whether to propagate the error gradients down to the bottom blob at
   *     the corresponding index
   * @param bottom
   *     the input blobs, whose diff fields will store the gradient of the error
   *     with respect to themselves after Backward is run
   *
   * The Backward wrapper calls the relevant device wrapper function
   * (Backward_cpu or Backward_gpu) to compute the bottom blob diffs given the
   * top blob diffs.
   *
   * Your layer should implement Backward_cpu and (optionally) Backward_gpu.
   */
  inline void Backward(const vector<Blob<Dtype>*>& top,
                       const vector<bool>& propagate_down,
                       const vector<Blob<Dtype>*>& bottom);

#ifdef USE_MLU
  /**
   * @brief Returns wheter FUSION mode is supported.
   */
  virtual inline bool mfus_supported() { return false; }
  /**
   * @brief Returns hardware computing time.
   */
  float get_event_time() {return event_time_; }
#endif  // USE_MLU

  /**
   * @brief Returns the vector of learnable parameter blobs.
   */
  vector<shared_ptr<Blob<Dtype> > >& blobs() { return blobs_; }

  /**
   * @brief Returns the layer parameter.
   */
  const LayerParameter& layer_param() const { return layer_param_; }
  LayerParameter* mutable_layer_param() { return &layer_param_; }

  /**
   * @brief Writes the layer parameter to a protocol buffer
   */
  virtual void ToProto(LayerParameter* param, bool write_diff = false);

  /**
   * @brief Returns the scalar loss associated with a top blob at a given index.
   */
  inline Dtype loss(const int top_index) const {
    return (loss_.size() > top_index) ? loss_[top_index] : Dtype(0);
  }

  /**
   * @brief Sets the loss associated with a top blob at a given index.
   */
  inline void set_loss(const int top_index, const Dtype value) {
    if (loss_.size() <= top_index) {
      loss_.resize(top_index + 1, Dtype(0));
    }
    loss_[top_index] = value;
  }

  /**
   * @brief Returns the layer type.
   */
  virtual inline const char* type() const { return ""; }

  /**
   * @brief Set the optimization level of layer.
   */
  virtual void set_optimization_level(int level) {}

  /**
   * @brief Set the int8 context of layer.
   */
  virtual void set_int8_context(bool int8_mode) {}

  /**
   * @brief Returns the exact number of bottom blobs required by the layer,
   *        or -1 if no exact number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some exact number of bottom blobs.
   */
  virtual inline int ExactNumBottomBlobs() const { return -1; }
  /**
   * @brief Returns the minimum number of bottom blobs required by the layer,
   *        or -1 if no minimum number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some minimum number of bottom blobs.
   */
  virtual inline int MinBottomBlobs() const { return -1; }
  /**
   * @brief Returns the maximum number of bottom blobs required by the layer,
   *        or -1 if no maximum number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some maximum number of bottom blobs.
   */
  virtual inline int MaxBottomBlobs() const { return -1; }
  /**
   * @brief Returns the exact number of top blobs required by the layer,
   *        or -1 if no exact number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some exact number of top blobs.
   */
  virtual inline int ExactNumTopBlobs() const { return -1; }
  /**
   * @brief Returns the minimum number of top blobs required by the layer,
   *        or -1 if no minimum number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some minimum number of top blobs.
   */
  virtual inline int MinTopBlobs() const { return -1; }
  /**
   * @brief Returns the maximum number of top blobs required by the layer,
   *        or -1 if no maximum number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some maximum number of top blobs.
   */
  virtual inline int MaxTopBlobs() const { return -1; }
  /**
   * @brief Returns true if the layer requires an equal number of bottom and
   *        top blobs.
   *
   * This method should be overridden to return true if your layer expects an
   * equal number of bottom and top blobs.
   */
  virtual inline bool EqualNumBottomTopBlobs() const { return false; }

  /**
   * @brief Return whether "anonymous" top blobs are created automatically
   *        by the layer.
   *
   * If this method returns true, Net::Init will create enough "anonymous" top
   * blobs to fulfill the requirement specified by ExactNumTopBlobs() or
   * MinTopBlobs().
   */
  virtual inline bool AutoTopBlobs() const { return false; }

  /**
   * @brief Return whether to allow force_backward for a given bottom blob
   *        index.
   *
   * If AllowForceBackward(i) == false, we will ignore the force_backward
   * setting and backpropagate to blob i only if it needs gradient information
   * (as is done when force_backward == false).
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

  /**
   * @brief Specifies whether the layer should compute gradients w.r.t. a
   *        parameter at a particular index given by param_id.
   *
   * You can safely ignore false values and always compute gradients
   * for all parameters, but possibly with wasteful computation.
   */
  inline bool param_propagate_down(const int param_id) {
    return (param_propagate_down_.size() > param_id)
               ? param_propagate_down_[param_id]
               : false;
  }
  /**
   * @brief Sets whether the layer should compute gradients w.r.t. a
   *        parameter at a particular index given by param_id.
   */
  inline void set_param_propagate_down(const int param_id, const bool value) {
    if (param_propagate_down_.size() <= param_id) {
      param_propagate_down_.resize(param_id + 1, true);
    }
    param_propagate_down_[param_id] = value;
  }

#ifdef USE_MLU
  inline bool externalOutput() { return external_output_; }
  inline void addExternalOutput(MFusion<Dtype>* fuser,
                                const vector<Blob<Dtype>*>& top) {
    for (int top_id = 0; top_id < top.size(); top_id++)
      fuser->addOutput(top[top_id]);
  }
#endif

  protected:
  /** The protobuf that stores the layer parameters */
  LayerParameter layer_param_;
  /** The phase: TRAIN or TEST */
  Phase phase_;
  /** The vector that stores the learnable parameters as a set of blobs. */
  vector<shared_ptr<Blob<Dtype> > > blobs_;
  /** Vector indicating whether to compute the diff of each param blob. */
  vector<bool> param_propagate_down_;

  /** The vector that indicates whether each top blob has a non-zero weight in
   *  the objective function. */
  vector<Dtype> loss_;
  /* generate output or not in offline cambricon models. */
  bool external_output_;
  /** Hardware computing time **/
  float event_time_;

#ifdef USE_MLU
  virtual void MLUDestroyOp() {}
  virtual void MLUCreateOpBindData(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top) {}
  virtual void MLUCompileOp() {}
#endif

  /** @brief Using the CPU device, compute the layer output. */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top) = 0;
  /**
   * @brief Using the GPU device, compute the layer output.
   *        Fall back to Forward_cpu() if unavailable.
   */
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top) {
    // LOG(WARNING) << "Using CPU code as backup.";
    return Forward_cpu(bottom, top);
  }

#ifdef USE_MLU
  /**
   * @brief Using the MLU device, compute the layer output.
   *        Fall back to Forward_cpu() if unavailable.
   */
  virtual void Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top) {
    // LOG(WARNING) << "Using CPU code as backup.";
    return Forward_cpu(bottom, top);
  }

  /**
   * @brief Using the MLU device, compute the layer output in Fusion mode.
   */
  virtual void Forward_mfus(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top) {
    if (!boost::iequals(type(), "Input"))
      LOG(WARNING) << __func__ << ": Using Forward_mlu() for " << type()
                 << " as no Forward_mfus().";
    CHECK(!mfus_supported());
    return Forward_mlu(bottom, top);
  }
#endif

  /**
   * @brief Using the CPU device, compute the gradients for any parameters and
   *        for the bottom blobs if propagate_down is true.
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom) = 0;
  /**
   * @brief Using the GPU device, compute the gradients for any parameters and
   *        for the bottom blobs if propagate_down is true.
   *        Fall back to Backward_cpu() if unavailable.
   */
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom) {
    // LOG(WARNING) << "Using CPU code as backup.";
    Backward_cpu(top, propagate_down, bottom);
  }

#ifdef USE_MLU
  /**
   * @brief Using the MLU device, compute the gradients for any parameters and
   *        for the bottom blobs if propagate_down is true.
   *        Fall back to Backward_cpu() if unavailable.
   */
  virtual void Backward_mlu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED << " Backward not supported yet!";
  }

  /**
   * @brief Using the MLU device, compute the gradients for any parameters and
   *        for the bottom blobs if propagate_down is true in Fusion mode.
   *        Fall back to Backward_cpu() if unavailable.
   */
  virtual void Backward_mfus(const vector<Blob<Dtype>*>& top,
                             const vector<bool>& propagate_down,
                             const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED << " Backward not supported yet!";
  }
#endif

  /**
   * Called by the parent Layer's SetUp to check that the number of bottom
   * and top Blobs provided as input match the expected numbers specified by
   * the {ExactNum,Min,Max}{Bottom,Top}Blobs() functions.
   */
  virtual void CheckBlobCounts(const vector<Blob<Dtype>*>& bottom,
                               const vector<Blob<Dtype>*>& top) {
    if (ExactNumBottomBlobs() >= 0) {
      CHECK_EQ(ExactNumBottomBlobs(), bottom.size())
          << type() << " Layer takes " << ExactNumBottomBlobs()
          << " bottom blob(s) as input.";
    }
    if (MinBottomBlobs() >= 0) {
      CHECK_LE(MinBottomBlobs(), bottom.size())
          << type() << " Layer takes at least " << MinBottomBlobs()
          << " bottom blob(s) as input.";
    }
    if (MaxBottomBlobs() >= 0) {
      CHECK_GE(MaxBottomBlobs(), bottom.size())
          << type() << " Layer takes at most " << MaxBottomBlobs()
          << " bottom blob(s) as input.";
    }
    if (ExactNumTopBlobs() >= 0) {
      CHECK_EQ(ExactNumTopBlobs(), top.size()) << type() << " Layer produces "
                                               << ExactNumTopBlobs()
                                               << " top blob(s) as output.";
    }
    if (MinTopBlobs() >= 0) {
      CHECK_LE(MinTopBlobs(), top.size())
          << type() << " Layer produces at least " << MinTopBlobs()
          << " top blob(s) as output.";
    }
    if (MaxTopBlobs() >= 0) {
      CHECK_GE(MaxTopBlobs(), top.size())
          << type() << " Layer produces at most " << MaxTopBlobs()
          << " top blob(s) as output.";
    }
    if (EqualNumBottomTopBlobs()) {
      CHECK_EQ(bottom.size(), top.size())
          << type() << " Layer produces one top blob as output for each "
          << "bottom blob input.";
    }
  }

  /**
   * Called by SetUp to initialize the weights associated with any top blobs in
   * the loss function. Store non-zero loss weights in the diff blob.
   */
  inline void SetLossWeights(const vector<Blob<Dtype>*>& top) {
    const int num_loss_weights = layer_param_.loss_weight_size();
    if (num_loss_weights) {
      CHECK_EQ(top.size(), num_loss_weights)
          << "loss_weight must be "
             "unspecified or specified once per top blob.";
      for (int top_id = 0; top_id < top.size(); ++top_id) {
        const Dtype loss_weight = layer_param_.loss_weight(top_id);
        if (loss_weight == Dtype(0)) {
          continue;
        }
        this->set_loss(top_id, loss_weight);
        const int count = top[top_id]->count();
        Dtype* loss_multiplier = top[top_id]->mutable_cpu_diff();
        caffe_set(count, loss_weight, loss_multiplier);
      }
    }
  }

  private:
  DISABLE_COPY_AND_ASSIGN(Layer);
};  // class Layer

#ifdef USE_MLU
/**
 * @brief Dispatch reshape task to different reshape routines as
 *        MLU has specific job for different mode.
 *
 * This is similar with Forward();
 */
template <typename Dtype>
inline void Layer<Dtype>::Reshape_dispatch(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {
  switch (Caffe::mode()) {
    case Caffe::GPU:
    case Caffe::CPU:
      Reshape(bottom, top);
      break;
    case Caffe::MLU:
      if (layer_param_.engine() == caffe::Engine::CAFFE)
        Reshape(bottom, top);
      else
        Reshape_mlu(bottom, top);
      break;
    case Caffe::MFUS:
      Reshape_mfus(bottom, top);
      break;
    default:
      LOG(FATAL) << "Unknown caffe mode.";
  }
}
#endif

// Forward and backward wrappers. You should implement the cpu and
// gpu specific implementations instead, and should not change these
// functions.
template <typename Dtype>
inline Dtype Layer<Dtype>::Forward(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top) {
  Dtype loss = 0;
  switch (Caffe::mode()) {
  case Caffe::CPU:
    Reshape(bottom, top);
    Forward_cpu(bottom, top);
    for (int top_id = 0; top_id < top.size(); ++top_id) {
      if (!this->loss(top_id)) { continue; }
      const int count = top[top_id]->count();
      const Dtype* data = top[top_id]->cpu_data();
      const Dtype* loss_weights = top[top_id]->cpu_diff();
      loss += caffe_cpu_dot(count, data, loss_weights);
    }
    break;
  case Caffe::GPU:
    Reshape(bottom, top);
    Forward_gpu(bottom, top);
#ifdef USE_CUDA
    for (int top_id = 0; top_id < top.size(); ++top_id) {
      if (!this->loss(top_id)) { continue; }
      const int count = top[top_id]->count();
      const Dtype* data = top[top_id]->gpu_data();
      const Dtype* loss_weights = top[top_id]->gpu_diff();
      Dtype blob_loss = 0;
      caffe_gpu_dot(count, data, loss_weights, &blob_loss);
      loss += blob_loss;
    }
#endif
      break;
    case Caffe::MLU:
#ifdef USE_MLU
      event_time_ = 0;
      if (layer_param_.engine() == caffe::Engine::CAFFE) {
        if (Caffe::reshapeMode() == Caffe::ReshapeMode::ALWAYS) {
          Reshape(bottom, top);
        }
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        Forward_cpu(bottom, top);
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        std::chrono::duration<double> time_span =
                 std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        event_time_ = time_span.count() * 1e6;
        LOG(INFO) << "layer " << layer_param_.name()
                  << " cpu time: " << event_time_ << " us";
      } else {
        if (Caffe::reshapeMode() == Caffe::ReshapeMode::ALWAYS) {
          Reshape_mlu(bottom, top);
        }
        // create start_event and end_event
        cnrtNotifier_t notifierBeginning, notifierEnd;
        cnrtCreateNotifier(&notifierBeginning);
        cnrtCreateNotifier(&notifierEnd);
        cnrtPlaceNotifier(notifierBeginning, Caffe::queue());
        Forward_mlu(bottom, top);
        cnrtPlaceNotifier(notifierEnd, Caffe::queue());
        CNRT_CHECK(cnrtSyncQueue(Caffe::queue()));
        cnrtNotifierDuration(notifierBeginning, notifierEnd, &event_time_);
        LOG(INFO) << "layer " << layer_param_.name()
                  << " hardware time: " << event_time_ << " us";
        cnrtDestroyNotifier(&notifierBeginning);
        cnrtDestroyNotifier(&notifierEnd);
      }
      for (int top_id = 0; top_id < top.size(); ++top_id) {
        if (!this->loss(top_id)) {
          continue;
        }
        const int count = top[top_id]->count();
        const Dtype* data = top[top_id]->cpu_data();
        const Dtype* loss_weights = top[top_id]->cpu_diff();
        loss += caffe_cpu_dot(count, data, loss_weights);
      }
#endif
      break;
    case Caffe::MFUS:
#ifdef USE_MLU
      if (layer_param_.engine() == caffe::Engine::CAFFE) {
        Forward_cpu(bottom, top);
      } else {
        Forward_mfus(bottom, top);
        CNRT_CHECK(cnrtSyncQueue(Caffe::queue()));
      }
#endif
      break;
    default:
      LOG(FATAL) << "Unknown caffe mode.";
  }
  return loss;
}


// backward is not supported on MLU
template <typename Dtype>
inline void Layer<Dtype>::Backward(const vector<Blob<Dtype>*>& top,
                                   const vector<bool>& propagate_down,
                                   const vector<Blob<Dtype>*>& bottom) {
  switch (Caffe::mode()) {
    case Caffe::CPU:
      Backward_cpu(top, propagate_down, bottom);
      break;
    case Caffe::GPU:
      Backward_gpu(top, propagate_down, bottom);
      break;
    case Caffe::MLU:
#ifdef USE_MLU
      Backward_mlu(top, propagate_down, bottom);
#endif
      break;
    case Caffe::MFUS:
#ifdef USE_MLU
      Backward_mfus(top, propagate_down, bottom);
#endif
      break;
    default:
      LOG(FATAL) << "Unknown caffe mode.";
  }
}

// Serialize LayerParameter to protocol buffer
template <typename Dtype>
void Layer<Dtype>::ToProto(LayerParameter* param, bool write_diff) {
  param->Clear();
  param->CopyFrom(layer_param_);
  param->clear_blobs();
#ifdef USE_MLU
  BlobDataType blob_dtype;
  float sparsity = 0.0;
  if (layer_param_.has_sparsity()) {
    sparsity = layer_param_.sparsity();
    for (int i = 0; i < blobs_.size(); ++i) {
      blobs_[i]->ToProto(param->add_blobs(), write_diff, sparsity);
    }
  } else {
    for (int i = 0; i < blobs_.size(); ++i) {
      blobs_[i]->ToProto(param->add_blobs(), write_diff);
    }
  }
#else
  for (int i = 0; i < blobs_.size(); ++i) {
    blobs_[i]->ToProto(param->add_blobs(), write_diff);
  }
#endif
}

}  // namespace caffe

#endif  // INCLUDE_CAFFE_LAYER_HPP_
