#ifndef INCLUDE_CAFFE_LAYERS_MLU_CRELU_LAYER_HPP_
#define INCLUDE_CAFFE_LAYERS_MLU_CRELU_LAYER_HPP_

#include <vector>

#include "caffe/layers/crelu_layer.hpp"

namespace caffe {

#ifdef USE_MLU

/**
 *  @brief CNML implementation of CReLULayer.
 *
 * CReLU(x) = [ ReLU(x), ReLU(-x)]
 */
template <typename Dtype>
class MLUCReLULayer : public CReLULayer<Dtype> {
  public:
  explicit MLUCReLULayer(const LayerParameter& param)
    : CReLULayer<Dtype>(param), minus_op_ptr_(nullptr),
      concat_op_ptr_(nullptr), prelu_op_ptr_(nullptr),
      concat_param_ptr_(nullptr) {}
  virtual inline bool mfus_supported() { return true; }
  virtual void fuse(MFusion<Dtype>* fuser) {
    fuser->fuse(minus_op_ptr_);
    fuser->fuse(concat_op_ptr_);
    fuser->fuse(prelu_op_ptr_);
  }
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                              const vector<Blob<Dtype>*>& top);
  virtual ~MLUCReLULayer();

  protected:
  virtual void MLUDestroyOp();
  virtual void MLUCreateOpBindData(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top);
  virtual void MLUCompileOp() {
    const unsigned int p = Caffe::model_parallel();  //  avoid the line wrap
    MLU_CHECK(cnmlCompileBaseOp(minus_op_ptr_, Caffe::rt_core(), p));
    MLU_CHECK(cnmlCompileBaseOp(concat_op_ptr_, Caffe::rt_core(), p));
    MLU_CHECK(cnmlCompileBaseOp(prelu_op_ptr_, Caffe::rt_core(), p));
  }
  virtual void Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);

  cnmlBaseOp_t minus_op_ptr_;
  cnmlBaseOp_t concat_op_ptr_;
  cnmlBaseOp_t prelu_op_ptr_;
  cnmlConcatOpParam_t concat_param_ptr_;

  Blob<Dtype> negative_input_;
  Blob<Dtype> concated_data_;
  // negative slope for relu
  Blob<Dtype> negative_slope_b_;
};
#endif

}  // namespace caffe
#endif  // INCLUDE_CAFFE_LAYERS_MLU_CRELU_LAYER_HPP_
