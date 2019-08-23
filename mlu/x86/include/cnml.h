/*************************************************************************
 * Copyright (C) [2019] by Cambricon, Inc.
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *************************************************************************/

/************************************************************************
 *
 *  @file cnml.h
 *
 *  @brief cnml APIs provide programmable interfaces for users to develop
 *  their-owned programs, which includes operation generation, tensor
 *  management, neural network construction and inference on both sides
 *  (devices and hosts), etc.
 *
 **************************************************************************/

#ifndef CNML_H_
#define CNML_H_

#include "cnrt.h"
#if defined(__cplusplus)
extern "C" {
#endif

#if defined(WIN32) || defined(WINDOWS)
#include <time.h>

#ifdef USE_CNML_DLL
#ifdef CNML_DLL_EXPORTS
#define CNML_DLL_API __declspec(dllexport)
#else /* CNML_DLL_EXPORTS */
#define CNML_DLL_API __declspec(dllimport)
#endif /* CNML_DLL_EXPORTS */
#else
#define CNML_DLL_API
#endif /* USE_CNML_DLL */

#else
#define CNML_DLL_API
#endif

typedef enum {
  CNML_1H8 = 1,
  CNML_1H16 = 2,
  CNML_C10 = 3,
  CNML_MLU100 = CNML_C10
} cnmlCoreVersion_t;

typedef enum {
  CNML_NCHW = 0,
  CNML_NCWH = 1,
  CNML_NHCW = 2,
  CNML_NHWC = 3,
  CNML_NWHC = 4,
  CNML_NWCH = 5,
  CNML_CNHW = 6,
  CNML_CNWH = 7,
  CNML_CHNW = 8,
  CNML_CHWN = 9,
  CNML_CWHN = 10,
  CNML_CWNH = 11,
  CNML_HCNW = 12,
  CNML_HCWN = 13,
  CNML_HNCW = 14,
  CNML_HNWC = 15,
  CNML_HWNC = 16,
  CNML_HWCN = 17,
  CNML_WCHN = 18,
  CNML_WCNH = 19,
  CNML_WHCN = 20,
  CNML_WHNC = 21,
  CNML_WNHC = 22,
  CNML_WNCH = 23
} cnmlDataOrder_t;

typedef enum {
  CNML_DATA_INVALID = 0,
  CNML_DATA_FLOAT16 = 1,
  CNML_DATA_FLOAT32 = 2,
  CNML_DATA_DOUBLE = 3,
  CNML_DATA_FIX8 = 4,
  CNML_DATA_INT8 = 4,
  CNML_DATA_INT16 = 6,
  CNML_DATA_INT32 = 7,
  CNML_DATA_UINT8 = 8,
  CNML_DATA_UINT16 = 9,
  CNML_DATA_UINT32 = 10,
  CNML_DATA_QUANT8 = 11,
  CNML_DATA_BINARY = 12,
  CNML_DATA_BOOL = 13
} cnmlDataType_t;

typedef enum {
  CNML_CPU_MLU_BALANCE = 0,
  CNML_CPU_PRIORITY = 1,
  CNML_MLU_PRIORITY = 2,
  CNML_NO_PREPROCESS = 3
} cnmlDataPreprocessStrategy_t;

typedef enum {
  CNML_STATUS_NODEVICE = -1,
  CNML_STATUS_SUCCESS = 0,
  CNML_STATUS_DOMAINERR = 1,
  CNML_STATUS_INVALIDARG = 2,
  CNML_STATUS_LENGTHERR = 3,
  CNML_STATUS_OUTOFRANGE = 4,
  CNML_STATUS_RANGEERR = 5,
  CNML_STATUS_OVERFLOWERR = 6,
  CNML_STATUS_UNDERFLOWERR = 7,
  CNML_STATUS_INVALIDPARAM = 8,
  CNML_STATUS_BADALLOC = 9,
  CNML_STATUS_BADTYPEID = 10,
  CNML_STATUS_BADCAST = 11,
  CNML_STATUS_UNSUPPORT = 12
} cnmlStatus_t;

/**
 * @brief An enum.
 *  It is an enumerated type passed to ``cnmlCreatePoolOpParam`` to select
 *  the pooling value-selecting method to be used by ``cnmlCreatePoolOp``.
 */
typedef enum {
  CNML_POOL_AVG = 0,
  /* The average value inside the pooling window is used. */
  CNML_POOL_MAX = 1,
  /* The maximum value inside the pooling window is used. */
  CNML_POOL_MAXINDEX = 2
} cnmlPoolMode_t;

/**
 * @brief An enum.
 *  It is an enumerated type passed to ``cnmlCreatePoolOpParam`` to select
 *  pooling strategy method to be used by ``cnmlCreatePoolOp``, which may
 *  cause different output size.
 */
typedef enum {
  CNML_POOL_KFULL = 0,
  /* The pooing window can be out of bounds. */
  CNML_POOL_KVALID = 1
  /* The pooling window must be within the bounds. */
} cnmlPoolStrategyMode_t;

typedef enum {
  CNML_UNPOOL = 0,
  CNML_ROWWISE_UNPOOL = 1,
  CNML_MAXPOOLBP = 2,
  CNML_AVGPOOLBP = 3,
  CNML_MIDUNPOOL = 4,
  CNML_DIV = 5,
  CNML_REP = 6
} cnmlUnpoolMode_t;

typedef enum {
  CNML_NoSparse = 0,
  CNML_Sparse = 1
} cnmlSparseMode_t;

typedef enum {
  CNML_SOFTMAX_DIM_C = 1,
  CNML_SOFTMAX_DIM_W = 2,
  CNML_SOFTMAX_DIM_H = 3,
  CNML_SOFTMAX_DIM_N = 4
} cnmlSoftmaxDim_t;

typedef enum {
  CNML_ACTIVE_NONE = 0,
  CNML_ACTIVE_SIGMOID = 1,
  CNML_ACTIVE_RELU = 2,
  CNML_ACTIVE_TANH = 3,
  CNML_ACTIVE_RELU1 = 4,
  CNML_ACTIVE_RELU6 = 5
} cnmlActiveFunction_t;

typedef enum {
  CNML_REDUCEMAX_BATCH = 0,
  CNML_REDUCEMAX_FEAT = 1,
  CNML_REDUCEMAX_HIGHT = 2,
  CNML_REDUCEMAX_WIDTH = 3
} cnmlReduceMaxMode_t;

typedef enum {
  CNML_DIM_N = 0,
  CNML_DIM_C = 1,
  CNML_DIM_H = 2,
  CNML_DIM_W = 3
} cnmlDimension_t;

typedef enum {
  CNML_CONCAT_FEAT = 0,
  CNML_CONCAT_BATCH = 1,
  CNML_CONCAT_HIGHT = 2,
  CNML_CONCAT_WIDTH = 3
} cnmlConcatMode_t;

typedef enum {
  CNML_SPLIT_FEAT = 0,
  CNML_SPLIT_BATCH = 1,
  CNML_SPLIT_HIGHT = 2,
  CNML_SPLIT_WIDTH = 3
} cnmlSplitMode_t;

typedef enum {
  CNML_TENSOR = 0,
  CNML_FILTER = 1,
  CNML_CONST = 2
} cnmlTensorType_t;

typedef enum {
  CNML_CAST_FLOAT32_TO_UINT8 = 0,
  CNML_CAST_UINT8_TO_FLOAT32 = 1,
  CNML_CAST_INT8_TO_FLOAT16 = 3,
  CNML_CAST_FIX8_TO_FLOAT16 = 3,
  CNML_CAST_FLOAT16_TO_FLOAT32 = 4,
  CNML_CAST_FLOAT16_TO_FIX8 = 5,
  CNML_CAST_FLOAT16_TO_INT8 = 5,
  CNML_CAST_FLOAT32_TO_FLOAT16 = 6,
  CNML_CAST_INT16_TO_FLOAT16 = 7,
  CNML_CAST_FLOAT16_TO_INT16 = 8
} cnmlCastType_t;

typedef enum {
  CNML_ARGMAX_AXIS_N = 0,
  CNML_ARGMAX_AXIS_C = 1,
  CNML_ARGMAX_AXIS_H = 2,
  CNML_ARGMAX_AXIS_W = 3
} cnmlArgmaxAxis_t;

typedef enum {
  CNML_REVERSE_AXIS_N,
  CNML_REVERSE_AXIS_C,
  CNML_REVERSE_AXIS_H,
  CNML_REVERSE_AXIS_W,
  CNML_REVERSE_AXIS_HW
} cnmlReverseAxis_t;

typedef enum {
  CNML_YUV420SP_NV12 = 0,
  CNML_YUV420SP_NV21 = 1
} cnmlYuvType_t;

typedef enum {
  CNML_RGB0 = 0,
  CNML_BGR0 = 1,
  CNML_ARGB = 2
} cnmlRgbType_t;

/**
 * @brief An enum about three different lrn algorithms.
 *  V1:Yi = Xi / [(alpha * sum(Xj^2) / m + k) ^ beta],
 *  m = min(local_size, 2*ci-1)
 *  V2:Yi = Xi / [(alpha * sum(Xj^2) + k) ^ beta]
 *  V3:Yi = Xi / [(alpha * sum(Xj^2) / local_size + k) ^ beta]
 */

typedef enum {
  CNML_LRN_V1,
  CNML_LRN_V2,
  CNML_LRN_V3
} cnmlLrnType_t;

//////////////////////// common /////////////////////////

CNML_DLL_API cnmlStatus_t cnmlInit(int flag);

CNML_DLL_API cnmlStatus_t cnmlExit();

CNML_DLL_API cnmlStatus_t cnmlGetVersion(unsigned int *ver);

// cnml tensor
struct cnmlTensor;
typedef struct cnmlTensor *cnmlTensor_t;

CNML_DLL_API cnmlStatus_t cnmlCreateTensor(cnmlTensor_t *tensor,
                                           cnmlTensorType_t tensor_type,
                                           cnmlDataType_t data_type,
                                           int n,
                                           int c,
                                           int h,
                                           int w);

CNML_DLL_API cnmlStatus_t cnmlDestroyTensor(cnmlTensor_t *tensor);

CNML_DLL_API cnmlStatus_t cnmlSetTensorDataType(cnmlTensor_t tensor, cnmlDataType_t dtype);

CNML_DLL_API cnmlDataType_t cnmlGetTensorDataType(cnmlTensor_t tensor);

CNML_DLL_API cnmlStatus_t cnmlSetTensorShape(cnmlTensor_t tensor, int dim_num, int dim_values[]);

CNML_DLL_API void cnmlGetTensorShape(cnmlTensor_t tensor, int *shape);

CNML_DLL_API size_t cnmlGetTensorSize(cnmlTensor_t tensor);

CNML_DLL_API void cnmlDumpTensorFromDevice(cnmlTensor_t tensor,
                                           const char *filename,
                                           int opt_level);

CNML_DLL_API cnmlStatus_t cnmlSetFix8Position(cnmlTensor_t tensor, int position);

CNML_DLL_API cnmlStatus_t cnmlGetFix8Position(cnmlTensor_t tensor, int *position);

CNML_DLL_API cnmlStatus_t cnmlSetFix8Scale(cnmlTensor_t tensor, float scale);

CNML_DLL_API cnmlStatus_t cnmlGetFix8Scale(cnmlTensor_t tensor, float *scale);

CNML_DLL_API cnmlStatus_t cnmlSetFix8PositionByChannel(cnmlTensor_t tensor,
                                                       int *positions,
                                                       int positions_size);

CNML_DLL_API cnmlStatus_t cnmlSetFix8ScaleByChannel(cnmlTensor_t tensor,
                                                    float *scales,
                                                    int scales_size);

CNML_DLL_API cnmlStatus_t cnmlSetQuantizedPosition(cnmlTensor_t tensor, int position);

CNML_DLL_API cnmlStatus_t cnmlGetQuantizedPosition(cnmlTensor_t tensor, int *position);

CNML_DLL_API cnmlStatus_t cnmlSetQuantizedScale(cnmlTensor_t tensor, float scale);

CNML_DLL_API cnmlStatus_t cnmlGetQuantizedScale(cnmlTensor_t tensor, float *scale);

CNML_DLL_API cnmlStatus_t cnmlSetQuantizedPositionByChannel(cnmlTensor_t tensor,
                                                            int *positions,
                                                            int positions_size);

CNML_DLL_API cnmlStatus_t cnmlSetQuantizedScaleByChannel(cnmlTensor_t tensor,
                                                         float *scales,
                                                         int scales_size);

CNML_DLL_API cnmlStatus_t cnmlSetQuant8Param(cnmlTensor_t tensor, float scale, float offset);

CNML_DLL_API cnmlStatus_t cnmlGetQuant8Param(cnmlTensor_t tensor, float *scale, float *offset);

CNML_DLL_API cnmlStatus_t cnmlEnableHardwareReshape(cnmlTensor_t tensor,
                                                    cnmlDataType_t cpu_type,
                                                    cnmlDataOrder_t cpu_order);

CNML_DLL_API void cnmlPrintTensor(cnmlTensor_t tensor, cnmlTensorType_t type);

CNML_DLL_API cnmlStatus_t cnmlSetWeightName(cnmlTensor_t tensor, char *name);

CNML_DLL_API cnmlStatus_t cnmlGetWeightName(cnmlTensor_t tensor, char *name);

CNML_DLL_API cnmlStatus_t cnmlUpdateWeight(cnmlTensor_t tensor, void *name);

// cpu tensor
struct cnmlCpuTensor;

typedef struct cnmlCpuTensor *cnmlCpuTensor_t;

CNML_DLL_API cnmlStatus_t cnmlCreateCpuTensor(cnmlCpuTensor_t *cpu_tensor,
                                              cnmlTensorType_t tensor_type,
                                              cnmlDataType_t data_type,
                                              cnmlDataOrder_t data_order,
                                              int n,
                                              int c,
                                              int h,
                                              int w);

CNML_DLL_API cnmlStatus_t cnmlBindCpuDataInfo(cnmlTensor_t tensor, cnmlCpuTensor_t cpu_tensor);

CNML_DLL_API cnmlStatus_t cnmlSetDataPreprocessStrategy(cnmlTensor_t tensor,
                                                        cnmlCpuTensor_t twins_cpu_tensor,
                                                        cnmlDataPreprocessStrategy_t dps);

CNML_DLL_API cnmlStatus_t cnmlDestroyCpuTensor(cnmlCpuTensor_t *cpu_tensor);

CNML_DLL_API cnmlStatus_t cnmlSetCpuTensorShape(cnmlCpuTensor_t tensor,
                                                int dim_num,
                                                int dim_values[]);

CNML_DLL_API cnmlStatus_t cnmlSetCpuTensorDataType(cnmlCpuTensor_t tensor, cnmlDataType_t dtype);

CNML_DLL_API cnmlStatus_t cnmlSetCpuTensorDataOrder(cnmlCpuTensor_t tensor, cnmlDataOrder_t order);

CNML_DLL_API cnmlStatus_t cnmlSetCpuTensorFix8Position(cnmlCpuTensor_t tensor, int position);

CNML_DLL_API cnmlStatus_t cnmlSetCpuTensorPosition(cnmlCpuTensor_t tensor, int position);

CNML_DLL_API void cnmlPrintCpuTensor(cnmlCpuTensor_t tensor, cnmlTensorType_t type);

CNML_DLL_API void cnmlDumpTensor2File(const char *filename,
                                      cnmlCpuTensor_t tensor,
                                      cnmlTensorType_t type,
                                      void *output_addr,
                                      bool app);

CNML_DLL_API cnmlStatus_t cnmlLoadTensorFromFile(const char *filename,
                                                 cnmlCpuTensor_t tensor,
                                                 void *output_addr);

// cnml bind data
CNML_DLL_API cnmlStatus_t cnmlBindConstData(cnmlTensor_t tensor,
                                            cnmlCpuTensor_t cpu_tensor,
                                            void *cpu_tensor_ptr);

// mlu memory management
CNML_DLL_API void *cnmlMalloc(size_t size);

CNML_DLL_API void *cnmlMallocBatch(size_t size, int data_parallelism);

CNML_DLL_API void *cnmlMallocBuffer(cnmlTensor_t tensor);

CNML_DLL_API void *cnmlMallocBatchBuffer(cnmlTensor_t tensor, int data_parallelism);

CNML_DLL_API cnmlStatus_t cnmlFreeBuffer(void *ptr);

CNML_DLL_API cnmlStatus_t cnmlMemcpyTensorToDevice(cnmlCpuTensor_t cpu_tensor,
                                                   void *src,
                                                   cnmlTensor_t cnml_tensor,
                                                   void *dst);

CNML_DLL_API cnmlStatus_t cnmlMemcpyTensorToHost(cnmlTensor_t cnml_tensor,
                                                 void *src,
                                                 cnmlCpuTensor_t cpu_tensor,
                                                 void *dst);

CNML_DLL_API cnmlStatus_t cnmlMemcpyBatchTensorToDevice(cnmlCpuTensor_t cpu_tensor,
                                                        void *src,
                                                        cnmlTensor_t cnml_tensor,
                                                        void *dst,
                                                        int data_parallelism);

CNML_DLL_API cnmlStatus_t cnmlMemcpyBatchTensorToHost(cnmlTensor_t cnml_tensor,
                                                      void *src,
                                                      cnmlCpuTensor_t cpu_tensor,
                                                      void *dst,
                                                      int data_parallelism);

////////////////////////// operation /////////////////////////
/* base operation start */
struct cnmlBaseOp;
typedef struct cnmlBaseOp *cnmlBaseOp_t;

CNML_DLL_API cnmlStatus_t cnmlGetMaxMemUsed(cnmlBaseOp_t op,
                                            int64_t *totalmem,
                                            int64_t *sharemem,
                                            int64_t *privatemem);

CNML_DLL_API cnmlStatus_t cnmlGetIOCount(cnmlBaseOp_t op, int64_t *iocount);

CNML_DLL_API cnmlStatus_t cnmlCheckBaseOpRunnable(cnmlBaseOp_t op, cnmlCoreVersion_t version);

CNML_DLL_API cnmlStatus_t cnmlCompileBaseOp(cnmlBaseOp_t op,
                                            cnmlCoreVersion_t version,
                                            int model_parallelism);

CNML_DLL_API cnmlStatus_t cnmlDestroyBaseOp(cnmlBaseOp_t *op);

CNML_DLL_API cnmlStatus_t cnmlSetFix8ThreadContext(bool fix8_mode);

CNML_DLL_API cnmlStatus_t cnmlSetQuantizedThreadContext(bool fix8_mode);

CNML_DLL_API cnmlStatus_t cnmlGetBaseOpRequiredStackSize(cnmlBaseOp_t op, int64_t *size);
/* base operation end */

/* fuse op start */
struct cnmlFusionOp;
typedef struct cnmlFusionOp *cnmlFusionOp_t;

CNML_DLL_API cnmlStatus_t cnmlGetFusionMaxMemUsed(cnmlFusionOp_t op,
                                                  int64_t *totalmem,
                                                  int64_t *sharemem,
                                                  int64_t *privatemem);

CNML_DLL_API cnmlStatus_t cnmlGetFusionIOCount(cnmlFusionOp_t op, int64_t *iocount);

CNML_DLL_API cnmlStatus_t cnmlFuseOp(cnmlBaseOp_t op, cnmlFusionOp_t fusion_op);

CNML_DLL_API cnmlStatus_t cnmlCompileFusionOp(cnmlFusionOp_t op,
                                              cnmlCoreVersion_t version,
                                              int model_parallelism);

CNML_DLL_API cnmlStatus_t cnmlComputeFusionOpForward_V3(cnmlFusionOp_t op,
                                                        void *inputs[],
                                                        int input_num,
                                                        void *outputs[],
                                                        int output_num,
                                                        cnrtInvokeFuncParam_t *compute_forw_param,
                                                        cnrtQueue_t queue);

CNML_DLL_API cnmlStatus_t cnmlInitFusionOpInstMemory(cnmlFusionOp_t op,
                                                     cnrtInvokeFuncParam_t *compute_forw_param,
                                                     void **inst_addr);

CNML_DLL_API cnmlStatus_t cnmlInitFusionOpConstMemory(cnmlFusionOp_t op,
                                                      cnrtInvokeFuncParam_t *compute_forw_param,
                                                      void **const_addr);

CNML_DLL_API cnmlStatus_t cnmlInitFusionOpIntmdMemory(cnmlFusionOp_t op,
                                                      cnrtInvokeFuncParam_t *compute_forw_param,
                                                      void **intmd_addr);

CNML_DLL_API cnmlStatus_t
cnmlComputeFusionOpForwardExtra_V2(cnmlFusionOp_t op,
                                   void *inputs[],
                                   int input_num,
                                   void *outputs[],
                                   int output_num,
                                   cnrtInvokeFuncParam_t *compute_forw_param,
                                   cnrtQueue_t queue,
                                   void *inst_addr,
                                   void *const_addr,
                                   void *intmd_addr);

CNML_DLL_API cnmlStatus_t cnmlEnableFusionOpInstConstSplit(cnmlFusionOp_t op,
                                                           bool is_support_inst_const_split);
CNML_DLL_API cnmlStatus_t cnmlCreateFusionOp(cnmlFusionOp_t *op);

CNML_DLL_API cnmlStatus_t cnmlDestroyFusionOp(cnmlFusionOp_t *op);

CNML_DLL_API cnmlStatus_t cnmlSetFusionIO(cnmlFusionOp_t op,
                                          cnmlTensor_t *inputs,
                                          int input_num,
                                          cnmlTensor_t *outputs,
                                          int output_num);

CNML_DLL_API cnmlStatus_t cnmlAddFusionInput(cnmlFusionOp_t op, cnmlTensor_t input);

CNML_DLL_API cnmlStatus_t cnmlAddFusionOutput(cnmlFusionOp_t op, cnmlTensor_t output);

CNML_DLL_API cnmlStatus_t cnmlGetFusionOpRequiredStackSize(cnmlFusionOp_t op, int64_t *size);
/* fusion op end */

/* basic rnn operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateBasicRNNOp(cnmlBaseOp_t *op,
                                               cnmlTensor_t input_tensor,
                                               cnmlTensor_t output_tensor,
                                               cnmlTensor_t weight_tensor,
                                               cnmlTensor_t bias_tensor,
                                               cnmlTensor_t state_input_tensor,
                                               cnmlTensor_t state_output_tensor,
                                               cnmlTensor_t state_weight_tensor,
                                               cnmlActiveFunction_t active_func);

CNML_DLL_API cnmlStatus_t cnmlComputeBasicRNNOpForward_V3(cnmlBaseOp_t op,
                                                          void *input,
                                                          void *state,
                                                          void *output,
                                                          void *state_output,
                                                          cnrtInvokeFuncParam_t *compute_forw_param,
                                                          cnrtQueue_t queue);
/* basic rnn operation end */

/* deconv operation start */
struct cnmlDeconvOpParam;
typedef cnmlDeconvOpParam *cnmlDeconvOpParam_t;

struct cnmlDeconvOptLevel;
typedef cnmlDeconvOptLevel *cnmlDeconvOptLevel_t;

CNML_DLL_API cnmlStatus_t cnmlCreateDeconvOpParam(cnmlDeconvOpParam_t *param,
                                                  int stride_height,
                                                  int stride_width,
                                                  int hu_crop,
                                                  int hd_crop,
                                                  int wl_crop,
                                                  int wr_crop);
CNML_DLL_API cnmlStatus_t cnmlDestroyDeconvOpParam(cnmlDeconvOpParam_t *param);

CNML_DLL_API cnmlStatus_t cnmlCreateDeconvOptLevel(cnmlDeconvOptLevel_t *optlevel, int opt);

CNML_DLL_API cnmlStatus_t cnmlDestroyDeconvOptLevel(cnmlDeconvOptLevel_t *optlevel);

CNML_DLL_API cnmlStatus_t cnmlCreateDeconvOp(cnmlBaseOp_t *op,
                                             cnmlDeconvOpParam_t param,
                                             cnmlTensor_t input_tensor,
                                             cnmlTensor_t output_tensor,
                                             cnmlTensor_t filter_tensor,
                                             cnmlTensor_t bias_tensor,
                                             cnmlDeconvOptLevel_t opt);

CNML_DLL_API cnmlStatus_t cnmlComputeDeConvOpForward_V3(cnmlBaseOp_t op,
                                                        void *input,
                                                        void *output,
                                                        cnrtInvokeFuncParam_t *compute_forw_param,
                                                        cnrtQueue_t queue);

CNML_DLL_API cnmlStatus_t cnmlEnableDeconvOpFix8Mode(cnmlBaseOp_t op);

CNML_DLL_API cnmlStatus_t cnmlEnableDeconvOpInt8Mode(cnmlBaseOp_t op);
/* deconv operation end */

/* conv operation start */
struct cnmlConvOpParam;
typedef cnmlConvOpParam *cnmlConvOpParam_t;

CNML_DLL_API cnmlStatus_t cnmlCreateConvOpParam(cnmlConvOpParam_t *param,
                                                int stride_height,
                                                int stride_width,
                                                int dilation_height,
                                                int dilation_width,
                                                int pad_height,
                                                int pad_width,
                                                cnmlSparseMode_t sparse);
CNML_DLL_API cnmlStatus_t cnmlDestroyConvOpParam(cnmlConvOpParam_t *param);

CNML_DLL_API cnmlStatus_t cnmlCreateConvOp(cnmlBaseOp_t *op,
                                           cnmlConvOpParam_t param,
                                           cnmlTensor_t input_tensor,
                                           cnmlTensor_t output_tensor,
                                           cnmlTensor_t filter_tensor,
                                           cnmlTensor_t bias_tensor);

CNML_DLL_API cnmlStatus_t cnmlComputeConvOpForward_V3(cnmlBaseOp_t op,
                                                      void *input,
                                                      void *output,
                                                      cnrtInvokeFuncParam_t *compute_forw_param,
                                                      cnrtQueue_t queue);

CNML_DLL_API cnmlStatus_t cnmlEnableConvOpFix8Mode(cnmlBaseOp_t op);

CNML_DLL_API cnmlStatus_t cnmlEnableConvOpInt8Mode(cnmlBaseOp_t op);
/* conv operation end */

/* conv first operation start */
struct cnmlConvFirstOpParam;
typedef cnmlConvFirstOpParam *cnmlConvFirstOpParam_t;

CNML_DLL_API cnmlStatus_t cnmlCreateConvFirstOpParam(cnmlConvFirstOpParam_t *param,
                                                     int stride_height,
                                                     int stride_width,
                                                     int pad_l,
                                                     int pad_r,
                                                     int pad_t,
                                                     int pad_b,
                                                     cnmlSparseMode_t sparse);

CNML_DLL_API cnmlStatus_t cnmlDestroyConvFirstOpParam(cnmlConvFirstOpParam_t *param);

CNML_DLL_API cnmlStatus_t cnmlCreateConvFirstOp(cnmlBaseOp_t *op,
                                                cnmlConvFirstOpParam_t param,
                                                cnmlTensor_t input_tensor,
                                                cnmlTensor_t mean_tensor,
                                                cnmlTensor_t output_tensor,
                                                cnmlTensor_t filter_tensor,
                                                cnmlTensor_t bias_tensor,
                                                cnmlTensor_t stdt_tensor);

CNML_DLL_API cnmlStatus_t
cnmlComputeConvFirstOpForward_V3(cnmlBaseOp_t op,
                                 void *input,
                                 void *output,
                                 cnrtInvokeFuncParam_t *compute_forw_param,
                                 cnrtQueue_t queue);

CNML_DLL_API cnmlStatus_t cnmlEnableConvFirstOpFix8Mode(cnmlBaseOp_t op);

CNML_DLL_API cnmlStatus_t cnmlEnableConvFirstOpInt8Mode(cnmlBaseOp_t op);

CNML_DLL_API cnmlStatus_t cnmlEnableConvFirstOpBgraMode(cnmlBaseOp_t op);

CNML_DLL_API cnmlStatus_t cnmlEnableConvFirstOpFusionPadMode(cnmlBaseOp_t op);
/* conv first operation end */

/* conv group operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateConvGroupOp(cnmlBaseOp_t *op,
                                                cnmlConvOpParam_t param,
                                                cnmlTensor_t input_tensor,
                                                cnmlTensor_t output_tensor,
                                                cnmlTensor_t filter_tensor,
                                                cnmlTensor_t bias_tensor,
                                                int group);

CNML_DLL_API cnmlStatus_t
cnmlComputeConvGroupOpForward_V3(cnmlBaseOp_t op,
                                 void *input,
                                 void *output,
                                 cnrtInvokeFuncParam_t *compute_forw_param,
                                 cnrtQueue_t queue);
/* conv group operation end */

/* conv_depthwise operation start */
struct cnmlConvDepthwiseOpParam;
typedef cnmlConvDepthwiseOpParam *cnmlConvDepthwiseOpParam_t;

CNML_DLL_API cnmlStatus_t cnmlCreateConvDepthwiseOpParam(cnmlConvDepthwiseOpParam_t *param,
                                                         int stride_height,
                                                         int stride_width);

CNML_DLL_API cnmlStatus_t cnmlCreateConvDepthwiseOpParam_V3(cnmlConvDepthwiseOpParam_t *param,
                                                            int stride_height,
                                                            int stride_width,
                                                            int pad_height,
                                                            int pad_width);

CNML_DLL_API cnmlStatus_t cnmlDestroyConvDepthwiseOpParam(cnmlConvDepthwiseOpParam_t *param);

CNML_DLL_API cnmlStatus_t cnmlCreateConvDepthwiseOp(cnmlBaseOp_t *op,
                                                    cnmlConvDepthwiseOpParam_t param,
                                                    cnmlTensor_t input_tensor,
                                                    cnmlTensor_t output_tensor,
                                                    cnmlTensor_t filter_tensor,
                                                    cnmlTensor_t bias_tensor);

CNML_DLL_API cnmlStatus_t
cnmlComputeConvDepthwiseOpForward_V3(cnmlBaseOp_t op,
                                     void *input,
                                     void *output,
                                     cnrtInvokeFuncParam_t *compute_forw_param,
                                     cnrtQueue_t queue);
/* conv_depthwise operation end */

/* add pad operation start */
struct cnmlAddPadOpParam;
typedef struct cnmlAddPadOpParam *cnmlAddPadOpParam_t;

CNML_DLL_API cnmlStatus_t cnmlCreateAddPadOpParam(cnmlAddPadOpParam_t *param,
                                                  int pad_h,
                                                  int pad_w,
                                                  float pad_value);

CNML_DLL_API cnmlStatus_t cnmlCreateAddPadOp4Param(cnmlAddPadOpParam_t *param,
                                                   int pad_htop,
                                                   int pad_hbottom,
                                                   int pad_wleft,
                                                   int pad_wright,
                                                   float pad_value);

CNML_DLL_API cnmlStatus_t cnmlDestroyAddPadOpParam(cnmlAddPadOpParam_t *param);

CNML_DLL_API cnmlStatus_t cnmlCreateAddPadOp(cnmlBaseOp_t *op,
                                             cnmlAddPadOpParam_t param,
                                             cnmlTensor_t input_tensor,
                                             cnmlTensor_t output_tensor);

CNML_DLL_API cnmlStatus_t cnmlComputeAddPadOpForward_V3(cnmlBaseOp_t op,
                                                        void *input,
                                                        void *output,
                                                        cnrtInvokeFuncParam_t *compute_forw_param,
                                                        cnrtQueue_t queue);
/* add pad operation end */

/* add pad channel operation start */
struct cnmlAddPadChannelOpParam;
typedef struct cnmlAddPadChannelOpParam *cnmlAddPadChannelOpParam_t;

CNML_DLL_API cnmlStatus_t cnmlCreateAddPadChannelOpParam(cnmlAddPadChannelOpParam_t *param,
                                                         int channel_,
                                                         float pad_value);

CNML_DLL_API cnmlStatus_t cnmlCreateAddPadChannelOp4Param(cnmlAddPadChannelOpParam_t *param,
                                                          int c_front_,
                                                          int c_back_,
                                                          float pad_value);

CNML_DLL_API cnmlStatus_t cnmlDestroyAddPadChannelOpParam(cnmlAddPadChannelOpParam_t *param);

CNML_DLL_API cnmlStatus_t cnmlCreateAddPadChannelOp(cnmlBaseOp_t *op,
                                                    cnmlAddPadChannelOpParam_t param,
                                                    cnmlTensor_t input_tensor,
                                                    cnmlTensor_t output_tensor);

CNML_DLL_API cnmlStatus_t
cnmlComputeAddPadChannelOpForward_V3(cnmlBaseOp_t op,
                                     void *input,
                                     void *output,
                                     cnrtInvokeFuncParam_t *compute_forw_param,
                                     cnrtQueue_t queue);
/* add pad channel operation end */

/* normalize operation start */
struct cnmlNormalizeOpParam;
typedef struct cnmlNormalizeOpParam *cnmlNormalizeOpParam_t;
CNML_DLL_API cnmlStatus_t cnmlCreateNormalizeOpParam(cnmlNormalizeOpParam_t *param,
                                                     int p,
                                                     int scale,
                                                     int across_spatial,
                                                     float weight);

CNML_DLL_API cnmlStatus_t cnmlDestroyNormalizeOpParam(cnmlNormalizeOpParam_t *param);

CNML_DLL_API cnmlStatus_t cnmlCreateNormalizeOp(cnmlBaseOp_t *op,
                                                cnmlNormalizeOpParam_t param,
                                                cnmlTensor_t input_tensor,
                                                cnmlTensor_t output_tensor,
                                                cnmlTensor_t scale_tensor);

CNML_DLL_API cnmlStatus_t
cnmlComputeNormalizeOpForward_V3(cnmlBaseOp_t op,
                                 void *input,
                                 void *output,
                                 void *scale,
                                 cnrtInvokeFuncParam_t *compute_forw_param,
                                 cnrtQueue_t queue);
/* normalize operation end */

/* grep channel operation start */
struct cnmlGrepChannelOpParam;
typedef struct cnmlGrepChannelOpParam *cnmlGrepChannelOpParam_t;

CNML_DLL_API cnmlStatus_t cnmlCreateGrepChannelOpParam(cnmlGrepChannelOpParam_t *param,
                                                       int channel_);

CNML_DLL_API cnmlStatus_t cnmlCreateGrepChannelOp2Param(cnmlGrepChannelOpParam_t *param,
                                                        int c_front_,
                                                        int c_back_);

CNML_DLL_API cnmlStatus_t cnmlDestroyGrepChannelOpParam(cnmlGrepChannelOpParam_t *param);

CNML_DLL_API cnmlStatus_t cnmlCreateGrepChannelOp(cnmlBaseOp_t *op,
                                                  cnmlGrepChannelOpParam_t param,
                                                  cnmlTensor_t input_tensor,
                                                  cnmlTensor_t output_tensor);

CNML_DLL_API cnmlStatus_t
cnmlComputeGrepChannelOpForward_V3(cnmlBaseOp_t op,
                                   void *input,
                                   void *output,
                                   cnrtInvokeFuncParam_t *compute_forw_param,
                                   cnrtQueue_t queue);
/* grep channel operation end */

/* pool operation start */
/*!
 *  @struct cnmlPoolOpParam
 *  @brief A struct.
 *
 *  ``cnmlPoolOpParam`` is a structure holding the description of
 *  a pooling operation param. */
struct cnmlPoolOpParam;
/*! ``cnmlPoolOpParam_t`` is a pointer to ``cnmlPoolOpParam`` which is a
    structure holding the description of a pooling operation param. */
typedef cnmlPoolOpParam *cnmlPoolOpParam_t;

/*!
 *  @brief A function.
 *
 *  This function creates a pooling param object by allocating the
 *  memory needed to hold its opaque structure.
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] window_height
 *    Input. Height of the pooling window.
 *  @param[in] window_width
 *    Input. Width of the pooling window.
 *  @param[in] stride_height
 *    Input. Pooling vertical stride.
 *  @param[in] stride_width
 *    Input. Pooling horizontal stride.
 *  @param[in] pad_height
 *    Input. Size of vertical padding.
 *  @param[in] pad_width
 *    Input. Size of horizontal padding.
 *  @param[in] dilation_height
 *    Input. Size of vertical dilation.
 *  @param[in] dilation_width
 *    Input. Size of horizontal dilation.
 *  @param[in] pool_mode Input.
 *    Enumerant to specify the pooling mode.
 *  @param[in] strategy_mode
 *    Input. Enumerant to specify the pooling strategy mode.
 *  @param[in] real
 *    Input. real
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions are met:
 *    - Reason1 TODO.
 *    - Reason2 TODO.
 */
CNML_DLL_API cnmlStatus_t cnmlCreatePoolOpParam(cnmlPoolOpParam_t *param,
                                                int window_height,
                                                int window_width,
                                                int stride_height,
                                                int stride_width,
                                                int pad_height,
                                                int pad_width,
                                                int dilation_height,
                                                int dilation_width,
                                                cnmlPoolMode_t pool_mode,
                                                cnmlPoolStrategyMode_t strategy_mode,
                                                bool real);

/*!
 *  @brief A function.
 *
 *  This function destroys a previously created pooling param
 *  descriptor object
 *
 *  @param[in] param
 *    Input. Pointer to the structure holding the description of the
 *    pooling param to be deleted.
 *  @retval CNML_STATUS_SUCCESS
 *    The param object was destroyed successfully
 */
CNML_DLL_API cnmlStatus_t cnmlDestroyPoolOpParam(cnmlPoolOpParam_t *param);

CNML_DLL_API cnmlStatus_t cnmlGetPoolOpOutputDim(cnmlTensor_t input,
                                                 cnmlPoolOpParam_t param,
                                                 int *no,
                                                 int *co,
                                                 int *ho,
                                                 int *wo);

/*!
 *  @brief A function.
 *
 *  This function creates a pooling op object by allocating the memory
 *  needed to hold its opaque structure.
 *
 *  @param[out] op
 *    Output. The returning op descriptor.
 *  @param[in] param
 *    Input. Param of this pooling op.
 *  @param[in] input
 *    Input. Input cnml tensor of this pooling op.
 *  @param[in] output
 *    Input. Input cnml tensor of this pooling op.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 *  @retval CNML_STATUS_BADALLOC
 *    At least one of the following conditions are met:
 *    - Reason1 TODO.
 *    - Reason2 TODO.
 */
CNML_DLL_API cnmlStatus_t cnmlCreatePoolOp(cnmlBaseOp_t *op,
                                           cnmlPoolOpParam_t param,
                                           cnmlTensor_t input,
                                           cnmlTensor_t output);

CNML_DLL_API cnmlStatus_t cnmlComputePoolOpForward_V3(cnmlBaseOp_t op,
                                                      void *input,
                                                      void *output,
                                                      cnrtInvokeFuncParam_t *compute_forw_param,
                                                      cnrtQueue_t queue);
/* pool operation end */

/* l2_pool operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateL2_PoolOpParam(cnmlPoolOpParam_t *param,
                                                   int window_height,
                                                   int window_width,
                                                   int stride_height,
                                                   int stride_width,
                                                   int pad_height,
                                                   int pad_width,
                                                   cnmlPoolMode_t pool_mode);

CNML_DLL_API cnmlStatus_t cnmlDestroyL2_PoolOpParam(cnmlPoolOpParam_t *param);

CNML_DLL_API cnmlStatus_t cnmlCreateL2_PoolOp(cnmlBaseOp_t *op,
                                              cnmlPoolOpParam_t param,
                                              cnmlTensor_t input,
                                              cnmlTensor_t output);

CNML_DLL_API cnmlStatus_t cnmlComputeL2_PoolOpForward_V3(cnmlBaseOp_t op,
                                                         void *input,
                                                         void *output,
                                                         cnrtInvokeFuncParam_t *compute_forw_param,
                                                         cnrtQueue_t queue);
/* l2_pool operation end */

/* active operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateActiveOp(cnmlBaseOp_t *op,
                                             cnmlActiveFunction_t function,
                                             cnmlTensor_t input,
                                             cnmlTensor_t output);

CNML_DLL_API cnmlStatus_t cnmlComputeActiveOpForward_V3(cnmlBaseOp_t op,
                                                        void *input,
                                                        void *output,
                                                        cnrtInvokeFuncParam_t *compute_forw_param,
                                                        cnrtQueue_t queue);
/* active operation end */

/* customized active operation start */
struct cnmlCustomizedActiveOpParam;
typedef cnmlCustomizedActiveOpParam *cnmlCustomizedActiveOpParam_t;

CNML_DLL_API cnmlStatus_t cnmlCreateCustomizedActiveOpParam(cnmlCustomizedActiveOpParam_t *param,
                                                            float x_start,
                                                            float x_end,
                                                            float y_min,
                                                            int segment_num);

CNML_DLL_API cnmlStatus_t cnmlDestroyCustomizedActiveOpParam(cnmlCustomizedActiveOpParam_t *param);

CNML_DLL_API cnmlStatus_t cnmlCreateCustomizedActiveOp(cnmlBaseOp_t *op,
                                                       void *active_func_ptr,
                                                       cnmlCustomizedActiveOpParam_t param,
                                                       cnmlTensor_t input,
                                                       cnmlTensor_t output);

CNML_DLL_API cnmlStatus_t
cnmlComputeCustomizedActiveForward_V3(cnmlBaseOp_t op,
                                      void *input,
                                      void *output,
                                      cnrtInvokeFuncParam_t *compute_forw_param,
                                      cnrtQueue_t queue);

/* customized active operation end */

/* device memcpy operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateDeviceMemcpyOp(cnmlBaseOp_t *op,
                                                   cnmlTensor_t input,
                                                   cnmlTensor_t output);

CNML_DLL_API cnmlStatus_t
cnmlComputeDeviceMemcpyOpForward_V3(cnmlBaseOp_t op,
                                    void *input,
                                    void *output,
                                    cnrtInvokeFuncParam_t *compute_forw_param,
                                    cnrtQueue_t queue);
/* active operation end */

/* transpose pro operation start */
struct cnmlTransposeOpParam;
typedef struct cnmlTransposeOpParam *cnmlTransposeOpParam_t;

CNML_DLL_API cnmlStatus_t cnmlCreateTransposeOpParam(cnmlTransposeOpParam_t *param,
                                                     cnmlDataOrder_t cpu_data_order,
                                                     int dim_id_0,
                                                     int dim_id_1,
                                                     int dim_id_2,
                                                     int dim_id_3);

CNML_DLL_API cnmlStatus_t cnmlDestroyTransposeOpParam(cnmlTransposeOpParam_t *param);

CNML_DLL_API cnmlStatus_t cnmlCreateTransposeProOp(cnmlBaseOp_t *op,
                                                   cnmlTensor_t input,
                                                   cnmlTensor_t output,
                                                   cnmlTransposeOpParam_t param);

CNML_DLL_API cnmlStatus_t
cnmlComputeTransposeProOpForward_V3(cnmlBaseOp_t op,
                                    void *input,
                                    void *output,
                                    cnrtInvokeFuncParam_t *compute_forw_param,
                                    cnrtQueue_t queue);
/* transpose pro operation end */

/* transpose operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateTransposeOp(cnmlBaseOp_t *op,
                                                cnmlTensor_t input,
                                                cnmlTensor_t output);

CNML_DLL_API cnmlStatus_t
cnmlComputeTransposeOpForward_V3(cnmlBaseOp_t op,
                                 void *input,
                                 void *output,
                                 cnrtInvokeFuncParam_t *compute_forw_param,
                                 cnrtQueue_t queue);
/* transpose operation end */

/* mlp operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateMlpOp(cnmlBaseOp_t *op,
                                          cnmlTensor_t input_tensor,
                                          cnmlTensor_t output_tensor,
                                          cnmlTensor_t filter_tensor,
                                          cnmlTensor_t bias_tensor,
                                          cnmlSparseMode_t sparse);

CNML_DLL_API cnmlStatus_t cnmlComputeMlpOpForward_V3(cnmlBaseOp_t op,
                                                     void *inputs,
                                                     void *outputs,
                                                     cnrtInvokeFuncParam_t *compute_forw_param,
                                                     cnrtQueue_t queue);

CNML_DLL_API cnmlStatus_t cnmlEnableMlpOpFix8Mode(cnmlBaseOp_t op);

CNML_DLL_API cnmlStatus_t cnmlEnableMlpOpInt8Mode(cnmlBaseOp_t op);

CNML_DLL_API cnmlStatus_t cnmlEnableMlpOpBinaryMode(cnmlBaseOp_t op);
/* mlp operation end */

/* matrix_mult operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateMatrixMultOp(cnmlBaseOp_t *op,
                                                 cnmlTensor_t lhs_tensor,
                                                 cnmlTensor_t rhs_tensor,
                                                 cnmlTensor_t output_tensor,
                                                 cnmlTensor_t bias_tensor);

CNML_DLL_API cnmlStatus_t
cnmlComputeMatrixMultOpForward_V3(cnmlBaseOp_t op,
                                  void *lhs,
                                  void *rhs,
                                  void *output,
                                  cnrtInvokeFuncParam_t *compute_forw_param,
                                  cnrtQueue_t queue);
/* matrix_mult operation end */

/* gatherv2 operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateGatherV2Op(cnmlBaseOp_t *op,
                                               cnmlTensor_t input_tensor,
                                               cnmlTensor_t indexTensor,
                                               cnmlTensor_t output_tensor,
                                               cnmlDimension_t axies);

CNML_DLL_API cnmlStatus_t cnmlComputeGatherV2OpForward_V3(cnmlBaseOp_t op,
                                                          void *input,
                                                          void *index,
                                                          void *output,
                                                          cnrtInvokeFuncParam_t *compute_forw_param,
                                                          cnrtQueue_t queue);
/* gatherv2 operation end */

/* batchdot operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateBatchDotOp(cnmlBaseOp_t *op,
                                               const cnmlTensor_t input_tensor_1,
                                               const cnmlTensor_t input_tensor_2,
                                               const cnmlTensor_t output_tensor,
                                               bool trans_a,
                                               bool trans_b);

CNML_DLL_API cnmlStatus_t cnmlComputeBatchDotOpForward_V3(cnmlBaseOp_t op,
                                                          void *input_1,
                                                          void *input_2,
                                                          void *output,
                                                          cnrtInvokeFuncParam_t *compute_forw_param,
                                                          cnrtQueue_t queue);
/* batchdot operation end*/

/* xw_plus_b operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateXwPlusBOp(cnmlBaseOp_t *op,
                                              cnmlTensor_t input_tensor,
                                              cnmlTensor_t output_tensor,
                                              cnmlTensor_t filter_tensor,
                                              cnmlTensor_t bias_tensor,
                                              cnmlSparseMode_t sparse);

CNML_DLL_API cnmlStatus_t cnmlComputeXwPlusBOpForward_V3(cnmlBaseOp_t op,
                                                         void *input,
                                                         void *output,
                                                         cnrtInvokeFuncParam_t *compute_forw_param,
                                                         cnrtQueue_t queue);

CNML_DLL_API cnmlStatus_t cnmlEnableXwPlusBOpFix8Mode(cnmlBaseOp_t op);

CNML_DLL_API cnmlStatus_t cnmlEnableXwPlusBOpInt8Mode(cnmlBaseOp_t op);
/* xw_plus_b operation end */

/* add operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateAddOp(cnmlBaseOp_t *op,
                                          cnmlTensor_t input_tensor_1,
                                          cnmlTensor_t input_tensor_2,
                                          cnmlTensor_t output_tensor);

CNML_DLL_API cnmlStatus_t cnmlComputeAddOpForward_V3(cnmlBaseOp_t op,
                                                     void *input_1,
                                                     void *input_2,
                                                     void *output,
                                                     cnrtInvokeFuncParam_t *compute_forw_param,
                                                     cnrtQueue_t queue);
/* add operation end */

/* coeff add operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateCoeffAddOp(cnmlBaseOp_t *op,
                                               float coeff1,
                                               float coeff2,
                                               cnmlTensor_t input_tensor_1,
                                               cnmlTensor_t input_tensor_2,
                                               cnmlTensor_t output_tensor);

CNML_DLL_API cnmlStatus_t cnmlComputeCoeffAddOpForward_V3(cnmlBaseOp_t op,
                                                          void *input_1,
                                                          void *input_2,
                                                          void *output,
                                                          cnrtInvokeFuncParam_t *compute_forw_param,
                                                          cnrtQueue_t queue);
/* coeff add operation end */

/* real div operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateRealDivOp(cnmlBaseOp_t *op,
                                              cnmlTensor_t input_tensor_1,
                                              cnmlTensor_t input_tensor_2,
                                              cnmlTensor_t output_tensor);

CNML_DLL_API cnmlStatus_t cnmlComputeRealDivOpForward_V3(cnmlBaseOp_t op,
                                                         void *input_1,
                                                         void *input_2,
                                                         void *output,
                                                         cnrtInvokeFuncParam_t *compute_forw_param,
                                                         cnrtQueue_t queue);
/* real div operation end */

/* broadcast operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateBroadcastOp(cnmlBaseOp_t *op,
                                                cnmlTensor_t input_tensor,
                                                cnmlTensor_t output_tensor);

CNML_DLL_API cnmlStatus_t
cnmlComputeBroadcastOpForward_V3(cnmlBaseOp_t op,
                                 void *input,
                                 void *output,
                                 cnrtInvokeFuncParam_t *compute_forw_param,
                                 cnrtQueue_t queue);
/* broadcast operation end */

/* broadcast sub operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateBroadcastSubOp(cnmlBaseOp_t *op,
                                                   cnmlTensor_t input_tensor_1,
                                                   cnmlTensor_t input_tensor_2,
                                                   cnmlTensor_t output_tensor);

CNML_DLL_API cnmlStatus_t
cnmlComputeBroadcastSubOpForward_V3(cnmlBaseOp_t op,
                                    void *input_1,
                                    void *input_2,
                                    void *output,
                                    cnrtInvokeFuncParam_t *compute_forw_param,
                                    cnrtQueue_t queue);
/* broadcast sub operation end */

/* broadcast add operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateBroadcastAddOp(cnmlBaseOp_t *op,
                                                   cnmlTensor_t input_tensor_1,
                                                   cnmlTensor_t input_tensor_2,
                                                   cnmlTensor_t output_tensor);

CNML_DLL_API cnmlStatus_t
cnmlComputeBroadcastAddOpForward_V3(cnmlBaseOp_t op,
                                    void *input_1,
                                    void *input_2,
                                    void *output,
                                    cnrtInvokeFuncParam_t *compute_forw_param,
                                    cnrtQueue_t queue);
/* broadcast add operation end */

/* broadcast mult operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateBroadcastMultOp(cnmlBaseOp_t *op,
                                                    cnmlTensor_t input_tensor_1,
                                                    cnmlTensor_t input_tensor_2,
                                                    cnmlTensor_t output_tensor);

CNML_DLL_API cnmlStatus_t
cnmlComputeBroadcastMultOpForward_V3(cnmlBaseOp_t op,
                                     void *input_1,
                                     void *input_2,
                                     void *output,
                                     cnrtInvokeFuncParam_t *compute_forw_param,
                                     cnrtQueue_t queue);
/* broadcast mult operation end */

/* sub operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateSubOp(cnmlBaseOp_t *op,
                                          cnmlTensor_t input_tensor_1,
                                          cnmlTensor_t input_tensor_2,
                                          cnmlTensor_t output_tensor);

CNML_DLL_API cnmlStatus_t cnmlComputeSubOpForward_V3(cnmlBaseOp_t op,
                                                     void *input_1,
                                                     void *input_2,
                                                     void *output,
                                                     cnrtInvokeFuncParam_t *compute_forw_param,
                                                     cnrtQueue_t queue);
/* sub operation end */

/* mult operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateMultOp(cnmlBaseOp_t *op,
                                           cnmlTensor_t input_tensor_1,
                                           cnmlTensor_t input_tensor_2,
                                           cnmlTensor_t output_tensor);

CNML_DLL_API cnmlStatus_t cnmlComputeMultOpForward_V3(cnmlBaseOp_t op,
                                                      void *input_1,
                                                      void *input_2,
                                                      void *output,
                                                      cnrtInvokeFuncParam_t *compute_forw_param,
                                                      cnrtQueue_t queue);
/* mult operation end */

/* lrn operation start */
struct cnmlLrnOpParam;
typedef struct cnmlLrnOpParam *cnmlLrnOpParam_t;

CNML_DLL_API cnmlStatus_t cnmlCreateLrnOpParam(cnmlLrnOpParam_t *param,
                                               cnmlLrnType_t type,
                                               int local_size,
                                               double alpha,
                                               double beta,
                                               double k);

CNML_DLL_API cnmlStatus_t cnmlDestroyLrnOpParam(cnmlLrnOpParam_t *param);

CNML_DLL_API cnmlStatus_t cnmlCreateLrnOp(cnmlBaseOp_t *op,
                                          cnmlLrnOpParam_t param,
                                          cnmlTensor_t input,
                                          cnmlTensor_t output);

CNML_DLL_API cnmlStatus_t cnmlComputeLrnOpForward_V3(cnmlBaseOp_t op,
                                                     void *input,
                                                     void *output,
                                                     cnrtInvokeFuncParam_t *compute_forw_param,
                                                     cnrtQueue_t queue);

CNML_DLL_API cnmlStatus_t cnmlEnableLrnOpFix8Mode(cnmlBaseOp_t op,
                                                  int input_data_fix8_position,
                                                  float offset);

CNML_DLL_API cnmlStatus_t cnmlEnableLrnOpFix8ScaleMode(cnmlBaseOp_t op, float scale);

CNML_DLL_API cnmlStatus_t cnmlEnableLrnOpInt8Mode(cnmlBaseOp_t op,
                                                  int input_data_int8_position,
                                                  float offset);

CNML_DLL_API cnmlStatus_t cnmlEnableLrnOpInt8ScaleMode(cnmlBaseOp_t op, float scale);
/*  lrn operation end  */

/* batch_norm operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateBatchNormOp(cnmlBaseOp_t *op,
                                                cnmlTensor_t input,
                                                cnmlTensor_t output,
                                                cnmlTensor_t mean,
                                                cnmlTensor_t var);

CNML_DLL_API cnmlStatus_t
cnmlComputeBatchNormOpForward_V3(cnmlBaseOp_t op,
                                 void *input,
                                 void *output,
                                 cnrtInvokeFuncParam_t *compute_forw_param,
                                 cnrtQueue_t queue);
/* batch_norm operation end */

/* mean-var_norm operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateMeanVarNormOp(cnmlBaseOp_t *op,
                                                  cnmlTensor_t input,
                                                  cnmlTensor_t output,
                                                  bool use_variance,
                                                  bool across_spatial,
                                                  float eps);

CNML_DLL_API cnmlStatus_t cnmlComputeMeanVarNormOpForward(cnmlBaseOp_t op,
                                                          void *input,
                                                          void *output,
                                                          cnrtInvokeFuncParam_t *compute_forw_param,
                                                          cnrtQueue_t queue);
/* mean-var_norm operation end */

/* max operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateMaxOp(cnmlBaseOp_t *op,
                                          cnmlTensor_t input,
                                          cnmlTensor_t output,
                                          cnmlTensor_t index);

CNML_DLL_API cnmlStatus_t cnmlComputeMaxOpForward_V3(cnmlBaseOp_t op,
                                                     void *input,
                                                     void *output,
                                                     void *index,
                                                     cnrtInvokeFuncParam_t *compute_forw_param,
                                                     cnrtQueue_t queue);
/* max operation end */

/* reduce max operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateReduceMaxOp(cnmlBaseOp_t *op,
                                                cnmlReduceMaxMode_t mode,
                                                cnmlTensor_t input,
                                                cnmlTensor_t output);

CNML_DLL_API cnmlStatus_t
cnmlComputeReduceMaxOpForward_V3(cnmlBaseOp_t op,
                                 void *input,
                                 void *output,
                                 cnrtInvokeFuncParam_t *compute_forw_param,
                                 cnrtQueue_t queue);
/* reduce max operation end */

/* min operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateMinOp(cnmlBaseOp_t *op,
                                          cnmlTensor_t input,
                                          cnmlTensor_t output,
                                          cnmlTensor_t index);

CNML_DLL_API cnmlStatus_t cnmlComputeMinOpForward_V3(cnmlBaseOp_t op,
                                                     void *input,
                                                     void *output,
                                                     void *index,
                                                     cnrtInvokeFuncParam_t *compute_forw_param,
                                                     cnrtQueue_t queue);
/* min operation end */

/* reverse operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateReverseOp(cnmlBaseOp_t *op,
                                              cnmlTensor_t input,
                                              cnmlTensor_t output,
                                              cnmlReverseAxis_t reverse_axis);

CNML_DLL_API cnmlStatus_t cnmlComputeReverseOpForward_V3(cnmlBaseOp_t op,
                                                         void *input,
                                                         void *output,
                                                         cnrtInvokeFuncParam_t *compute_forw_param,
                                                         cnrtQueue_t queue);
/* reverse operation end */

/* resize operation start */
struct cnmlResizeOpParam;
typedef struct cnmlResizeOpParam *cnmlResizeOpParam_t;

CNML_DLL_API cnmlStatus_t cnmlCreateResizeOpParam(cnmlResizeOpParam_t *param,
                                                  float height_1,
                                                  float weight_1,
                                                  float height_2,
                                                  float weight_2);

CNML_DLL_API cnmlStatus_t cnmlDestroyResizeOpParam(cnmlResizeOpParam_t *param);

CNML_DLL_API cnmlStatus_t cnmlCreateResizeOp(cnmlBaseOp_t *op,
                                             cnmlTensor_t input,
                                             cnmlTensor_t output,
                                             cnmlResizeOpParam_t param);

CNML_DLL_API cnmlStatus_t cnmlComputeResizeOpForward_V3(cnmlBaseOp_t op,
                                                        void *input,
                                                        void *output,
                                                        cnrtInvokeFuncParam_t *compute_forw_param,
                                                        cnrtQueue_t queue);
/* resize operation end */

/* interp operation start */
struct cnmlInterpOpParam;
typedef struct cnmlInterpOpParam *cnmlInterpOpParam_t;

CNML_DLL_API cnmlStatus_t cnmlCreateInterpOpParam(cnmlInterpOpParam_t *param,
                                                  int output_width,
                                                  int output_height,
                                                  bool align_corners);

CNML_DLL_API cnmlStatus_t cnmlCreateInterpOpParamByRatio(cnmlInterpOpParam_t *param,
                                                         float zoom,
                                                         bool align_corners);

CNML_DLL_API cnmlStatus_t cnmlDestroyInterpOpParam(cnmlInterpOpParam_t *param);

CNML_DLL_API cnmlStatus_t cnmlCreateInterpOp(cnmlBaseOp_t *op,
                                             cnmlTensor_t input,
                                             cnmlTensor_t output,
                                             cnmlInterpOpParam_t param);

CNML_DLL_API cnmlStatus_t cnmlComputeInterpOpForward_V3(cnmlBaseOp_t op,
                                                        void *input,
                                                        void *output,
                                                        cnrtInvokeFuncParam_t *compute_forw_param,
                                                        cnrtQueue_t queue);
/* interp operation end */

/* scale operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateScaleOp(cnmlBaseOp_t *op,
                                            cnmlTensor_t input,
                                            cnmlTensor_t output,
                                            cnmlTensor_t alpha,
                                            cnmlTensor_t beta);

CNML_DLL_API cnmlStatus_t cnmlComputeScaleOpForward_V3(cnmlBaseOp_t op,
                                                       void *input,
                                                       void *output,
                                                       cnrtInvokeFuncParam_t *compute_forw_param,
                                                       cnrtQueue_t queue);

CNML_DLL_API cnmlStatus_t
cnmlComputeScaleOpForwardUltra_V3(cnmlBaseOp_t op,
                                  void *input,
                                  void *alpha,
                                  void *beta,
                                  void *output,
                                  cnrtInvokeFuncParam_t *compute_forw_param,
                                  cnrtQueue_t queue);
/* scale operation end */

/* concat operation start */
struct cnmlConcatOpParam;
typedef struct cnmlConcatOpParam *cnmlConcatOpParam_t;

CNML_DLL_API cnmlStatus_t cnmlCreateConcatOpParam(cnmlConcatOpParam_t *param,
                                                  int input_num,
                                                  int output_num,
                                                  cnmlConcatMode_t mode);

CNML_DLL_API cnmlStatus_t cnmlDestroyConcatOpParam(cnmlConcatOpParam_t *param);

CNML_DLL_API cnmlStatus_t cnmlCreateConcatOp(cnmlBaseOp_t *op,
                                             cnmlConcatOpParam_t param,
                                             cnmlTensor_t *inputs,
                                             int input_num,
                                             cnmlTensor_t *outputs,
                                             int output_num);

CNML_DLL_API cnmlStatus_t cnmlComputeConcatOpForward_V3(cnmlBaseOp_t op,
                                                        void *inputs[],
                                                        int input_num,
                                                        void *outputs[],
                                                        int output_num,
                                                        cnrtInvokeFuncParam_t *compute_forw_param,
                                                        cnrtQueue_t queue);
/* concat operation end */

/* slice operation start */
struct cnmlSplitOpParam;
typedef struct cnmlSplitOpParam *cnmlSplitOpParam_t;

CNML_DLL_API cnmlStatus_t cnmlCreateSplitOpParam(cnmlSplitOpParam_t *param,
                                                 int input_num,
                                                 int output_num,
                                                 cnmlSplitMode_t mode);

CNML_DLL_API cnmlStatus_t cnmlDestroySplitOpParam(cnmlSplitOpParam_t *param);

CNML_DLL_API cnmlStatus_t cnmlCreateSplitOp(cnmlBaseOp_t *op,
                                            cnmlSplitOpParam_t param,
                                            cnmlTensor_t *inputs,
                                            int input_num,
                                            cnmlTensor_t *outputs,
                                            int output_num);

CNML_DLL_API cnmlStatus_t cnmlComputeSplitOpForward_V3(cnmlBaseOp_t op,
                                                       void *inputs[],
                                                       int input_num,
                                                       void *outputs[],
                                                       int output_num,
                                                       cnrtInvokeFuncParam_t *compute_forw_param,
                                                       cnrtQueue_t queue);
/* slice operation end  */

/* shuffle channel start */
CNML_DLL_API cnmlStatus_t cnmlCreateShuffleChannelOp(cnmlBaseOp_t *op,
                                                     cnmlTensor_t *inputs,
                                                     cnmlTensor_t *outputs,
                                                     int group);

CNML_DLL_API cnmlStatus_t
cnmlComputeShuffleChannelOpForward_V3(cnmlBaseOp_t op,
                                      void *inputs[],
                                      void *outputs[],
                                      cnrtInvokeFuncParam_t *compute_forw_param,
                                      cnrtQueue_t queue);
/* shuffle channel operation end  */

/* not operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateNotOp(cnmlBaseOp_t *op,
                                          cnmlTensor_t input_tensor,
                                          cnmlTensor_t output_tensor);

CNML_DLL_API cnmlStatus_t cnmlComputeNotOpForward_V3(cnmlBaseOp_t op,
                                                     void *input,
                                                     void *output,
                                                     cnrtInvokeFuncParam_t *compute_forw_param,
                                                     cnrtQueue_t queue);
/* not operation end */

/* and operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateAndOp(cnmlBaseOp_t *op,
                                          cnmlTensor_t input_tensor_1,
                                          cnmlTensor_t input_tensor_2,
                                          cnmlTensor_t output_tensor);

CNML_DLL_API cnmlStatus_t cnmlComputeAndOpForward_V3(cnmlBaseOp_t op,
                                                     void *input_1,
                                                     void *input_2,
                                                     void *output,
                                                     cnrtInvokeFuncParam_t *compute_forw_param,
                                                     cnrtQueue_t queue);
/* and operation end */

/* cycleand operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateCycleAndOp(cnmlBaseOp_t *op,
                                               cnmlTensor_t input_tensor_1,
                                               cnmlTensor_t input_tensor_2,
                                               cnmlTensor_t output_tensor);

CNML_DLL_API cnmlStatus_t cnmlComputeCycleAndOpForward_V3(cnmlBaseOp_t op,
                                                          void *input_1,
                                                          void *input_2,
                                                          void *output,
                                                          cnrtInvokeFuncParam_t *compute_forw_param,
                                                          cnrtQueue_t queue);
/* cycleand operation end */

/* cyclexor operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateCycleXorOp(cnmlBaseOp_t *op,
                                               cnmlTensor_t input_tensor_1,
                                               cnmlTensor_t input_tensor_2,
                                               cnmlTensor_t output_tensor);

CNML_DLL_API cnmlStatus_t cnmlComputeCycleXorOpForward_V3(cnmlBaseOp_t op,
                                                          void *input_1,
                                                          void *input_2,
                                                          void *output,
                                                          cnrtInvokeFuncParam_t *compute_forw_param,
                                                          cnrtQueue_t queue);
/* cyclexor operation end */

/* xor operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateXorOp(cnmlBaseOp_t *op,
                                          cnmlTensor_t input_tensor_1,
                                          cnmlTensor_t input_tensor_2,
                                          cnmlTensor_t output_tensor);

CNML_DLL_API cnmlStatus_t cnmlComputeXorOpForward_V3(cnmlBaseOp_t op,
                                                     void *input_1,
                                                     void *input_2,
                                                     void *output,
                                                     cnrtInvokeFuncParam_t *compute_forw_param,
                                                     cnrtQueue_t queue);
/* xor operation end */

/* cycleor operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateCycleOrOp(cnmlBaseOp_t *op,
                                              cnmlTensor_t input_tensor_1,
                                              cnmlTensor_t input_tensor_2,
                                              cnmlTensor_t output_tensor);

CNML_DLL_API cnmlStatus_t cnmlComputeCycleOrOpForward_V3(cnmlBaseOp_t op,
                                                         void *input_1,
                                                         void *input_2,
                                                         void *output,
                                                         cnrtInvokeFuncParam_t *compute_forw_param,
                                                         cnrtQueue_t queue);
/* cycleor operation end */

/* or operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateOrOp(cnmlBaseOp_t *op,
                                         cnmlTensor_t input_tensor_1,
                                         cnmlTensor_t input_tensor_2,
                                         cnmlTensor_t output_tensor);

CNML_DLL_API cnmlStatus_t cnmlComputeOrOpForward_V3(cnmlBaseOp_t op,
                                                    void *input_1,
                                                    void *input_2,
                                                    void *output,
                                                    cnrtInvokeFuncParam_t *compute_forw_param,
                                                    cnrtQueue_t queue);
/* or operation end */

/* cycleadd operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateCycleAddOp(cnmlBaseOp_t *op,
                                               cnmlTensor_t input_tensor_1,
                                               cnmlTensor_t input_tensor_2,
                                               cnmlTensor_t output_tensor);

CNML_DLL_API cnmlStatus_t cnmlComputeCycleAddOpForward_V3(cnmlBaseOp_t op,
                                                          void *input_1,
                                                          void *input_2,
                                                          void *output,
                                                          cnrtInvokeFuncParam_t *compute_forw_param,
                                                          cnrtQueue_t queue);
/* cycleadd operation end */

/* cyclesub operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateCycleSubOp(cnmlBaseOp_t *op,
                                               cnmlTensor_t input_tensor_1,
                                               cnmlTensor_t input_tensor_2,
                                               cnmlTensor_t output_tensor);

CNML_DLL_API cnmlStatus_t cnmlComputeCycleSubOpForward_V3(cnmlBaseOp_t op,
                                                          void *input_1,
                                                          void *input_2,
                                                          void *output,
                                                          cnrtInvokeFuncParam_t *compute_forw_param,
                                                          cnrtQueue_t queue);
/* cyclesub operation end */

/* cyclemult operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateCycleMultOp(cnmlBaseOp_t *op,
                                                cnmlTensor_t input_tensor_1,
                                                cnmlTensor_t input_tensor_2,
                                                cnmlTensor_t output_tensor);

CNML_DLL_API cnmlStatus_t
cnmlComputeCycleMultOpForward_V3(cnmlBaseOp_t op,
                                 void *input_1,
                                 void *input_2,
                                 void *output,
                                 cnrtInvokeFuncParam_t *compute_forw_param,
                                 cnrtQueue_t queue);
/* cyclemult operation end */

/* cycle equal operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateCycleEqualOp(cnmlBaseOp_t *op,
                                                 cnmlTensor_t input_tensor_1,
                                                 cnmlTensor_t input_tensor_2,
                                                 cnmlTensor_t output_tensor);

CNML_DLL_API cnmlStatus_t
cnmlComputeCycleEqualOpForward_V3(cnmlBaseOp_t op,
                                  void *input_1,
                                  void *input_2,
                                  void *output,
                                  cnrtInvokeFuncParam_t *compute_forw_param,
                                  cnrtQueue_t queue);
/* cycle equal operation end */

/* cycle n equal operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateCycleNEqualOp(cnmlBaseOp_t *op,
                                                  cnmlTensor_t input_tensor_1,
                                                  cnmlTensor_t input_tensor_2,
                                                  cnmlTensor_t output_tensor);

CNML_DLL_API cnmlStatus_t
cnmlComputeCycleNEqualOpForward_V3(cnmlBaseOp_t op,
                                   void *input_1,
                                   void *input_2,
                                   void *output,
                                   cnrtInvokeFuncParam_t *compute_forw_param,
                                   cnrtQueue_t queue);
/* cycle n equal operation end */

/* cycle less equal operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateCycleLessEqualOp(cnmlBaseOp_t *op,
                                                     cnmlTensor_t input_tensor_1,
                                                     cnmlTensor_t input_tensor_2,
                                                     cnmlTensor_t output_tensor);

CNML_DLL_API cnmlStatus_t
cnmlComputeCycleLessEqualOpForward_V3(cnmlBaseOp_t op,
                                      void *input_1,
                                      void *input_2,
                                      void *output,
                                      cnrtInvokeFuncParam_t *compute_forw_param,
                                      cnrtQueue_t queue);
/* cycle less equal operation end */

/* cycle less operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateCycleLessOp(cnmlBaseOp_t *op,
                                                cnmlTensor_t input_tensor_1,
                                                cnmlTensor_t input_tensor_2,
                                                cnmlTensor_t output_tensor);

CNML_DLL_API cnmlStatus_t
cnmlComputeCycleLessOpForward_V3(cnmlBaseOp_t op,
                                 void *input_1,
                                 void *input_2,
                                 void *output,
                                 cnrtInvokeFuncParam_t *compute_forw_param,
                                 cnrtQueue_t queue);
/* cycle less operation end */

/* cycle greater equal operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateCycleGreaterEqualOp(cnmlBaseOp_t *op,
                                                        cnmlTensor_t input_tensor_1,
                                                        cnmlTensor_t input_tensor_2,
                                                        cnmlTensor_t output_tensor);

CNML_DLL_API cnmlStatus_t
cnmlComputeCycleGreaterEqualOpForward_V3(cnmlBaseOp_t op,
                                         void *input_1,
                                         void *input_2,
                                         void *output,
                                         cnrtInvokeFuncParam_t *compute_forw_param,
                                         cnrtQueue_t queue);
/* cycle greater equal operation end */

/* cycle greater operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateCycleGreaterOp(cnmlBaseOp_t *op,
                                                   cnmlTensor_t input_tensor_1,
                                                   cnmlTensor_t input_tensor_2,
                                                   cnmlTensor_t output_tensor);

CNML_DLL_API cnmlStatus_t
cnmlComputeCycleGreaterOpForward_V3(cnmlBaseOp_t op,
                                    void *input_1,
                                    void *input_2,
                                    void *output,
                                    cnrtInvokeFuncParam_t *compute_forw_param,
                                    cnrtQueue_t queue);
/* cycle greater operation end */

/* equal operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateEqualOp(cnmlBaseOp_t *op,
                                            cnmlTensor_t input_tensor_1,
                                            cnmlTensor_t input_tensor_2,
                                            cnmlTensor_t output_tensor);

CNML_DLL_API cnmlStatus_t cnmlComputeEqualOpForward_V3(cnmlBaseOp_t op,
                                                       void *input_1,
                                                       void *input_2,
                                                       void *output,
                                                       cnrtInvokeFuncParam_t *compute_forw_param,
                                                       cnrtQueue_t queue);
/* equal operation end */

/* not equal operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateNEqualOp(cnmlBaseOp_t *op,
                                             cnmlTensor_t input_tensor_1,
                                             cnmlTensor_t input_tensor_2,
                                             cnmlTensor_t output_tensor);

CNML_DLL_API cnmlStatus_t cnmlComputeNEqualOpForward_V3(cnmlBaseOp_t op,
                                                        void *input_1,
                                                        void *input_2,
                                                        void *output,
                                                        cnrtInvokeFuncParam_t *compute_forw_param,
                                                        cnrtQueue_t queue);
/* not equal operation end */

/* less equal operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateLessEqualOp(cnmlBaseOp_t *op,
                                                cnmlTensor_t input_tensor_1,
                                                cnmlTensor_t input_tensor_2,
                                                cnmlTensor_t output_tensor);

CNML_DLL_API cnmlStatus_t
cnmlComputeLessEqualOpForward_V3(cnmlBaseOp_t op,
                                 void *input_1,
                                 void *input_2,
                                 void *output,
                                 cnrtInvokeFuncParam_t *compute_forw_param,
                                 cnrtQueue_t queue);
/* less equal operation end */

/* less operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateLessOp(cnmlBaseOp_t *op,
                                           cnmlTensor_t input_tensor_1,
                                           cnmlTensor_t input_tensor_2,
                                           cnmlTensor_t output_tensor);

CNML_DLL_API cnmlStatus_t cnmlComputeLessOpForward_V3(cnmlBaseOp_t op,
                                                      void *input_1,
                                                      void *input_2,
                                                      void *output,
                                                      cnrtInvokeFuncParam_t *compute_forw_param,
                                                      cnrtQueue_t queue);
/* less operation end */

/* greater equal operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateGreaterEqualOp(cnmlBaseOp_t *op,
                                                   cnmlTensor_t input_tensor_1,
                                                   cnmlTensor_t input_tensor_2,
                                                   cnmlTensor_t output_tensor);

CNML_DLL_API cnmlStatus_t
cnmlComputeGreaterEqualOpForward_V3(cnmlBaseOp_t op,
                                    void *input_1,
                                    void *input_2,
                                    void *output,
                                    cnrtInvokeFuncParam_t *compute_forw_param,
                                    cnrtQueue_t queue);
/* greater equal operation end */

/* greater operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateGreaterOp(cnmlBaseOp_t *op,
                                              cnmlTensor_t input_tensor_1,
                                              cnmlTensor_t input_tensor_2,
                                              cnmlTensor_t output_tensor);

CNML_DLL_API cnmlStatus_t cnmlComputeGreaterOpForward_V3(cnmlBaseOp_t op,
                                                         void *input_1,
                                                         void *input_2,
                                                         void *output,
                                                         cnrtInvokeFuncParam_t *compute_forw_param,
                                                         cnrtQueue_t queue);
/* greater operation end */

/* maxtt operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateMaxTTOp(cnmlBaseOp_t *op,
                                            cnmlTensor_t input_1,
                                            cnmlTensor_t input_2,
                                            cnmlTensor_t output);

CNML_DLL_API cnmlStatus_t cnmlComputeMaxTTOpForward_V3(cnmlBaseOp_t op,
                                                       void *input_1,
                                                       void *input_2,
                                                       void *output,
                                                       cnrtInvokeFuncParam_t *compute_forw_param,
                                                       cnrtQueue_t queue);
/* maxtt operation end */

/* maxtc operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateMaxTCOp(cnmlBaseOp_t *op,
                                            cnmlTensor_t input,
                                            cnmlTensor_t output,
                                            float cc);

CNML_DLL_API cnmlStatus_t cnmlComputeMaxTCOpForward_V3(cnmlBaseOp_t op,
                                                       void *input,
                                                       void *output,
                                                       cnrtInvokeFuncParam_t *compute_forw_param,
                                                       cnrtQueue_t queue);
/* maxtc operation end */

/* mintc operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateMinTCOp(cnmlBaseOp_t *op,
                                            cnmlTensor_t input,
                                            cnmlTensor_t output,
                                            float cc);

CNML_DLL_API cnmlStatus_t cnmlComputeMinTCOpForward_V3(cnmlBaseOp_t op,
                                                       void *input,
                                                       void *output,
                                                       cnrtInvokeFuncParam_t *compute_forw_param,
                                                       cnrtQueue_t queue);
/* mintc operation end */

/* clip operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateClipOp(cnmlBaseOp_t *op,
                                           cnmlTensor_t input_tensor,
                                           cnmlTensor_t output_tensor,
                                           double lower_bound,
                                           double upper_bound);

CNML_DLL_API cnmlStatus_t cnmlComputeClipOpForward_V3(cnmlBaseOp_t op,
                                                      void *input,
                                                      void *output,
                                                      cnrtInvokeFuncParam_t *compute_forw_param,
                                                      cnrtQueue_t queue);
/* clip operation end */

/* nearest_neighbor operation start */
struct cnmlNearestNeighborOpParam;
typedef struct cnmlNearestNeighborOpParam *cnmlNearestNeighborOpParam_t;

CNML_DLL_API cnmlStatus_t cnmlCreateNearestNeighborOpParam(cnmlNearestNeighborOpParam_t *param,
                                                           int output_width,
                                                           int output_height);

CNML_DLL_API cnmlStatus_t
cnmlCreateNearestNeighborOpParamByRatio(cnmlNearestNeighborOpParam_t *param, int zoom);

CNML_DLL_API cnmlStatus_t cnmlDestroyNearestNeighborOpParam(cnmlNearestNeighborOpParam_t *param);

CNML_DLL_API cnmlStatus_t cnmlCreateNearestNeighborOp(cnmlBaseOp_t *op,
                                                      cnmlTensor_t input,
                                                      cnmlTensor_t output,
                                                      cnmlNearestNeighborOpParam_t param);

CNML_DLL_API cnmlStatus_t
cnmlComputeNearestNeighborOpForward_V3(cnmlBaseOp_t op,
                                       void *input,
                                       void *output,
                                       cnrtInvokeFuncParam_t *compute_forw_param,
                                       cnrtQueue_t queue);
/* nearest_neighbor operation end */

/* prelu operation start */
CNML_DLL_API cnmlStatus_t cnmlCreatePreluOp(cnmlBaseOp_t *op,
                                            cnmlTensor_t input,
                                            cnmlTensor_t output,
                                            cnmlTensor_t prelu_param);

CNML_DLL_API cnmlStatus_t cnmlComputePreluOpForward_V3(cnmlBaseOp_t op,
                                                       void *input,
                                                       void *output,
                                                       cnrtInvokeFuncParam_t *compute_forw_param,
                                                       cnrtQueue_t queue);
/* prelu operation end */

/* sqrt operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateSqrtOp(cnmlBaseOp_t *op,
                                           cnmlTensor_t input,
                                           cnmlTensor_t output);

CNML_DLL_API cnmlStatus_t cnmlComputeSqrtOpForward_V3(cnmlBaseOp_t op,
                                                      void *input,
                                                      void *output,
                                                      cnrtInvokeFuncParam_t *compute_forw_param,
                                                      cnrtQueue_t queue);
/* sqrt operation end */

/* Rsqrt operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateRsqrtOp(cnmlBaseOp_t *op,
                                            cnmlTensor_t input,
                                            cnmlTensor_t output);

CNML_DLL_API cnmlStatus_t cnmlComputeRsqrtOpForward_V3(cnmlBaseOp_t op,
                                                       void *input,
                                                       void *output,
                                                       cnrtInvokeFuncParam_t *compute_forw_param,
                                                       cnrtQueue_t queue);
/* rsqrt operation end */

/* exp operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateExpOp(cnmlBaseOp_t *op,
                                          cnmlTensor_t input,
                                          cnmlTensor_t output);

CNML_DLL_API cnmlStatus_t cnmlComputeExpOpForward_V3(cnmlBaseOp_t op,
                                                     void *input,
                                                     void *output,
                                                     cnrtInvokeFuncParam_t *compute_forw_param,
                                                     cnrtQueue_t queue);
/* exp operation end */

/* softmax operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateSoftmaxOp(cnmlBaseOp_t *op,
                                              cnmlSoftmaxDim_t dim,
                                              cnmlTensor_t input,
                                              cnmlTensor_t output);

CNML_DLL_API cnmlStatus_t cnmlComputeSoftmaxOpForward_V3(cnmlBaseOp_t op,
                                                         void *input,
                                                         void *output,
                                                         cnrtInvokeFuncParam_t *compute_forw_param,
                                                         cnrtQueue_t queue);
/* softmax operation end */

/* log operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateLogOp(cnmlBaseOp_t *op,
                                          cnmlTensor_t input,
                                          cnmlTensor_t output);

CNML_DLL_API cnmlStatus_t cnmlComputeLogOpForward_V3(cnmlBaseOp_t op,
                                                     void *input,
                                                     void *output,
                                                     cnrtInvokeFuncParam_t *compute_forw_param,
                                                     cnrtQueue_t queue);
/* log operation end */

/* floor operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateFloorOp(cnmlBaseOp_t *op,
                                            cnmlTensor_t input,
                                            cnmlTensor_t output);

CNML_DLL_API cnmlStatus_t cnmlComputeFloorOpForward_V3(cnmlBaseOp_t op,
                                                       void *input,
                                                       void *output,
                                                       cnrtInvokeFuncParam_t *compute_forw_param,
                                                       cnrtQueue_t queue);
/* floor operation end */

/* power operation start */
CNML_DLL_API cnmlStatus_t cnmlCreatePowerOp(cnmlBaseOp_t *op,
                                            cnmlTensor_t input,
                                            cnmlTensor_t output,
                                            float power_c);

CNML_DLL_API cnmlStatus_t cnmlComputePowerOpForward_V3(cnmlBaseOp_t op,
                                                       void *input,
                                                       void *output,
                                                       cnrtInvokeFuncParam_t *compute_forw_param,
                                                       cnrtQueue_t queue);
/* power operation end */

/* unarySelect operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateUnarySelectOp(cnmlBaseOp_t *op,
                                                  cnmlTensor_t input_tensor_1,
                                                  cnmlTensor_t input_tensor_2,
                                                  cnmlTensor_t output_tensor,
                                                  cnmlTensor_t count_cnml);

CNML_DLL_API cnmlStatus_t
cnmlComputeUnarySelectOpForward_V3(cnmlBaseOp_t op,
                                   void *input_1,
                                   void *input_2,
                                   void *output,
                                   void *count,
                                   cnrtInvokeFuncParam_t *compute_forw_param,
                                   cnrtQueue_t queue);

/* unarySelect operation end */

/* dyadicSelect operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateDyadicSelectOp(cnmlBaseOp_t *op,
                                                   cnmlTensor_t input_tensor_1,
                                                   cnmlTensor_t input_tensor_2,
                                                   cnmlTensor_t input_tensor_3,
                                                   cnmlTensor_t output_tensor);

CNML_DLL_API cnmlStatus_t
cnmlComputeDyadicSelectOpForward_V3(cnmlBaseOp_t op,
                                    void *input_1,
                                    void *input_2,
                                    void *input_3,
                                    void *output,
                                    cnrtInvokeFuncParam_t *compute_forw_param,
                                    cnrtQueue_t queue);
/* dyadicSelect operation end */

/* svdf operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateSvdfOp(cnmlBaseOp_t *op,
                                           cnmlTensor_t input_tensor,
                                           cnmlTensor_t weights_feature,
                                           cnmlTensor_t weights_time,
                                           cnmlTensor_t bias_tensor,
                                           cnmlTensor_t state_in,
                                           cnmlTensor_t state_out,
                                           cnmlTensor_t output_tensor);

CNML_DLL_API cnmlStatus_t cnmlComputeSvdfOpForward_V3(cnmlBaseOp_t op,
                                                      void *input,
                                                      void *state_in,
                                                      void *state_out,
                                                      void *output,
                                                      cnrtInvokeFuncParam_t *compute_forw_param,
                                                      cnrtQueue_t queue);
/* svdf operation end */

/* abs operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateAbsOp(cnmlBaseOp_t *op,
                                          cnmlTensor_t input,
                                          cnmlTensor_t output);

CNML_DLL_API cnmlStatus_t cnmlComputeAbsOpForward_V3(cnmlBaseOp_t op,
                                                     void *input,
                                                     void *output,
                                                     cnrtInvokeFuncParam_t *compute_forw_param,
                                                     cnrtQueue_t queue);
/* abs operation end */

/* softplus operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateSoftplusOp(cnmlBaseOp_t *op,
                                               cnmlTensor_t input,
                                               cnmlTensor_t output);

CNML_DLL_API cnmlStatus_t cnmlComputeSoftplusOpForward_V3(cnmlBaseOp_t op,
                                                          void *input,
                                                          void *output,
                                                          cnrtInvokeFuncParam_t *compute_forw_param,
                                                          cnrtQueue_t queue);
/* softplus operation end */

/* minus operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateMinusOp(cnmlBaseOp_t *op,
                                            cnmlTensor_t input,
                                            cnmlTensor_t output);

CNML_DLL_API cnmlStatus_t cnmlComputeMinusOpForward_V3(cnmlBaseOp_t op,
                                                       void *input,
                                                       void *output,
                                                       cnrtInvokeFuncParam_t *compute_forw_param,
                                                       cnrtQueue_t queue);
/* minus operation end */

/* fakequant operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateFakeQuantOp(cnmlBaseOp_t *op,
                                                cnmlTensor_t input,
                                                cnmlTensor_t output,
                                                float scale,
                                                int offset);

CNML_DLL_API cnmlStatus_t
cnmlComputeFakeQuantOpForward_V3(cnmlBaseOp_t op,
                                 void *input,
                                 void *output,
                                 cnrtInvokeFuncParam_t *compute_forw_param,
                                 cnrtQueue_t queue);
/* fakequant operation end */

/* avg operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateAvgOp(cnmlBaseOp_t *op,
                                          cnmlTensor_t input,
                                          cnmlTensor_t output);

CNML_DLL_API cnmlStatus_t cnmlComputeAvgOpForward_V3(cnmlBaseOp_t op,
                                                     void *input,
                                                     void *output,
                                                     cnrtInvokeFuncParam_t *compute_forw_param,
                                                     cnrtQueue_t queue);
/* avg operation end */

/* vector2norm operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateVector2NormOp(cnmlBaseOp_t *op,
                                                  cnmlTensor_t input,
                                                  cnmlTensor_t output);

CNML_DLL_API cnmlStatus_t
cnmlComputeVector2NormOpForward_V3(cnmlBaseOp_t op,
                                   void *input,
                                   void *output,
                                   cnrtInvokeFuncParam_t *compute_forw_param,
                                   cnrtQueue_t queue);
/* vector2norm operation end */

/* cast operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateCastOp(cnmlBaseOp_t *op,
                                           cnmlCastType_t cast_type,
                                           cnmlTensor_t input,
                                           cnmlTensor_t output);

CNML_DLL_API cnmlStatus_t cnmlComputeCastOpForward_V3(cnmlBaseOp_t op,
                                                      void *input,
                                                      void *output,
                                                      cnrtInvokeFuncParam_t *compute_forw_param,
                                                      cnrtQueue_t queue);

void cnmlDumpCastTensor2File(const char *filename,
                             cnmlCpuTensor_t tensor,
                             cnmlTensorType_t type,
                             void *output_addr,
                             bool app);
/* cast operation end */

/* proposal operation start */
struct cnmlProposalOpParam;
typedef cnmlProposalOpParam *cnmlProposalOpParam_t;

CNML_DLL_API cnmlStatus_t cnmlCreateProposalOpParam(cnmlProposalOpParam_t *param,
                                                    int H,
                                                    int W,
                                                    int A,
                                                    int im_h,
                                                    int im_w,
                                                    int min_h,
                                                    int min_w,
                                                    int out_size,
                                                    float filter_scale,
                                                    float nms_thresh,
                                                    float nms_scale,
                                                    int feat_stride);

CNML_DLL_API cnmlStatus_t cnmlCreateProposalOpParamV2(cnmlProposalOpParam_t *param,
                                                      int H,
                                                      int W,
                                                      int A,
                                                      int im_h,
                                                      int im_w,
                                                      int min_h,
                                                      int min_w,
                                                      int out_size,
                                                      float filter_scale,
                                                      float nms_thresh,
                                                      float nms_scale,
                                                      int feat_stride,
                                                      float *anchor_scales,
                                                      int anchor_scales_size,
                                                      float *anchor_ratios,
                                                      int anchor_ratios_size);

CNML_DLL_API
cnmlStatus_t cnmlEnableProposalOpScoreSoftmax(cnmlProposalOpParam_t *param);

CNML_DLL_API
cnmlStatus_t cnmlEnableProposalOpPVANetMode(cnmlProposalOpParam_t *param);

CNML_DLL_API cnmlStatus_t cnmlDestroyProposalOpParam(cnmlProposalOpParam_t *param);

CNML_DLL_API cnmlStatus_t cnmlCreateProposalOp(cnmlBaseOp_t *op,
                                               cnmlProposalOpParam_t param,
                                               cnmlTensor_t bbox_deltas,
                                               cnmlTensor_t scores,
                                               cnmlTensor_t output);

CNML_DLL_API cnmlStatus_t cnmlComputeProposalOpForward_V3(cnmlBaseOp_t op,
                                                          void *bbox_deltas,
                                                          void *scores,
                                                          void *output,
                                                          cnrtInvokeFuncParam_t *compute_forw_param,
                                                          cnrtQueue_t queue);
/* proposal operation end */

/* nms operation start */
struct cnmlNmsOpParam;
typedef cnmlNmsOpParam *cnmlNmsOpParam_t;

CNML_DLL_API cnmlStatus_t cnmlCreateNmsOpParam(cnmlNmsOpParam_t *param,
                                               int box_size,
                                               int out_size,
                                               float nms_thresh,
                                               float nms_scale,
                                               float score_thresh,
                                               bool filter_scores,
                                               bool normalized_bbox);

CNML_DLL_API cnmlStatus_t cnmlDestroyNmsOpParam(cnmlNmsOpParam_t *param);

CNML_DLL_API cnmlStatus_t cnmlCreateNmsOp(cnmlBaseOp_t *op,
                                          cnmlNmsOpParam_t param,
                                          cnmlTensor_t input,
                                          cnmlTensor_t output);

CNML_DLL_API cnmlStatus_t cnmlComputeNmsOpForward_V3(cnmlBaseOp_t op,
                                                     void *input,
                                                     void *output,
                                                     cnrtInvokeFuncParam_t *compute_forw_param,
                                                     cnrtQueue_t queue);
/* nms operation end */

/* roi pooling operation start */
struct cnmlRoiPoolOpParam;
typedef struct cnmlRoiPoolOpParam *cnmlRoiPoolOpParam_t;

CNML_DLL_API cnmlStatus_t cnmlCreateRoiPoolOpParam(cnmlRoiPoolOpParam_t *param,
                                                   int pooled_h,
                                                   int pooled_w,
                                                   float spatial_scale);
CNML_DLL_API cnmlStatus_t cnmlDestroyRoiPoolOpParam(cnmlRoiPoolOpParam_t *param);

CNML_DLL_API cnmlStatus_t cnmlCreateRoiPoolOp(cnmlBaseOp_t *op,
                                              cnmlRoiPoolOpParam_t param,
                                              cnmlTensor_t input_tensor,
                                              cnmlTensor_t rois,
                                              cnmlTensor_t output_tensor);

CNML_DLL_API cnmlStatus_t cnmlComputeRoiPoolOpForward_V3(cnmlBaseOp_t op,
                                                         void *input,
                                                         void *rois,
                                                         void *output,
                                                         cnrtInvokeFuncParam_t *compute_forw_param,
                                                         cnrtQueue_t queue);
/* roi pooling operation end */

/* image_detect operation start */
struct cnmlImageDetectOpParam;
typedef cnmlImageDetectOpParam *cnmlImageDetectOpParam_t;

CNML_DLL_API cnmlStatus_t cnmlCreateImageDetectOpParam(cnmlImageDetectOpParam_t *param,
                                                       int num_class,
                                                       int box_size_per_class,
                                                       int im_h,
                                                       int im_w,
                                                       float score_thresh,
                                                       float nms_thresh,
                                                       float nms_scale);

CNML_DLL_API cnmlStatus_t cnmlDestroyImageDetectOpParam(cnmlImageDetectOpParam_t *param);

CNML_DLL_API cnmlStatus_t cnmlCreateImageDetectOp(cnmlBaseOp_t *op,
                                                  cnmlImageDetectOpParam_t param,
                                                  cnmlTensor_t bbox_deltas,
                                                  cnmlTensor_t scores,
                                                  cnmlTensor_t anchors,
                                                  cnmlTensor_t output);

CNML_DLL_API cnmlStatus_t
cnmlComputeImageDetectOpForward_V3(cnmlBaseOp_t op,
                                   void *bbox_deltas,
                                   void *scores,
                                   void *anchors,
                                   void *output,
                                   cnrtInvokeFuncParam_t *compute_forw_param,
                                   cnrtQueue_t queue);
/* image_detect operation end */

/* grep operation start */
struct cnmlGrepOpParam;
typedef struct cnmlGrepOpParam *cnmlGrepOpParam_t;

CNML_DLL_API cnmlStatus_t cnmlCreateGrepOpParam(cnmlGrepOpParam_t *param,
                                                int start_index_of_N,
                                                int start_index_of_H,
                                                int start_index_of_W,
                                                float space_number);

CNML_DLL_API cnmlStatus_t cnmlDestroyGrepOpParam(cnmlGrepOpParam_t *param);

CNML_DLL_API cnmlStatus_t cnmlCreateGrepOp(cnmlBaseOp_t *op,
                                           cnmlGrepOpParam_t param,
                                           cnmlTensor_t input_tensor,
                                           cnmlTensor_t output_tensor);

CNML_DLL_API cnmlStatus_t cnmlComputeGrepOpForward_V3(cnmlBaseOp_t op,
                                                      void *input,
                                                      void *output,
                                                      cnrtInvokeFuncParam_t *compute_forw_param,
                                                      cnrtQueue_t queue);
/* grep operation end */

/* argmax operation start */
CNML_DLL_API cnmlStatus_t cnmlGetArgmaxOpOutputDim(cnmlArgmaxAxis_t argmax_axis,
                                                   int ni,
                                                   int ci,
                                                   int hi,
                                                   int wi,
                                                   int *no,
                                                   int *co,
                                                   int *ho,
                                                   int *wo);

CNML_DLL_API cnmlStatus_t cnmlCreateArgmaxOp(cnmlBaseOp_t *op,
                                             cnmlArgmaxAxis_t argmax_mode,
                                             cnmlTensor_t input,
                                             cnmlTensor_t output);

CNML_DLL_API cnmlStatus_t cnmlComputeArgmaxOpForward_V3(cnmlBaseOp_t op,
                                                        void *input,
                                                        void *output,
                                                        cnrtInvokeFuncParam_t *compute_forw_param,
                                                        cnrtQueue_t queue);
/* argmax operation end */

/* reorg operation start */
struct cnmlReorgOpParam;
typedef struct cnmlReorgOpParam *cnmlReorgOpParam_t;

CNML_DLL_API cnmlStatus_t cnmlGetReorgOpOutputDim(int reorg_h,
                                                  int reorg_w,
                                                  int ni,
                                                  int ci,
                                                  int hi,
                                                  int wi,
                                                  int *no,
                                                  int *co,
                                                  int *ho,
                                                  int *wo,
                                                  bool reverse);

CNML_DLL_API cnmlStatus_t cnmlCreateReorgOpParam(cnmlReorgOpParam_t *param,
                                                 int reorg_h,
                                                 int reorg_w,
                                                 bool reverse);

CNML_DLL_API cnmlStatus_t cnmlDestroyReorgOpParam(cnmlReorgOpParam_t *param);

CNML_DLL_API cnmlStatus_t cnmlCreateReorgOp(cnmlBaseOp_t *op,
                                            cnmlReorgOpParam_t param,
                                            cnmlTensor_t input_tensor,
                                            cnmlTensor_t output_tensor);

CNML_DLL_API cnmlStatus_t cnmlEnableReorgOpFix8Mode(cnmlBaseOp_t op,
                                                    int input_position,
                                                    float input_scale);

CNML_DLL_API cnmlStatus_t cnmlEnableReorgOpInt8Mode(cnmlBaseOp_t op,
                                                    int input_position,
                                                    float input_scale);

CNML_DLL_API cnmlStatus_t cnmlComputeReorgOpForward_V3(cnmlBaseOp_t op,
                                                       void *input,
                                                       void *output,
                                                       cnrtInvokeFuncParam_t *compute_forw_param,
                                                       cnrtQueue_t queue);
/* reorg operation end */

/* yuv_to_rgb operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateYUVtoRGBOp(cnmlBaseOp_t *op,
                                               cnmlTensor_t input,
                                               cnmlTensor_t output,
                                               cnmlYuvType_t yuv_type,
                                               cnmlRgbType_t rgb_type);

CNML_DLL_API cnmlStatus_t cnmlComputeYUVtoRGBOpForward_V3(cnmlBaseOp_t op,
                                                          void *input,
                                                          void *output,
                                                          cnrtInvokeFuncParam_t *compute_forw_param,
                                                          cnrtQueue_t queue);

CNML_DLL_API cnmlStatus_t cnmlCreateYUVtoGrayOp(cnmlBaseOp_t *op,
                                                cnmlTensor_t input,
                                                cnmlTensor_t output,
                                                cnmlYuvType_t yuv_type);

CNML_DLL_API cnmlStatus_t
cnmlComputeYUVtoGrayOpForward_V3(cnmlBaseOp_t op,
                                 void *input,
                                 void *output,
                                 cnrtInvokeFuncParam_t *compute_forw_param,
                                 cnrtQueue_t queue);
/* yuv_to_rgb operation end */

/* reshape operation start */
struct cnmlReshapeOpParam;
typedef cnmlReshapeOpParam *cnmlReshapeOpParam_t;
/*
 * when create cpu tensor, the DataOrder is up to you.
 * when create mlu tensor, the DataOrder always is NHWC.
 * no, co, ho, wo is the output tensor shape.
 * df in reshape param is the same as cpu tensor DataOrder.
 */
CNML_DLL_API cnmlStatus_t cnmlCreateReshapeOpParam(cnmlReshapeOpParam_t *param,
                                                   int no,
                                                   int co,
                                                   int ho,
                                                   int wo,
                                                   cnmlDataOrder_t df);
CNML_DLL_API cnmlStatus_t cnmlDestroyReshapeOpParam(cnmlReshapeOpParam_t *param);

CNML_DLL_API cnmlStatus_t cnmlCreateReshapeOp(cnmlBaseOp_t *op,
                                              cnmlReshapeOpParam_t param,
                                              cnmlTensor_t input,
                                              cnmlTensor_t output);

CNML_DLL_API cnmlStatus_t cnmlComputeReshapeOpForward_V3(cnmlBaseOp_t op,
                                                         void *input,
                                                         void *output,
                                                         cnrtInvokeFuncParam_t *compute_forw_param,
                                                         cnrtQueue_t queue);
/* reshape operation end */

/* space2batch operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateSpace2batchOp(cnmlBaseOp_t *op,
                                                  int w_block_size,
                                                  int h_block_size,
                                                  cnmlTensor_t input,
                                                  cnmlTensor_t output);

CNML_DLL_API cnmlStatus_t
cnmlComputeSpace2batchOpForward_V3(cnmlBaseOp_t op,
                                   void *input,
                                   void *output,
                                   cnrtInvokeFuncParam_t *compute_forw_param,
                                   cnrtQueue_t queue);
/* space2batch operation end */

/* batch2space operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateBatch2spaceOp(cnmlBaseOp_t *op,
                                                  int w_block_size,
                                                  int h_block_size,
                                                  cnmlTensor_t input,
                                                  cnmlTensor_t output);

CNML_DLL_API cnmlStatus_t
cnmlComputeBatch2spaceOpForward_V3(cnmlBaseOp_t op,
                                   void *input,
                                   void *output,
                                   cnrtInvokeFuncParam_t *compute_forw_param,
                                   cnrtQueue_t queue);
/* batch2space operation end */

/* unpool operation start */
struct cnmlUnpoolOpParam;
typedef cnmlUnpoolOpParam *cnmlUnpoolOpParam_t;

CNML_DLL_API cnmlStatus_t cnmlCreateUnpoolOpParam(cnmlUnpoolOpParam_t *param,
                                                  int window_height,
                                                  int window_width,
                                                  int stride_height,
                                                  int stride_width,
                                                  cnmlUnpoolMode_t unpool_mode);

CNML_DLL_API cnmlStatus_t cnmlDestroyUnpoolOpParam(cnmlUnpoolOpParam_t *param);

CNML_DLL_API cnmlStatus_t cnmlCreateUnpoolOp(cnmlBaseOp_t *op,
                                             cnmlTensor_t input,
                                             cnmlTensor_t index,
                                             cnmlTensor_t output,
                                             cnmlUnpoolOpParam_t unpool_param);

CNML_DLL_API cnmlStatus_t cnmlComputeUnpoolOpForward_V3(cnmlBaseOp_t op,
                                                        void *input,
                                                        void *index,
                                                        void *output,
                                                        cnrtInvokeFuncParam_t *compute_forw_param,
                                                        cnrtQueue_t queue);
/* unpool operation end */

/* top_k operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateTopkOp(cnmlBaseOp_t *op,
                                           int k,
                                           cnmlTensor_t input,
                                           cnmlTensor_t output,
                                           cnmlTensor_t index,
                                           cnmlDimension_t ch);

CNML_DLL_API cnmlStatus_t cnmlComputeTopkOpForward_V3(cnmlBaseOp_t op,
                                                      void *input,
                                                      void *output,
                                                      void *index,
                                                      cnrtInvokeFuncParam_t *compute_forw_param,
                                                      cnrtQueue_t queue);
/* top_k operation end */

/* where operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateWhereOp(cnmlBaseOp_t *op,
                                            cnmlTensor_t input,
                                            cnmlTensor_t output,
                                            cnmlTensor_t count);

CNML_DLL_API cnmlStatus_t cnmlComputeWhereOpForward_V3(cnmlBaseOp_t op,
                                                       void *input,
                                                       void *output,
                                                       void *count,
                                                       cnrtInvokeFuncParam_t *compute_forw_param,
                                                       cnrtQueue_t queue);
/* where operation end */

/* ssd_detection_output operation start */
struct cnmlSsdDetectionOutputOpParam;
typedef cnmlSsdDetectionOutputOpParam *cnmlSsdDetectionOutputOpParam_t;

CNML_DLL_API cnmlStatus_t
cnmlCreateSsdDetectionOutputOpParam(cnmlSsdDetectionOutputOpParam_t *param,
                                    int num_classes,
                                    bool share_location,
                                    int background_label_id,
                                    int code_type,
                                    bool variance_encoded_in_target,
                                    float confidence_threshold,
                                    float nms_threshold,
                                    int nms_topk,
                                    int keep_topk,
                                    cnmlDataOrder_t input_layout);

CNML_DLL_API cnmlStatus_t
cnmlDestroySsdDetectionOutputOpParam(cnmlSsdDetectionOutputOpParam_t *param);

CNML_DLL_API cnmlStatus_t cnmlCreateSsdDetectionOutputOp(cnmlBaseOp_t *op,
                                                         cnmlTensor_t pred_loc,
                                                         cnmlTensor_t conf,
                                                         cnmlTensor_t priors,
                                                         cnmlTensor_t output,
                                                         cnmlSsdDetectionOutputOpParam_t param);

CNML_DLL_API cnmlStatus_t
cnmlComputeSsdDetectionOutputOpForward_V3(cnmlBaseOp_t op,
                                          void *pred_loc,
                                          void *conf,
                                          void *priors,
                                          void *output,
                                          cnrtInvokeFuncParam_t *compute_forw_param,
                                          cnrtQueue_t queue);
/* ssd_detection_output operation end */

/* ssd_detection operation start */
struct cnmlSsdDetectionOpParam;
typedef cnmlSsdDetectionOpParam *cnmlSsdDetectionOpParam_t;

CNML_DLL_API cnmlStatus_t cnmlCreateSsdDetectionOpParam(cnmlSsdDetectionOpParam_t *param,
                                                        int num_classes,
                                                        bool share_location,
                                                        int background_label_id,
                                                        int code_type,
                                                        bool variance_encoded_in_target,
                                                        float confidence_threshold,
                                                        float nms_threshold,
                                                        int nms_topk,
                                                        int keep_topk,
                                                        cnmlDataOrder_t input_layout);

CNML_DLL_API cnmlStatus_t cnmlDestroySsdDetectionOpParam(cnmlSsdDetectionOpParam_t *param);

CNML_DLL_API cnmlStatus_t cnmlCreateSsdDetectionOpV2(cnmlBaseOp_t *op,
                                                     cnmlTensor_t *locs,
                                                     cnmlTensor_t *confs,
                                                     int num,
                                                     cnmlTensor_t priors,
                                                     cnmlTensor_t output,
                                                     cnmlSsdDetectionOpParam_t param);

CNML_DLL_API cnmlStatus_t
cnmlComputeSsdDetectionOpForward_V3(cnmlBaseOp_t op,
                                    void *locs[],
                                    void *confs[],
                                    int num,
                                    void *priors,
                                    void *output,
                                    cnrtInvokeFuncParam_t *compute_forw_param,
                                    cnrtQueue_t queue);
/* ssd_detection operation end */

/* ssd_detection_pose_output operation start */
struct cnmlSsdDetectionPoseOutputOpParam;
typedef cnmlSsdDetectionPoseOutputOpParam *cnmlSsdDetectionPoseOutputOpParam_t;

CNML_DLL_API cnmlStatus_t
cnmlCreateSsdDetectionPoseOutputOpParam(cnmlSsdDetectionPoseOutputOpParam_t *param,
                                        int num_classes,
                                        bool share_location,
                                        int background_label_id,
                                        int code_type,
                                        bool variance_encoded_in_target,
                                        float confidence_threshold,
                                        float nms_threshold,
                                        int nms_topk,
                                        int keep_topk,
                                        bool share_pose,
                                        int num_poses,
                                        cnmlDataOrder_t input_layout);

CNML_DLL_API cnmlStatus_t
cnmlDestroySsdDetectionPoseOutputOpParam(cnmlSsdDetectionPoseOutputOpParam_t *param);

CNML_DLL_API cnmlStatus_t
cnmlCreateSsdDetectionPoseOutputOp(cnmlBaseOp_t *op,
                                   cnmlTensor_t pred_loc,
                                   cnmlTensor_t conf,
                                   cnmlTensor_t pose,
                                   cnmlTensor_t priors,
                                   cnmlTensor_t output,
                                   cnmlSsdDetectionPoseOutputOpParam_t param);

CNML_DLL_API cnmlStatus_t
cnmlComputeSsdDetectionPoseOutputOpForward_V3(cnmlBaseOp_t op,
                                              void *pred_loc,
                                              void *conf,
                                              void *pose,
                                              void *priors,
                                              void *output,
                                              cnrtInvokeFuncParam_t *compute_forw_param,
                                              cnrtQueue_t queue);
/* ssd_detection_pose_output operation end */

/* ssd_detection_pose operation start */
struct cnmlSsdDetectionPoseOpParam;
typedef cnmlSsdDetectionPoseOpParam *cnmlSsdDetectionPoseOpParam_t;

CNML_DLL_API cnmlStatus_t cnmlCreateSsdDetectionPoseOpParam(cnmlSsdDetectionPoseOpParam_t *param,
                                                            int num_classes,
                                                            bool share_location,
                                                            int background_label_id,
                                                            int code_type,
                                                            bool variance_encoded_in_target,
                                                            float confidence_threshold,
                                                            float nms_threshold,
                                                            int nms_topk,
                                                            int keep_topk,
                                                            bool share_pose,
                                                            int num_poses,
                                                            cnmlDataOrder_t input_layout);

CNML_DLL_API cnmlStatus_t cnmlDestroySsdDetectionPoseOpParam(cnmlSsdDetectionPoseOpParam_t *param);

CNML_DLL_API cnmlStatus_t cnmlCreateSsdDetectionPoseOp(cnmlBaseOp_t *op,
                                                       cnmlTensor_t *locs,
                                                       cnmlTensor_t *confs,
                                                       cnmlTensor_t *poses,
                                                       int num,
                                                       cnmlTensor_t priors,
                                                       cnmlTensor_t output,
                                                       cnmlSsdDetectionPoseOpParam_t param);

CNML_DLL_API cnmlStatus_t
cnmlComputeSsdDetectionPoseOpForward_V3(cnmlBaseOp_t op,
                                        void *locs[],
                                        void *confs[],
                                        void *poses[],
                                        int num,
                                        void *priors,
                                        void *output,
                                        cnrtInvokeFuncParam_t *compute_forw_param,
                                        cnrtQueue_t queue);
/* ssd_detection_pose operation end */

/* roialign operation start */
struct cnmlRoiAlignOpParam;
typedef cnmlRoiAlignOpParam *cnmlRoiAlignOpParam_t;

CNML_DLL_API cnmlStatus_t cnmlCreateRoiAlignOpParam(cnmlRoiAlignOpParam_t *param,
                                                    int pooled_w,
                                                    int pooled_h,
                                                    float scale,
                                                    float sampling_ratio);

CNML_DLL_API cnmlStatus_t cnmlDestroyRoiAlignOpParam(cnmlRoiAlignOpParam_t *param);

CNML_DLL_API cnmlStatus_t cnmlCreateRoiAlignOp(cnmlBaseOp_t *op,
                                               cnmlRoiAlignOpParam_t param,
                                               cnmlTensor_t input,
                                               cnmlTensor_t rois,
                                               cnmlTensor_t output);

CNML_DLL_API cnmlStatus_t cnmlComputeRoiAlignOpForward_V3(cnmlBaseOp_t op,
                                                          void *input,
                                                          void *rois,
                                                          void *output,
                                                          cnrtInvokeFuncParam_t *compute_forw_param,
                                                          cnrtQueue_t queue);
/* roialign operation end */

/* Fractional pool operation start */
struct cnmlFractionalPoolOpParam;
typedef cnmlFractionalPoolOpParam *cnmlFractionalPoolOpParam_t;

CNML_DLL_API cnmlStatus_t cnmlCreateFractionalPoolOpParam(cnmlFractionalPoolOpParam_t *param,
                                                          cnmlPoolMode_t mode,
                                                          int *row_sequence,
                                                          int row_num,
                                                          int *col_sequence,
                                                          int col_num,
                                                          bool overlapping);

CNML_DLL_API cnmlStatus_t cnmlDestroyFractionalPoolOpParam(cnmlFractionalPoolOpParam_t *param);

CNML_DLL_API cnmlStatus_t cnmlCreateFractionalPoolOp(cnmlBaseOp_t *op,
                                                     cnmlFractionalPoolOpParam_t param,
                                                     cnmlTensor_t input,
                                                     cnmlTensor_t output);

CNML_DLL_API cnmlStatus_t
cnmlComputeFractionalPoolOpForward_V3(cnmlBaseOp_t op,
                                      void *input,
                                      void *output,
                                      cnrtInvokeFuncParam_t *compute_forw_param,
                                      cnrtQueue_t queue);
/* Fractional pool operation end */

/* Log Softmax operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateLogSoftmaxOp(cnmlBaseOp_t *op,
                                                 cnmlTensor_t input,
                                                 cnmlTensor_t output,
                                                 cnmlDimension_t dim);

CNML_DLL_API cnmlStatus_t
cnmlComputeLogSoftmaxOpForward_V3(cnmlBaseOp_t op,
                                  void *input,
                                  void *output,
                                  cnrtInvokeFuncParam_t *compute_forw_param,
                                  cnrtQueue_t queue);
/* Log Softmax operation end */

/* roi pooling operation start */
struct cnmliPsRoiPoolOpParam;
typedef struct cnmlPsRoiPoolOpParam *cnmlPsRoiPoolOpParam_t;

CNML_DLL_API cnmlStatus_t cnmlCreatePsRoiPoolOpParam(cnmlPsRoiPoolOpParam_t *param,
                                                     int pooled_w,
                                                     int pooled_h,
                                                     float spatial_scale,
                                                     int roi_num,
                                                     int output_dim);
CNML_DLL_API cnmlStatus_t cnmlDestroyPsRoiPoolOpParam(cnmlPsRoiPoolOpParam_t *param);

CNML_DLL_API cnmlStatus_t cnmlCreatePsRoiPoolOp(cnmlBaseOp_t *op,
                                                cnmlPsRoiPoolOpParam_t param,
                                                cnmlTensor_t input_tensor,
                                                cnmlTensor_t rois,
                                                cnmlTensor_t output_tensor);

CNML_DLL_API cnmlStatus_t
cnmlComputePsRoiPoolOpForward_V3(cnmlBaseOp_t op,
                                 void *input,
                                 void *rois,
                                 void *output,
                                 cnrtInvokeFuncParam_t *compute_forw_param,
                                 cnrtQueue_t queue);
/* roi pooling operation end */

/* Strided Slice operation start */
struct cnmlStridedSliceOpParam;
typedef cnmlStridedSliceOpParam *cnmlStridedSliceOpParam_t;

CNML_DLL_API cnmlStatus_t cnmlCreateStridedSliceOpParam(cnmlStridedSliceOpParam_t *param,
                                                        int nb,
                                                        int cb,
                                                        int hb,
                                                        int wb,
                                                        int ne,
                                                        int ce,
                                                        int he,
                                                        int we,
                                                        int ns,
                                                        int cs,
                                                        int hs,
                                                        int ws);

CNML_DLL_API cnmlStatus_t cnmlDestroyStridedSliceOpParam(cnmlStridedSliceOpParam_t *param);

CNML_DLL_API cnmlStatus_t cnmlCreateStridedSliceOp(cnmlBaseOp_t *op,
                                                   cnmlStridedSliceOpParam_t param,
                                                   cnmlTensor_t input,
                                                   cnmlTensor_t output);

CNML_DLL_API cnmlStatus_t
cnmlComputeStridedSliceOpForward_V3(cnmlBaseOp_t op,
                                    void *input,
                                    void *output,
                                    cnrtInvokeFuncParam_t *compute_forw_param,
                                    cnrtQueue_t queue);
/* Strided Slice operation end */

/* Yolo Detect operation start */
struct cnmlYoloDetectOpParam;
typedef cnmlYoloDetectOpParam *cnmlYoloDetectOpParam_t;

CNML_DLL_API cnmlStatus_t cnmlCreateYoloDetectOpParam(cnmlYoloDetectOpParam_t *param,
                                                      int side,
                                                      int num_class,
                                                      int num_box,
                                                      float confidence_threshold,
                                                      float nms_thresh,
                                                      float *biases,
                                                      int bias_num);

CNML_DLL_API cnmlStatus_t cnmlDestroyYoloDetectOpParam(cnmlYoloDetectOpParam_t *param);

CNML_DLL_API cnmlStatus_t cnmlCreateYoloDetectOp(cnmlBaseOp_t *op,
                                                 cnmlYoloDetectOpParam_t param,
                                                 cnmlTensor_t input,
                                                 cnmlTensor_t output);

CNML_DLL_API cnmlStatus_t
cnmlComputeYoloDetectOpForward_V3(cnmlBaseOp_t op,
                                  void *input,
                                  void *output,
                                  cnrtInvokeFuncParam_t *compute_forw_param,
                                  cnrtQueue_t queue);
/* Yolo Detect operation end */

/* axpy start */
typedef struct cnmlAxpyOpParam *cnmlAxpyOpParam_t;

CNML_DLL_API cnmlStatus_t cnmlCreateAxpyOp(cnmlBaseOp_t *op,
                                           cnmlTensor_t input_a_tensor,
                                           cnmlTensor_t input_x_tensor,
                                           cnmlTensor_t input_y_tensor,
                                           cnmlTensor_t outputs);

CNML_DLL_API cnmlStatus_t cnmlComputeAxpyOpForward_V3(cnmlBaseOp_t op,
                                                      void *input_a,
                                                      void *input_x,
                                                      void *input_y,
                                                      void *outputs,
                                                      cnrtInvokeFuncParam_t *compute_forw_param,
                                                      cnrtQueue_t queue);
/* axpy end */

/* reduce sum operation start  */
CNML_DLL_API cnmlStatus_t cnmlCreateReduceSumOp(cnmlBaseOp_t *op,
                                                cnmlDimension_t mode,
                                                cnmlTensor_t input,
                                                cnmlTensor_t output);

CNML_DLL_API cnmlStatus_t
cnmlComputeReduceSumOpForward_V3(cnmlBaseOp_t op,
                                 void *input,
                                 void *output,
                                 cnrtInvokeFuncParam_t *compute_forw_param,
                                 cnrtQueue_t queue);
/* reduce sum operation end   */

/* reduce mean operation start  */
CNML_DLL_API cnmlStatus_t cnmlCreateReduceMeanOp(cnmlBaseOp_t *op,
                                                 cnmlDimension_t mode,
                                                 cnmlTensor_t input,
                                                 cnmlTensor_t output);

CNML_DLL_API cnmlStatus_t
cnmlComputeReduceMeanOpForward_V3(cnmlBaseOp_t op,
                                  void *input,
                                  void *output,
                                  cnrtInvokeFuncParam_t *compute_forw_param,
                                  cnrtQueue_t queue);
/* reduce mean operation end   */

/* sin operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateSinOp(cnmlBaseOp_t *op,
                                          cnmlTensor_t input,
                                          cnmlTensor_t output);

CNML_DLL_API cnmlStatus_t cnmlComputeSinOpForward_V3(cnmlBaseOp_t op,
                                                     void *input,
                                                     void *output,
                                                     cnrtInvokeFuncParam_t *compute_forw_param,
                                                     cnrtQueue_t queue);
/* sin operation end */

/* cos operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateCosOp(cnmlBaseOp_t *op,
                                          cnmlTensor_t input,
                                          cnmlTensor_t output);

CNML_DLL_API cnmlStatus_t cnmlComputeCosOpForward_V3(cnmlBaseOp_t op,
                                                     void *input,
                                                     void *output,
                                                     cnrtInvokeFuncParam_t *compute_forw_param,
                                                     cnrtQueue_t queue);
/* cos operation end */

/* threshold operation start */
CNML_DLL_API cnmlStatus_t cnmlCreateThrsOp(cnmlBaseOp_t *op,
                                           float threshold,
                                           cnmlTensor_t input_tensor,
                                           cnmlTensor_t output_tensor);

CNML_DLL_API cnmlStatus_t cnmlComputeThrsOpForward_V3(cnmlBaseOp_t op,
                                                      void *input,
                                                      void *output,
                                                      cnrtInvokeFuncParam_t *compute_forw_param,
                                                      cnrtQueue_t queue);
/* threshold operation end */

////////////////////// cpu operation /////////////////////
cnmlStatus_t cnmlCpuComputePoolOpForward(cnmlPoolOpParam_t param,
                                         cnmlCpuTensor_t input_tensor,
                                         void *input,
                                         cnmlCpuTensor_t output_tensor,
                                         void *output);

cnmlStatus_t cnmlCpuComputeL2_PoolOpForward(cnmlPoolOpParam_t param,
                                            cnmlCpuTensor_t input_tensor,
                                            void *input,
                                            cnmlCpuTensor_t output_tensor,
                                            void *output);

cnmlStatus_t cnmlCpuComputeRoiAlignOpForward(cnmlRoiAlignOpParam_t param,
                                             cnmlCpuTensor_t input_tensor,
                                             void *input,
                                             cnmlCpuTensor_t rois_tensor,
                                             void *rois,
                                             cnmlCpuTensor_t output_tensor,
                                             void *output);

cnmlStatus_t cnmlCpuComputePsRoiPoolOpForward(cnmlPsRoiPoolOpParam_t param,
                                              cnmlCpuTensor_t input_tensor,
                                              void *input,
                                              cnmlCpuTensor_t rois_tensor,
                                              void *rois,
                                              cnmlCpuTensor_t output_tensor,
                                              void *output);

cnmlStatus_t cnmlCpuComputeAddPadOpForward(cnmlAddPadOpParam_t param,
                                           cnmlCpuTensor_t input_tensor,
                                           void *input,
                                           cnmlCpuTensor_t output_tensor,
                                           void *output);

cnmlStatus_t cnmlCpuComputeAddPadChannelOpForward(cnmlAddPadChannelOpParam_t param,
                                                  cnmlCpuTensor_t input_tensor,
                                                  void *input,
                                                  cnmlCpuTensor_t output_tensor,
                                                  void *output);

cnmlStatus_t cnmlCpuComputeNormalizeOpForward(cnmlNormalizeOpParam_t param,
                                              cnmlCpuTensor_t input_tensor,
                                              void *input,
                                              cnmlCpuTensor_t output_tensor,
                                              void *output,
                                              cnmlCpuTensor_t scale_tensor,
                                              void *scale);

cnmlStatus_t cnmlCpuComputeGrepChannelOpForward(cnmlGrepChannelOpParam_t param,
                                                cnmlCpuTensor_t input_tensor,
                                                void *input,
                                                cnmlCpuTensor_t output_tensor,
                                                void *output);

cnmlStatus_t cnmlCpuComputeScaleOpForward(cnmlCpuTensor_t input_tensor,
                                          void *input,
                                          cnmlCpuTensor_t output_tensor,
                                          void *output,
                                          cnmlCpuTensor_t alpha_tensor,
                                          void *alpha,
                                          cnmlCpuTensor_t beta_tensor,
                                          void *beta);

cnmlStatus_t cnmlCpuComputeSubOpForward(cnmlCpuTensor_t input_tensor_1,
                                        void *input_1,
                                        cnmlCpuTensor_t input_tensor_2,
                                        void *input_2,
                                        cnmlCpuTensor_t output_tensor,
                                        void *output);

cnmlStatus_t cnmlCpuComputeMultOpForward(cnmlCpuTensor_t input_tensor_1,
                                         void *input_1,
                                         cnmlCpuTensor_t input_tensor_2,
                                         void *input_2,
                                         cnmlCpuTensor_t output_tensor,
                                         void *output);

cnmlStatus_t cnmlCpuComputeSplitOpForward(cnmlSplitOpParam_t param,
                                          cnmlCpuTensor_t *inputs_tensor_ptr,
                                          void *inputs[],
                                          int input_num,
                                          cnmlCpuTensor_t *outputs_tensor_ptr,
                                          void *outputs[],
                                          int output_num);
cnmlStatus_t cnmlCpuComputeShuffleChannelOpForward(cnmlCpuTensor_t *inputs_tensor_ptr,
                                                   void *inputs[],
                                                   cnmlCpuTensor_t *outputs_tensor_ptr,
                                                   void *outputs[],
                                                   int group);

cnmlStatus_t cnmlCpuComputeConcatOpForward(cnmlConcatOpParam_t param,
                                           cnmlCpuTensor_t *inputs_tensor_ptr,
                                           void *inputs[],
                                           int input_num,
                                           cnmlCpuTensor_t *outputs_tensor_ptr,
                                           void *outputs[],
                                           int output_num);

cnmlStatus_t cnmlCpuComputeAddOpForward(cnmlCpuTensor_t input_tensor_1,
                                        void *input_1,
                                        cnmlCpuTensor_t input_tensor_2,
                                        void *input_2,
                                        cnmlCpuTensor_t output_tensor,
                                        void *output);

cnmlStatus_t cnmlCpuComputeCoeffAddOpForward(cnmlCpuTensor_t input_tensor_1,
                                             void *input_1,
                                             cnmlCpuTensor_t input_tensor_2,
                                             void *input_2,
                                             cnmlCpuTensor_t output_tensor,
                                             void *output,
                                             float coeff1,
                                             float coeff2);

cnmlStatus_t cnmlCpuComputeRealDivOpForward(cnmlCpuTensor_t input_tensor_1,
                                            void *input_1,
                                            cnmlCpuTensor_t input_tensor_2,
                                            void *input_2,
                                            cnmlCpuTensor_t output_tensor,
                                            void *output);

cnmlStatus_t cnmlCpuComputeBroadcastOpForward(cnmlCpuTensor_t input_tensor,
                                              void *input,
                                              cnmlCpuTensor_t output_tensor,
                                              void *output);

cnmlStatus_t cnmlCpuComputeBroadcastAddOpForward(cnmlCpuTensor_t input_tensor_1,
                                                 void *input_1,
                                                 cnmlCpuTensor_t input_tensor_2,
                                                 void *input_2,
                                                 cnmlCpuTensor_t output_tensor,
                                                 void *output);

cnmlStatus_t cnmlCpuComputeBroadcastSubOpForward(cnmlCpuTensor_t input_tensor_1,
                                                 void *input_1,
                                                 cnmlCpuTensor_t input_tensor_2,
                                                 void *input_2,
                                                 cnmlCpuTensor_t output_tensor,
                                                 void *output);

cnmlStatus_t cnmlCpuComputeBroadcastMultOpForward(cnmlCpuTensor_t input_tensor_1,
                                                  void *input_1,
                                                  cnmlCpuTensor_t input_tensor_2,
                                                  void *input_2,
                                                  cnmlCpuTensor_t output_tensor,
                                                  void *output);

cnmlStatus_t cnmlCpuComputeActiveOpForward(cnmlActiveFunction_t fn,
                                           cnmlCpuTensor_t input_tensor,
                                           void *input,
                                           cnmlCpuTensor_t output_tensor,
                                           void *output);

cnmlStatus_t cnmlCpuComputeMlpOpForward(cnmlCpuTensor_t input_tensor,
                                        void *input,
                                        cnmlCpuTensor_t output_tensor,
                                        void *output,
                                        cnmlCpuTensor_t filter_tensor,
                                        void *filter,
                                        cnmlCpuTensor_t bias_tensor,
                                        void *bias);

cnmlStatus_t cnmlCpuComputeMatrixMultOpForward(cnmlCpuTensor_t input_tensor,
                                               void *input,
                                               cnmlCpuTensor_t filter_tensor,
                                               void *filter,
                                               cnmlCpuTensor_t output_tensor,
                                               void *output,
                                               cnmlCpuTensor_t bias_tensor,
                                               void *bias);

cnmlStatus_t cnmlCpuComputeGatherV2OpForward(cnmlCpuTensor_t input_tensor,
                                             void *input,
                                             cnmlCpuTensor_t index_tensor,
                                             void *index,
                                             cnmlCpuTensor_t output_tensor,
                                             void *output,
                                             cnmlDimension_t axies);

cnmlStatus_t cnmlCpuComputeBatchDotOpForward(cnmlCpuTensor_t input_tensor_1,
                                             void *input_1,
                                             cnmlCpuTensor_t input_tensor_2,
                                             void *input_2,
                                             cnmlCpuTensor_t output_tensor,
                                             void *output,
                                             bool trans_a,
                                             bool trans_b);

cnmlStatus_t cnmlCpuComputeXwPlusBOpForward(cnmlCpuTensor_t input_tensor,
                                            void *input,
                                            cnmlCpuTensor_t output_tensor,
                                            void *output,
                                            cnmlCpuTensor_t filter_tensor,
                                            void *filter,
                                            cnmlCpuTensor_t bias_tensor,
                                            void *bias);

cnmlStatus_t cnmlCpuComputeBasicRNNOpForward(cnmlCpuTensor_t input_tensor,
                                             void *input,
                                             cnmlCpuTensor_t output_tensor,
                                             void *output,
                                             cnmlCpuTensor_t weight_tensor,
                                             void *weight,
                                             cnmlCpuTensor_t state_input_tensor,
                                             void *state_input,
                                             cnmlCpuTensor_t state_output_tensor,
                                             void *state_output,
                                             cnmlCpuTensor_t state_weight_tensor,
                                             void *state_weight,
                                             cnmlCpuTensor_t bias_tensor,
                                             void *bias,
                                             cnmlActiveFunction_t active_func);

cnmlStatus_t cnmlCpuComputeConvGroupOpForward(cnmlConvOpParam_t param,
                                              cnmlCpuTensor_t input_tensor,
                                              void *input,
                                              cnmlCpuTensor_t filter_tensor,
                                              void *filter,
                                              cnmlCpuTensor_t output_tensor,
                                              void *output,
                                              cnmlCpuTensor_t bias_tensor,
                                              void *bias,
                                              int group);

cnmlStatus_t cnmlCpuComputeDeconvOpForward(cnmlDeconvOpParam_t param,
                                           cnmlCpuTensor_t input_tensor,
                                           void *input,
                                           cnmlCpuTensor_t filter_tensor,
                                           void *filter,
                                           cnmlCpuTensor_t output_tensor,
                                           void *output,
                                           cnmlCpuTensor_t bias_tensor,
                                           void *bias);

cnmlStatus_t cnmlCpuComputeConvOpForward(cnmlConvOpParam_t param,
                                         cnmlCpuTensor_t input_tensor,
                                         void *input,
                                         cnmlCpuTensor_t filter_tensor,
                                         void *filter,
                                         cnmlCpuTensor_t output_tensor,
                                         void *output,
                                         cnmlCpuTensor_t bias_tensor,
                                         void *bias);

cnmlStatus_t cnmlCpuComputeConvFirstOpForward(cnmlConvFirstOpParam_t param,
                                              cnmlCpuTensor_t input_tensor,
                                              void *input,
                                              cnmlCpuTensor_t mean_tensor,
                                              void *mean,
                                              cnmlCpuTensor_t filter_tensor,
                                              void *filter,
                                              cnmlCpuTensor_t output_tensor,
                                              void *output,
                                              cnmlCpuTensor_t bias_tensor,
                                              void *bias,
                                              cnmlCpuTensor_t stdt_tensor,
                                              void *stdt);

cnmlStatus_t cnmlCpuComputeConvDepthwiseOpForward(cnmlConvDepthwiseOpParam_t param,
                                                  cnmlCpuTensor_t input_tensor,
                                                  void *input,
                                                  cnmlCpuTensor_t filter_tensor,
                                                  void *filter,
                                                  cnmlCpuTensor_t output_tensor,
                                                  void *output,
                                                  cnmlCpuTensor_t bias_tensor,
                                                  void *bias);

cnmlStatus_t cnmlCpuComputeBatchNormOpForward(cnmlCpuTensor_t input_tensor,
                                              void *input_buf,
                                              cnmlCpuTensor_t output_tensor,
                                              void *output_buf,
                                              cnmlCpuTensor_t mean_tensor,
                                              void *mean_buf,
                                              cnmlCpuTensor_t var_tensor,
                                              void *var_buf);

cnmlStatus_t cnmlCpuComputeOrOpForward(cnmlCpuTensor_t input_tensor_1,
                                       void *input_1,
                                       cnmlCpuTensor_t input_tensor_2,
                                       void *input_2,
                                       cnmlCpuTensor_t output_tensor,
                                       void *output);

cnmlStatus_t cnmlCpuComputeCycleOrOpForward(cnmlCpuTensor_t input_tensor_1,
                                            void *input_1,
                                            cnmlCpuTensor_t input_tensor_2,
                                            void *input_2,
                                            cnmlCpuTensor_t output_tensor,
                                            void *output);

cnmlStatus_t cnmlCpuComputeAndOpForward(cnmlCpuTensor_t input_tensor_1,
                                        void *input_1,
                                        cnmlCpuTensor_t input_tensor_2,
                                        void *input_2,
                                        cnmlCpuTensor_t output_tensor,
                                        void *output);

cnmlStatus_t cnmlCpuComputeCycleAndOpForward(cnmlCpuTensor_t input_tensor_1,
                                             void *input_1,
                                             cnmlCpuTensor_t input_tensor_2,
                                             void *input_2,
                                             cnmlCpuTensor_t output_tensor,
                                             void *output);

cnmlStatus_t cnmlCpuComputeXorOpForward(cnmlCpuTensor_t input_tensor_1,
                                        void *input_1,
                                        cnmlCpuTensor_t input_tensor_2,
                                        void *input_2,
                                        cnmlCpuTensor_t output_tensor,
                                        void *output);

cnmlStatus_t cnmlCpuComputeCycleXorOpForward(cnmlCpuTensor_t input_tensor_1,
                                             void *input_1,
                                             cnmlCpuTensor_t input_tensor_2,
                                             void *input_2,
                                             cnmlCpuTensor_t output_tensor,
                                             void *output);

cnmlStatus_t cnmlCpuComputeNotOpForward(cnmlCpuTensor_t input_tensor,
                                        void *input,
                                        cnmlCpuTensor_t output_tensor,
                                        void *output);

cnmlStatus_t cnmlCpuComputeCycleAddOpForward(cnmlCpuTensor_t input_tensor_1,
                                             void *input_1,
                                             cnmlCpuTensor_t input_tensor_2,
                                             void *input_2,
                                             cnmlCpuTensor_t output_tensor,
                                             void *output);

cnmlStatus_t cnmlCpuComputeCycleSubOpForward(cnmlCpuTensor_t input_tensor_1,
                                             void *input_1,
                                             cnmlCpuTensor_t input_tensor_2,
                                             void *input_2,
                                             cnmlCpuTensor_t output_tensor,
                                             void *output);

cnmlStatus_t cnmlCpuComputeCycleMultOpForward(cnmlCpuTensor_t input_tensor_1,
                                              void *input_1,
                                              cnmlCpuTensor_t input_tensor_2,
                                              void *input_2,
                                              cnmlCpuTensor_t output_tensor,
                                              void *output);

cnmlStatus_t cnmlCpuComputeMaxOpForward(cnmlCpuTensor_t input_tensor,
                                        void *input_buf,
                                        cnmlCpuTensor_t output_tensor,
                                        void *output_buf,
                                        cnmlCpuTensor_t index_tensor,
                                        void *index_buf);

cnmlStatus_t cnmlCpuComputeReduceMaxOpForward(cnmlReduceMaxMode_t mode,
                                              cnmlCpuTensor_t input_tensor,
                                              void *input_buf,
                                              cnmlCpuTensor_t output_tensor,
                                              void *output_buf);

cnmlStatus_t cnmlCpuComputeMinOpForward(cnmlCpuTensor_t input_tensor,
                                        void *input_buf,
                                        cnmlCpuTensor_t output_tensor,
                                        void *output_buf,
                                        cnmlCpuTensor_t index_tensor,
                                        void *index_buf);

cnmlStatus_t cnmlCpuComputeReverseOpForward(cnmlCpuTensor_t input_tensor,
                                            void *input_buf,
                                            cnmlCpuTensor_t output_tensor,
                                            void *output_buf,
                                            cnmlReverseAxis_t param);

cnmlStatus_t cnmlCpuComputeTransposeOpForward(cnmlCpuTensor_t input_tensor,
                                              void *input_buf,
                                              cnmlCpuTensor_t output_tensor,
                                              void *output_buf);

cnmlStatus_t cnmlCpuComputeTransposeProOpForward(cnmlCpuTensor_t input_tensor,
                                                 void *input_buf,
                                                 cnmlCpuTensor_t output_tensor,
                                                 void *output_buf,
                                                 cnmlTransposeOpParam_t param);

cnmlStatus_t cnmlCpuComputeInterpOpForward(cnmlCpuTensor_t input_tensor,
                                           void *input_buf,
                                           cnmlCpuTensor_t output_tensor,
                                           void *output_buf,
                                           cnmlInterpOpParam_t param);

cnmlStatus_t cnmlCpuComputeResizeOpForward(cnmlCpuTensor_t input_tensor,
                                           void *input_buf,
                                           cnmlCpuTensor_t output_tensor,
                                           void *output_buf,
                                           cnmlResizeOpParam_t param);

cnmlStatus_t cnmlCpuComputeLrnOpForward(cnmlLrnOpParam_t param,
                                        cnmlCpuTensor_t input_tensor,
                                        void *input_buf,
                                        cnmlCpuTensor_t output_tensor,
                                        void *output_buf);

cnmlStatus_t cnmlCpuComputeMaxTTOpForward(cnmlCpuTensor_t input_tensor_1,
                                          void *input_buf_1,
                                          cnmlCpuTensor_t input_tensor_2,
                                          void *input_buf_2,
                                          cnmlCpuTensor_t output_tensor,
                                          void *output_buf);

cnmlStatus_t cnmlCpuComputeUnpoolOpForward(cnmlUnpoolOpParam_t unpool_param,
                                           cnmlCpuTensor_t input_tensor,
                                           void *input_buf,
                                           cnmlCpuTensor_t index_tensor,
                                           void *index_buf,
                                           cnmlCpuTensor_t output_tensor,
                                           void *output_buf);

cnmlStatus_t cnmlCpuComputeMaxTCOpForward(cnmlCpuTensor_t input_tensor_1,
                                          void *input_buf_1,
                                          cnmlCpuTensor_t output_tensor,
                                          void *output_buf,
                                          float cc);

cnmlStatus_t cnmlCpuComputeMinTCOpForward(cnmlCpuTensor_t input_tensor,
                                          void *input_buf,
                                          cnmlCpuTensor_t output_tensor,
                                          void *output_buf,
                                          float cc);

cnmlStatus_t cnmlCpuComputeCycleEqualOpForward(cnmlCpuTensor_t input_tensor_1,
                                               void *input_1,
                                               cnmlCpuTensor_t input_tensor_2,
                                               void *input_2,
                                               cnmlCpuTensor_t output_tensor,
                                               void *output);

cnmlStatus_t cnmlCpuComputeCycleNEqualOpForward(cnmlCpuTensor_t input_tensor_1,
                                                void *input_1,
                                                cnmlCpuTensor_t input_tensor_2,
                                                void *input_2,
                                                cnmlCpuTensor_t output_tensor,
                                                void *output);

cnmlStatus_t cnmlCpuComputeCycleLessEqualOpForward(cnmlCpuTensor_t input_tensor_1,
                                                   void *input_1,
                                                   cnmlCpuTensor_t input_tensor_2,
                                                   void *input_2,
                                                   cnmlCpuTensor_t output_tensor,
                                                   void *output);

cnmlStatus_t cnmlCpuComputeCycleLessOpForward(cnmlCpuTensor_t input_tensor_1,
                                              void *input_1,
                                              cnmlCpuTensor_t input_tensor_2,
                                              void *input_2,
                                              cnmlCpuTensor_t output_tensor,
                                              void *output);

cnmlStatus_t cnmlCpuComputeCycleGreaterOpForward(cnmlCpuTensor_t input_tensor_1,
                                                 void *input_1,
                                                 cnmlCpuTensor_t input_tensor_2,
                                                 void *input_2,
                                                 cnmlCpuTensor_t output_tensor,
                                                 void *output);

cnmlStatus_t cnmlCpuComputeCycleGreaterEqualOpForward(cnmlCpuTensor_t input_tensor_1,
                                                      void *input_1,
                                                      cnmlCpuTensor_t input_tensor_2,
                                                      void *input_2,
                                                      cnmlCpuTensor_t output_tensor,
                                                      void *output);

cnmlStatus_t cnmlCpuComputeEqualOpForward(cnmlCpuTensor_t input_tensor_1,
                                          void *input_1,
                                          cnmlCpuTensor_t input_tensor_2,
                                          void *input_2,
                                          cnmlCpuTensor_t output_tensor,
                                          void *output);

cnmlStatus_t cnmlCpuComputeNEqualOpForward(cnmlCpuTensor_t input_tensor_1,
                                           void *input_1,
                                           cnmlCpuTensor_t input_tensor_2,
                                           void *input_2,
                                           cnmlCpuTensor_t output_tensor,
                                           void *output);

cnmlStatus_t cnmlCpuComputeLessEqualOpForward(cnmlCpuTensor_t input_tensor_1,
                                              void *input_1,
                                              cnmlCpuTensor_t input_tensor_2,
                                              void *input_2,
                                              cnmlCpuTensor_t output_tensor,
                                              void *output);

cnmlStatus_t cnmlCpuComputeLessOpForward(cnmlCpuTensor_t input_tensor_1,
                                         void *input_1,
                                         cnmlCpuTensor_t input_tensor_2,
                                         void *input_2,
                                         cnmlCpuTensor_t output_tensor,
                                         void *output);

cnmlStatus_t cnmlCpuComputeGreaterEqualOpForward(cnmlCpuTensor_t input_tensor_1,
                                                 void *input_1,
                                                 cnmlCpuTensor_t input_tensor_2,
                                                 void *input_2,
                                                 cnmlCpuTensor_t output_tensor,
                                                 void *output);

cnmlStatus_t cnmlCpuComputeGreaterOpForward(cnmlCpuTensor_t input_tensor_1,
                                            void *input_1,
                                            cnmlCpuTensor_t input_tensor_2,
                                            void *input_2,
                                            cnmlCpuTensor_t output_tensor,
                                            void *output);

cnmlStatus_t cnmlCpuComputeClipOpForward(cnmlCpuTensor_t input_tensor,
                                         void *input_buf,
                                         cnmlCpuTensor_t output_tensor,
                                         void *output_buf,
                                         double lower_bound,
                                         double upper_bound);

cnmlStatus_t cnmlCpuComputeNearestNeighborOpForward(cnmlCpuTensor_t input_tensor,
                                                    void *input_buf,
                                                    cnmlCpuTensor_t output_tensor,
                                                    void *output_buf,
                                                    cnmlNearestNeighborOpParam_t param);

cnmlStatus_t cnmlCpuComputePreluOpForward(cnmlCpuTensor_t input_tensor,
                                          void *input_buf,
                                          cnmlCpuTensor_t output_tensor,
                                          void *output_buf,
                                          cnmlCpuTensor_t prelu_param_tensor,
                                          void *prelu_param_buf);

cnmlStatus_t cnmlCpuComputeSqrtOpForward(cnmlCpuTensor_t input_tensor,
                                         void *input_buf,
                                         cnmlCpuTensor_t output_tensor,
                                         void *output_buf);

cnmlStatus_t cnmlCpuComputeExpOpForward(cnmlCpuTensor_t input_tensor,
                                        void *input_buf,
                                        cnmlCpuTensor_t output_tensor,
                                        void *output_buf);

cnmlStatus_t cnmlCpuComputeLogOpForward(cnmlCpuTensor_t input_tensor,
                                        void *input_buf,
                                        cnmlCpuTensor_t output_tensor,
                                        void *output_buf);

cnmlStatus_t cnmlCpuComputeRsqrtOpForward(cnmlCpuTensor_t input_tensor,
                                          void *input_buf,
                                          cnmlCpuTensor_t output_tensor,
                                          void *output_buf);

cnmlStatus_t cnmlCpuComputeFloorOpForward(cnmlCpuTensor_t input_tensor,
                                          void *input,
                                          cnmlCpuTensor_t output_tensor,
                                          void *output);

cnmlStatus_t cnmlCpuComputePowerOpForward(cnmlCpuTensor_t input_tensor,
                                          void *input_buf,
                                          cnmlCpuTensor_t output_tensor,
                                          void *output_buf,
                                          float power_c);

cnmlStatus_t cnmlCpuComputeDyadicSelectOpForward(cnmlCpuTensor_t input_tensor_1,
                                                 void *input_1,
                                                 cnmlCpuTensor_t input_tensor_2,
                                                 void *input_2,
                                                 cnmlCpuTensor_t input_tensor_3,
                                                 void *input_3,
                                                 cnmlCpuTensor_t output_tensor,
                                                 void *output);

cnmlStatus_t cnmlCpuComputeUnarySelectOpForward(cnmlCpuTensor_t input_tensor_1,
                                                void *input_1,
                                                cnmlCpuTensor_t input_tensor_2,
                                                void *input_2,
                                                cnmlCpuTensor_t output_cpu,
                                                void *output_cpu_ptr,
                                                cnmlCpuTensor_t count_tensor,
                                                void *count);

cnmlStatus_t cnmlCpuComputeSvdfOpForward(cnmlCpuTensor_t input_tensor,
                                         void *input_ptr,
                                         cnmlCpuTensor_t weights_feature,
                                         void *weights_feature_ptr,
                                         cnmlCpuTensor_t weights_time,
                                         void *weights_time_ptr,
                                         cnmlCpuTensor_t bias_tensor,
                                         void *bias_ptr,
                                         cnmlCpuTensor_t state_in,
                                         void *state_in_ptr,
                                         cnmlCpuTensor_t state_out,
                                         void *state_out_ptr,
                                         cnmlCpuTensor_t output_tensor,
                                         void *output_ptr);

cnmlStatus_t cnmlCpuComputeAbsOpForward(cnmlCpuTensor_t input_tensor,
                                        void *input_buf,
                                        cnmlCpuTensor_t output_tensor,
                                        void *output_buf);

cnmlStatus_t cnmlCpuComputeSoftplusOpForward(cnmlCpuTensor_t input_tensor,
                                             void *input_buf,
                                             cnmlCpuTensor_t output_tensor,
                                             void *output_buf);

cnmlStatus_t cnmlCpuComputeMinusOpForward(cnmlCpuTensor_t input_tensor,
                                          void *input_buf,
                                          cnmlCpuTensor_t output_tensor,
                                          void *output_buf);

cnmlStatus_t cnmlCpuComputeFakeQuantOpForward(cnmlCpuTensor_t input_tensor,
                                              void *input_buf,
                                              cnmlCpuTensor_t output_tensor,
                                              void *output_buf,
                                              float scale,
                                              int offset);

cnmlStatus_t cnmlCpuComputeAvgOpForward(cnmlCpuTensor_t input_tensor,
                                        void *input_buf,
                                        cnmlCpuTensor_t output_tensor,
                                        void *output_buf);

cnmlStatus_t cnmlCpuComputeVector2NormOpForward(cnmlCpuTensor_t input_tensor,
                                                void *input_buf,
                                                cnmlCpuTensor_t output_tensor,
                                                void *output_buf);

cnmlStatus_t cnmlCpuComputeDeviceMemcpyOpForward(cnmlCpuTensor_t input_tensor,
                                                 void *input,
                                                 cnmlCpuTensor_t output_tensor,
                                                 void *output);

cnmlStatus_t cnmlCpuComputeTransposeOpForward(cnmlCpuTensor_t input_tensor,
                                              void *input,
                                              cnmlCpuTensor_t output_tensor,
                                              void *output);

cnmlStatus_t cnmlCpuComputeCastOpForward(cnmlCastType_t fn,
                                         cnmlCpuTensor_t input_tensor,
                                         void *input,
                                         cnmlCpuTensor_t output_tensor,
                                         void *output);

cnmlStatus_t cnmlCpuComputeSoftmaxOpForward(cnmlSoftmaxDim_t dim,
                                            cnmlCpuTensor_t input_tensor,
                                            void *input_buf,
                                            cnmlCpuTensor_t output_tensor,
                                            void *output_buf);

cnmlStatus_t cnmlCpuComputeGrepOpForward(cnmlGrepOpParam_t param,
                                         cnmlCpuTensor_t input_tensor,
                                         void *input,
                                         cnmlCpuTensor_t output_tensor,
                                         void *output);

cnmlStatus_t cnmlCpuComputeArgmaxOpForward(cnmlArgmaxAxis_t argmax_mode,
                                           cnmlCpuTensor_t input_tensor,
                                           void *input,
                                           cnmlCpuTensor_t output_tensor,
                                           void *output);

cnmlStatus_t cnmlCpuComputeReorgOpForward(cnmlReorgOpParam_t param,
                                          cnmlCpuTensor_t input_tensor,
                                          void *input,
                                          cnmlCpuTensor_t output_tensor,
                                          void *output);

cnmlStatus_t cnmlCpuComputeYUVtoRGBOpForward(cnmlCpuTensor_t input_tensor,
                                             void *input_buf,
                                             cnmlCpuTensor_t output_tensor,
                                             void *output_buf,
                                             cnmlYuvType_t yuv_type,
                                             cnmlRgbType_t rgb_type);

cnmlStatus_t cnmlCpuComputeYUVtoGrayOpForward(cnmlCpuTensor_t input_tensor,
                                              void *input_buf,
                                              cnmlCpuTensor_t output_tensor,
                                              void *output_buf,
                                              cnmlYuvType_t yuv_type);

cnmlStatus_t cnmlCpuComputeReshapeOpForward(cnmlReshapeOpParam_t param,
                                            cnmlCpuTensor_t input_tensor,
                                            void *input,
                                            cnmlCpuTensor_t output_tensor,
                                            void *output);

cnmlStatus_t cnmlCpuComputeTopkOpForward(int k,
                                         cnmlCpuTensor_t input_tensor,
                                         void *input,
                                         cnmlCpuTensor_t output_tensor,
                                         void *output,
                                         cnmlCpuTensor_t index_tensor,
                                         void *index,
                                         cnmlDimension_t ch);

cnmlStatus_t cnmlCpuComputeWhereOpForward(cnmlCpuTensor_t input_tensor,
                                          void *input,
                                          cnmlCpuTensor_t output_tensor,
                                          void *output,
                                          cnmlCpuTensor_t count_tensor,
                                          void *count);

cnmlStatus_t cnmlCpuComputeSpace2batchOpForward(int w_block_size,
                                                int h_block_size,
                                                cnmlCpuTensor_t input_tensor,
                                                void *input,
                                                cnmlCpuTensor_t output_tensor,
                                                void *output);

cnmlStatus_t cnmlCpuComputeBatch2spaceOpForward(int w_block_size,
                                                int h_block_size,
                                                cnmlCpuTensor_t input_tensor,
                                                void *input,
                                                cnmlCpuTensor_t output_tensor,
                                                void *output);

cnmlStatus_t cnmlCpuComputeProposalOpForward(cnmlProposalOpParam_t param,
                                             cnmlCpuTensor_t bbox_deltas_tensor,
                                             void *bbox_deltas,
                                             cnmlCpuTensor_t scores_tensor,
                                             void *scores,
                                             cnmlCpuTensor_t output_tensor,
                                             void *output);

cnmlStatus_t cnmlCpuNmsOpGenerateRois(void *input, int num_box, int height, int width);

cnmlStatus_t cnmlCpuComputeNmsOpForward(cnmlNmsOpParam_t param,
                                        cnmlCpuTensor_t input_tensor,
                                        void *input,
                                        cnmlCpuTensor_t output_tensor,
                                        void *output);

cnmlStatus_t cnmlCpuComputeRoiPoolOpForward(cnmlRoiPoolOpParam_t param,
                                            cnmlCpuTensor_t input_tensor,
                                            void *input,
                                            cnmlCpuTensor_t rois_tensor,
                                            void *rois,
                                            cnmlCpuTensor_t output_tensor,
                                            void *output);

cnmlStatus_t cnmlCpuComputeFractionalPoolOpForward(cnmlFractionalPoolOpParam_t param,
                                                   cnmlCpuTensor_t input_tensor,
                                                   void *input,
                                                   cnmlCpuTensor_t output_tensor,
                                                   void *output);

cnmlStatus_t cnmlCpuComputeLogSoftmaxOpForward(cnmlDimension_t dim,
                                               cnmlCpuTensor_t input_tensor,
                                               void *input,
                                               cnmlCpuTensor_t output_tensor,
                                               void *output);

cnmlStatus_t cnmlCpuComputeStridedSliceOpForward(cnmlStridedSliceOpParam_t param,
                                                 cnmlCpuTensor_t input_tensor,
                                                 void *input,
                                                 cnmlCpuTensor_t output_tensor,
                                                 void *output);

cnmlStatus_t cnmlCpuComputeYoloDetectOpForward(cnmlYoloDetectOpParam_t param,
                                               cnmlCpuTensor_t input_tensor,
                                               void *input,
                                               cnmlCpuTensor_t output_tensor,
                                               void *output);

cnmlStatus_t cnmlCpuComputeImageDetectOpForward(cnmlImageDetectOpParam_t param,
                                                cnmlCpuTensor_t bbox_deltas_tensor,
                                                void *bbox_deltas_ptr,
                                                cnmlCpuTensor_t scores_tensor,
                                                void *scores_ptr,
                                                cnmlCpuTensor_t anchors_tensor,
                                                void *anchors_ptr,
                                                cnmlCpuTensor_t output_tensor,
                                                void *output_ptr);

cnmlStatus_t cnmlCpuComputeAxpyOpForward(cnmlCpuTensor_t input_tensor_1,
                                         void *input_1,
                                         cnmlCpuTensor_t input_tensor_2,
                                         void *input_2,
                                         cnmlCpuTensor_t input_tensor_3,
                                         void *input_3,
                                         cnmlCpuTensor_t output_tensor,
                                         void *output);

cnmlStatus_t cnmlCpuComputeReduceMeanOpForward(cnmlDimension_t mode,
                                               cnmlCpuTensor_t input_tensor,
                                               void *input_buf,
                                               cnmlCpuTensor_t output_tensor,
                                               void *output_buf);

cnmlStatus_t cnmlCpuComputeReduceSumOpForward(cnmlDimension_t mode,
                                              cnmlCpuTensor_t input_tensor,
                                              void *input_buf,
                                              cnmlCpuTensor_t output_tensor,
                                              void *output_buf);

cnmlStatus_t cnmlCpuComputeSinOpForward(cnmlCpuTensor_t input_tensor,
                                        void *input_buf,
                                        cnmlCpuTensor_t output_tensor,
                                        void *output_buf);

cnmlStatus_t cnmlCpuComputeCosOpForward(cnmlCpuTensor_t input_tensor,
                                        void *input_buf,
                                        cnmlCpuTensor_t output_tensor,
                                        void *output_buf);

cnmlStatus_t cnmlCpuComputeSvdfNNOpForward(cnmlCpuTensor_t input_tensor,
                                           void *input_ptr,
                                           cnmlCpuTensor_t weights_feature,
                                           void *weights_feature_ptr,
                                           cnmlCpuTensor_t weights_time,
                                           void *weights_time_ptr,
                                           cnmlCpuTensor_t bias_tensor,
                                           void *bias_ptr,
                                           cnmlCpuTensor_t state_in,
                                           void *state_in_ptr,
                                           cnmlCpuTensor_t state_out,
                                           void *state_out_ptr,
                                           cnmlCpuTensor_t output_tensor,
                                           void *output_ptr);

cnmlStatus_t cnmlCpuComputeMeanVarNormOpForward(cnmlCpuTensor_t input_tensor,
                                                void *input,
                                                cnmlCpuTensor_t output_tensor,
                                                void *output,
                                                bool use_variance,
                                                bool across_spatial);

////////////////////////////////// offline API ///////////////////////////////
struct cnmlModel;
typedef struct cnmlModel *cnmlModel_t;

/**
 * @brief Create a offline model handler.
 * @param model[out] Model pointer.
 * @param name[in] Model name.
 * @return CNML_STATUS_SUCCESS if success.
 *         otherwise the error code is returned.
 */
CNML_DLL_API cnmlStatus_t cnmlCreateModel(cnmlModel_t *model, const char *name);

/**
 * @brief Destroy a offline model handler.
 * @param model[in] Model pointer
 * @return CNML_STATUS_SUCCESS if success.
 *         otherwise the error code is returned.
 */
CNML_DLL_API cnmlStatus_t cnmlDestroyModel(cnmlModel_t model);

/**
 * @brief Add a compiled cnmlBaseOp in to given model. You should give this operation
 *        a symbol name for sake of the future reference. If given op has not
 *        been compiled, error code will be returned.
 * @param model[in] Model pointer
 * @param op[in] Pointer to a base operation.
 * @param symbol[in] Name of added base operation.
 * @return CNML_STATUS_SUCCESS if success.
 *         otherwise the error code is returned.
 */
CNML_DLL_API cnmlStatus_t cnmlAddBaseOpToModel(cnmlModel_t model,
                                               cnmlBaseOp_t op,
                                               const char *symbol);

/**
 * @brief Add a compiled cnmlFusionOp in to given model. You should give this operation
 *        a symbol name for sake of the future reference.
 * @param model[in] Model pointer
 * @param op[in] Pointer to a fusion operation.
 * @param symbol[in] Name of added base operation.
 * @return CNML_STATUS_SUCCESS if success.
 *         otherwise the error code is returned.
 */
CNML_DLL_API cnmlStatus_t cnmlAddFusionOpToModel(cnmlModel_t model,
                                                 cnmlFusionOp_t op,
                                                 const char *symbol);

/**
 * @brief Save a model to the file whose name is fname. If there is
 *        no operation in model, error code is returned.
 * @param model[in] Model pointer
 * @param fname[in] Offline model file name.
 * @return CNML_STATUS_SUCCESS if success.
 *         otherwise the error code is returned.
 */
CNML_DLL_API cnmlStatus_t cnmlSaveModel(cnmlModel_t model, const char *fname);

/**
 * @brief Save a model to specific memory space start with ptr. If there is
 *        no operation in model or memroy is not efficient, error code is returned.
 * @param model[in] Model pointer
 * @param ptr[in] Memory space pointer.
 * @param len[in] Memory space size.
 * @param size[out] actual size of model.
 * @return CNML_STATUS_SUCCESS if success.
 *         otherwise the error code is returned.
 */
CNML_DLL_API cnmlStatus_t cnmlSaveModelToMem(cnmlModel_t model,
                                             void *ptr,
                                             uint64_t len,
                                             uint64_t *size);

#if defined(__cplusplus)
}
#endif

#endif  // CNML_H_
