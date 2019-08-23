/*************************************************************************
 * Copyright (C) [2018] by Cambricon, Inc.
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
 *  @file cnrt.h
 *
 *  @brief Runtime APIs provide programmable interfaces for users to develop
 *  their-owned programs, which includes device managment, context
 *  management, memory managment of both sides (devices and hosts), etc.
 *
 **************************************************************************/

#ifndef __CNRT_H
#define __CNRT_H

/************************************************************************
 *  Include files
 ************************************************************************/
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#if defined(__cplusplus)
extern "C" {
#endif /*__cplusplus*/

/************************************************************************
 *  Definitions
 ************************************************************************/
/**< DLL exports controller. */
#if defined(WIN32) || defined(WINDOWS)
#ifdef USE_CNRT_DLL
#ifdef CNRT_DLL_EXPORTS
#define CNRT_DLL_API __declspec(dllexport)
#else /*CNRT_DLL_EXPORTS*/
#define CNRT_DLL_API __declspec(dllimport)
#endif /*CNRT_DLL_EXPORTS*/
#else
#define CNRT_DLL_API
#endif /*USE_CNRT_DLL*/
#else  /*WIN32 || WINDOWS*/
#define CNRT_DLL_API
#endif /*WIN32 || WINDOWS*/

/**< struct tailed */
#define CNRT_PARAM_END (void *)0xFFFFFFFF

/************************************************************************
 *  Data type declaration
 ************************************************************************/

#define CNRT_CHECK(statment)                                                  \
  do {                                                                        \
    int ret_code = (statment);                                                \
    if (ret_code != CNRT_RET_SUCCESS) {                                       \
      printf("[%s:%d] CNRT error, code: %d\n", __FILE__, __LINE__, ret_code); \
      exit(1);                                                                \
    }                                                                         \
  } while (false);

/**< Error codes */
typedef enum {
  CNRT_RET_SUCCESS = 0,                   /**< No error */
  CNRT_RET_WARNING_FAKE_DEVICE = 1,       /**< Use fake device */
  CNRT_RET_ERR_INVALID = 632007,          /**< Invalid argument */
  CNRT_RET_ERR_NOMEM = 632008,            /**< Out of memory */
  CNRT_RET_ERR_NODEV = 632009,            /**< No such device */
  CNRT_RET_ERR_IO = 632010,               /**< I/O error */
  CNRT_RET_ERR_SYS = 632011,              /**< System error */
  CNRT_RET_ERR_ACCES = 632012,            /**< Permission denied */
  CNRT_RET_ERR_FAULT = 632013,            /**< Bad address */
  CNRT_RET_ERR_BUSY = 632014,             /**< Device or resource busy */
  CNRT_RET_ERR_TIMEOUT = 632015,          /**< Time expired */
  CNRT_RET_ERR_EXIST = 632016,            /**< Resource or file already exists */
  CNRT_RET_ERR_NOSYS = 632017,            /**< Function not implemenmted */
  CNRT_RET_ERR_AGAIN = 632018,            /**< try again later */
  CNRT_RET_ERR_NORES = 632019,            /**< Out of resource */
  CNRT_RET_ERR_UNSUPPORTED = 632020,      /**< Unsupported operation */
  CNRT_RET_ERR_INVALID_POINTER = 632021,  /**< Invalid pointer */
  CNRT_RET_ERR_NO_EXIST = 632022,         /**< Resource or file doesn't exist */
  CNRT_RET_ERR_BROKEN = 632023,           /**< Data transmission is broken */
  CNRT_RET_ERR_INIT = 632024,             /**< Uninitialized */
  CNRT_RET_ERR_QUEUE = 632025,            /**< Failure on Queue */
  CNRT_RET_ERR_OUT_RANGE = 632026,        /**< Number out of range */
  CNRT_RET_ERR_MATH_OVERFLOW = 632027,    /**< Math result not representable */
  CNRT_RET_ERR_FUNC_CALL = 632028,        /**< Failure to call runtime functions */
  CNRT_RET_ERR_UNHANDLED = 632029,        /**< Unhandled error */
  CNRT_RET_ERR_INVALID_TYPE = 632030,     /**< Invalid type */
  CNRT_RET_ERR_INVALID_OP = 632031,       /**< Invalid operation */
  CNRT_RET_ERR_MLU = 632032,              /**< MLU error */
  CNRT_RET_ERR_ONCHIP_CORE = 632033,      /**< Onchip core error */
  CNRT_RET_ERR_NOTIFIER = 632034,         /**< Failure on notifier operation */
  CNRT_RET_ERR_RESHAPE = 632035,          /**< Failure on data reshape */
  CNRT_RET_ERR_MEMCPY = 632036,           /**< Failure on memory copy */
  CNRT_RET_ERR_ENCRYPT = 632037,          /**< Failure on encrypt */
  CNRT_RET_ERR_INVALID_DATADESC = 632038, /**< Invalid data descriptor */
  CNRT_RET_ERR_UNKNOWN = 999991,          /**< Unknown error */
  CNRT_RET_ERR_MAX                        /**< The last one */
} cnrtRet_t;

/**< Memory types available for allocator */
typedef enum {
  CNRT_MEMTYPE_DEFAULT = 0, /**< User space pagable memory */
  CNRT_MEMTYPE_LOCKED,      /**< Pinned memory */
  CNRT_MEMTYPE_DEV          /**< Device memory */
} cnrtMemType_t;

/**< Execution modes of tasks on MLU */
typedef enum {
  CNRT_FUNC_TYPE_BLOCK = 1,
  CNRT_FUNC_TYPE_UNION1 = 4,
  CNRT_FUNC_TYPE_UNION2,
  CNRT_FUNC_TYPE_UNION4,
  CNRT_FUNC_TYPE_UNION8,
  CNRT_FUNC_TYPE_MUTABLE
} cnrtFunctionType_t;

/**< DDR Channel for tasks used on MLU, only MLU100 support */
typedef enum {
  CNRT_CHANNEL_TYPE_NONE = -1,
  CNRT_CHANNEL_TYPE_0 = 0,
  CNRT_CHANNEL_TYPE_1,
  CNRT_CHANNEL_TYPE_2,
  CNRT_CHANNEL_TYPE_3
} cnrtChannelType_t;

/**< Direction of data transmission */
typedef enum {
  CNRT_MEM_TRANS_DIR_HOST2DEV = 0, /**< Host to Device*/
  CNRT_MEM_TRANS_DIR_DEV2DEV,      /**< Device to Device, not support currently */
  CNRT_MEM_TRANS_DIR_DEV2HOST,     /**< Device to Host */
  CNRT_MEM_TRANS_DIR_HOST2HOST,    /**< Host to Host, not support currently */
  CNRT_MEM_TRANS_DIR_NODIR         /**< no direction for init */
} cnrtMemTransDir_t;

/**< Parameter for function call */
typedef struct {
  unsigned int x; /**< x aixs */
  unsigned int y; /**< y aixs */
  unsigned int z; /**< x aixs */
} cnrtDim3_t;

/**< Parameter for init function call*/
typedef struct {
  bool *muta;             /**< mutable option*/
  int *data_parallelism;  /**< data parallelism*/
  unsigned int *affinity; /**< affinity*/
  void *end;              /**< end of struct*/
} cnrtInitFuncParam_t;

/**< Parameter for invoke function call*/
typedef struct {
  int *data_parallelism;  /**< data parallelism*/
  unsigned int *affinity; /**< affinity*/
  void *end;              /**< end of struct*/
} cnrtInvokeFuncParam_t;

/**< Data type and data order*/
typedef enum cnrtDataType {
  CNRT_INVALID = 0x0,
  CNRT_FLOAT16 = 0x12,
  CNRT_FLOAT32 = 0x13,
  CNRT_FLOAT64 = 0x14,
  CNRT_INT8 = 0x21,
  CNRT_INT16 = 0x22,
  CNRT_INT32 = 0x23,
  CNRT_UINT8 = 0x31,
  CNRT_UINT32 = 0x33,
  CNRT_FIX8 = 0x41,
  CNRT_QUANT8 = 0x51,
  CNRT_BOOL = 0x61
} cnrtDataType_t;

typedef enum cnrtDimOrder {
  CNRT_NCHW = 0x0123,
  CNRT_NHWC = 0x0231,
  CNRT_HWCN = 0x2310,
} cnrtDimOrder_t;

typedef enum cnrtCoreVersion {
  CNRT_1H8 = 0,
  CNRT_1H16 = 1
} cnrtCoreVersion_t;

/**< Model and function */
struct cnrtModel;
typedef struct cnrtModel *cnrtModel_t;

struct cnrtFunction;
typedef struct cnrtFunction *cnrtFunction_t;

/**< Input/Output data description */
struct cnrtDataDesc;
typedef struct cnrtDataDesc *cnrtDataDesc_t, **cnrtDataDescArray_t;

/**< Queue, Notifier, MLU device */
struct cnrtQueue;
typedef struct cnrtQueue *cnrtQueue_t;

struct cnrtNotifier;
typedef struct cnrtNotifier *cnrtNotifier_t;

typedef uint64_t cnrtDev_t;

/************************************************************************
 * Function prototype declaration
 ************************************************************************/

/************************************************************************
 * Error handling
 ************************************************************************/

/**
 * @brief Return string pointer that describes
 *     the error code passed in the argument errCode.
 *
 * The function returns a read only string that is corresponding
 * to the argument @p errcode.
 *
 * @param  err_code[in] the error code was returned by previous function call.
 * @return a pointer that points to a constant string.
 */
extern CNRT_DLL_API const char *cnrtGetErrorStr(cnrtRet_t err_code);

/**
 * @brief Get the error code set by any runtime calls.
 *     Its value is meaningful only when the return value indicating an error.
 *
 * @return error code of the last call of runtime functions.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetLastErr(void);

/*************************************************************************
 * Initialization and destroy
 *************************************************************************/

/**
 * @brief Initialize runtime environment in current process space.
 *
 * Initializes this API must be called before any other runtime API calls.
 *
 * @param  flags[in] reserved for further use, pass 0 as well.
 * @return CNRT_RET_SUCCESS if success, otherwise with the error code.
 */
extern CNRT_DLL_API cnrtRet_t cnrtInit(unsigned int flags);

/**
 * @brief Destroy everything that allocated by runtime API calls.
 *
 * This API should be called after any other runtime API calls.
 *
 * @return void (None).
 */
extern CNRT_DLL_API void cnrtDestroy(void);

/******************************************************************************
 * Version and revision
 ******************************************************************************/

/**
 * @brief Return the version of the cnrt software.
 *
 * Higher version usually offers more features provided by this library.
 *
 * @param  ver[out] pointer to retrieve the version.
 * @return unsigned int for version number.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetVersion(unsigned int *ver);

/******************************************************************************
 * Device managment
 ******************************************************************************/

/**
 * @brief Get the device handle by a given device ordinal.
 *
 *  The function returns the device handle given a specific device ordinal.
 *
 * @param  pdev[out] pointer to retrieve the device handle.
 * @param  ordinal[in] the device ordinal to get the device handle.
 * @note   the ordinal should be in the range [0~cnrtGetDeviceCount() - 1].
 * @return CNRT_RET_SUCCESS if success, otherwise the error code is returned.
 */

extern CNRT_DLL_API cnrtRet_t cnrtGetDeviceHandle(cnrtDev_t *pdev, int ordinal);

/**
 * @brief Set the device handle for current thread execution context.
 *
 *  It implies that any subsequent runtime API calls are for this device.
 *
 * @param  dev[in] the device handle.
 * @return CNRT_RET_SUCCESS if success, otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtSetCurrentDevice(cnrtDev_t dev);

/**
 * @brief Get the cnrtDevice handle from current thread execution context.
 *
 * The handle has been set by calling cnrtSetCurrentDevice().
 *
 * @param  pdev[out] pointer to retrieve the device handle.
 * @return CNRT_RET_SUCCESS if success, otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetCurrentDevice(cnrtDev_t *pdev);

/**
 * @brief Get the number of MLU devices in the system.
 *
 * @param  dev_num[out] pointer to retrieve the number of devices.
 * @return CNRT_RET_SUCCESS if success, otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetDeviceCount(unsigned int *devNum);

/**
 * @brief  Wait for the device to complete precedent tasks.
 *
 * @return CNRT_RET_SUCCESS if success, otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtSyncDevice(void);

/******************************************************************************
 * Queue managment
 ******************************************************************************/

/**
 * @brief Create a new queue after calling this function,
 *        it works in asynchronous mode by default.
 *
 * @param pQueue[out] pointer to retrieve the new created Queue handle.
 * @return CNRT_RET_SUCCESS if success, otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtCreateQueue(cnrtQueue_t *pQueue);

/**
 * @brief Destroy a queue created by calling cnrtCreateQueue.
 *
 * @param queue[in] queue handle created by calling cnrtCreateQueue.
 * @return CNRT_RET_SUCCESS if success, otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtDestroyQueue(cnrtQueue_t queue);

/**
 * @brief Function should be blocked until all precedent tasks in the queue are completed.
 *
 * @param queue[in] queue handle created by calling cnrtCreateQueue.
 * @return CNRT_RET_SUCCESS if success, otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtSyncQueue(cnrtQueue_t queue);

/*********************************************************************************
 * Notifier, only MLU100 support
 *********************************************************************************/

/**
 * @brief Create a notifier corresponding to the current device.
 *
 * @param notifier[out] point to an notifier handle to retrieve newly created notifier.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtCreateNotifier(cnrtNotifier_t *notifier);

/**
 * @brief Destroy a notifier that was created by calling cnrtCreateNotifier.
 *
 * @param notifier[in] notifier handle to be destroyed.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtDestroyNotifier(cnrtNotifier_t *notifier);

/**
 * @brief Place a notifier in specified queue. This function will not block the CPU thread.
 *        All computation tasks submitted to the queue will wait until notifier reports
 *        completion before starting execution.
 *
 * @param notifier[in] signal handle created by calling cnrtCreateNotifier.
 * @param queue[in] queue handle created by calling cnrtCreateQueue.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtPlaceNotifier(cnrtNotifier_t notifier, cnrtQueue_t queue);

/**
 * @brief Get duration time of two notifiers.
 *
 * @param start[in] notifier handle created by calling cnrtCreateNotifier.
 * @param end[in] notifier handle created by calling cnrtCreateNotifier.
 * @param us[out] duartion time between start and end.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtNotifierDuration(cnrtNotifier_t start,
                                                   cnrtNotifier_t end,
                                                   float *us);

/*********************************************************************************
 * Model load and Function call
 *********************************************************************************/

/**
 * @brief Load a model from a given model file.
 *
 * @param pmodel[out] point to a cnrtModel_t.
 * @param fname[in]  file name of a cambricon model.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtLoadModel(cnrtModel_t *pmodel, const char *fname);

/**
 * @brief Load a model from memory
 *
 * @param pmodel[out] point to a cnrtModel_t.
 * @param ptr[in] memory ptr.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtLoadModelFromMem(cnrtModel_t *pmodel, char *ptr);

/**
 * @brief Unload a model.
 *
 * @param model[in] point to a cnrtModel_t.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtUnloadModel(cnrtModel_t model);

/**
 * @brief  Get actual size of model in offline file.
 *
 * @param fname[in] file name of a cambricon model.
 * @param size[out] pointer to model's actual size.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetModelSize(const char *fname, int *size);

/**
 * @brief  Query model's core version, 1H8 or 1H16.
 *
 * @param model[in] point to a loaded model.
 * @param coreVersion[out] pointer to model's core version.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtQueryCoreVersion(cnrtModel_t model,
                                                   cnrtCoreVersion_t *coreVersion);

/**
 * @brief  Query model's parallelism, which means the core number
 * involved to compute this model.
 *
 * @param model[in] point to a loaded model.
 * @param modelParallelism[out] pointer to model's parallelism.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtQueryModelParallelism(cnrtModel_t model, int *modelParallelism);

/**
 * @brief  Query model's stack size, which is the biggest stack size(MB)
 * in all the kernels in the model.
 *
 * @param model[in] point to a loaded model.
 * @param size[out] pointer to the stack size.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtQueryModelStackSize(cnrtModel_t model, uint64_t *stack_size);

/**
 * @brief Get function number of a given model
 *
 * @param model[in] pointer of a cnrt model
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetFunctionNumber(cnrtModel_t model, int *func_num);

/**
 * @brief Extract the symbol from the given model if symbol exists,
 *        otherwise error code will be returned.
 *
 * @param function[out] point to a cnrtFunction_t.
 * @param model[in]  point to a loaded model.
 * @param symbol[in] symbol name.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtExtractFunction(cnrtFunction_t *pfunction,
                                                  cnrtModel_t model,
                                                  const char *symbol);

/**
 * @brief Create a mlu function.
 * @param function[in] pointer of cnrtFunction_t.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtCreateFunction(cnrtFunction_t *pfunction);

/**
 * @brief Destroy a function.
 *
 * @param function[in] point to a function generated by cnrtExtractFunction.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtDestroyFunction(cnrtFunction_t function);

/**
* @brief Initialize instrucion and runtime data of a function on current MLU device.
*
*        cnrtInitFunctionMemory has two modes, distinguished by affinity.
*        The first mode is affinity == CNRT_FUNC_TYPE_MUTABLE.
*        The second mode is affinity != CNRT_FUNC_TYPE_MUTABLE.
*
*        The first mode is more flexble.
*        Under this mode, the same function can be invoked by different
*        cnrtInvokeFunction with different parallelism simultaneously.
*        For example, you can write the following code:
*        dataParallelism int the cnrtInitFunctionMemory_V2 is just initialization.
*        cnrtInitFuncParam_t init_func_param;
*        bool muta = true;
*        init_func_param.muta = &muta;
*        int dataParallelism = 1;
*        init_func_param.dataParallelism = &dataParallelism;
*        init_func_param.end = CNRT_PARAM_END;
*        cnrtInitFunctionMemory_V2(function, &init_func_param);
*
*        dataParallelism in the cnrtInvokeFunction is mutalble.
*        cnrtInvokeFuncParam_t invoke_func_param;
*        int dataParallelism = 2;
*        invoke_func_param.dataParallelism = &dataParallelism;
*        invoke_func_param.end = CNRT_PARAM_END;
*        cnrtFunctionType_t func_type = (cnrtFunctionType_t)0;
*        cnrtInvokeFunction(function, ..., (void *)&invoke_func_param);
*
*        The second mode is more efficient.
*        Under this mode, the same function can also be invoked by different
*        cnrtInvokeFunction. But the parallelism is limited. It should be
*        the same as affinity.
*        For example, you can write the following code:
*        dataParallelism in the cnrtInvokeFunction is not mutalble.
*        cnrtInitFuncParam_t init_func_param;
*        bool muta = false;
*        init_func_param.muta = &muta;
*        int dataParallelism = 1;
*        init_func_param.dataParallelism = &dataParallelism;
*        init_func_param.end = CNRT_PARAM_END;
*        cnrtInitFunctionMemory_V2(function, &init_func_param);
*
*        dataParallelism in the cnrtInvokeFunction should be same as the
*        dataParallelism int the cnrtInitFunctionMemory_V2.
*        cnrtInvokeFuncParam_t invoke_func_param;
*        int dataParallelism = 1;
*        invoke_func_param.dataParallelism = &dataParallelism;
*        invoke_func_param.end = CNRT_PARAM_END;
*        cnrtFunctionType_t func_type = (cnrtFunctionType_t)0;
*        cnrtInvokeFunction(function, ..., (void *)&invoke_func_param);
*
*        notice: cnrtInitFunctionMemory should be called before
*        cnrtInvokeFunction and after cnrtSetCurrentDevice.
*
* @param function[in] pointer of cnrtFunction_t.
* @param param[in] pointer of cnrtInitfuncParam_t.
* @return CNRT_RET_SUCCESS if success,
*         otherwise the error code is returned.
*/
extern CNRT_DLL_API cnrtRet_t cnrtInitFunctionMemory_V2(cnrtFunction_t function,
                                                        cnrtInitFuncParam_t *param);

/**
 * @brief Invoke a function with given params on MLU.
 * @param function[in] point to the MLU function.
 * @param dim[in] how many grid dimentions.
 * @param params[in] point to arguments.
 * @param func_type[in] function type. @see cnrtFunctionType_t.
 * @param queue[in] queue associated to the function call.
 * @param extra_param[in] pointer to cnrtInvokeFuncParam_t as extra param.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtInvokeFunction_V2(cnrtFunction_t function,
                                                    cnrtDim3_t dim,
                                                    void **params,
                                                    cnrtFunctionType_t func_type,
                                                    cnrtQueue_t queue,
                                                    void *extra_param);

/**
 * @brief Generate a copy of source MLU function. src and dst function share the
 *        same kernel on host, but they have different device space, so model
 *        data(include instruction) is doubled on device.
 *
 * @param src[in] Pointer to a source MLU function
 * @param dst[out] Pointer to a destination MLU function pointer
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtCopyFunction(cnrtFunction_t *dst, cnrtFunction_t src);

/*********************************************************************************
 * Memory management
 *********************************************************************************/

/**
 * @brief Allocate nByte bytes and place a pointer to pointer
 *        in pPtr to the allocated host memory. If bytes is 0, then
 *        cnrtMallocHost returns either NULL, or a unique pointer value
 *        that can later be passed to cnrtFreeHost.
 *
 * @param pPtr[out]  a pointer to pointer for retrieving allocated host memory.
 * @param bytes[in] number bytes of memory to be allocated.
 * @param type[in]   memory type to be allocated,
 *                   @see CNRT_HOST_MEMORY_TYPE_LOCK and CNRT_HOST_MEMORY_TYPE_MAPPED.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtMallocHost(void **pPtr, size_t bytes, cnrtMemType_t type);

/**
 * @brief Free the memory space pointed by ptr, which must be
 *        returned by a previous call of cnrtMallocHost.
 *
 * @param ptr[in]  point to the address of memory to be free.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtFreeHost(void *ptr);

/**
 * @brief Allocate memory on MLU device.
 *
 * @param pPtr[out] a pointer to pointer for retrieving allocated device memory.
 * @param bytes[in] allocate size.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtMalloc(void **pPtr, size_t bytes);

/**
 * @brief Allocate continuous memory for muti-way data on MLU device.
 *        This API should be used under data parallel mode.
 *        Data size of each way will be aligned automatically for sake of
 *        high performance memory access. So the truely allocate size is
 *        align(bytes) * dataParallelism.
 *
 * @param pPtr[out] a pointer to pointer for retrieving allocated device memory.
 * @param bytes[in] allocate size.
 * @param dataParallelism[in] data parallelism
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtMallocBatch(void **pPtr, size_t bytes, int dataParallelism);

/**
 * @brief Allocate memory on MLU device.
 *        Compared with cnrtMalloc, cnrtMallocByDesc use cnrtDataDesc_t
 *        object to determine the allocate size.
 *
 * @param pPtr[out] point to allocated memory.
 * @param dataDesc[in] data descriptor.
 * @return CNRT_RET_SUCCESS if success.
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtMallocByDesc(void **pPtr, cnrtDataDesc_t dataDesc);

/**
 * @brief Allocate continuous memory for muti-way data on MLU device.
 *        Compared with cnrtMallocBatch, cnrtMallocBatchByDesc use
 *        cnrtDataDesc_t object to determine the allocate size.
 *
 * @param pPtr[out] point to allocated memory.
 * @param dataDesc[in] data descriptor.
 * @param dataParallelism[in] data parallelism
 * @return CNRT_RET_SUCCESS if success.
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtMallocBatchByDesc(void **pPtr,
                                                    cnrtDataDesc_t dataDesc,
                                                    int dataParallelism);

/**
 * @brief Allocate multiple addresses for multiple data objects on MLU device.
 *        Multiple addresses and data descriptors is present in array format.
 *        This API is a reinforced version of cnrtMallocByDesc. You can call
 *        cnrtMallocByDesc more than once to realize the same function.
 *
 * @param pPtrArray[out] point to the allocated memory array.
 * @param dataDescArray[in] data descriptor array.
 * @param lentgh[in] array length.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtMallocByDescArray(void ***pPtrArray,
                                                    cnrtDataDescArray_t dataDescArray,
                                                    int length);

/**
 * @brief Allocate multiple addresses for multiple data objects
 *        of multiple way on MLU device. This API is a mult-way
 *        version of cnrtMallocByDescArray.
 *
 * @param pPtrArray[out] point to the allocated memory array.
 * @param dataDescArray[in] data descriptor array.
 * @param lentgh[in] array length.
 * @param dataParallelism[in] way size of data.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtMallocBatchByDescArray(void ***pPtrArray,
                                                         cnrtDataDescArray_t dataDescArray,
                                                         int length,
                                                         int dataParallelism);

/**
 * @brief Deallocate MLU device Memory.
 *
 * @param ptr[in] point to the memory to be free.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtFree(void *ptr);

/**
 * @brief Deallocate MLU multiple device memory addresses allocated
 *        by cnrtMallocBatchByDescArray, cnrtMallocByDescArray.
 *
 * @param ptr[in] a pointer array.
 * @param length[in] array length.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtFreeArray(void **ptr, int length);

/**
 * @brief Copy data from src address to dst address. The copy direction
 *        is specified by input parameter dir. The copy operation is
 *        always performed on current device which is set by cnrtSetCurrentDevice.
 *
 * @param dst[in] destination address.
 * @param src[in] source address.
 * @param bytes[in] number of bytes to be copied.
 * @param dir[in] direction of transfer.
 *                @see  CNRT_MEM_TRANS_DIR_HOST2DEV,
 *                      CNRT_MEM_TRANS_DIR_DEV2DEV,
 *                      CNRT_MEM_TRANS_DIR_DEV2HOST,
 *                      CNRT_MEM_TRANS_DIR_HOST2HOST,
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtMemcpy(void *dst, void *src, size_t bytes, cnrtMemTransDir_t dir);

/**
 * @brief Copy multi-way data from src address to dst address.
 *        The device address should be allocated by cnrtMallocBatch.
 *        The host address should contain dataParallelism number of data arranged
 *        continuously. This API should be used under data parallel mode.
 *        More infomation about device address @see cnrtMallocBatch.
 *
 * @param dst[in] destination address.
 * @param src[in] source address.
 * @param bytes[in] size of single way data.
 * @param dataParallelism[in] data parallelism.
 * @param dir[in] direction of transfer.
 *                @see  CNRT_MEM_TRANS_DIR_HOST2DEV,
 *                      CNRT_MEM_TRANS_DIR_DEV2DEV,
 *                      CNRT_MEM_TRANS_DIR_DEV2HOST,
 *                      CNRT_MEM_TRANS_DIR_HOST2HOST,
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t
cnrtMemcpyBatch(void *dst, void *src, size_t bytes, int dataParallelism, cnrtMemTransDir_t dir);

/**
 * @brief Copy data from src address to dst address.
 *        Compared with cnrtMemcpy, cnrtMemcpyByDesc receives data descriptor as
 *        a input parameter. Because we need to carry out data layout optimization
 *        for MLU device. This API is typically used in image process situation.
 *
 * @param dst[in] destination address.
 * @param src[in] source address.
 * @param dataDesc[in] data descriptor.
 * @param dir[in] direction of transfer.
 *                @see  CNRT_MEM_TRANS_DIR_HOST2DEV,
 *                      CNRT_MEM_TRANS_DIR_DEV2DEV,
 *                      CNRT_MEM_TRANS_DIR_DEV2HOST,
 *                      CNRT_MEM_TRANS_DIR_HOST2HOST,
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtMemcpyByDesc(void *dst,
                                               void *src,
                                               cnrtDataDesc_t dataDesc,
                                               cnrtMemTransDir_t dir);

/**
 * @brief Copy multi-way data from src address to dst address.
 *        This is a multi-way version of cnrtMemcpyByDesc.
 *        The host address should contain dataParallelism number of data arranged
 *        continuously.
 *        The device address should be allocated by cnrtMallocBatchByDesc.
 *        To get more infomation about multi-way, @see cnrtMallocBatchByDesc.
 *
 * @param dst[in] destination address.
 * @param src[in] source address.
 * @param dataDesc[in] data descriptor.
 * @param dataParallelism[in] data parallelism.
 * @param dir[in] direction of transfer.
 *                @see  CNRT_MEM_TRANS_DIR_HOST2DEV,
 *                      CNRT_MEM_TRANS_DIR_DEV2DEV,
 *                      CNRT_MEM_TRANS_DIR_DEV2HOST,
 *                      CNRT_MEM_TRANS_DIR_HOST2HOST,
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtMemcpyBatchByDesc(void *dst,
                                                    void *src,
                                                    cnrtDataDesc_t dataDesc,
                                                    int dataParallelism,
                                                    cnrtMemTransDir_t dir);

/**
 * @brief Copy multiple data objects from src addresses to dst addresses.
 *        Multiple addresses and data descriptors is present in array format.
 *        This API is a reinforced version of cnrtMemcpyByDesc. You can call
 *        cnrtMemcpyByDesc more than once to realize the same function.
 *
 * @param dstArray[in] pointer to destination address array.
 * @param srcArray[in] pointer to source address array.
 * @param dataDescArray[in] data descriptor array.
 * @param length[in] array length.
 * @param dir[in] direction of transfer.
 *                @see  CNRT_MEM_TRANS_DIR_HOST2DEV,
 *                      CNRT_MEM_TRANS_DIR_DEV2DEV,
 *                      CNRT_MEM_TRANS_DIR_DEV2HOST,
 *                      CNRT_MEM_TRANS_DIR_HOST2HOST,
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtMemcpyByDescArray(void **dstArray,
                                                    void **srcArray,
                                                    cnrtDataDescArray_t dataDescArray,
                                                    int length,
                                                    cnrtMemTransDir_t dir);

/**
 * @brief Copy multiple data objects of multi-way from src
 *        addresses to dst addresses.
 *        This API is the multi-way of cnrtMemcpyByDescArray.
 *
 * @param dstArray[in] pointer to destination address array.
 * @param srcArray[in] pointer to source address array.
 * @param dataDescArray[in] data descriptor array.
 * @param length[in] array length.
 * @param dataParallelism[in] data parallelism.
 * @param dir[in] direction of transfer.
 *                @see  CNRT_MEM_TRANS_DIR_HOST2DEV,
 *                      CNRT_MEM_TRANS_DIR_DEV2DEV,
 *                      CNRT_MEM_TRANS_DIR_DEV2HOST,
 *                      CNRT_MEM_TRANS_DIR_HOST2HOST,
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtMemcpyBatchByDescArray(void **dstArray,
                                                         void **srcArray,
                                                         cnrtDataDescArray_t dataDescArray,
                                                         int length,
                                                         int dataParallelism,
                                                         cnrtMemTransDir_t dir);

/**
 * @brief Fill the bytes of the device memory space
 *        pointed by devPtr with the constant value c.
 *
 * @param ptr[in] device memory address.
 * @param c[in] value to be filled.
 * @param bytes[in] number of bytes to be filled.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtMemset(void *ptr, int c, size_t bytes);

/**
 * @brief set mlu stack space memory to stack_size(MB).
 *        Only MLU100 support.
 * @param stacksize[in] the size of mlu stack space memory will be set.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise CNRT_RET_ERR_MLU is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtSetStackMem(unsigned int stacksize);

/**
 * @brief get mlu stack space memory to stack_size(MB).
 *        Only MLU100 support.
 * @param pStacksize[out] the size of mlu stack space memory will be get.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise CNRT_RET_ERR_MLU is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetStackMem(unsigned int *pStacksize);

/**
 * @brief get max memory used of function
 * @param function[in] point to the MLU function.
 * @brief cnrt get max memory used
 * @param function[in] point to the model.
 * @param pMemused[out] return value.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetFunctionMemUsed(cnrtFunction_t function, int64_t *pMemused);

/**
 * @brief get max memory used of model
 * @param model[in] point to the model.
 * @param pMemused[out] return value.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetModelMemUsed(cnrtModel_t model, int64_t *pMemused);

/*********************************************************************************
 * Channel control, only MLU100 support
 *********************************************************************************/

/**
 * @brief Set memory and computation channel on current MLU device. Once
 *        a channel is configured, all memory allocation(eg. cnrtMalloc)
 *        will be performed on this channel. And all function invokation
 *        (cnrtInvokeFunction) will be performed on this channel too.
 *        Attention: The above policy only take effect when model parallelism
 *        is 1.
 *        This function is base on CPU thread context. So it's action scope
 *        is within current CPU thread. This function should be called after
 *        cnrtSetCurrentDevice;
 *
 * @param cnrtChannelType_t[in] channel.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtSetCurrentChannel(cnrtChannelType_t channel);

/**
* @brief Get current channel of current CPU thread.
*
* @param pChannel[out] Pointer to channel.
* @return CNRT_RET_SUCCESS if success,
*         otherwise the error code is returned.
*/
extern CNRT_DLL_API cnrtRet_t cnrtGetCurrentChannel(cnrtChannelType_t *pChannel);

/*********************************************************************************
 * Data descriptor
 *********************************************************************************/

/**
 * @brief Get a series of input data descriptors from a given function.
 *
 * @param descArray[out] point to Data descriptor array.
 * @param num[out] length of the data descriptor array.
 * @param function[in] MLU function pointer.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetInputDataDesc(cnrtDataDescArray_t *descArray,
                                                   int *num,
                                                   cnrtFunction_t function);

/**
 * @brief Get a series of input data descriptors from a given function.
 *
 * @param descArray[out] point to the data descriptor array.
 * @param num[out] length of the data descriptor array.
 * @param function[in] MLU function pointer.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetOutputDataDesc(cnrtDataDescArray_t *descArray,
                                                    int *num,
                                                    cnrtFunction_t function);

/**
 * @brief Set data layout(eg. type, dim order) on the host according to the data descriptor.
 *
 * @param desc[in] point to data descriptor.
 * @param dtype[in] host data type. @see cnrtDataDesc_t.
 * @param order[in] host data dim order. @see cnrtDimOrder_t.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtSetHostDataLayout(cnrtDataDesc_t desc,
                                                    cnrtDataType_t dtype,
                                                    cnrtDimOrder_t order);

/**
 * @brief get a DataDesc's n, c, h ,w.
 *
 * @param desc[in] point to the data descriptor pointer.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t
cnrtGetDataShape(cnrtDataDesc_t desc, unsigned *n, unsigned *c, unsigned *h, unsigned *w);

/**
 * @brief Get host data count (e.g. for tensor with dim nchw, the count is n*c*h*w).
 *
 * @param count[out] host data count.
 * @param desc[in] point to the data descriptor.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetHostDataCount(cnrtDataDesc_t desc, int *count);

#if defined(__cplusplus)
}
#endif /*__cplusplus*/
#endif /*__CNRT_H*/
