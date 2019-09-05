#!/bin/bash

shopt -s nocasematch

# check default ndk if user doesn't export ARM64_R17_NDK_ROOT
if [ -z "${ARM64_R17_NDK_ROOT}" ]; then
    echo "[ERROR] ARM64_R17_NDK_ROOT NOT SET. EXIT."
    exit 1
else
    echo "ARM64_R17_NDK_ROOT already be defined by user."
fi

if [ ! -z "ARM32_LINUX_LIB_ROOT" ]; then
    unset ARM32_LINUX_LIB_ROOT
fi

echo "ARM64_R17_NDK_ROOT=${ARM64_R17_NDK_ROOT}"

if [ ! -d "${ARM64_R17_NDK_ROOT}" ]; then
    echo "[ERROR] ${ARM64_R17_NDK_ROOT} does not exist."
    exit 1
fi

# check default android lib if user doesn't export ARM64_R17_ANDROID_LIB_ROOT
if [ -z "${ARM64_R17_ANDROID_LIB_ROOT}" ]; then
    echo "[ERROR] ARM64_R17_ANDROID_LIB_ROOT NOT SET. EXIT."
    exit 1
else
    echo "ARM64_R17_ANDROID_LIB_ROOT already be defined by user."
fi
echo "ARM64_R17_ANDROID_LIB_ROOT=${ARM64_R17_ANDROID_LIB_ROOT}"

if [ ! -d "${ARM64_R17_ANDROID_LIB_ROOT}" ]; then
    echo "[ERROR] ${ARM64_R17_ANDROID_LIB_ROOT} does not exist."
    exit 1
fi

export NDK_ROOT="${ARM64_R17_NDK_ROOT}"
export ANDROID_LIB_ROOT="${ARM64_R17_ANDROID_LIB_ROOT}"
export BOOST_HOME="${ANDROID_LIB_ROOT}/boost"
export GFLAGS_HOME="${ANDROID_LIB_ROOT}/gflags"
export GLOG_HOME="${ANDROID_LIB_ROOT}/glog"
export LMDB_HOME="${ANDROID_LIB_ROOT}/lmdb"
export HDF5_HOME="${ANDROID_LIB_ROOT}/hdf5"
export HDF5_ROOT="${HDF5_HOME}"
export OPENBLAS_HOME="${ANDROID_LIB_ROOT}/openblas"
export OPENCV_ROOT="${ANDROID_LIB_ROOT}/opencv/sdk/native/jni"
export PROTOBUF_HOME="${ANDROID_LIB_ROOT}/protobuf"


echo "=== cmake ==============================================================="
pushd $BUILD_DIR

# Since HDF5 doesn't support cross-compile, it is turned off for now
cmake -DANDROID_NDK="${NDK_ROOT}" \
      -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
      -DANDROID_STL="c++_shared" \
      -DANDROID_NATIVE_API_LEVEL="21" \
      -DADDITIONAL_FIND_PATH="${ANDROID_LIB_ROOT}" \
      -DBUILD_python=OFF \
      -DBUILD_docs=OFF \
      -DCPU_ONLY=ON \
      -DUSE_MLU=ON \
      -DDISABLE_HDF5_FEAT=ON \
      -DUSE_LMDB=ON \
      -DLMDB_INCLUDE_DIR="${LMDB_HOME}/include" \
      -DLMDB_LIBRARIES="${LMDB_HOME}/lib/liblmdb.a" \
      -DUSE_LEVELDB=OFF \
      -DBLAS="open" \
      -DOpenBLAS_INCLUDE_DIR="${OPENBLAS_HOME}/include" \
      -DOpenBLAS_LIB="${OPENBLAS_HOME}/lib/libopenblas.a" \
      -DBOOST_ROOT="${BOOST_HOME}" \
      -DGFLAGS_INCLUDE_DIR="${GFLAGS_HOME}/include" \
      -DGFLAGS_LIBRARY="${GFLAGS_HOME}/lib/libgflags.a" \
      -DGLOG_INCLUDE_DIR="${GLOG_HOME}/include" \
      -DGLOG_LIBRARY="${GLOG_HOME}/lib/libglog.a" \
      -DOpenCV_DIR="${OPENCV_ROOT}" \
      -DPROTOBUF_PROTOC_EXECUTABLE="${ANDROID_LIB_ROOT}/protobuf_host/bin/protoc" \
      -DPROTOBUF_INCLUDE_DIR="${PROTOBUF_HOME}/include" \
      -DPROTOBUF_LIBRARY="${PROTOBUF_HOME}/lib/libprotobuf.a" \
      -DNEUWARE_HOME=${NEUWARE_HOME} \
      -DCMAKE_TOOLCHAIN_FILE="${CAFFE_DIR}/cmake/android.toolchain.cmake" \
      -DCROSS_COMPILE_ARM64=ON \
      -DTEST_COVERAGE=${TEST_COVERAGE} \
      ..

echo "=== build =============================================================="

make 

if [ $? -ne 0 ]; then
    popd
    exit 1
fi

popd
