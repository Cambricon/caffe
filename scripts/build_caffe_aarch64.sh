#!/bin/bash

shopt -s nocasematch

# check default cross toolchain if user doesn't export CROSS_TOOLCHAIN_PATH
if [ -z "${CROSS_TOOLCHAIN_PATH}" ]; then
    echo "[ERROR] CROSS_TOOLCHAIN_PATH NOT SET. EXIT."
    exit 1
else
    echo "CROSS_TOOLCHAIN_PATH already be defined by user."
fi
echo "CROSS_TOOLCHAIN_PATH=$CROSS_TOOLCHAIN_PATH"

export PATH=${CROSS_TOOLCHAIN_PATH}:$PATH
CROSS_TOOLCHAIN_PATH=`which aarch64-linux-gnu-gcc`
TOOLCHAIN_PREFIX="`echo ${CROSS_TOOLCHAIN_PATH%\-*}`-"
CROSS_COMPILE=aarch64-linux-gnu-

# check default android lib if user doesn't export AARCH64_LINUX_LIB_ROOT
if [ -z "${AARCH64_LINUX_LIB_ROOT}" ]; then
    echo "[ERROR] AARCH64_LINUX_LIB_ROOT NOT SET. EXIT."
    exit 1
else
    echo "AARCH64_LINUX_LIB_ROOT already be defined by user."
fi
echo "AARCH64_LINUX_LIB_ROOT=$AARCH64_LINUX_LIB_ROOT"

if [ ! -d "${AARCH64_LINUX_LIB_ROOT}" ]; then
    echo "[ERROR] ${AARCH64_LINUX_LIB_ROOT} does not exist."
    exit 1
else
    export AARCH64_LINUX_LIB_ROOT
fi

export PROTOBUF_HOME="${AARCH64_LINUX_LIB_ROOT}/protobuf"
export GFLAGS_HOME="${AARCH64_LINUX_LIB_ROOT}/gflags"
export GLOG_HOME="${AARCH64_LINUX_LIB_ROOT}/glog"
export BOOST_HOME="${AARCH64_LINUX_LIB_ROOT}/boost"
export OPENBLAS_HOME="${AARCH64_LINUX_LIB_ROOT}/openblas"
export LMDB_HOME="${AARCH64_LINUX_LIB_ROOT}/lmdb"
export HDF5_HOME="${AARCH64_LINUX_LIB_ROOT}/hdf5"
export OPENCV_ROOT="${AARCH64_LINUX_LIB_ROOT}/opencv/share/OpenCV"

echo "=== cmake =============================================================="
pushd $BUILD_DIR
cmake -DCROSS_COMPILE="${TOOLCHAIN_PREFIX}" \
      -DCMAKE_C_COMPILER="${CROSS_COMPILE}gcc" \
      -DCMAKE_CXX_COMPILER="${CROSS_COMPILE}g++" \
      -DCMAKE_CXX_FLAGS=" -fPIC -lgomp" \
      -DCMAKE_C_FLAGS=" -fPIC -lgomp" \
      -DOpenCV_MODULES_SUFFIX="" \
      -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
      -DDISABLE_HDF5_FEAT=ON \
      -DBUILD_python=OFF \
      -DBUILD_docs=OFF \
      -DCPU_ONLY=OFF \
      -DUSE_LMDB=ON \
      -DUSE_MLU=ON \
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
      -DPROTOBUF_PROTOC_EXECUTABLE="${AARCH64_LINUX_LIB_ROOT}/protobuf_host/bin/protoc" \
      -DPROTOBUF_INCLUDE_DIR="${PROTOBUF_HOME}/include" \
      -DPROTOBUF_LIBRARY="${PROTOBUF_HOME}/lib/libprotobuf.a" \
      -DNEUWARE_HOME=${NEUWARE_HOME} \
      -DTEST_COVERAGE=${TEST_COVERAGE} \
      ..

echo "=== build =============================================================="
make -j${JOB_NUM}

if [ $? -ne 0 ]; then
    popd
    exit 1
fi

popd
