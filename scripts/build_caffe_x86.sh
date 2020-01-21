#!/bin/bash
shopt -s nocasematch
if [ ! -z "$ARM32_LINUX_LIB_ROOT"  ]; then
  unset ARM32_LINUX_LIB_ROOT
fi
pushd $BUILD_DIR

cmake -DNEUWARE_HOME=${NEUWARE_HOME} \
      -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
      -DTEST_COVERAGE=${TEST_COVERAGE} \
      -DMLU_VENTI=${MLU_VENTI} \
      -DPRE_READ=${PRE_READ} \
      ..

make -j${JOB_NUM}

if [ $? -ne 0 ]; then
    popd
    exit 1
fi

popd
