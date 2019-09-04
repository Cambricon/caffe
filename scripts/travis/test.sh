#!/bin/bash
# test the project

#BASEDIR=$(dirname $0)
#source $BASEDIR/defaults.sh

#if $WITH_CUDA ; then
#  echo "Skipping tests for CUDA build"
#  exit 0
#fi
#
#if ! $WITH_CMAKE ; then
#  make runtest
#  make pytest
#else
#  cd build
#  make runtest
#  make pytest
#fi
echo `pwd`
echo "===============test start====================="
cd build 
#do nothing
#make testlist
