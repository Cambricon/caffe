#!/bin/bash
# build the project

#BASEDIR=$(dirname $0)
#source $BASEDIR/defaults.sh
#
#if ! $WITH_CMAKE ; then
#  make --jobs $NUM_THREADS all test pycaffe warn
#else
#  cd build
#  make --jobs $NUM_THREADS all test.testbin
#fi
#make lint
 
echo `pwd`
echo "Prepare swaps before build."
free -m
mkdir ./images
rm -rf ./images/swap
dd if=/dev/zero of=./images/swap bs=10240 count=1M
sudo -E mkswap ./images/swap
sudo -E swapon ./images/swap
echo "After set swap space."
free -m
echo "===========build start================"
./scripts/build_cambriconcaffe.sh
