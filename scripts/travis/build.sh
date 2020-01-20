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
sudo -s
mkdir /opt/images/
rm -rf /opt/images/swap
dd if=/dev/zero of=/opt/images/swap bs=1024 count=2048000
mkswap /opt/images/swap
swapon /opt/images/swap
echo "After set swap space."
free -m
echo "===========build start================"
./scripts/build_cambriconcaffe.sh 
