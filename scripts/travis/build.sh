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
mkdir /opt/images/
rm -rf /opt/images/swap
travis_wait 21 dd if=/dev/zero of=/opt/images/swap bs=8M count=4096
sudo -E mkswap /opt/images/swap
sudo -E swapon /opt/images/swap
echo "After set swap space."
free -m
echo "===========build start================"
./scripts/travis/echo_time.sh & ./scripts/build_cambriconcaffe.sh -c
