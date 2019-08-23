#! /bin/bash

# please define DEBUG_CAFFE if you'd like to compile with debug version
if [ ! -z ${DEBUG_CAFFE} ]; then
    SUFFIX="-d"
fi

if [ ! -z ${CAFFE_EXTERNAL} ]; then
    ROOT=$(dirname $(dirname $(dirname $(dirname $(readlink -f $0)))))
    CAFFE_MODELS_DIR=$ROOT/"caffe_mp"
    VOC_PATH=$ROOT/"datasets/VOC2012/Annotations"
    COCO_PATH=$ROOT/"datasets/COCO"
fi

FILE_LIST="file_list"
