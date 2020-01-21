# Cambricon Caffe

[![Build Status](https://travis-ci.com/Cambricon/caffe.svg?branch=master)](https://travis-ci.com/Cambricon/caffe)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

To support Cambricon deep learning processor, the open source deep learning programming framework [BVLC-Caffe](https://github.com/BVLC/caffe) has been modified. New functions such as off-line multi-core inference, online fusion mode, support of multiple cards and others are developed. Cambricon Caffe focuses on inference, it is dedicated to improving [BVLC-Caffe](https://github.com/BVLC/caffe) performance when running on Machine Learning Unit(MLU).

## Prerequisites
Cambricon Caffe has several dependencies as same as [BVLC-Caffe](https://github.com/BVLC/caffe) does, please refer to [caffe.berkeleyvision.org](https://caffe.berkeleyvision.org/installation.html) for details .

## Building
You need to firstly clone [Cambricon Caffe](https://github.com/Cambricon/caffe), and then go to **scripts** folder to compile Cambricon Caffe: 
```
git clone git@github.com:Cambricon/caffe.git
cd caffe/scripts
```
To build Cambricon Caffe, you could use **build_cambriconcaffe.sh**, which is in the scripts folder. It has three options:
- -debug: build Cambricon Caffe with debug information.
- -release: build Cambricon Caffe for production environment. This is the default build type.
- -platform: specify platform argument. Supported arguments are x86, arm32, arm64. Default platform is x86.

### x86
```
./build_cambriconcaffe.sh -platform x86
```

### arm32
Please download [cross toolchain](https://releases.linaro.org), e.g. arm-linux-gnueabihf-4.8.3-201404, and set **CROSS_TOOLCHAIN_PATH** environment to point to the tool chain downloaded:
```
export CROSS_TOOLCHAIN_PATH=your_toolchain_path/bin  // please replace your_toolchain_path with your actual path
```
There is another dependent library set **arm32_linux_lib**, which is necessary for the compiling of Cambricon Caffe. It has been pre-compiled and is available for downloading on Cambricon's FTP site. Please clone caffe_boost, then run **download_dependency.sh**. The script will help download it.

The download_dependency.sh script needs one argument. The argument meaning is listed below: 
- 1: download arm32_linux_lib.
- 2: download android_lib_r17b.

Firstly, download the library set.
```
git clone git@github.com:Cambricon/caffe_boost.git
cd caffe_boost/scripts
./download_dependency.sh 1
```
Then set **ARM32_LINUX_LIB_ROOT** environment variable for arm32_linux_lib:
```
export ARM32_LINUX_LIB_ROOT=your_lib_path  // please replace your_lib_path with your actual path
```
Once you have set up above two environment variables, you are ready to compile Cambricon Caffe for arm32 platform.
```
./build_cambriconcaffe.sh -platform arm32
```

### arm64
For arm64, please download **android-ndk-r17b** firstly, it can be downloaded from [NDK's](https://developer.android.google.cn/ndk) official website. Then place it in **/opt/shared/Android/Ndk/** directory(if not exists, please create one) and set **ARM64_R17_NDK_ROOT** environment as follows:
```
export ARM64_R17_NDK_ROOT=/opt/shared/Android/Ndk/android-ndk-r17b
```
There is another dependent library set **android_lib_r17b**. It also has been pre-compiled and is available for downloading on Cambricon's FTP site. Please input parameter **2** for download_dependency.sh.
```
git clone git@github.com:Cambricon/caffe_boost.git
cd caffe_boost/scripts
./download_dependency.sh 2
```
Finally, set **ARM64_R17_ANDROID_LIB_ROOT** environment variable for android_lib_r17b: 
```
export ARM64_R17_ANDROID_LIB_ROOT=your_android_lib_path  // please replace your_android_lib_path with your actual path
```
Once you have finished setting the environment variables, you could compile Cambricon Caffe for arm64 platform:
```
./build_cambriconcaffe.sh -platform arm64
```

## License and Citation
Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }

***
 *Other names and brands may be claimed as the property of others

# SSD: Single Shot MultiBox Detector
This repository contains merged code issued as pull request to BVLC caffe written by:
[Wei Liu](http://www.cs.unc.edu/~wliu/), [Dragomir Anguelov](https://www.linkedin.com/in/dragomiranguelov), [Dumitru Erhan](http://research.google.com/pubs/DumitruErhan.html), [Christian Szegedy](http://research.google.com/pubs/ChristianSzegedy.html), [Scott Reed](http://www-personal.umich.edu/~reedscot/), [Cheng-Yang Fu](http://www.cs.unc.edu/~cyfu/), [Alexander C. Berg](http://acberg.com).

Original branch can be found at https://github.com/weiliu89/caffe/tree/ssd.

Read our [wiki page](https://github.com/intel/caffe/wiki/SSD:-Single-Shot-MultiBox-Detector) for more details.

# Darknet
If you use Darknet in your research please cite it!

    @misc{darknet13,
      author =   {Joseph Redmon},
      title =    {Darknet: Open Source Neural Networks in C},
      howpublished = {\url{http://pjreddie.com/darknet/}},
      year = {2013--2016}
    }

# YOLOv2
If you use YOLOv2 in your work please cite it!

    @article{redmon2016yolo9000,
      title={YOLO9000: Better, Faster, Stronger},
      author={Redmon, Joseph and Farhadi, Ali},
      journal={arXiv preprint arXiv:1612.08242},
      year={2016}
    }

# YOLOv3
If you use YOLOv3 in your work please cite it!

    @article{yolov3,
      title={YOLOv3: An Incremental Improvement},
      author={Redmon, Joseph and Farhadi, Ali},
      journal = {arXiv},
      year={2018}
    }
