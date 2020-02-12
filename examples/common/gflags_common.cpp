/*
All modification made by Cambricon Corporation: © 2019 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2019, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <gflags/gflags.h>

DEFINE_string(meanfile, "", "mean file used to subtract from the input image.");
DEFINE_string(meanvalue, "",
    "If specified, can be one value or can be same as image channels"
    " - would subtract from the corresponding channel). Separated by ','."
    "Either meanfile or meanvalue should be provided, not both.");
DEFINE_int32(threads, 1, "threads, should "
                         "be lower than or equal to 32 ");
DEFINE_int32(channel_dup, 1, "Enable const memory auto channel duplication. "
                         "Could improve performance when multithreading."
                         "Works only with apiversion 3");
DEFINE_int32(simple_compile, 1, "Use simple compile interface or not.");
DEFINE_string(images, "", "input file list");
DEFINE_string(labels, "", "label to name");
DEFINE_string(labelmapfile, "",
    "prototxt with infomation about mapping from label to name");
DEFINE_int32(fix8, 0, "fp16(0) or fix8(1) mode. Default is fp16");
DEFINE_int32(int8, -1, "invalid(-1), fp16(0) or int8(1) mode. Default is invalid(-1)."
    "If specified, use int8 value, else, use fix8 value");
DEFINE_int32(yuv, 0, "bgr(0) or yuv(1) mode. Default is bgr");
DEFINE_double(scale, 1, "scale for input data, mobilenet...");
DEFINE_string(logdir, "", "path to dump log file, to terminal stderr by default");
DEFINE_int32(fifosize, 2, "set FIFO size of mlu input and output buffer, default is 2");
DEFINE_string(mludevice, "0",
    "set using mlu device number, set multidevice seperated by ','"
    "eg 0,1 when you use device number 0 and 1, default: 0");
DEFINE_int32(apiversion, 2, "specify the version of CNRT to run.");
DEFINE_string(functype, "1H16",
    "Specify the core to run on the arm device."
    "Set the options to 1H16 or 1H8, the default is 1H16.");
DEFINE_int32(Bangop, 0, "Use Bang Operator or not");
DEFINE_int32(preprocess_option, 0, "Use it to choose Image preprocess:"
    "0: image resize to input size,"
    "1: center input size crop from resized image with shorter size = 256,"
    "2: center input size crop from resized image into 256 x 256.");
DEFINE_string(output_dtype, "INVALID",
    "Specifies the type of output in the middle of the model.");
