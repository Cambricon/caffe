#!/usr/bin/env python
# note in python2 1/3 = 0
# but in python3 1/3 = 0.3333
from __future__ import division
import sys
import os
import numpy as np


def cmpData(filenamea, filenameb):
    #os.system("rm tmpfilea tmpfileb")
    #os.system("cat %s | grep 'caffe.cpp:322' | grep -v 'label' > tmpfilea" % filenamea)
    #os.system("cat %s | grep 'caffe.cpp:322' | grep -v 'label' > tmpfileb" % filenameb)
    lineCount = -1
    lineCount1 = -1
    for lineCount, _ in enumerate(open(filenamea)):
        pass
    for lineCount1, _ in enumerate(open(filenameb)):
        pass
    if (lineCount != lineCount1):
        print "The length is not same"
        sys.exit(1)

    lineCount += 1
    rfa = open(filenamea, "r")
    rfb = open(filenameb, "r")
    numa_list = []
    numb_list = []
    err_list = []
    total_err = 0.0
    numerator = 0.0
    denominator = 0.0
    for i in range(lineCount):
        numa_list.append(float(rfa.readline().strip("\n").split(" ")[-1]))
        numb_list.append(float(rfb.readline().strip("\n").split(" ")[-1]))
    for a,b in zip(numa_list,numb_list):
        if not (a == 0 and b == 0):
            err_list.append(abs(a-b)/((a ** 2 + b ** 2) ** 0.5))
        else:
            err_list.append(0)
    std = np.std(np.array(err_list))
    if any(numa_list) or any(numb_list):
        for a,b in zip(numa_list,numb_list):
            numerator += (a-b) ** 2
            denominator += a ** 2 + b ** 2
        total_err = (numerator / denominator) ** 0.5
    rfa.close()
    rfb.close()

    return total_err,std


if __name__ == "__main__":
    totalErr,std = cmpData(sys.argv[1], sys.argv[2])
    print "errRate = %f" % totalErr
    print "std = %f(should be close to 0)" % std
