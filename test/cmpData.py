#!/usr/bin/env python
# note in python2 1/3 = 0
# but in python3 1/3 = 0.3333
from __future__ import division
import sys
import os


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
    totalErr = 0.0
    totalNum = 0.0
    errorList = []
    for i in range(lineCount):
        numa = float(rfa.readline().strip("\n").split(" ")[-1])
        numb = float(rfb.readline().strip("\n").split(" ")[-1])
        totalErr += abs(numa - numb)
        totalNum += abs(numb)
        if numb != numa:
            errorList.append(i)

    rfa.close()
    rfb.close()

    return totalErr,totalNum,errorList


if __name__ == "__main__":
    totalErr,totalNum, errorList = cmpData(sys.argv[1], sys.argv[2])
    # sometimes the output is all zero.
    if totalNum == 0.0:
        print "errRate = %f" % totalErr
    else:
        print "errRate = %f" % (totalErr/totalNum)

#    if len(errorList) > 0:
#        print "errorList[0] %f" % errorList[0]
