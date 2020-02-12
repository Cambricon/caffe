#!/usr/bin/env python

import sys
import os

def cmpData(filenamea, filenameb):
    #os.system("rm tmpfilea tmpfileb")
    #os.system("cat %s | grep 'caffe.cpp:322' | grep -v 'label' > tmpfilea" % filenamea)
    #os.system("cat %s | grep 'caffe.cpp:322' | grep -v 'label' > tmpfileb" % filenameb)
    lineCount = -1
    for lineCount,line in enumerate(open(filenamea)):
        pass
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
    return (totalErr/totalNum),errorList

def cmpDirData(dir1, dir2):
    dir1 = dir1.rstrip('/')
    dir2 = dir2.rstrip('/')
    cmd_str = "ls %s > list_tmp" %dir1
    os.system(cmd_str)
    files = open("list_tmp")
    input_data = True
    for item in files.readlines():
        if (input_data):
           input_data=False
           continue
        item = item.strip()
        file1 = dir1 + '/' + item
        file2 = dir2 + '/' + item
        errRate, errorList = cmpData(file1, file2)
        print (file1 + " errRate = %f" %errRate)

if __name__ == "__main__":
    cmpDirData(sys.argv[1], sys.argv[2])
