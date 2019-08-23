#-*- coding: utf-8 -*-
"""
All modification made by Cambricon Corporation: © 2018 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2018, the respective contributors
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
"""

import os
import unittest

from google.protobuf import text_format

import caffe.draw
from caffe.proto import caffe_pb2

def getFilenames():
    """Yields files in the source tree which are Net prototxts."""
    result = []

    root_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', '..'))
    assert os.path.exists(root_dir)

    for dirname in ('models', 'examples'):
        dirname = os.path.join(root_dir, dirname)
        assert os.path.exists(dirname)
        for cwd, _, filenames in os.walk(dirname):
            for filename in filenames:
                filename = os.path.join(cwd, filename)
                if filename.endswith('.prototxt') and 'solver' not in filename:
                    yield os.path.join(dirname, filename)


class TestDraw(unittest.TestCase):
    def test_draw_net(self):
        for filename in getFilenames():
            net = caffe_pb2.NetParameter()
            with open(filename) as infile:
                text_format.Merge(infile.read(), net)
            caffe.draw.draw_net(net, 'LR')


if __name__ == "__main__":
    unittest.main()
