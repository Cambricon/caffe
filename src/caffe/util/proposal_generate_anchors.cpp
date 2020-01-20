/*
All modification made by Cambricon Corporation: © 2018-2019 Cambricon Corporation
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
#include "caffe/util/proposal_generate_anchors.hpp"

float *whctrs(std::vector<float> anchor) {
    float w, h, x_ctr, y_ctr;
    static float ret[4];
    w = anchor[2] - anchor[0] + 1;
    h = anchor[3] - anchor[1] + 1;
    x_ctr = anchor[0] + 0.5 * (w - 1);
    y_ctr = anchor[1] + 0.5 * (h - 1);
    ret[0] = w;
    ret[1] = h;
    ret[2] = x_ctr;
    ret[3] = y_ctr;
    return ret;
}

std::vector<std::vector<float> >
mk_anchors(std::vector<int> ws, std::vector<int> hs, float x_ctr, float y_ctr) {
    // ws hs should be int
    std::vector<std::vector<float> > anchors;
    for (unsigned int i = 0; i < ws.size(); i++) {
        std::vector<float> anchor(4);
        anchor[0] = x_ctr - 0.5 * (ws[i] - 1);
        anchor[1] = y_ctr - 0.5 * (hs[i] - 1);
        anchor[2] = x_ctr + 0.5 * (ws[i] - 1);
        anchor[3] = y_ctr + 0.5 * (hs[i] - 1);
        anchors.push_back(anchor);
    }
    return anchors;
}

std::vector<std::vector<float> >
ratio_enum(std::vector<float> anchor, std::vector<float> ratios) {
    float *ret = whctrs(anchor);
    float w = ret[0];
    float h = ret[1];
    float x_ctr = ret[2];
    float y_ctr = ret[3];
    float size = w * h;
    std::vector<float> size_ratios(ratios.size());
    std::vector<int> ws, hs;
    for (unsigned int i = 0; i < ratios.size(); i++) {
        size_ratios[i] = size / ratios[i];
        ws.push_back((int)(sqrt(size_ratios[i]) + 0.5));
        hs.push_back((int)(ws[i] * ratios[i] + 0.5));
    }
    std::vector<std::vector<float> > anchors = mk_anchors(ws, hs, x_ctr, y_ctr);
    return anchors;
}

void
generate_anchor_box(int H, int W, int feat_stride, int base_size,
                    std::vector<float> scales, std::vector<float> ratios,
                    bool pad_hw, float *anchors_cxcywh) {
    /*return the size of preprocessed anchors*/
    std::vector<std::vector<float> > anchors;
    anchors = generate_anchors(ratios, scales, base_size);

    preprocess_anchors(feat_stride, H, W, anchors, anchors_cxcywh, pad_hw);
}

void preprocess_anchors(int feat_stride, int H, int W,
                        std::vector<std::vector<float> > anchors, float *coords,
                        bool pad_hw) {
    int A = anchors.size();
    // A = 9 in official case

    int padd;
    if (pad_hw) {
        // TODO(fangzhou) fix this
        padd = (16 - (H * W) % 16) % 16;
    } else {
        padd = 0;
    }

    // create xyxy form anchor boxes
    // 4 A H W
    float x0, y0, x1, y1;
    for (int a = 0; a < A; a++) {
        for (int w = 0; w < W; w++) {
            for (int h = 0; h < H; h++) {
                x0 = w * feat_stride + anchors[a][0];
                y0 = h * feat_stride + anchors[a][1];
                x1 = w * feat_stride + anchors[a][2];
                y1 = h * feat_stride + anchors[a][3];
                coords[(0 * A + a) * (H * W + padd) + h * W + w] = x0 + 0.5 * (x1 - x0);
                coords[(1 * A + a) * (H * W + padd) + h * W + w] = y0 + 0.5 * (y1 - y0);
                coords[(2 * A + a) * (H * W + padd) + h * W + w] = x1 - x0 + 1;
                coords[(3 * A + a) * (H * W + padd) + h * W + w] = y1 - y0 + 1;
            }
        }
    }

    // 1 A*4 H W
    //for (int a = 0; a < A; a++) {
    //    for (int w = 0; w < W; w++) {
    //        for (int h = 0; h < H; h++) {
    //            coords[(a * 4 + 0) * (H * W + padd) + h * W + w] =
    //                   w * feat_stride + anchors[a][0];
    //            coords[(a * 4 + 1) * (H * W + padd) + h * W + w] =
    //                   h * feat_stride + anchors[a][1];
    //            coords[(a * 4 + 2) * (H * W + padd) + h * W + w] =
    //                   w * feat_stride + anchors[a][2];
    //            coords[(a * 4 + 3) * (H * W + padd) + h * W + w] =
    //                   h * feat_stride + anchors[a][3];
    //        }
    //    }
    //}

    // transfer to (c)x(c)ywh
    // float x0, y0, x1, y1;

    //  for (float i = 0; i < (H * W + padd) * A; i++) {
    //    x0 = coords[i + 0 * ( H * W + padd ) * A];
    //    y0 = coords[i + 1 * ( H * W + padd ) * A];
    //    x1 = coords[i + 2 * ( H * W + padd ) * A];
    //    y1 = coords[i + 3 * ( H * W + padd ) * A];
    //    coords[i + 2 * ( H * W + padd ) * A] = x1 - x0 + 1;//w'
    //    coords[i + 3 * ( H * W + padd ) * A] = y1 - y0 + 1;//h'
    //    coords[i + 0 * ( H * W + padd ) * A] = x0 + 0.5 * coords[i + 2 * ( H * W
    // + padd ) * A];//x_ctr
    //    coords[i + 1 * ( H * W + padd ) * A] = y0 + 0.5 * coords[i + 3 * ( H * W
    // + padd ) * A];//y_ctr
    // for (int i = 0; i < (H * W); i++) {
    //    for (int a = 0; a < A; a++) {
    //        x0 = coords[i + a * (H * W + padd) + 0 * (H * W + padd) * A];
    //        y0 = coords[i + a * (H * W + padd) + 1 * (H * W + padd) * A];
    //        x1 = coords[i + a * (H * W + padd) + 2 * (H * W + padd) * A];
    //        y1 = coords[i + a * (H * W + padd) + 3 * (H * W + padd) * A];
    //        coords[i + a * (H * W + padd) + 2 * (H * W + padd) * A] =
    //                x1 - x0 + 1;  // w'
    //        coords[i + a * (H * W + padd) + 3 * (H * W + padd) * A] =
    //                y1 - y0 + 1;  // h'
    //        coords[i + a * (H * W + padd) + 0 * (H * W + padd) * A] =
    //                x0 +
    //                0.5 *
    //                coords[i + a * (H * W + padd) + 2 * (H * W + padd) * A];  // x_ctr
    //        coords[i + a * (H * W + padd) + 1 * (H * W + padd) * A] =
    //                y0 +
    //                0.5 *
    //                coords[i + a * (H * W + padd) + 3 * (H * W + padd) * A];  // y_ctr
    //    }
    //}
}

std::vector<std::vector<float> >
generate_anchors(std::vector<float> ratios, std::vector<float> scales,
                 int base_size) {
    std::vector<float> base_anchor(4);
    base_anchor[0] = base_anchor[1] = 0;
    base_anchor[2] = base_anchor[3] = base_size - 1;
    std::vector<std::vector<float> > ratio_anchors =
            ratio_enum(base_anchor, ratios);

    std::vector<std::vector<float> > anchors;
    for (unsigned int i = 0; i < ratio_anchors.size(); i++) {
        std::vector<std::vector<float> > tmp = scale_enum(ratio_anchors[i], scales);
        for (unsigned int j = 0; j < tmp.size(); j++) {
            anchors.push_back(tmp[j]);
        }
    }
    return anchors;
}

std::vector<std::vector<float> >
scale_enum(std::vector<float> anchor, std::vector<float> scales) {
    float *ret = whctrs(anchor);
    float w = ret[0];
    float h = ret[1];
    float x_ctr = ret[2];
    float y_ctr = ret[3];

    std::vector<int> ws(scales.size()), hs(scales.size());
    for (unsigned int i = 0; i < scales.size(); i++) {
        ws[i] = (int)(w * scales[i]);
        hs[i] = (int)(h * scales[i]);
    }
    std::vector<std::vector<float> > anchors = mk_anchors(ws, hs, x_ctr, y_ctr);
    return anchors;
}

std::vector<std::vector<float> >
generate_anchors_pvanet(std::vector<float> ratios, std::vector<float> scales,
                        int base_size) {
#define ROUND(x) ((int)((x) + (float)0.5))

    // base box's width & height & center location
    const float base_area = (float)(base_size * base_size);
    const float center = (float)0.5 * (base_size - (float)1);
    // enumerate all transformed boxes
    std::vector<float> anchors_1d(4 * ratios.size() * scales.size());
    float* p_anchors = anchors_1d.data();
    for (int i = 0; i < ratios.size(); ++i) {
        // transformed width & height for given ratio factors
        const float ratio_w = (float)ROUND(sqrt(base_area / ratios[i]));
        const float ratio_h = (float)ROUND(ratio_w * ratios[i]);

        for (int j = 0; j < scales.size(); ++j) {
            // transformed width & height for given scale factors
            const float scale_w = (float)0.5 * (ratio_w * scales[j] - (float)1);
            const float scale_h = (float)0.5 * (ratio_h * scales[j] - (float)1);

            // (x1, y1, x2, y2) for transformed box
            p_anchors[0] = center - scale_w;
            p_anchors[1] = center - scale_h;
            p_anchors[2] = center + scale_w;
            p_anchors[3] = center + scale_h;

            p_anchors += 4;
        }  // endfor j
    }

    // pack anchors_1d to 2d vector (A, 4)
    std::vector<std::vector<float> > result;
    for (int i = 0; i < ratios.size() * scales.size(); ++i) {
        std::vector<float> tmp;
        tmp.push_back(anchors_1d[4 * i + 0]);
        tmp.push_back(anchors_1d[4 * i + 1]);
        tmp.push_back(anchors_1d[4 * i + 2]);
        tmp.push_back(anchors_1d[4 * i + 3]);
        result.push_back(tmp);
    }
    return result;
#undef ROUND
}

void
generate_anchor_box_pvanet(int H, int W, int feat_stride, int base_size,
                           std::vector<float> scales, std::vector<float> ratios,
                           bool pad_hw, float *anchors_cxcywh) {
    /*return the size of preprocessed anchors (A, 4) */
    std::vector<std::vector<float> > anchors;
    anchors = generate_anchors_pvanet(ratios, scales, base_size);

    preprocess_anchors(feat_stride, H, W, anchors, anchors_cxcywh, pad_hw);
}
