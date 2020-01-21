import xml.etree.ElementTree as ET
import sys
import os
import numpy as np
import json

def parse_rec(filename):
    tree = ET.parse(filename)
    root = tree.getroot()

    width = 0
    height = 0
    objects = []

    for elem in root.iterfind('size/width'):
        width = int(elem.text)
    for elem in root.iterfind('size/height'):
        height = int(elem.text)

    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        #obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [float(bbox.find('xmin').text) / width,
                              float(bbox.find('ymin').text) / height,
                              float(bbox.find('xmax').text) / width,
                              float(bbox.find('ymax').text) / height]
        objects.append(obj_struct)

    return objects

def voc_ap(rec, prec, use_07_metric=False):
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def IOU(box0, box1):
    ixmin = np.maximum(box0[0], box1[0])
    iymin = np.maximum(box0[1], box1[1])
    ixmax = np.minimum(box0[2], box1[2])
    iymax = np.minimum(box0[3], box1[3])
    iw = np.maximum(ixmax - ixmin, 0.)
    ih = np.maximum(iymax - iymin, 0.)
    inter = iw * ih
    union = (box0[2] - box0[0]) * (box0[3] - box0[1]) + \
            (box1[2] - box1[0]) * (box1[3] - box1[1]) - \
            inter
    return float(inter) /union

def voc_eval(gt, result, min_overlap_threshold = 0.5):
    ap_sum = 0.

    for classname in gt.keys():
        if result.has_key(classname):
            class_gt = gt[classname]
            class_result = result[classname]
            score = np.array([obj[1] for obj in class_result])
            sorted_index = np.argsort(-score)

            ## Get the the total number of reference bbox and the number
            ## doesn't include these bbox which belong to 'difficult'.
            npos = 0;
            for idx, gt_bbox in enumerate(class_gt):
                if not gt_bbox[2]:
                    npos = npos + 1

            result_num = len(class_result)
            tp = np.zeros(result_num)
            fp = np.zeros(result_num)
            for i in range(result_num):
                index = sorted_index[i]
                overmax = 0.
                idx_max = 0
                for idx, gt_bbox in enumerate(class_gt):
                    if gt_bbox[0] == class_result[index][0]:
                        overlap = IOU(gt_bbox[1], class_result[index][2])
                        if overlap > overmax:
                            overmax = overlap
                            idx_max = idx

                if overmax > min_overlap_threshold:
                    ## judge whether the bbox is difficult or not.
                    if not class_gt[idx_max][2]:
                        tp[i] = 1.
                        ## if the reference box is used, then sheild it to
                        ## avoid repeat detection.
                        class_gt[idx_max][0] = "xxx"
                else:
                    fp[i] = 1.
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / float(npos)
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ## if using VOCdevkit 2007, please set use_07_metric to True
            ## Currently, Caffe uses VOCdevkit 2012 to test, so use_07_metric is
            ## set to False
            cls_ap = voc_ap(rec, prec, use_07_metric=False)
            ap_sum = ap_sum + cls_ap
            print "AP of %s: %f"%(classname, cls_ap)
        else:
            continue
    return ap_sum / len(gt)

def parse_output(input):
    objs = []
    f = open(input)
    for line in f.readlines():
        objs.append(line.split(' '))
    f.close()
    return objs

def meanAP(image_list, result_dir, golden_dir):
    result_dir = result_dir.rstrip('/')
    golden_dir = golden_dir.rstrip('/')

    detect_final = {}
    f = open(image_list)
    for img in f.readlines():
        img_name = os.path.splitext(os.path.basename(img))[0]
        result_objs = parse_output(result_dir + '/' + img_name + '.txt')
        for i in range(len(result_objs)):
            label = result_objs[i][0]
            confidence = float(result_objs[i][1])
            BB = [float(result_objs[i][2]), float(result_objs[i][3]),
                  float(result_objs[i][4]), float(result_objs[i][5])]
            if detect_final.has_key(label):
                detect_final.get(label).append([img_name, confidence, BB])
            else:
                detect_final.setdefault(label, []).append([img_name, confidence, BB])
    f.close()

    ground_truth = {}
    f = open(image_list)
    for img in f.readlines():
        img_name = os.path.splitext(os.path.basename(img))[0]
        gt_objs = parse_rec(golden_dir + '/' + img_name + '.xml')
        gt_num = len(gt_objs)
        for i in range(gt_num):
            label = gt_objs[i]['name']
            difficult = gt_objs[i]['difficult']
            if  ground_truth.has_key(label):
                ground_truth.get(label).append([img_name, gt_objs[i]['bbox'], difficult])
            else:
                ground_truth.setdefault(label, []).append([img_name, gt_objs[i]['bbox'], difficult])
    f.close()

    return voc_eval(ground_truth, detect_final)

def update_json_meanAp(json_data, meanAp, key):
    if isinstance(json_data, dict):
        for k in json_data:
            if k == key:
               json_data[k] = round(meanAp, 2)
            elif isinstance(json_data[k], dict):
               update_json_meanAp(json_data[k], meanAp, key)

if __name__ == "__main__":
    mAp = meanAP(sys.argv[1], sys.argv[2], sys.argv[3])
    print  "mAP: %f"%(mAp)
    input_json_file = os.getenv('OUTPUT_JSON_FILE','')
    if os.path.isfile(input_json_file):
        file_in = open(input_json_file, "r")
        json_data = json.load(file_in)
        update_json_meanAp(json_data, mAp * 100, 'meanAp')
        file_in.close()
        file_out = open(input_json_file, "w")
        json.dump(json_data, file_out,indent=2)
        file_out.close()
    else:
        exit()
