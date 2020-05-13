# !/usr/bin/env python
# coding: utf-8

"""
@File      : demo
@Copyright : INNNO Co.Ltd
@author    : xiaoyu wang
@Date      : 2019/7/29
@Desc      :
"""

import os
import cv2
import time
import socket
import argparse
import numpy as np
from common import *
from core import Segment
from pipeClient import label_rect, remove_unreasonable_box, last_process

path = os.path

cmap = [
    [0, 0, 0],
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
    [0, 255, 255],
    [255, 0, 255],
]


def str_rect_to_rgn(str_ret):
    results = []
    rets = str_ret.split("-")
    for ret in rets:
        if ret == "":
            continue
        [x, y, w, h, type] = [int(v) for v in ret.split(",")]
        results.append([x, y, w, h, type])
    return results


def draw_ret(img, results):
    les = [1, 2, 6]  # 建筑、水、裂缝
    colors = [[255, 0, 0], [0, 255, 0], [255, 0, 255]]
    h, w = img.shape[:2]
    for ret in results:
        [x1, y1, x2, y2, t] = ret
        x1 = max(0, x1 - 5)
        y1 = max(0, y1 - 5)
        x2 = min(x2 + 5, w)
        y2 = min(y2 + 5, h)
        rgn = [x1, y1, x2, y2]
        # img = draw_rectangle(img, [rgn], colors[les.index(t)])
        img = draw_rectangle(img, [rgn], [0, 0, 255])
    return img


def init_config(gpu_id, fraction):
    print("run func init_config: gpu_id=%s, fraction=%f" % (gpu_id, fraction))
    gpu_config(gpu_id, fraction)


def load_model(model_dir):
    print("run func load_model: model_dir=%s" % model_dir)
    seg = Segment()
    seg.load_model(model_dir)
    return seg


def detect(seg, img):
    label = seg.predict(img)  # 分割label
    results = label_rect(label)  # 分割转外接矩形
    results = remove_unreasonable_box(img, label, results)  # 区域去重
    results = last_process(img, label, results)  # 后处理
    return label, results


def test():
    # img = cv2.imread(path.join(desktop, "0071_DSC_0366.JPG"))
    img = cv2.imread('E:\\wxy\\demo1205\\fake_B\\01000_fake_B.png')
    lalel, results = detect(seg, img)
    img1 = draw_ret(img, results)
    img2 = draw_mask(img, lalel)
    ret = merge_ret(img1, img2)
    # cv2.imwrite(path.join(desktop, "ret.jpg"), ret)
    cv2.imwrite('E:\\wxy\\demo1205\\result\\01000_fake_B.png', ret)
    pass


def test_for_dir(in_dir, out_dir):
    mkdir(out_dir)
    file_names = os.listdir(in_dir)
    for idx, file_name in enumerate(file_names):
        print(idx, file_name)
        img = cv2.imread(path.join(in_dir, file_name))
        if img is None:
            print("img is None.")
            continue
        lalel, results = detect(seg, img)
        img = draw_mask(img, lalel, alfa=0.4)
        cv2.imwrite(path.join(out_dir, file_name), img)
        cv2.imwrite(path.join(out_dir, file_name[:-4]+"_mask.jpg"), lalel*255)
    pass


if __name__ == '__main__':
    init_config("0", 1)
    seg = load_model("./model")
    seg.set_type_cfg(1, 0, 0)  # 房、水、裂缝

    # test()
    input_dir = "E:\\wxy\\demo1205\\fake_A"
    output_dir = "E:\\wxy\\demo1205\\resultA"
    test_for_dir(input_dir, output_dir)
    pass
