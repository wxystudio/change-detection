# !/usr/bin/env python
# coding: utf-8

"""
@File      : common
@Copyright : INNNO Co.Ltd
@author    : xiaoyu wang
@Date      : 2019/7/29
@Desc      :
"""

import os
import re
import cv2
import copy
import shutil
import numpy as np
import tensorflow as tf
from keras import backend as K

path = os.path
desktop = r"C:\Users\Administrator\Desktop"


def gpu_config(gpuid, fraction=0.8):
    if gpuid == "-1":
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        return

    _gpu_options = tf.GPUOptions(allow_growth=False,
                                 per_process_gpu_memory_fraction=fraction,
                                 visible_device_list=gpuid)
    if not os.environ.get('OMP_NUM_THREADS'):
        config = tf.ConfigProto(allow_soft_placement=True,
                                gpu_options=_gpu_options)
    else:
        num_thread = int(os.environ.get('OMP_NUM_THREADS'))
        config = tf.ConfigProto(intra_op_parallelism_threads=num_thread,
                                allow_soft_placement=True,
                                gpu_options=_gpu_options)
    _SESSION = tf.Session(config=config)
    K.set_session(_SESSION)


def draw_mask(img, mask, cmap=None, alfa=0.5):
    """
    将mask按颜色显示在image上
    :param img: 原图
    :param mask: 按类别1、2、3、4...编码
    :param cmap: 颜色数组，默认由colormap()产生
    :param alfa: 不透明度
    :return: 显示的label mask
    """

    assert mask.dtype == np.uint8
    assert len(mask.shape) == 2

    if cmap is None:
        cmap = [
            [0, 0, 0],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [0, 255, 255],
            [255, 0, 255],
        ]

    max_label = np.max(mask)
    label_img = copy.deepcopy(img)
    for id in np.arange(1, max_label + 1):
        label_img[mask == id] = cmap[id]
    label_img = (label_img * alfa + (1 - alfa) * img).astype(np.uint8)
    return label_img


class CText(object):
    def __init__(self, path, is_clear=False):
        self.path = path
        fp = open(path, 'a')  # 无则创建，有则打开
        fp.close()
        if is_clear:
            self.clear()

    def clear(self):
        fp = open(self.path, 'r+')
        fp.truncate()
        fp.close()

    def append(self, text):
        fp = open(self.path, 'a')
        fp.write(text)
        fp.close()

    def readlines(self, is_split=False):
        lines = []
        fp = open(self.path)
        for item in fp.readlines():
            if is_split:
                lines.append(item.strip().split())
            else:
                lines.append(item.strip())
        fp.close()
        return lines


def walk_dir_file(dir, sub_dir="", _filter=""):
    '''
    递归获取文件
    :param dir:
    :param sub_dir: 子文件夹名
    :param _filter:
    :return:
    '''

    file_paths = []
    for dirpath, dirnames, filenames in os.walk(dir):
        if sub_dir and not re.search(sub_dir, dirpath):
            continue
        for filename in filenames:
            if _filter and not re.search(_filter, filename):
                continue
            file_paths.append(os.path.join(dirpath, filename))
    return file_paths


def mkdir(dir_name):
    """ 创建文件夹
    """
    if not os.path.exists(dir_name):
        print("mkdir: ", dir_name)
        os.makedirs(dir_name)


def rm(dist):
    # print('rm ' + dist + '\r\n')
    if os.path.isdir(dist):
        shutil.rmtree(dist)
    elif os.path.exists(dist):
        os.remove(dist)


def merge_ret(*imgs):
    """
    但类别图像结果显示
    :param *imgs: 图像序列连接
    :return: 合并连接后图像
    """
    if len(imgs) == 0:
        return None
    elif len(imgs) == 1:
        return imgs[0]

    seg_line = np.ones((imgs[0].shape[0], 5, imgs[0].shape[2])) * 128

    ret_imgs = []
    for idx, img in enumerate(imgs):
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if idx != 0:
            ret_imgs.append(seg_line)
        ret_imgs.append(img)
    return np.hstack(tuple(ret_imgs))


def draw_rectangle(image, bboxs, color=(0, 0, 255), thickness=2):
    if 2 == len(image.shape):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image = copy.deepcopy(image)
    for box in bboxs:
        [x0, y0, x1, y1] = box[:4]
        cv2.rectangle(image, (x0, y0), (x1, y1),
                      color=color,
                      thickness=thickness)
    return image


def cp(src, dist, is_print=True):
    '''
    :describe: 文件、文件夹拷贝
    :param src: 源文件、文件夹地址
    :param dist: 目标文件、文件夹地址
    :return:
    '''

    if path.isfile(src):
        dirname = path.dirname(dist)
        if not path.isdir(dirname):
            mkdir(dirname)
            if is_print:
                print('cp file:\r\n', 'src:' + src + '\r\n', 'dist:' + dist + '\r\n')
        return shutil.copy(src, dist)
    elif path.exists(src):
        if is_print:
            print('cp folder:\r\n', 'src:' + src + '\r\n', 'dist:' + dist + '\r\n')
        if path.exists(dist):
            shutil.rmtree(dist)
        return shutil.copytree(src, dist)
    else:
        print(src + " not existence!")
