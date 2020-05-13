# !/usr/bin/env python
# coding: utf-8

"""
@File      : __init__.py
@Copyright : INNNO Co.Ltd
@author    : xiaoyu wang
@Date      : 2019/7/29
@Desc      :
"""

import os
import cv2
import numpy as np
from keras import backend as K
from .building import predict as building_predict
from .building import get_model as building_get_model
from .water_hole import predict as water_hole_predict
from .water_hole import get_model as water_hole_get_model

from .post_process import remove_unreasonable_box, label_rect, results_to_str, to_interface_ret, last_process


class Segment(object):
    def __init__(self):
        self._nets = []  # 保存所有网络结构
        self._classify_model = None

        self.is_detect_water = None
        self.is_detect_crack = None
        self.is_detect_building = None
        self.is_use_segment = False

        # init.
        self._init_model()

    def _init_model(self):
        self._nets.append([building_get_model(), "building"])  # 加载建筑模型
        self._nets.append([water_hole_get_model(), "water_hole"])  # 加载水坑模型
        # self._nets.append([crack_get_model(), "crack"])  # 加载裂缝模型
        # self._nets.append([multi_category_get_model(), "multi_category"])  # 加载多类别模型
        # self._nets.append([classify_get_model(), "classify"])  # 加载分类去误检模型
        pass

    def load_model(self, model_dir):
        """加载模型，成功返回0，否则返回1
        """
        try:
            for idx, [_, name] in enumerate(self._nets):
                model_path = os.path.join(model_dir, name + "_model.h5")
                # model_path = model_dir + "/" + name + "_model.h5"
                print("model_path:", model_path)
                (self._nets[idx][0]).load_weights(model_path)
                if "classify" == name:
                    self._classify_model = (self._nets[idx][0])
            return 0
        except Exception as err:
            print("load_model error info:\n", err)

    def set_type_cfg(self, is_detect_building, is_detect_water, is_detect_crack):
        self.is_detect_water = is_detect_water
        self.is_detect_crack = is_detect_crack
        self.is_detect_building = is_detect_building
        log_str_segment_type = "py-segment type is: "
        detect_types = []
        for key, value in zip(["water", "crack", "building"],
                              [self.is_detect_water, self.is_detect_crack, self.is_detect_building]):
            if value == 1:
                detect_types.append(key)
                self.is_use_segment = True

        if len(detect_types) == 0:
            log_str_segment_type += "empty."
        else:
            log_str_segment_type += ", ".join(detect_types)
        print(log_str_segment_type)

    def predict(self, img):
        if not self.is_use_segment:
            print("py-warning: py-segment type is empty, you may need call set_type_cfg fist.")

        min_s = 512
        h, w, _ = img.shape
        if min(h, w) < min_s:
            if h < min_s:
                img = cv2.resize(img, (int(min_s / h * w), min_s))
            if img.shape[1] < min_s:  # w
                img = cv2.resize(img, (min_s, int(min_s / img.shape[1] * img.shape[0])))

        # 1.房、2.水、3.路、4、树、5.地、6.裂缝
        label_ret = np.zeros(img.shape[:2], dtype=np.uint8)
        for idx, net in enumerate(self._nets):
            if "building" == net[1]:
                if not self.is_detect_building:
                    continue
                label = building_predict(net[0], img)
                label_ret[label == 1] = 1

            elif "water_hole" == net[1]:
                if not self.is_detect_water:
                    continue
                label = water_hole_predict(net[0], img)
                label_ret[(label_ret != 1) & (label == 1)] = 2
            else:
                raise ValueError(net[1] + " model not define.")

        if min(h, w) < min_s:
            label_ret = cv2.resize(label_ret, (w, h), cv2.INTER_NEAREST)
        return label_ret

    def clear(self):
        self._nets = []
        K.clear_session()
        pass
