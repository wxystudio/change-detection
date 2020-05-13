# !/usr/bin/env python
# coding: utf-8

"""
@File      : post_process
@Copyright : INNNO Co.Ltd
@author    : xiaoyu wang
@Date      : 2019/9/4
@Desc      :
"""

import cv2
import copy
import math
import numpy as np


def label_rect(label):
    """
    求取label的外界矩形坐标及类型
    :param label: 1:房、2:水、3:路、4:树、5:其它、6:裂缝
    :return: [(x1, y1, x2, y2, type)]
    """

    h, w = label.shape
    min_area = int((h / 60.0) * (w / 60.0))
    cls_ids = [1, 2, 6]  # 房、水、裂缝

    results = []
    for cls_id in cls_ids:
        mask = copy.deepcopy(label)
        mask[mask != cls_id] = 0
        mask[mask == cls_id] = 255
        counters, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in counters:
            (x0, y0, w0, h0) = cv2.boundingRect(cnt)
            if cls_id in [1]:  # 房
                if w0 * h0 < min_area:
                    continue
            elif cls_id in [2]:  # 水
                if w0 * h0 < min_area * 4:
                    continue
            elif 6 == cls_id:  # 裂缝
                mini_len = math.sqrt(min_area)
                if w0 < mini_len and h0 < mini_len:
                    continue
            item_ret = [int(x0), int(y0), int(x0 + w0 - 1), int(y0 + h0 - 1), int(cls_id)]
            results.append(item_ret)
    return results


def compute_iou_ex(box, boxes, box_area, boxes_area):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = np.minimum(box_area, boxes_area[:])
    iou = intersection / union
    return iou


def non_max_suppression(boxes, scores, threshold):
    assert boxes.shape[0] > 0
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (y2 - y1) * (x2 - x1)
    ixs = scores.argsort()[::-1]
    pick = []
    while len(ixs) > 0:
        i = ixs[0]
        pick.append(i)
        iou = compute_iou_ex(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])
        remove_ixs = np.where(iou > threshold)[0] + 1
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)


def same_cls_nms(results, th=0.7):
    nms_results = []
    clses = list(set([ret[4] for ret in results]))
    for cls in clses:
        boxes, scores = [], []
        for ret in results:
            [x1, y1, x2, y2, t] = ret
            if cls != t:
                continue
            scores.append(x2 - x1 + y2 - y1)
            boxes.append([x1, y1, x2, y2])
        idxs = non_max_suppression(np.array(boxes), np.array(scores), th)
        for id in idxs:
            [x1, y1, x2, y2] = boxes[id]
            nms_results.append([int(x1), int(y1), int(x2), int(y2), int(cls)])
    return nms_results


def compute_overlap(rgn1, rgn2):
    [x1, y1, x2, y2] = rgn1
    [a1, b1, a2, b2] = rgn2
    intersection = max(min(x2, a2) - max(x1, a1), 0) * max(min(y2, b2) - max(y1, b1), 0)
    union = np.minimum((x2 - x1) * (y2 - y1), (b2 - b1) * (a2 - a1))
    return intersection / union


def filter_by_cls(results):
    cls_ids = np.array([2, 1, 6])  # sort by priority, # 建筑、裂缝、水
    ret_rgns = []
    for idx, cls in enumerate(cls_ids):
        if idx == len(cls_ids) - 1:
            break
        pos_rgns = [ret for ret in results if ret[4] == cls_ids[idx]]
        neg_rgns = [ret for ret in results if ret[4] in cls_ids[(idx + 1):]]
        for rgn in pos_rgns:
            [x1, y1, x2, y2, _] = rgn
            for i, n_rgn in enumerate(neg_rgns):
                if n_rgn is None:
                    continue
                [nx1, ny1, nx2, ny2, _] = n_rgn
                if compute_overlap([x1, y1, x2, y2], [nx1, ny1, nx2, ny2]) > 0:
                    neg_rgns[i] = None
        neg_rgns = [r for r in neg_rgns if r is not None]
        results = neg_rgns
        ret_rgns += pos_rgns
    ret_rgns += neg_rgns
    return ret_rgns


def water_filter(label, results):
    """按照均值方差去除误检
       去除整图上较大区域
    """
    water_id = 2  # label类别水坑
    height, width = label.shape
    max_size = min(height, width) * 0.4
    for idx, ret in enumerate(results):
        [x1, y1, x2, y2, t] = ret
        if t != water_id:
            continue
        if (x2 - x1) > max_size or (y2 - y1) > max_size:  # 最大水坑 < min(height, width) * 0.4
            results[idx] = None
    results = [ret for ret in results if ret is not None]
    return results


def crack_filter(label, results, dis=100):
    """去除房、水、路附近的裂缝
    """
    crack_id = 6  # label类别裂缝
    height, width = label.shape
    max_size = min(height, width) * 0.3
    mask = copy.deepcopy(label)
    mask[(mask == 1) | (mask == 2) | (mask == 3)] = 1  # 房、水、路
    mask[mask != 1] = 0
    rgns = []
    for idx, ret in enumerate(results):
        [x1, y1, x2, y2, t] = ret
        if t != crack_id:
            continue
        if (x2 - x1) > max_size or (y2 - y1) > max_size:  # 最大裂缝 < min(height, width) * 0.3
            results[idx] = None
        x1 = max(x1 - dis, 0)
        y1 = max(y1 - dis, 0)
        x2 = min(x2 + dis, width)
        y2 = min(y2 + dis, height)
        rgns.append([x1, y1, x2, y2])
        if np.sum(mask[y1:y2, x1:x2]) > 0:  # 房、水、路 dis 距离内，裂缝删除
            results[idx] = None

    results = [ret for ret in results if ret is not None]
    return results


def crack_morphology(img, results):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_result = cv2.blur(gray, (3, 3))
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    morp_result = cv2.morphologyEx(blur_result, cv2.MORPH_BLACKHAT, element)
    max_val = np.max(morp_result)
    _, imgOpened = cv2.threshold(morp_result, max_val * 0.5, 255, cv2.THRESH_BINARY)
    element1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    print(imgOpened.dtype, np.max(imgOpened))
    ret_img = cv2.morphologyEx(imgOpened.astype(np.uint8), cv2.MORPH_CLOSE, element1)
    return ret_img


def remove_unreasonable_box(img, label, results):
    # step1: NMS in same class, th=0.7
    results = same_cls_nms(results, 0.7)

    # step2: priority filter in class. building > water > crack
    # results = filter_by_cls(results)

    # step3: water filter
    # results = water_filter(label, results)

    # step4: crack filter
    # results = crack_filter(label, results, 100)

    # step4: crack morphology
    # ret_img = crack_morphology(img, results)
    # if len(ret_img.shape) == 2:
    #     ret_img = cv2.cvtColor(ret_img, cv2.COLOR_GRAY2BGR)
    # return results, ret_img
    return results


def water_last_process(img, label, results):
    water_id = 2
    for idx, ret in enumerate(results):
        [x1, y1, x2, y2, cls] = ret
        if cls != water_id:
            continue
        h, w = label.shape
        bh = y2 - y1
        bw = x2 - x1
        bimg = img[y1:y2, x1:x2, :]
        bmask = label[y1:y2, x1:x2]
        if np.sum(bmask[bmask == water_id]) / water_id > bw * bh * 0.6:
            extend_ratio = 0.2
            x1 = max(0, x1 - int(extend_ratio * bw))
            x2 = min(w, x2 + int(extend_ratio * bw))
            y1 = max(0, y1 - int(extend_ratio * bh))
            y2 = min(h, y2 + int(extend_ratio * bh))
            bimg = img[y1:y2, x1:x2, :]
            bmask = label[y1:y2, x1:x2]

        m_mean = np.mean(bimg[bmask == water_id])
        m_std = np.std(bimg[bmask == water_id])
        b_mean = np.mean(bimg[bmask != water_id])
        b_std = np.std(bimg)

        if not (math.fabs(m_mean - b_mean) > max(m_mean * 0.01, 3) or
                math.fabs(m_std - b_std) > max(m_std * 0.05, 2.3)):
            results[idx] = None

    # step2:
    results = [ret for ret in results if ret is not None]
    return results


def last_process(img, label, results):
    """1.水坑后处理，降假阳；
       2.裂缝后处理，降采样。
    """
    # step1: 水坑降假阳
    results = water_last_process(img, label, results)

    # step2: 裂缝降采样
    # crack_id = 6
    # crack_ids = [[idx, (ret[2] - ret[0] + ret[3] - ret[1])] for idx, ret in enumerate(results) if ret[4] == crack_id]
    # max_crack_num = 5
    # if len(crack_ids) > max_crack_num:
    #     crack_ids = sorted(crack_ids, key=lambda x: x[1], reverse=True)
    #     crack_ids = crack_ids[:min(max_crack_num + 3, len(crack_ids))]
    #     pos_crack_ids = np.random.choice([v[0] for v in crack_ids], max_crack_num, replace=False).tolist()
    #     for i, ret in enumerate(results):
    #         if ret[4] != crack_id:
    #             continue
    #         if i not in pos_crack_ids:
    #             results[i] = None
    #     results = [ret for ret in results if ret is not None]
    return results


def results_to_str(results):
    """输出结果转str,格式：x.y.w.h.t-x.y.w.h.t-...
    """
    ret_str = ""
    for item_ret in results:
        ret_str += ",".join([str(v) for v in item_ret]) + "-"
    return ret_str[:-1] if ret_str != "" else ret_str


def to_interface_ret(results):
    """
      输出mask转rect，与c++接口文档对接,(14:RESIDENTIAL，13：WATER，4:CRACK)
      :param label: 1:房、2:水、3:路、4:树、5:其它、6:裂缝
      :return: [(x, y, w, h, type)]
      """
    type_map = {1: 14, 2: 13, 6: 4}
    for idx, ret in enumerate(results):
        [x1, y1, x2, y2, cls] = ret
        results[idx] = [x1, y1, x2 - x1, y2 - y1, type_map[cls]]

    return results_to_str(results)


if __name__ == '__main__':
    pass
