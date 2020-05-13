#!/usr/bin/env python
# coding: utf-8

"""
@File     : utils
@Copyright: INNNO Co.Ltd
@author   : xiaoyu wang
@Date     : 2019/11/12
@Desc     :
"""


def single_sliding_window(total_len, block_size, overlap):
    '''
    根据单尺度滑动窗，生成滑动结果list
    :param total_len: 长度
    :param block_size: 块大小
    :param overlap: 重叠像素数
    :return: 输出滑动坐标list
    '''

    if overlap is None:
        if 0 == int(total_len / block_size - 1):
            overlap = 0
        else:
            overlap = int((total_len % block_size) / int(total_len / block_size - 1))
    win_list = []
    bottom = 0
    bottom_border = total_len - block_size
    while bottom <= bottom_border:
        top = bottom + block_size
        win_list.append([bottom, top])
        bottom += block_size - overlap
    valid_len = total_len - overlap - bottom
    if valid_len > 0:
        win_list.append([bottom_border, total_len])
    return win_list


def sliding_window(shape, block_size, overlap=None):
    '''
    :describe: 滑动窗，输出坐标
    :param shape: (h, w) = shape
    :param block_size: 块大小
    :param overlap: 重叠大小
    :return: 输出图像block_size*block_size的框list
    '''

    (h, w) = shape
    if block_size > h or block_size > w:
        block_size = min(h, w)
    x_win_list = single_sliding_window(w, block_size, overlap)
    y_win_list = single_sliding_window(h, block_size, overlap)
    s_wins = []
    for y_win in y_win_list:
        for x_win in x_win_list:
            [x0, x1] = x_win
            [y0, y1] = y_win
            s_wins.append([x0, y0, x1, y1])
    return s_wins


if __name__ == '__main__':
    pass
