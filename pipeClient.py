# !/usr/bin/env python
# coding: utf-8

"""
@File      : main
@Copyright : INNNO Co.Ltd
@author    : xiaoyu wang
@Date      : 2019/7/29
@Desc      :
"""

import os
import cv2
import time
import argparse
import numpy as np
import tensorflow as tf
from keras import backend as K
import win32pipe, win32file, win32con
from core import Segment, label_rect, remove_unreasonable_box, last_process, to_interface_ret

path = os.path
MSG_OK = "OK"


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


def send_msg(conn, msg):
    win32file.WriteFile(conn, msg_encode(msg))


def msg_encode(msg):
    if type(msg) == str:
        return msg.encode("utf-8")
    elif type(msg) == np.ndarray:
        msg = cv2.imencode('.png', msg)[1]
        return (np.array(msg)).tostring()
    else:
        raise ValueError("nonsupport type: %s ." % type(msg))


def msg_decode(msg, is_img=False):
    if not is_img:
        return msg.decode("utf-8")
    else:
        return cv2.imdecode(np.fromstring(msg, np.uint8), cv2.IMREAD_COLOR)


def pipe_img_read(conn, buff_size):
    # 接收数据
    img_data = b""
    while True:
        data = win32file.ReadFile(conn, buff_size, None)
        if data is None or len(data) < 2:
            raise AttributeError("read data error.", data)
        if data[1] == MSG_OK.encode("utf-8"):
            break
        img_data += data[1]
        send_msg(conn, MSG_OK)

    try:
        img = msg_decode(img_data, True)
        send_msg(conn, MSG_OK)
    except Exception as err:
        print("py-recv img error:", err)
        img = None
        send_msg(conn, err)
    return img


def pipe_img_send(conn, img, buff_size):
    str_encode = msg_encode(img)
    msg_len = len(str_encode)

    iter = 0
    while iter < msg_len:
        istart = iter
        iter = min(iter + buff_size, msg_len)
        win32file.WriteFile(conn, str_encode[istart:iter])

        data = win32file.ReadFile(conn, 1024, None)
        if data is None or len(data) < 2:
            raise AttributeError("py-read pipe data error.", data)
        msg = msg_decode(data[1])
        if msg == "OK":
            continue
        else:
            raise AttributeError("py-read pipe msg error.", msg)

    send_msg(conn, MSG_OK)


def argv_parse():
    parser = argparse.ArgumentParser(usage="it is use for start semantic segmentation sever.",
                                     description="change detect demo.")
    parser.add_argument("--pipe_name", required=True, type=str, help="pipe name.")
    parser.add_argument("--gpu_id", required=True, type=str, help="gpu id")
    parser.add_argument("--fraction", required=True, type=float, help="the fraction of gpu used")
    parser.add_argument("--model_dir", required=True, type=str, help="the h5 model directory.")
    return parser.parse_args()


def connect_pipe(pipe_name, wait_time=10):
    print("py-waite for pipe connect.")
    retPipe = -1
    errorPipe = None
    while True:
        time.sleep(0.2)
        try:
            retPipe = win32pipe.WaitNamedPipe(pipe_name, win32con.NMPWAIT_NOWAIT)
            if retPipe is None:
                break
        except Exception as err:
            errorPipe = err
            pass
        wait_time -= 1

    if retPipe is not None:
        print("py-link error: ", errorPipe)
        return

    conn = None
    try:
        conn = win32file.CreateFile(pipe_name,
                                    win32file.GENERIC_READ | win32file.GENERIC_WRITE,
                                    0, None,
                                    win32file.OPEN_EXISTING, win32file.FILE_ATTRIBUTE_NORMAL, None)
    except Exception as err:
        print("py-create file error: ", err)
        try:
            win32file.CloseHandle(conn)
        except:
            pass
    return conn


def process(argv, conn):
    gpu_id = argv.gpu_id
    fraction = argv.fraction
    model_dir = argv.model_dir
    print("argv:", argv)

    seg = None
    try:
        while True:
            try:
                data = win32file.ReadFile(conn, 1024, None)
                if data is None or len(data) < 2:
                    raise AttributeError("py-read pipe data error.", data)

                msg = msg_decode(data[1])
                items = msg.split("-")
                if len(items) == 0:
                    send_msg(conn, "py-params is missing.")
                    continue

                elif "init_py" == items[0]:
                    try:
                        gpu_config(gpu_id, fraction)
                        seg = Segment()
                        seg.load_model(model_dir)
                        send_msg(conn, MSG_OK)
                    except Exception as err:
                        send_msg(conn, "py-init env error.")
                        try:
                            seg.clear()
                            seg = None
                        except:
                            pass
                        try:
                            win32pipe.DisconnectNamedPipe(conn)
                        except Exception as err:
                            pass
                        break

                elif "exit" == items[0]:
                    if seg is not None:
                        try:
                            seg.clear()
                            seg = None
                        except:
                            # win32file.WriteFile(conn, msg_encode("py-clear session error."))
                            pass

                    time.sleep(1)
                    send_msg(conn, MSG_OK)
                    try:
                        win32pipe.DisconnectNamedPipe(conn)
                    except Exception as err:
                        pass
                    break

                elif "predict" == items[0]:
                    # 第一次应答
                    if seg is None:
                        send_msg(conn, "py-Py env not init.")
                        continue
                    try:
                        buff_size = int(items[1])
                        assert buff_size > 0
                        send_msg(conn, MSG_OK)
                    except:
                        send_msg(conn, "py-predict func param error.")
                        continue

                    # 接收图像
                    # print("py-recv img...")
                    img = pipe_img_read(conn, buff_size)
                    if img is None:  # 消息内已处理异常
                        print("py-img is none.")
                        continue

                    label = seg.predict(img)  # 分割label
                    # # 发送图像
                    # pipe_img_send(conn, label, buff_size)
                    # 发送外接矩形
                    # print("py-send ret...")
                    results = label_rect(label)  # 分割转外接矩形
                    results = remove_unreasonable_box(img, label, results)  # 区域去重
                    # results = seg.classify(img, label, results)  # 分类降假阳
                    results = last_process(img, label, results)  # 后处理

                    # # debug code...
                    # from test import draw_ret, cmap
                    # label[(label != 1) & (label != 2) & (label != 6)] = 0
                    # label_ret = draw_ret(draw_mask(img, label, cmap, alfa=0.4), results)
                    # cv2.imwrite(r"C:\Users\Administrator\Desktop\ret.jpg", label_ret)

                    str_ret = to_interface_ret(results)  # 转c++传输接口形式
                    print("py-ret:", str_ret)

                    if str_ret == "":
                        str_ret = MSG_OK
                    send_msg(conn, str_ret)
                    # print("py-img proc ok.")

                elif "type_cfg" == items[0]:
                    if seg is None:
                        send_msg(conn, "py-Py env not init.")
                        continue

                    if len(items) != 2:
                        send_msg(conn, "py-input type config param is empty.")
                        continue

                    try:
                        cfgs = [int(v) for v in items[1].split(",")]
                        [is_building, is_water, is_crack] = cfgs
                        seg.set_type_cfg(is_building, is_water, is_crack)
                        send_msg(conn, MSG_OK)
                    except:
                        send_msg(conn, "py-input type config param is error.")
                    pass

                else:
                    send_msg(conn, "py-params error.")
                    continue

            except Exception as err:
                print("py-exception:", err)
                break
    finally:
        try:
            win32pipe.DisconnectNamedPipe(conn)
        except:
            pass


def main():
    argv = argv_parse()
    print("py-argv:", argv)
    print("py-pid:", os.getpid())
    conn = connect_pipe(argv.pipe_name, 10)
    if conn is None:
        print("py-pip connect error.")
        return
    print("py-pip connected.")

    process(argv, conn)
    print("py-pipe client end.")


if __name__ == '__main__':
    main()

    # pyinstaller -F pipeClient.py
    # pyinstaller -D pipeClient.py
    # pyinstaller -D -w pipeClient.py # 去dos框
    pass
