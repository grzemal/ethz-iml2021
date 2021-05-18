import timeit
import os
import numpy as np
import cv2
from PIL import Image

CHANNEL_NUM = 3

def cal_dir_stat(root):
    print("Starting Mean/Std Calculation")
    start = timeit.default_timer()

    im_pths = [os.path.join(root, d) for d in os.listdir(root) if d.endswith('.jpg')]
    pixel_num = 0  # store all pixel number in the dataset
    channel_sum = np.zeros(CHANNEL_NUM)
    channel_sum_squared = np.zeros(CHANNEL_NUM)

    for path in im_pths:
        im = cv2.imread(path)  # image in M*N*CHANNEL_NUM shape, channel in BGR order
        im = im / 255.0
        pixel_num += (im.size / CHANNEL_NUM)
        channel_sum += np.sum(im, axis=(0, 1))
        channel_sum_squared += np.sum(np.square(im), axis=(0, 1))

    bgr_mean = channel_sum / pixel_num
    bgr_std = np.sqrt(channel_sum_squared / pixel_num - np.square(bgr_mean))

    # change the format from bgr to rgb
    rgb_mean = list(bgr_mean)[::-1]
    rgb_std = list(bgr_std)[::-1]

    end = timeit.default_timer()
    print("elapsed time: {}".format(end - start))
    print("mean:{}\nstd:{}".format(rgb_mean, rgb_std))

    return rgb_mean, rgb_std

def get_min_size(directory):
    widths = []
    heights = []
    for path in os.listdir(directory):
        if path.endswith('.jpg'):
            im = Image.open(os.path.join(directory, path))
            widths.append(im.size[0])
            heights.append(im.size[1])

    print("Max Width: %f" % (max(widths)))
    print("Max Height: %f" % (max(heights)))

    print("Min Width: %f" % (min(widths)))
    print("Min Height: %f" % (min(heights)))

    return min(widths), min(heights)