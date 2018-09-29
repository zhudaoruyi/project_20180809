# coding=utf-8
"""柱面投影脚本"""

import os
import cv2
import math
import numpy as np


def cylindrical_projection(img, f, vertical=True):
    """柱面投影"""
    rows = img.shape[0]
    cols = img.shape[1]

    blank = np.zeros_like(img)
    w = int(cols / 2)  # 图像宽度的一半
    h = int(rows / 2)  # 图像高度的一半

    for y in range(rows):
        for x in range(cols):
            if vertical:
                # 竖直柱面投影 |||
                point_x = f * (x - w) / math.sqrt(math.pow(x - w, 2) + math.pow(f, 2)) + w
                point_y = f * (y - h) / math.sqrt(math.pow(x - w, 2) + math.pow(f, 2)) + h
            else:
                # 水平柱面投影 ==
                point_y = int(f * (y - h) / math.sqrt(math.pow(y - h, 2) + math.pow(f, 2)) + h)
                point_x = int(f * (x - w) / math.sqrt(math.pow(y - h, 2) + math.pow(f, 2)) + w)
            blank[point_y, point_x, :] = img[y, x, :]
    return blank


if __name__ == '__main__':
    flag = 0
    if flag:
        img_dir = "/home/pzw/hdd/projects/stitching/output/"
        img = cv2.imread(os.path.join(img_dir, '0824/fcnk2/a11.png'))
        waved_img = cylindrical_projection(img, 5000, vertical=True)
        cv2.imwrite(os.path.join(img_dir, '0827/a11.png'), waved_img)

        cv2.namedWindow("Horizontal cylindrical projection result image", 500)
        # cv2.imshow("Vertical cylindrical projection result image", cylindrical_projection(img, 800))
        cv2.imshow("Horizontal cylindrical projection result image", waved_img)
        cv2.waitKey(0)

    if not flag:
        img_dir = "/home/pzw/hdd/projects/stitching/output/0824/fcnk/2/"
        img = cv2.imread(os.path.join(img_dir, 'IMG_180518_014731_0174_RGB.JPG'))
        waved_img = cylindrical_projection(img, 4000, vertical=True)
        cv2.imwrite(os.path.join(img_dir, 'a95.png'), waved_img)

        # cv2.namedWindow("Vertical cylindrical projection result image", 600)
        cv2.imshow("Vertical cylindrical projection result image", waved_img)

        # cv2.namedWindow("Horizontal cylindrical projection result image", 600)
        # cv2.imshow("Horizontal cylindrical projection result image", waved_img)
        cv2.waitKey(0)
