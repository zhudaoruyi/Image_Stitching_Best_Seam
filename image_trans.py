#!/usr/bin/python
# coding=utf-8

"""Templates"""

import os
import cv2
import logging
import numpy as np
from argparse import ArgumentParser

from stitch import show_image

logging.basicConfig(format='%(asctime)s %(levelname)s [%(module)s] %(message)s', level=logging.INFO)
log = logging.getLogger()


def main():
    # img = cv2.imread("/home/pzw/hdd/dataset/stitch_data/288x216/fcdp20.jpg")
    # rows, cols = img.shape[:2]
    # print "origin image shape is", rows, cols
    # show_image("origin img", img)
    # # 平移变换
    # H = np.float32([[1, 0, 100], [0, 1, 50]])
    # img1 = cv2.warpAffine(img, H, (cols, rows))
    # show_image("flip img1", img1)
    # # 旋转变换
    # H = np.float32([[1, 0.5, 10], [0, 1, 5]])
    # img2 = cv2.warpAffine(img, H, (cols, rows))
    # show_image("rotate img2", img2)
    #
    # center = (cols / 2, rows / 2)
    # angle = 45
    # scale = 0.5
    # M = cv2.getRotationMatrix2D(center=center, angle=angle, scale=scale)
    # img3 = cv2.warpAffine(img, M, (cols, rows))
    # show_image("rotate img3", img3)
    #
    # # 仿射变换，需要仿射变换前后3个点
    # pts1 = np.float32([[20, 20], [80, 20], [20, 80]])
    # pts2 = np.float32([[10, 50], [60, 10], [50, 120]])
    # M = cv2.getAffineTransform(pts1, pts2)
    # img4 = cv2.warpAffine(img, M, (cols, rows))
    # show_image("img4", img4)
    #
    # # 投影变换，需要投影变换前后4个点
    # pts1 = np.float32([[50, 50], [150, 50], [150, 150], [50, 150]])
    # pts2 = np.float32([[0, 0], [100, 0], [100, 100], [0, 100]])
    # M = cv2.getPerspectiveTransform(pts1, pts2)
    # img5 = cv2.warpPerspective(img, M, (300, 300))
    # show_image("img5", img5)
    # img6 = cv2.warpPerspective(img, M, (100, 100))
    # show_image("img6", img6)
    img = cv2.imread("/home/pzw/hdd/dataset/stitch_data/21-43/IMG_180502_044507_0121_RGB.JPG")
    show_image("origin img", img)
    per = 0.5
    img = cv2.resize(img, (int(img.shape[1] * per), int(img.shape[0] * per)))
    show_image("src", img)

    return


if __name__ == "__main__":
    main()
