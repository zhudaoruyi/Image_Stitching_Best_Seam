#!/usr/bin/python
# coding=utf-8

"""融合算法提升9：能量法搜索最优拼接缝"""

import os
import cv2
import logging
import argparse
import numpy as np
import scipy.signal

from utils import *

logging.basicConfig(format='%(asctime)s %(levelname)s [%(module)s] %(message)s', level=logging.INFO)
log = logging.getLogger()


class Images:
    def __init__(self):
        self.imageList = []  # 所有需要拼接图像的像素矩阵集合
        self.poseList = None
        self.imageWidth = 100
        self.imageHeight = 100
        self.filenames = []

    def loadFromDirectory(self, dirPath=None):  # 主操作函数，输入所有拼接图像的文件夹路径
        log.info("Searching for images in: {}".format(dirPath))

        if dirPath == None:
            raise Exception("You must specify a directory path to the source images")
        if not os.path.isdir(dirPath):
            raise Exception("Directory does not exist!")

        self.filenames = self.getFilenames(dirPath)
        if self.filenames == None:
            log.error("Error reading filenames, was directory empty?")
            return False

        for i, img in enumerate(self.filenames):
            log.info("Opening file: {}".format(img))
            self.imageList.append(cv2.imread(img))  # 图像像素矩阵

        (self.imageWidth, self.imageHeight) = self.getImageAttributes(self.imageList[0])
        log.info("Data loaded successfully.")

    def getImageAttributes(self, img):
        return (img.shape[1], img.shape[0])

    def getFilenames(self, sPath):
        filenames = []
        for sChild in os.listdir(sPath):
            if os.path.splitext(sChild)[1][1:] in ['jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG']:
                sChildPath = os.path.join(sPath, sChild)
                filenames.append(sChildPath)
        if len(filenames) == 0:
            return None
        else:
            log.info("Found {} files in directory: {}".format(len(filenames), sPath))
            return sorted(filenames)  # 排序


class Stitch:
    def __init__(self, imagesObj):
        self.images = imagesObj.imageList
        self.imageWidth = imagesObj.imageWidth
        self.imageHeight = imagesObj.imageHeight
        self.filenames = imagesObj.filenames

    def rotateImageAndCenter(self, img, degreesCCW=0):
        scaleFactor = 1.0
        (oldY, oldX, oldC) = img.shape  # note: numpy uses (y,x) convention but most OpenCV functions use (x,y)
        M = cv2.getRotationMatrix2D(center=(oldX / 2, oldY / 2), angle=degreesCCW,
                                    scale=scaleFactor)  # rotate about center of image.
        newX, newY = oldX * scaleFactor, oldY * scaleFactor
        r = np.deg2rad(degreesCCW)
        newX, newY = (abs(np.sin(r) * newY) + abs(np.cos(r) * newX), abs(np.sin(r) * newX) + abs(np.cos(r) * newY))
        (tx, ty) = ((newX - oldX) / 2, (newY - oldY) / 2)
        M[0, 2] += tx  # third column of matrix holds translation, which takes effect after rotation.
        M[1, 2] += ty
        rotatedImg = cv2.warpAffine(img, M, dsize=(int(newX), int(newY)))
        return rotatedImg

    def scaleAndCrop(self, img, gray=False):
        """将最后的大图，去掉黑边（通过阈值分割，锁定感兴趣区域，找到最小外接矩形）"""
        if gray:
            grey = img
        else:
            grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(grey, 10, 255, cv2.THRESH_BINARY)
        out = cv2.findContours(thresh, 1, 2)
        cnt = out[0]
        x,y,w,h = cv2.boundingRect(cnt)
        crop = img[y:y+h, x:x+w]
        return x, y, w, h, crop

    def initScaling(self, imageWidth, inScale, outScale):
        # compute scaling values for input and output images 定义输入图片的尺寸，要不要做resize
        inWidth = int(imageWidth*inScale)
        # 定义画布的大小（拼接后图片的大小）
        windowSize = (inWidth*3, inWidth*3)  # this should be a large canvas, used to create container size
        outWidth = int(windowSize[0]*outScale)
        windowShift = [inWidth/2, inWidth/2]
        log.info("Scaling input image widths from {} to {}".format(imageWidth,inWidth))
        log.info("Using canvas container width (input x2): {}".format(windowSize[0]))
        log.info("Scaling output image width from {} to {}".format(windowSize[0], outWidth))
        return (inWidth, outWidth, windowSize, windowShift)

    def imageCombine(self, image):
        """相邻两图像放在一个list中，奇数复制"""
        imgt = list()
        i = 0
        iternum = int(len(image) / 2.)
        for _ in range(iternum):
            imgt.append([image[i], image[i + 1]])
            i += 2
        if len(image) % 2 != 0:
            imgt.append([image[-1], image[-1]])
        log.info("{} pairs of images".format(len(imgt)))
        return imgt

    def process(self,
                ratio=0.75,
                reprojThresh=4.0,
                save_dir="output/",
                showmatches=False):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        image_pairs = self.imageCombine(self.images)
        img_merge = list()

        for n, imgp in enumerate(image_pairs):
            imgt_sc = self.stitch(imgp, ratio=ratio, reprojThresh=reprojThresh, direc="horiz")
            img_name = "a" + str(n).zfill(4) + ".png"
            if showmatches:
                show_image(img_name, imgt_sc)
            # cv2.imwrite(save_dir + img_name, imgt_sc)
            # log.info("Stitched image {} saved successfully!".format(img_name))
            img_merge.append(imgt_sc)

        def stitch_loop(img_merge):
            image_two = self.imageCombine(img_merge)
            img_merge = list()
            for imgt in image_two:
                print "size of image 0 is {}, size of image 1 is {}".format(imgt[0].shape, imgt[1].shape)
                flag = False
                if max(imgt[0].shape[0], imgt[0].shape[1], imgt[1].shape[0], imgt[1].shape[1]) > 3456:
                    flag = True
                imgt_sc = self.stitch(imgt, ratio=ratio, reprojThresh=reprojThresh, direc="horiz", resize=flag)
                img_merge.append(imgt_sc)
                show_image("resu", imgt_sc)
            return img_merge

        c = 1
        flags = 1
        while flags:
            log.info("After merging {}, {} images left".format(c, len(img_merge)))
            img_merge = stitch_loop(img_merge)
            c += 1
            if len(img_merge) == 1:  # 最后拼接的图像数量为1时，拼接结束
                flags = 0
        cv2.imwrite(os.path.join(save_dir, "result.jpg"), img_merge[0])
        log.info("Stitched image saved successfully!")
        # self.logger.info("Found {} merged images".format(len(img_merge)))
        # for n, im in enumerate(img_merge):
        #     img_name = "a" + str(n).zfill(3) + ".png"
        #     cv2.imwrite(save_dir + img_name, im)
        #     self.logger.info("Stitched image {} saved successfully!".format(img_name))
        # if showmatches:
        #     for i in range(len(self.filenames)):
        #         show_image(os.path.basename(self.filenames[i]), self.images[i])

    def stitch(self, imgt, ratio=0.75, reprojThresh=1.0, direc="vertical", resize=False):
        """两张图像拼接"""
        imgA = imgt[0]
        imgB = imgt[1]
        # imgA = imgt[1]
        # imgB = imgt[0]
        if (np.array(imgA == imgB, dtype=int)).all():  # 单数的图片，避免做重复拼接
            return imgA
        if resize:
            per = 0.5
            imgA = cv2.resize(imgA, (int(imgA.shape[1] * per), int(imgA.shape[0] * per)))
            imgB = cv2.resize(imgB, (int(imgB.shape[1] * per), int(imgB.shape[0] * per)))
        win = max(imgA.shape[0], imgA.shape[1]) * 2
        base = np.zeros((win, win, 3), np.uint8)  # 创建一个大的空白窗口，将图像特征点对应一张张的贴上去
        if direc == "vertical":
            base[int(win / 8.): imgA.shape[0] + int(win / 8.),
            int(win / 4.): imgA.shape[1] + int(win / 4.)] = imgA  # 将第一张图片先贴到空白窗口上去，放在正中间
        if direc == "horiz":
            base[int(win / 4.): imgA.shape[0] + int(win / 4.),
            int(win / 4.): imgA.shape[1] + int(win / 4.)] = imgA  # 将第一张图片先贴到空白窗口上去，放在正中间
        container = np.array(base)
        show_image("cont", container)
        (containerKpts, containerFeats) = self.extractFeatures(container)
        (kps, feats) = self.extractFeatures(imgB)  # 第二张图片，被匹配的被图像变换的

        kpsMatches = self.matchKeypoints(kps,
                                         containerKpts,
                                         feats,
                                         containerFeats,
                                         ratio,
                                         reprojThresh)
        if kpsMatches == None:
            log.warning("kpsMatches == None!")
            return None

        (_, H, _) = kpsMatches
        res = cv2.warpPerspective(imgB, H, (win, win))
        # show_image("res", res)
        container = self.addImage(res, container)  # 交集拼接
        show_image("con", container)
        _, _, _, _, scaledContainer = self.scaleAndCrop(container)
        return scaledContainer

    def addImage(self, image, container):
        greyImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        greyContainer = cv2.cvtColor(container, cv2.COLOR_BGR2GRAY)
        ret, threshImage = cv2.threshold(greyImage, 10, 255, cv2.THRESH_BINARY)
        ret, threshContainer = cv2.threshold(greyContainer, 10, 255, cv2.THRESH_BINARY)
        intersect = cv2.bitwise_and(threshImage, threshContainer)  # 两个二值图片的交集，要求两图片的大小相同
        # kernel = np.ones((3, 3), dtype=np.int8)  # for dilation below

        img_c01 = cv2.bitwise_and(container, container, mask=intersect)  # container相交的部分
        mask_c02 = cv2.subtract(threshContainer, intersect)
        # mask_c02 = cv2.dilate(mask_c02, kernel, iterations=1)  # make the mask slightly larger so we don't get blank lines on the edges
        img_c02 = cv2.bitwise_and(container, container, mask=mask_c02)

        img_i01 = cv2.bitwise_and(image, image, mask=intersect)  # image相交的部分
        mask_i02 = cv2.subtract(threshImage, intersect)  # subtract the intersection, leaving just the new part to union
        # mask_i02 = cv2.dilate(mask_i02, kernel, iterations=1)  # make the mask slightly larger so we don't get blank lines on the edges
        img_i02 = cv2.bitwise_and(image, image, mask=mask_i02)  # apply mask

        # img_weighted = cv2.addWeighted(img_c01, 0.3, img_i01, 0.7, 0.0)  # 交集平均值（加权）
        # img_weighted = self.addweighted(img_c01, img_i01)  # 交集平均值（直接加权）
        img_weighted = self.addweighted_distance(img_c01, img_i01, intersect)  # 交集平均值（按照距离加权）
        # img_weighted = self.stitch_line(img_c01, img_i01, intersect)

        # show_image("img_c02", img_c02)
        # show_image("img_weighted", img_weighted)
        # show_image("img_i02", img_i02)

        con01 = cv2.add(img_c02, img_weighted)
        con02 = cv2.add(con01, img_i02)
        # show_image("con02", con02)
        # return con02
        return cv2.medianBlur(con02, 3)  # 中值滤波去掉椒盐噪声（毛刺边缘）

    def extractFeatures(self, image):
        """利用SIFT算法提取特征点，返回特征点的坐标和特征值"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        descriptor = cv2.xfeatures2d.SIFT_create()
        (kps, features) = descriptor.detectAndCompute(gray, None)
        log.info("Found {} keypoints in frame".format(len(kps)))
        kps = np.float32([kp.pt for kp in kps])
        return (kps, features)

    def matchKeypoints(self,
                       kpsA,
                       kpsB,
                       featuresA,
                       featuresB,
                       ratio,
                       reprojThresh):
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        log.info("Found {} raw matches".format(len(rawMatches)))

        matches = []
        # loop over the raw matches and remove outliers
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))
        log.info("Found {} matches after Lowe's test".format(len(matches)))
        # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

            # return the matches along with the homograpy matrix
            # and status of each matched point
            return (matches, H, status)
        log.warning("Homography could not be computed!")
        return None

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # initialize the output visualization image
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully
            # matched
            if s == 1:
                # draw the match
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        # return the visualization
        return vis

    def addweighted(self, image1, image2):
        """加权平均融合算法"""
        k = 0.3
        b1, g1, r1 = cv2.split(image1)
        b2, g2, r2 = cv2.split(image2)
        b = np.array(np.add(b1*k, b2*(1.0-k)), dtype=np.uint8)
        g = np.array(np.add(g1*k, g2*(1.0-k)), dtype=np.uint8)
        r = np.array(np.add(r1*k, r2*(1.0-k)), dtype=np.uint8)
        return cv2.merge([b, g, r])

    def addweighted_distance(self, image1, image2, intersect):
        """基于距离的渐入渐出加权平均融合算法"""
        log.info("Starting weighted merging...")
        rows, cols, dpt = image1.shape
        image = np.zeros((rows, cols, 3), np.uint8)
        x, y, w, h, _ = self.scaleAndCrop(intersect, gray=True)
        # for k in range(dpt):  # 逐通道逐像素的遍历，耗时较长，效率更低
        #     for i in range(rows):
        #         for j in range(cols):
        #             if intersect[i][j] != 0:
        #                 alp = float(i - y) / h  # todo: ***此处的横纵坐标容易弄错***
        #                 image[i][j][k] = image1[i][j][k] * (1. - alp) + image2[i][j][k] * alp
        # self.logger.info("Finishing weighted merging!")
        # return image
        b1, g1, r1 = cv2.split(image1)  # 通道分离，遍历一次，各通道计算赋值
        b2, g2, r2 = cv2.split(image2)
        b, g, r = cv2.split(image)
        for i in range(rows):
            for j in range(cols):
                if intersect[i][j] != 0:
                    alp = float(i - y) / h   # todo: ***此处的横纵坐标容易弄错***
                    b[i][j] = b1[i][j] * (1 - alp) + b2[i][j] * alp
                    g[i][j] = g1[i][j] * (1 - alp) + g2[i][j] * alp
                    r[i][j] = r1[i][j] * (1 - alp) + r2[i][j] * alp
        log.info("Finishing weighted merging!")
        return cv2.merge([b, g, r])

    def stitch_line(self, image1, image2, mask):
        energy_map = self.energy(image1, image2)  # 获取能量函数
        # show_image("mask", mask)
        line_points = self.minimum_seam(energy_map)  # 获取最佳拼接线
        image1_canvas = image1.copy()
        cv2.polylines(image1_canvas, [line_points], False, (0, 255, 0))
        # show_image("image1_canvas", image1_canvas)
        line_x = np.array([t[0] for t in line_points])
        line_y = np.array([t[1] for t in line_points])
        mask1 = mask.copy()
        mask2 = mask.copy()
        for i in range(mask1.shape[0]):
            if i < np.min(line_points, axis=0)[1] or i > np.max(line_points, axis=0)[1]:
                continue
            else:
                y_index = np.where(line_y == i)[0][0]
                for j in range(mask1.shape[1]):
                    mask1[i][j] = j < line_x[y_index]
                    mask2[i][j] = j >= line_x[y_index]
        # mask1 = np.multiply(mask1, mask)
        # mask2 = np.multiply(mask2, mask)
        # show_image("mask1", mask1)
        # show_image("mask2", mask2)
        mask1 = np.stack([mask1] * 3, axis=2)
        mask2 = np.stack([mask2] * 3, axis=2)
        image1 = np.multiply(image1, mask1)
        image2 = np.multiply(image2, mask2)
        # show_image("image1", image1)
        # show_image("image2", image2)
        return cv2.add(image1, image2)

    def energy(self, image1, image2):
        """计算能量函数：灰度差分图和纹理权重图"""
        w1, w2 = 0.1, 0.9
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        # 灰度差分图（亮度差异图）
        # gray_dif21 = np.abs(cv2.subtract(gray2, gray1))  # 这种像素值直接相减取绝对值的方式不可取，可以看到两个差结果不一样。
        gray_dif = cv2.absdiff(gray1, gray2)
        # show_image("gray diff", gray_dif)

        x = cv2.Sobel(gray_dif, -1, 1, 0)  # cv2.CV_16S
        y = cv2.Sobel(gray_dif, -1, 0, 1)
        text_dif = x + y
        # 梯度差分图（纹理结构差异图）
        text_dif = cv2.convertScaleAbs(text_dif)  # ***1
        # show_image("texture diff", text_dif)

        kernel = np.ones([3, 3], dtype=np.float32)
        kernel[1, 1] = 0
        # 亮度和纹理权重图
        weights_image1 = scipy.signal.convolve(gray_dif, kernel, "same")  # 此处卷积实质是对相邻8像素求和
        # print np.max(weights_image1)
        weights_image2 = scipy.signal.convolve(text_dif, kernel, "same")
        # print np.max(weights_image2)
        weights = w1 * weights_image1 + w2 * weights_image2  # ***2

        # 能量函数
        # energy1 = np.multiply(text_dif, weights)
        energy = np.multiply(text_dif, weights * (1.0 / (255. * 8.)))  # 归一化
        energy = np.array(energy, dtype=np.uint8)
        # show_image("energy_map", energy)
        return energy

    def minimum_seam(self, energy_map):
        x0, y0, w, h, cro = self.scaleAndCrop(energy_map, gray=True)  # 把重叠区取出来(crop)，为了计算拼接线
        out1 = self.find_seam(cro)  # 拼接线搜索算法之一
        # out1 = self.find_seam_v1(cro)  # 拼接线搜索算法之二
        # out1 = self.find_seam_v1(cro, begin_point=y0)  # 拼接线搜索算法之二

        re1 = zip(out1, range(len(out1)))
        pts = np.array([np.array(p) for p in re1])
        pts_origin = np.array([np.array([pt[0]+x0, pt[1]+y0]) for pt in pts])  # 小区域坐标转换到大画布上

        # green = (0, 255, 0)
        # # cv2.polylines(canvas, [pts], False, green)  # 画多段线
        # # 可视化：最佳拼接线可视化
        # inter = np.zeros_like(energy_map, dtype=np.uint8)
        # inter = np.stack([inter] * 3, axis=2)
        # cv2.polylines(inter, [pts_origin], False, green)
        # cv2.circle(inter, (x0, y0), 2, (0, 0, 255))
        # cv2.circle(inter, (x0+w, y0), 3, (0, 0, 255))
        # cv2.circle(inter, (x0, y0+h), 4, (0, 0, 255))
        # cv2.circle(inter, (x0+w, y0+h), 5, (0, 0, 255))
        # # inter[y0:y0+h, x0:x0+w] = canvas
        # show_image("inter", inter)
        return pts_origin

    def find_seam(self, cumulative_map):
        """利用动态规划算法，找到一条能量最低的拼接线"""
        r, c = cumulative_map.shape
        M = cumulative_map.copy()
        backtrack = np.zeros_like(M, dtype=np.int)
        for i in range(1, r):
            for j in range(0, c):
                # if intersect[i, j] != 0:   # 动态规划找的最低能量的点必须位于重叠区内
                    # 处理图像的左侧边缘，确保我们不会索引-1
                if j == 0:
                    idx = np.argmin(M[i - 1, j:j + 2])
                    backtrack[i, j] = idx + j
                    min_energy = M[i - 1, idx + j]
                else:
                    idx = np.argmin(M[i - 1, j - 1:j + 2])
                    backtrack[i, j] = idx + j - 1
                    min_energy = M[i - 1, idx + j - 1]
                M[i, j] += min_energy
        mask = np.ones((r, c), dtype=np.bool)
        j = np.argmin(M[-1])
        for i in reversed(range(r)):
            mask[i, j] = False
            j = backtrack[i, j]
        _, x = np.nonzero(np.bitwise_not(mask))
        return np.array(x)

    def find_seam_v1(self, cumulative_map, begin_point=None):
        """利用动态规划算法，找到一条能量最低的拼接线"""
        m, n = cumulative_map.shape
        output = np.zeros((m,), dtype=np.uint32)
        if not begin_point:
            output[-1] = np.argmin(cumulative_map[-1])
        else:
            output[-1] = begin_point
        for row in range(m - 2, -1, -1):
            prv_x = output[row + 1]
            if prv_x == 0:
                output[row] = np.argmin(cumulative_map[row, : 2])
            else:
                output[row] = np.argmin(cumulative_map[row, prv_x - 1: min(prv_x + 2, n - 1)]) + prv_x - 1
        return output


if __name__ == "__main__":
    flag = 0
    if flag:
        ap = argparse.ArgumentParser()
        ap.add_argument("-d", "--dir", default="/home/pzw/hdd/dataset/stitch_data/02feichengnongke1/",
                        help="directory of images (jpg, png)")
        ap.add_argument("-s", "--save_dir", default="output/0824/nk/",
                        help="save directory of stitched images")
        args = vars(ap.parse_args())

        imgs = Images()  # 图像预处理的类
        imgs.loadFromDirectory(args['dir'])  # 传入需要处理的图像的文件夹，该类返回一系列拼接类所需的参数

        mosaic = Stitch(imgs)  # 图像拼接的类
        mosaic.process(ratio=0.75,  # todo 可调参数
                       reprojThresh=4.0,
                       save_dir=args['save_dir'],
                       outScale=1.0)

    if not flag:
        imgs = Images()
        imgs.loadFromDirectory("data/test_images1/")

        mo = Stitch(imgs)

        mo.process(ratio=0.75,
                   reprojThresh=4.0,
                   save_dir="output/test_images1/",
                   showmatches=True)

