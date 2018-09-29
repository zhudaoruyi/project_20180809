#!/usr/bin/env python
# coding=utf-8

"""融合算法提升二：多分辨率融合算法"""

import os
import cv2
import imutils
import logging
import argparse
import numpy as np

logging.basicConfig(format='%(asctime)s %(levelname)s [%(module)s] %(message)s', level=logging.INFO)


# 把文件夹下的图片都读入进来，像素矩阵存入到list中
class Images:
    def __init__(self):
        self.logger = logging.getLogger()
        self.imageList = []  # 所有需要拼接图像的像素矩阵集合
        self.imageWidth = 100
        self.imageHeight = 100
        self.filenames = []

    def loadFromDirectory(self, dirPath=None):  # 主操作函数，输入所有拼接图像的文件夹路径
        self.logger.info("Searching for images in: {}".format(dirPath))

        if dirPath == None:
            raise Exception("You must specify a directory path to the source images")
        if not os.path.isdir(dirPath):
            raise Exception("Directory does not exist!")

        # grab filenames from directory
        self.filenames = self.getFilenames(dirPath)
        if self.filenames == None:
            self.logger.error("Error reading filenames, was directory empty?")
            return False

        # load the images
        for i,img in enumerate(self.filenames):
            self.logger.info("Opening file: {}".format(img))
            self.imageList.append(cv2.imread(img))  # 图像像素矩阵

        # set attributes for images (based on image 1), assumes all images are the same size
        (self.imageWidth, self.imageHeight) = self.getImageAttributes(self.imageList[0])

        self.logger.info("Data loaded successfully.")

    def getImageAttributes(self, img):
        return (img.shape[1], img.shape[0])

    def getFilenames(self, sPath):
        filenames = []
        for sChild in os.listdir(sPath):
            # check for valid file types here
            if os.path.splitext(sChild)[1][1:] in ['jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG']:
                sChildPath = os.path.join(sPath,sChild)
                filenames.append(sChildPath)
        if len(filenames) == 0:
            return None
        else:
            self.logger.info("Found {} files in directory: {}".format(len(filenames), sPath))
            return sorted(filenames)


class Stitch:
    def __init__(self, imagesObj):
        self.logger = logging.getLogger()
        self.images = imagesObj.imageList
        self.imageWidth = imagesObj.imageWidth
        self.imageHeight = imagesObj.imageHeight
        self.filenames = imagesObj.filenames

    def scaleAndCrop(self, img):
        """将最后的大图，去掉黑边（通过阈值分割，锁定感兴趣区域，找到最小外接矩形）"""
        # resized = imutils.resize(img, width=outWidth)
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
        self.logger.info("Scaling input image widths from {} to {}".format(imageWidth,inWidth))
        self.logger.info("Using canvas container width (input x2): {}".format(windowSize[0]))
        self.logger.info("Scaling output image width from {} to {}".format(windowSize[0], outWidth))
        return (inWidth, outWidth, windowSize, windowShift)

    def preprocessImages(self, inWidth):
        # pre-process the images: resize and align by vehicle yaw (helps the matcher)
        for i, img in enumerate(self.images):
            self.images[i] = imutils.resize(self.images[i], width=inWidth)  # reduce computation time 缩小图片，降低计算复杂度

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
        return imgt

    def process(self,
                ratio=0.75,
                reprojThresh=4.0,
                outScale=1.0,
                save_dir="output/",
                showmatches=False):
        inScale = 0.25
        # scale and rotate the input images accordingly 依次对输入图像缩放，降低计算复杂度（图像越大，特征点越多）
        (inWidth, outWidth, windowSize, windowShift) = self.initScaling(self.imageWidth, inScale, outScale)
        self.preprocessImages(inWidth)
        image_two = self.imageCombine(self.images)
        img_merge = list()
        for imgt in image_two:
            imgt_sc = self.stitch(windowSize, windowShift,
                                  outWidth, imgt,
                                  ratio=ratio, reprojThresh=reprojThresh)
            img_merge.append(imgt_sc)
        # def stitch_loop(img_merge):
        #     inScale = 0.8
        #     (inWidth, outWidth, windowSize, windowShift) = self.initScaling(img_merge[0].shape[0], inScale, outScale)
        #     self.preprocessImages(inWidth)
        #     image_two = self.imageCombine(img_merge)
        #     img_merge = list()
        #     for imgt in image_two:
        #         imgt_sc = self.stitch(windowSize, windowShift,
        #                               outWidth, imgt,
        #                               ratio=ratio, reprojThresh=reprojThresh)
        #         img_merge.append(imgt_sc)
        #     return img_merge
        #
        # c = 1
        # flags = 1
        # while flags:
        #     # print "01", len(img_merge)
        #     self.logger.info("After merging {}, {} images left".format(c, len(img_merge)))
        #     img_merge = stitch_loop(img_merge)
        #     c += 1
        #     if len(img_merge) == 1:  # 最后拼接的图像数量为1时，拼接结束
        #         flags = 0
        # 17-31张图片经历了以下循环
        # print "01", len(img_merge)
        # img_merge = stitch_loop(img_merge)
        # print "02", len(img_merge)
        # img_merge = stitch_loop(img_merge)
        # print "03", len(img_merge)
        # img_merge = stitch_loop(img_merge)
        # print "04", len(img_merge)
        # img_merge = stitch_loop(img_merge)
        # print "05", len(img_merge)
        # cv2.imwrite(save_dir + "/output03.png", img_merge[0])
        # self.logger.info("Stitched image saved successfully!")
        self.logger.info("Found {} merged images".format(len(img_merge)))
        for n, im in enumerate(img_merge):
            if n < 10:
                prefix = "wi0"
            else:
                prefix = "wi"
            cv2.imwrite(save_dir + prefix + str(n) + ".png", im)
            self.logger.info("Stitched image {} saved successfully!".format(n))

        if showmatches:
            cv2.imshow("Scaled Output1", img_merge[0])
            # cv2.imshow("Scaled Output2", img_merge[1])
            # cv2.imshow("Scaled Output3", img_merge[2])
            # cv2.imshow("Scaled Output4", img_merge[3])
            # cv2.imshow("Scaled Output5", img_merge[4])
            # cv2.imshow("Scaled Output6", img_merge[5])
            # cv2.imshow("Scaled Output7", img_merge[6])
            # cv2.imshow("Scaled Output8", img_merge[7])
            self.logger.info("Hit space bar to close viewer...")
            cv2.waitKey(0)

    def stitch(self, windowSize, windowShift,
               outWidth, imgt,
               ratio=0.75, reprojThresh=1.0):
        """两张图像拼接"""
        if (np.array(imgt[0] == imgt[1], dtype=int)).all():  # 单数的图片，避免做重复拼接
            return imgt[0]
        # create some empty images for use in combining results
        base = np.zeros((windowSize[1], windowSize[0], 3), np.uint8)  # 创建一个大的空白窗口，将图像特征点对应一张张的贴上去
        base[int(imgt[0].shape[0] / 2.):
             imgt[0].shape[0] + int(imgt[0].shape[0] / 2.),
        windowShift[0]*2 + int(imgt[0].shape[1] / 2.):
        imgt[0].shape[1] + windowShift[0]*2 + int(imgt[0].shape[1] / 2.)] = imgt[0]  # 将第一张图片先贴到空白窗口上去，放在正中间
        container = np.array(base)

        (containerKpts, containerFeats) = self.extractFeatures(container)
        (kps, feats) = self.extractFeatures(imgt[1])

        kpsMatches = self.matchKeypoints(kps,
                                         containerKpts,
                                         feats,
                                         containerFeats,
                                         ratio,
                                         reprojThresh)
        if kpsMatches == None:
            self.logger.warning("kpsMatches == None!")
            return None

        (_, H, _) = kpsMatches

        # apply transformation  匹配特征点所做的仿射变换
        res = cv2.warpPerspective(imgt[1], H, windowSize)
        cv2.namedWindow("res", 200)
        cv2.imshow("res", res)
        cv2.waitKey(0)
        # add image to container
        container = self.addImage(res, container, transparent=False)  # 交集拼接
        cv2.namedWindow("con", 200)
        cv2.imshow("con", container)
        cv2.waitKey(0)
        _, _, _, _, scaledContainer = self.scaleAndCrop(container)
        return scaledContainer

    def addImage(self, image, container, first=False, transparent=False):
        if transparent:
            con = cv2.addWeighted(container, 0.5, image, 0.5, 0.0)
            cv2.imshow("Container", con)
            cv2.waitKey(0)
            return con

        # if the container is empty, just return the full image
        if first:
            return image
        # else threshold both images, find non-overlapping sections, add to container
        greyImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        greyContainer = cv2.cvtColor(container, cv2.COLOR_BGR2GRAY)
        ret, threshImage = cv2.threshold(greyImage, 10, 255, cv2.THRESH_BINARY)
        ret, threshContainer = cv2.threshold(greyContainer, 10, 255, cv2.THRESH_BINARY)
        intersect = cv2.bitwise_and(threshImage, threshContainer)  # 两个二值图片的交集，要求两图片的大小相同

        img_c01 = cv2.bitwise_and(container, container, mask=intersect)  # container相交的部分
        mask_c02 = cv2.subtract(threshContainer, intersect)
        img_c02 = cv2.bitwise_and(container, container, mask=mask_c02)

        img_i01 = cv2.bitwise_and(image, image, mask=intersect)  # image相交的部分
        mask_i02 = cv2.subtract(threshImage, intersect)  # subtract the intersection, leaving just the new part to union
        img_i02 = cv2.bitwise_and(image, image, mask=mask_i02)  # apply mask

        img_weighted = self.mergeMulti(img_c01, img_i01, intersect)  # 高斯金字塔、拉普拉斯金字塔图像融合

        con01 = cv2.add(img_c02, img_weighted)
        con02 = cv2.add(con01, img_i02)
        return con02

    def extractFeatures(self, image):
        """利用SIFT算法提取特征点，返回特征点的坐标和特征值"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 特征点检测和提取
        descriptor = cv2.xfeatures2d.SIFT_create()  # SIFT算法提取特征点
        # descriptor = cv2.AKAZE_create()  # AKAZE算法提取特征点
        (kps, features) = descriptor.detectAndCompute(gray, None)
        self.logger.info("Found {} keypoints in frame".format(len(kps)))
        # convert the keypoints from KeyPoint objects to np
        kps = np.float32([kp.pt for kp in kps])
        # return a tuple of keypoints and features
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

        self.logger.info("Found {} raw matches".format(len(rawMatches)))

        matches = []
        # loop over the raw matches and remove outliers
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        self.logger.info("Found {} matches after Lowe's test".format(len(matches)))

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

        # otherwise, no homograpy could be computed
        self.logger.warning("Homography could not be computed!")
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

    def mergeMulti(self, image1, image2, intersect):
        self.logger.info("Multiple pyramid level merge...")
        image1_l = self.getLP(image1)
        image2_l = self.getLP(image2)
        inter_l = self.getLP(intersect)
        LS = list()
        for l1, l2, il in zip(image1_l, image2_l, inter_l):
            # ls = cv2.addWeighted(l1, 0.5, l2, 0.5, 0.0)  # 平均值加权
            ls = self.addweighted_distance(l1, l2, il)  # 依距离加权
            LS.append(ls)
        ls_ = LS[0]
        for i in range(1, len(LS)):
            ls_ = cv2.pyrUp((ls_))
            ls_ = cv2.add(ls_, LS[i])
        self.logger.info("Images merged successfully!")
        return ls_

    def getGP(self, image):
        G = image.copy()
        gp = [G]
        for i in range(3):
            self.logger.info("Shape of level {} is {}".format(i, gp[i].shape))
            G = cv2.pyrDown(G)
            gp.append(G)
        self.logger.info("Shape of level {} is {}".format(len(gp)-1, gp[-1].shape))
        return gp

    def getLP(self, image):
        gp = self.getGP(image)
        lp = [gp[-1]]
        # for i in range(len(gp)-1, 0, -1):
        for i in range(len(gp)-1, 0, -1):
            GE = cv2.pyrUp(gp[i])
            self.logger.info("gp {} shape {}".format(i, gp[i].shape))
            self.logger.info("GE {} shape {}".format(i, GE.shape))
            L = cv2.subtract(gp[i-1], GE)
            self.logger.info("gp {} shape {}".format(i-1, gp[i-1].shape))
            self.logger.info("L {} shape {}".format(i-1, L.shape))
            lp.append(L)
        return lp

    def addweighted_distance(self, image1, image2, intersect):
        """基于距离的加权平均融合算法"""
        self.logger.info("Starting weighted merging...")
        rows, cols, dpt = image1.shape
        image = np.zeros((rows, cols, 3), np.uint8)
        x, y, w, h, _ = self.scaleAndCrop(image1)
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
        self.logger.info("Finishing weighted merging!")
        return cv2.merge([b, g, r])


if __name__ == "__main__":
    if 0:
        ap = argparse.ArgumentParser()
        ap.add_argument("-d", "--dir", required=True, help="directory of images (jpg, png)")
        ap.add_argument("-os", "--outscale", default=1.0, type=float, help="ratio by which to scale the output image")
        ap.add_argument("-is", "--inscale", default=1.0, type=float,
                        help="ratio by which to scale the input images (faster processing)")
        args = vars(ap.parse_args())

        imgs = Images()  # 图像预处理的类
        imgs.loadFromDirectory(args['dir'])  # 传入需要处理的图像的文件夹，该类返回一系列拼接类所需的参数

        mosaic = Stitch(imgs)  # 图像拼接的类
        mosaic.process(ratio=0.75,  # todo 可调参数
                       reprojThresh=4.0,
                       outScale=args['outscale'],
                       inScale=args['inscale'])

    if 1:
        imgs = Images()
        # imgs.loadFromDirectory("/home/pzw/hdd/dataset/stitch_data/01feichengdongpo2/")
        imgs.loadFromDirectory("/home/pzw/hdd/dataset/stitch_data/01feichengdongpo/01/")
        # imgs.loadFromDirectory("/home/pzw/hdd/dataset/stitch_data/01feichengdongpo1/")
        # imgs.loadFromDirectory("/home/pzw/hdd/projects/stitching/output/0822/03/01/")

        mo = Stitch(imgs)
        mo.process(ratio=0.75,
                   reprojThresh=4.0,
                   save_dir="output/0822/07/02/",
                   outScale=1.0)

