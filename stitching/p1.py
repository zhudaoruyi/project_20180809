# coding=utf-8

import cv2
import logging
import numpy as np

logging.basicConfig(format='%(asctime)s %(levelname)s [%(module)s] %(message)s', level=logging.INFO)
log = logging.getLogger()

A = cv2.imread('/home/pzw/hdd/dataset/stitch_data/01feichengdongpo/01//IMG_180502_044437_0101_RGB.JPG')
B = cv2.imread('/home/pzw/hdd/dataset/stitch_data/01feichengdongpo/01/IMG_180502_044438_0102_RGB.JPG')
# A = cv2.resize(A, (200, 200))
# B = cv2.resize(B, (200, 200))
# generate Gaussian pyramid for A
G = A.copy()
gpA = [G]
for i in range(3):     # 将苹果进行高斯金字塔处理，总共3级处理
    log.info("size of level {} is {}".format(i, gpA[i].shape))
    G = cv2.pyrDown(G)
    gpA.append(G)
log.info("size of level {} is {}".format(len(gpA)-1, gpA[-1].shape))

j = 0
cv2.imshow("gpa%s"%j, gpA[j])
j += 1
cv2.imshow("gpa%s"%j, gpA[j])
j += 1
cv2.imshow("gpa%s"%j, gpA[j])
j += 1
cv2.imshow("gpa%s"%j, gpA[j])
cv2.waitKey(1)




if 1:
    # generate Gaussian pyramid for B
    G = B.copy()
    gpB = [G]
    for i in np.arange(3):  # #进行高斯金字塔处理，总共六级处理
        G = cv2.pyrDown(G)
        gpB.append(G)
    # generate Laplacian Pyramid for A
    lpA = [gpA[-1]]
    for i in np.arange(len(gpA)-1, 0, -1):  # 进行拉普拉斯金字塔处理，总共5级处理
        GE = cv2.pyrUp(gpA[i])
        log.info("gpA {} shape {}".format(i, gpA[i].shape))
        log.info("GE {} shape {}".format(i, GE.shape))
        L = cv2.subtract(gpA[i - 1], GE)
        log.info("gpA {} shape {}".format(i-1, gpA[i-1].shape))
        log.info("L {} shape {}".format(i-1, L.shape))
        lpA.append(L)
    # generate Laplacian Pyramid for B
    lpB = [gpB[-1]]
    for i in np.arange(len(gpA)-1,0,-1):    # 进行拉普拉斯金字塔处理，总共5级处理
        GE = cv2.pyrUp(gpB[i])
        L = cv2.subtract(gpB[i-1],GE)
        lpB.append(L)
    # Now add left and right halves of images in each level
    #numpy.hstack(tup)
    #Take a sequence of arrays and stack them horizontally
    #to make a single array.
    LS = []
    for la,lb in zip(lpA,lpB):
        rows,cols,dpt = la.shape
        ls = np.hstack((la[:,0:cols//2], lb[:,cols//2:]))    #将两个图像的矩阵的左半部分和右半部分拼接到一起
        LS.append(ls)
    # now reconstruct
    ls_ = LS[0]   #这里LS[0]为高斯金字塔的最小图片
    for i in xrange(1,len(gpA)):                        #第一次循环的图像为高斯金字塔的最小图片，依次通过拉普拉斯金字塔恢复到大图像
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, LS[i])                #采用金字塔拼接方法的图像
    # image with direct connecting each half
    # real = np.hstack((A[:,:cols/2],B[:,cols/2:]))   #直接的拼接
    # cv2.imwrite('Pyramid_blending2.jpg',ls_)
    cv2.imshow("Pyramid_blending", ls_)
    cv2.waitKey(0)
    # cv2.imwrite('Direct_blending.jpg',real)
