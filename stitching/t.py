# coding=utf-8

if 0:
    img = [str(i+1)+".png" for i in range(8)]

    print "原list ", img, '\n'


    def cut_list(image):
        imgt = list()
        i = 0
        iternum = int(len(image) / 2.)
        for _ in range(iternum):
            imgt.append([image[i], image[i+1]])
            i += 2
        if len(image) % 2 != 0:
            imgt.append([image[-1], image[-1]])
        return imgt


    imgb = cut_list(img)
    print "切片后的list ", imgb

if 0:
    import cv2
    img1 = cv2.imread("/home/pzw/hdd/dataset/wheat/01/2.jpeg")
    img2 = cv2.imread("/home/pzw/hdd/dataset/wheat/01/41.jpg")

    # cv2.imshow("img1", img1)
    # cv2.imshow("img2", img2)
    # cv2.waitKey(0)
    img3 = img2
    # cv2.imshow("img3", img3)
    # cv2.waitKey(1)
    img3[0:img1.shape[0], 0:img1.shape[1]] = img1
    cv2.imshow("img3", img3)
    cv2.waitKey(0)


if 0:
    import cv2
    import numpy as np
    from dense_sift import DenseRootSIFT

    image = cv2.imread('/home/pzw/hdd/dataset/stitch_data/01feichengdongpo/01/IMG_180502_044437_0101_RGB.JPG')
    image = cv2.resize(image, (400, 400))

    d2 = cv2.xfeatures2d.SIFT_create()
    (kps, feats) = d2.detectAndCompute(image, None)
    kps = np.float32([kp.pt for kp in kps])
    print feats.shape, '\n', kps.shape

    # dense_root_sift = DenseRootSIFT()
    # kps1, descriptors = dense_root_sift.detectAndCompute(image, window_size=None)
    # print descriptors.shape, '\n', kps1.shape
    # print kps1

    d2 = cv2.AKAZE_create()
    (kps, feats) = d2.detectAndCompute(image, None)
    kps = np.float32([kp.pt for kp in kps])
    print feats.shape, '\n', kps.shape


if 0:
    import cv2
    import matplotlib.pyplot as plt

    # cv2.namedWindow("i", 600)
    # cv2.imshow("i", cv2.imread("/home/pzw/hdd/projects/stitching/output/0822/07/02/p00.png"))
    # cv2.waitKey(0)
    img = plt.imread("/home/pzw/hdd/projects/stitching/output/0822/07/02/p00.png")
    plt.imshow(img)
    plt.show()


if 0:
    import cv2
    import os
    import logging
    import numpy as np

    logging.basicConfig(format='%(asctime)s %(levelname)s [%(module)s] %(message)s', level=logging.INFO)
    log = logging.getLogger()

    basedir = "/home/pzw/hdd/projects/stitching/output/0824/nk"
    imgs = sorted(os.listdir(basedir))
    log.info("拼接后的图片有{}张".format(len(imgs)))
    wh = list()
    for img in imgs:
        image = cv2.imread(os.path.join(basedir, img))
        w, h, _ = image.shape
        wh.append([w, h])
    wh = np.array(wh)
    log.info("拼接后的图片最大宽高是 {}, 最小的宽高是 {}".format(
        np.max(wh, axis=0), np.min(wh, axis=0)
    ))
    wm = np.array([np.abs(w-h) for (w,h) in wh])
    log.info("第{}张的宽高相差最大，相差{}像素".format(np.argmax(wm), np.max(wm)))

if 0:
    """柱面投影"""
    import cv2
    import math
    import numpy as np

    img = cv2.imread('/home/pzw/github/拼接/Python-Multiple-Image-Stitching/images/1.jpg')
    # img = cv2.imread('/home/pzw/github/拼接/Python-Multiple-Image-Stitching/test.jpg')
    # img = cv2.imread('/home/pzw/github/拼接/Python-Multiple-Image-Stitching/lunchroom_ultimate.jpg')


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


    # waved_img = cylindrical_projection(img, 800)
    cv2.imshow("Vertical cylindrical projection result image", cylindrical_projection(img, 800))
    cv2.imshow("Horizontal cylindrical projection result image", cylindrical_projection(img, 800, vertical=False))
    cv2.waitKey(0)

if 0:
    import cv2
    import numpy as np
    import scipy.signal

    k = np.ones([3, 3], dtype=np.float32)
    k[1,1] = 0
    print "卷积核\n", k, '\n'

    img = np.random.randn(9) * 10
    img = np.array(img, dtype=np.int)
    # print np.reshape(img, [-1, 3])
    # img = np.array([[1,4,3], [0, 5,1], [9,0,9]], dtype=np.float32)
    img = np.array(np.reshape(img, [-1, 3]), dtype=np.float32)
    print "图片为\n", img, '\n'

    con = np.multiply(img, k)
    print "mul\n", con, '\n'
    res = np.sum(con)
    print "result\n", res, '\n'

    print "res\n", scipy.signal.convolve(img, k, "same")


if 0:
    import numpy as np

    a = np.random.randn(20) * 10
    a = np.array(a, dtype=np.int)
    a = np.array(np.reshape(a, [-1, 4]), dtype=np.float32)
    print a, '\n'

    b = np.zeros_like(a)
    b[0:1] = a[0:1]
    print b
    # print np.min(a, axis=1), '\n'
    # print np.argmin(a, axis=1)

if 1:
    import numpy as np
    import cv2

    a = np.ones((5, 3), dtype=np.bool)
    print "a", a, '\n'
    a[2:4] = False
    print "a", a, '\n'
    a = np.bitwise_not(a)
    (y,x) = np.nonzero(a)
    print zip(x,y)
    print np.array(zip(x,y))

    canvas = np.zeros((300, 300, 3), dtype="uint8")
    green = (0, 255, 0)
    pts = np.array([[0,0],[50,0],[50,100],[80,200]])
    # cv2.line(canvas, (0, 0), (300, 300), green)
    cv2.polylines(canvas, [pts], False, green)
    cv2.imshow("Canvas", canvas)
    cv2.waitKey(0)

    red = (0, 0, 255)
    cv2.line(canvas, (300, 0), (0, 300), red, 3)
    cv2.imshow("Canvas", canvas)
    cv2.waitKey(0)

    cv2.rectangle(canvas, (10, 10), (60, 60), green)
    cv2.imshow("Canvas", canvas)
    cv2.waitKey(0)

    cv2.rectangle(canvas, (50, 200), (200, 225), red, 5)
    cv2.imshow("Canvas", canvas)
    cv2.waitKey(0)

    blue = (255, 0, 0)
    cv2.rectangle(canvas, (200, 50), (225, 125), blue, -1)
    cv2.imshow("Canvas", canvas)
    cv2.waitKey(0)

    canvas = np.zeros((300, 300, 3), dtype="uint8")
    (centerX, centerY) = (canvas.shape[1] // 2, canvas.shape[0] // 2)
    white = (255, 255, 255)

    for r in range(0, 175, 25):
        cv2.circle(canvas, (centerX, centerY), r, white)

    cv2.imshow("Canvas", canvas)
    cv2.waitKey(0)

    for i in range(0, 25):
        radius = np.random.randint(5, high=200)
        color = np.random.randint(0, high=256, size=(3,))

        pt = np.random.randint(0, high=300, size=(2,))

        cv2.circle(canvas, tuple(pt), radius, color, -1)

    cv2.imshow("Canvas", canvas)
    cv2.waitKey(0)





