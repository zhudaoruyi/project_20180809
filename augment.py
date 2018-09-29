# coding=utf-8

import cv2
import numpy as np


def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])
        h = cv2.add(h, hue_shift)
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def randomShiftScaleRotate(image,
                           shift_limit=(-0.0625, 0.0625),
                           scale_limit=(-0.1, 0.1),
                           rotate_limit=(-45, 45), aspect_limit=(0, 0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])  # degree
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(0, 0, 0,))
        # mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
        #                            borderValue=(0, 0, 0,))

    return image


def randomHorizontalFlip(image, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        # mask = cv2.flip(mask, 1)

    return image


def randomVerticallyFlip(image, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
    return image


def randomCrop(image, size=(299, 299), u=0.5):
    if np.random.random() < u:
        w, h = int(image.shape[0] * 1.4), int(image.shape[0] * 1.4)
        image = cv2.resize(image, (w, h))
        x = np.random.randint(0, w-size[0])
        y = np.random.randint(0, h-size[0])
        image = image[x:x + size[0], y:y + size[1]]
    return image


def resize(image, test=False):
    """针对于长宽比比较大的图像做黑边填充"""
    w, h = image.shape[:2]
    if float(w) / h < 1.2 and float(h) / w < 1.2:
        return image
    else:
        if float(w) / h >= 1.2:
            new_image = np.zeros([w, w, 3], dtype=np.uint8)
            pad = np.random.randint(0, w-h)
            if test:
                pad = int(0.5 * (w-h))
            new_image[0:w, pad:pad+h] = image
        else:
            new_image = np.zeros([h, h, 3], dtype=np.uint8)
            pad = np.random.randint(0, h-w)
            if test:
                pad = int(0.5 * (h-w))
            new_image[pad:pad+w, 0:h] = image
        return new_image


def resize_1(image, test=False):
    """对所有图像做黑边填充"""
    w, h = image.shape[:2]
    if w == h:
        return image
    else:
        if w > h:
            w1 = np.random.randint(h+1, w+1)
            pad = np.random.randint(0, w1-h)
            if test:
                w1 = w
                pad = int(0.5 * (w - h))
            new_image = np.zeros([w, w1, 3], dtype=np.uint8)
            new_image[0:w, pad:pad+h] = image
        else:
            h1 = np.random.randint(w+1, h+1)
            pad = np.random.randint(0, h1-w)
            if test:
                h1 = h
                pad = int(0.5 * (h - w))
            new_image = np.zeros([h1, h, 3], dtype=np.uint8)
            new_image[pad:pad+w, 0:h] = image
    return new_image


if __name__ == '__main__':
    # img = cv2.imread("/home/pzw/hdd/projects/ai_challenger/61_cla/output/low_conf/ecc4e38e-409c-4495-8cb2-1efd711fc162___FREC_Pwd.M 5146.JPG")[:, :, ::-1]
    img = cv2.imread("/home/pzw/hdd/projects/ai_challenger/61_cla/output/low_conf/e2043fde5b443190fe210bbe931d17e5.jpg")[:, :, ::-1]
    # img = resize(img)
    img = resize_1(img)
    print img.shape
    cv2.imshow("img", img)
    cv2.waitKey()
