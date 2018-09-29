# coding=utf-8

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from os.path import join
from augment import *
import numpy as np
import threading
import json
import cv2

train_base_path = '/home/pzw/hdd/dataset/ai_challenger/AgriculturalDisease_trainingset/'
valid_base_path = '/home/pzw/hdd/dataset/ai_challenger/AgriculturalDisease_validationset/'

data_path = "../dataset/train_valid/"
json_file_t = join(data_path, "AgriculturalDisease_train_annotations.json")
json_file_v = join(data_path, "AgriculturalDisease_validation_annotations.json")
with open(json_file_t, 'r') as fr:
    data_annos_t = json.load(fr, 'utf-8')
with open(json_file_v, 'r') as fr:
    data_annos_v = json.load(fr, 'utf-8')
data_annos = data_annos_t + data_annos_v
train_pairs, valid_pairs = train_test_split(data_annos, test_size=0.05, random_state=14)

CLASS_ = 61


class ThreadsafeIter:
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    def g(*args, **kw):
        return ThreadsafeIter(f(*args, **kw))
    return g


@threadsafe_generator
def data_generator(width, height, batch_size, train=True):
    """
    input:
        directories of train or validation,eg:train_dirs,valid_dirs
    output:
        yield X,y
    """
    if train:
        data_pairs = train_pairs
        np.random.shuffle(data_pairs)
    else:
        data_pairs = valid_pairs
    while True:
        for start in range(0, len(data_pairs), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(data_pairs))
            dirs_batch = data_pairs[start:end]
            # print(dirs_batch)
            for i, key in enumerate(dirs_batch):
                # print(key["image_id"])
                img = cv2.imread(join(data_path, "images/", (key["image_id"]).encode("utf-8")))[:, :, ::-1]
                # img = resize(img)   # 对于长宽比较大的图像做黑边填充
                img = resize_1(img)   # 对于所有图像做黑边填充
                img = cv2.resize(img, (width, height))
                if train:
                    img = randomCrop(img, size=(width, height))
                    img = randomHueSaturationValue(img,
                                                   hue_shift_limit=(-1, 1),  # 色调变换
                                                   sat_shift_limit=(-150, 150),
                                                   val_shift_limit=(-15, 15))
                    img = randomShiftScaleRotate(img,
                                                 shift_limit=(-0.1, 0.1),
                                                 scale_limit=(-0.5, 0.5),
                                                 rotate_limit=(-180, 180))
                    img = randomHorizontalFlip(img)
                    img = randomVerticallyFlip(img)
                x_batch.append(img)
                y = int(key["disease_class"])

                y = to_categorical(y, num_classes=CLASS_)
                y_batch.append(y)
            x_batch = np.array(x_batch, np.float32) / 255.
            y_batch = np.array(y_batch)
            yield x_batch, y_batch


def show_images(imgs, rows, cols, scale=2):
    import matplotlib.pyplot as plt

    figsize = (cols * scale, rows * scale)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    for i in range(rows):
        for j in range(cols):
            axes[i][j].imshow(imgs[i * cols + j])
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    plt.show()
    print "图像可视化结果"
    return axes


def data_visual(width, height, batch_size, flag=True):
    exam_x, exam_y = next(data_generator(width, height, batch_size, train=flag))
    show_images(exam_x, 4, 4, 2)


if __name__ == '__main__':
    if 0:
        with open(join(train_base_dir, 'AgriculturalDisease_train_annotations.json'), 'r') as fr:
            json_file = json.load(fr, 'utf-8')
        print type(json_file), len(json_file)
        print json_file[0]
        print json_file[0]['image_id']
        print json_file[0]['disease_class']
        for i in json_file:
            if i['image_id'] == 'b83f0d90-0dce-427a-9cdd-ab8d1f20724f___UF.GRC_YLCV_Lab 01690 - 副本.JPG'.decode('utf-8'):
                print i['disease_class']

        print train_dirs[0].split('/')[-1]

        y = 6
        y = y%6
        print "y=6", to_categorical(6%6, num_classes=CLASS_)
        print "y=7", to_categorical(7%6, num_classes=CLASS_)
        print "y=8", to_categorical(8%6, num_classes=CLASS_)
    data_visual(224, 224, 16)

