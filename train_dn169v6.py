#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import logging
from os.path import join
from data_generator2 import *   # 数据生成器
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.applications import InceptionV3, Xception, InceptionResNetV2, \
    DenseNet121, DenseNet169, DenseNet201, \
    NASNetLarge, NASNetMobile, \
    ResNet50
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, GlobalMaxPooling2D
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, TensorBoard

logging.basicConfig(format="%(asctime)s %(levelname)s [%(module)s] %(message)s", level=logging.INFO)
log = logging.getLogger()


def set_gpu(gpu_memory_frac=0.3):
    import tensorflow as tf
    import keras.backend as K
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_frac
    sess = tf.Session(config=config)
    K.set_session(sess)


def lr_schedule(epoch):
    lr = 1e-2
    if epoch > 200:
        lr = 1e-4
    elif epoch > 90:
        lr = 0.5e-5
    elif epoch > 60:
        lr = 1e-4
    elif epoch > 40:
        lr = 1e-3
    log.info("Learning rate:{}".format(lr))
    return lr


def get_model(MODEL, width, height):
    w = 1
    if w:
        base_model = MODEL(weights='imagenet', include_top=False, input_shape=(width, height, 3))
    else:
        base_model = MODEL(weights=None, include_top=False, input_shape=(width, height, 3))
    x = base_model.output
    # x = GlobalAveragePooling2D()(x)
    x = GlobalMaxPooling2D(name="gmp")(x)
    log.info("global average pooling shape \n{}".format(x))
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(CLASS_, activation='sigmoid')(x)
    # predictions = Dense(CLASS_, activation='softmax')(x)

    for layer in base_model.layers:
        layer.trainable = True

    model = Model(inputs=base_model.input, outputs=predictions)
    # log.info("model summary \n{}".format(model.summary()))
    return model


def train(epochs, batch_size, width, height, prefix='_01', save_dir='models_and_logs/'):
    # model = get_model(InceptionV3, width, height)
    # model = get_model(ResNet50, width, height)
    model = get_model(DenseNet169, width, height)
    # model = get_model(DenseNet201, width, height)
    # model = get_model(Xception, width, height)
    # model = get_model(InceptionResNetV2, width, height)
    # model = get_model(NASNetMobile, width, height)
    # model = resnet.ResnetBuilder.build_resnet_50((3, width, height), 16)

    model.compile(optimizer=SGD(lr=lr_schedule(0), momentum=0.9, nesterov=True),
                  loss=['binary_crossentropy'],
                  # loss=['categorical_crossentropy'],
                  metrics=['accuracy'])

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    csvlogger = CSVLogger(save_dir + 'log' + str(epochs) + str(batch_size) + prefix + '.log', append=True)
    model_check = ModelCheckpoint(save_dir + 'm' + str(epochs) + str(batch_size) + prefix + '_p.h5',
                                  monitor='loss', save_best_only=True)
    lr_scheduler = LearningRateScheduler(lr_schedule)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    tblogger = TensorBoard(log_dir=save_dir + 'tblogger' + prefix, histogram_freq=0,
                           write_graph=True, write_images=True)

    model.fit_generator(data_generator(width, height, batch_size, train=True),
                        steps_per_epoch=np.ceil(len(train_dirs)/batch_size),
                        epochs=epochs,
                        validation_data=data_generator(width, height, batch_size, train=False),
                        validation_steps=np.ceil(len(valid_dirs)/batch_size),
                        verbose=1,
                        workers=8,
                        max_q_size=48,
                        callbacks=[csvlogger, model_check, lr_scheduler, tblogger])

    # model.save_weights(save_dir + 'weight' + str(epochs) + str(batch_size) + prefix + '.h5')
    model.save(save_dir + 'm' + str(epochs) + str(batch_size) + prefix + '_l.h5')
    log.info("模型保存成功，保存的路径是{}".format(join(save_dir, 'm' + str(epochs) + str(batch_size) + prefix + '_l.h5')))


if __name__ == '__main__':
    # ***====================================***
    # 参数设置
    gpu_id = "0"
    gpu_memory_fraction = 0.8
    EPOCH = 70
    BATCH_SIZE = 32
    WIDTH, HEIGHT = 224, 224
    NET = "dn169"

    train_dirs = train_pairs
    valid_dirs = valid_pairs

    # ***====================================***
    # 日志输出参数使用情况
    log.info("正在使用{}号GPU".format(gpu_id))
    log.info("GPU显存使用率为{}%".format(gpu_memory_fraction * 100))
    log.info("迭代次数为{}".format(EPOCH))
    log.info("一次训练的图像数量为{}".format(BATCH_SIZE))
    log.info("采用的网络框架为{}".format(NET))
    log.info("网络的输入尺寸为{}x{}".format(WIDTH, HEIGHT))
    log.info("开始训练：\n训练集的图片数量为{},验证集的图片数量为{}".format(len(train_dirs), len(valid_dirs)))

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    set_gpu(gpu_memory_frac=gpu_memory_fraction)
    train(EPOCH, BATCH_SIZE, WIDTH, HEIGHT, prefix=NET+'v6_13')
    # train(70, 16, 299, 299, prefix='_Irnv6_01')
