#!/usr/bin/python
# coding=utf-8

import os
import cv2
import json
import logging
import numpy as np
from keras.models import load_model

from augment import resize_1  # 引入黑边填充的resize方式

logging.basicConfig(format='%(asctime)s %(levelname)s [%(module)s] %(message)s', level=logging.INFO)
log = logging.getLogger()


def set_gpu(gpu_memory_frac=0.3):
    import tensorflow as tf
    import keras.backend as K
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_frac
    sess = tf.Session(config=config)
    K.set_session(sess)


def predict_on_batch(model, test_set, width=224, height=224, batch_size=32, pad=False):
    predict_prob = list()  # 存放每张图像的预测概率矩阵
    predict_class = list()  # 存放每张图像的预测类别矩阵
    low_conf = list()  # 存放低置信度的图像名称和类别
    c = 0
    for start in range(0, len(test_set), batch_size):
        x_batch = list()
        end = min(start + batch_size, len(test_set))
        new_batch = test_set[start:end]
        for im in new_batch:
            img = cv2.imread(os.path.join(test_path, im))[:, :, ::-1]
            if pad:
                img = resize_1(img, test=True)   # 对图像做黑边填充，图像居中
                # img = resize(img)   # 对于长宽比较大的图像做黑边填充，黑边随机大小，图像不一定居中
            img = cv2.resize(img, (width, height))
            x_batch.append(img)
        x_batch = np.array(x_batch, np.float32) / 255.
        preds = model.predict_on_batch(x_batch)
        for i, pred in enumerate(preds):
            if np.max(pred) < 0.6:
                # print '低置信度图像', new_batch[i], '\n', pred
                low_conf.append(new_batch[i])
                c += 1
            predict_prob.append(pred)
        predict_class += list(np.argmax(preds, axis=1))
    print '低置信度的图片数量是', c
    return predict_class, low_conf, predict_prob


def get_result(prob_ensemble, fix=""):
    low_confidence = list()
    c = 0
    for i, pred in enumerate(prob_ensemble):
        if np.max(pred) < 0.6:
            # print '低置信度图像', test_dirs[i], '\n', pred
            low_confidence.append(test_dirs[i])
            c += 1
    print '低置信度的图片数量是', c
    pred_cla = np.argmax(prob_ensemble, axis=1)

    results = list()
    for i in range(len(test_dirs)):
        result = dict()
        result["image_id"] = test_dirs[i]
        result["disease_class"] = pred_cla[i]
        results.append(result)
    log.info("成功预测的图片数量是 %s 张" % len(results))

    with open("output/low_conf_"+fix+".json", "w") as fw:
        json.dump(low_confidence, fw, indent=4)  # indent设置tab键格式
    log.info("low confidence image names saved successfully!")

    with open("output/results_"+fix+".json", "w") as fw:
        json.dump(results, fw, indent=4)  # indent设置tab键格式
    log.info("Prediction results saved successfully!")


if __name__ == '__main__':
    w1, h1, bs1 = 224, 224, 32
    w2, h2, bs2 = 299, 299, 16

    # ***--------------------------------------------------***
    # ***--------------------------------------------------***
    # 加载测试集
    test_path = "../dataset/AgriculturalDisease_testA/images/"
    # test_path = "output/low_conf/"
    test_dirs = os.listdir(test_path)
    log.info("需要预测的图片数量是 %s 张" % len(test_dirs))

    ensemble_flag = 1
    if not ensemble_flag:
        # ***--------------------------------------------------***
        # ***--------------------------------------------------***
        # 引入模型文件
        log.info("Loading model...")
        # model_path = "models_and_logs/m7032_dn169v6_03_p.h5"  # 无黑边填充方式训练出来的模型
        # model_path = "models_and_logs/m7032_dn169v6_04_l.h5"  # 随机黑边填充方式训练出来的模型
        model_path = "models_and_logs/m7032_dn169v6_05_l.h5"  # 所有长宽比不为1的图像随机黑边填充方式训练出来的模型
        model = load_model(model_path)
        log.info("{} model loaded successfully!".format(model_path))

        log.info("Begin predicting...")
        pred_cla, low_confidence, _ = predict_on_batch(model, test_dirs, width=w1, height=h1, batch_size=bs1)
        log.info("Test sets predicted successfully!")

        # ***--------------------------------------------------***
        # ***--------------------------------------------------***
        # 预测的结果处理
        results = list()
        for i in range(len(test_dirs)):
            result = dict()
            result["image_id"] = test_dirs[i]
            result["disease_class"] = pred_cla[i]
            results.append(result)
        log.info("成功预测的图片数量是 %s 张" % len(results))
    else:
        # ***--------------------------------------------------***
        # ***--------------------------------------------------***
        # 引入模型文件
        log.info("Loading model...")
        model_path1 = "models_and_logs/m7032_dn169v6_03_p.h5"  # 无黑边填充方式训练出来的模型
        model_path2 = "models_and_logs/m7032_dn169v6_04_l.h5"  # 所有长宽比较大的图像随机黑边填充方式训练出来的模型
        model_path3 = "models_and_logs/m7032_dn169v6_05_p.h5"  # 所有长宽不等图像随机黑边填充方式训练出来的模型
        model_path4 = "models_and_logs/m6532_dn169v6_06_p.h5"  # 所有长宽不等图像随机黑边填充方式训练出来的模型
        model_path5 = "models_and_logs/m6532_dn169v6_07_p.h5"  # softmax所有长宽不等图像随机黑边填充方式训练出来的模型
        model_path6 = "models_and_logs/m6532_dn169v6_08_p.h5"  # 所有长宽不等图像随机黑边填充方式训练出来的模型
        # model_path = "models_and_logs/m7016_icprv1_p.h5"
        # model1 = load_model(model_path1)
        model2 = load_model(model_path2)
        # model3 = load_model(model_path3)
        model4 = load_model(model_path4)
        # model5 = load_model(model_path5)
        model6 = load_model(model_path6)
        log.info("{} model1 loaded successfully!".format(model_path1))
        log.info("{} model2 loaded successfully!".format(model_path2))
        log.info("{} model3 loaded successfully!".format(model_path3))
        log.info("{} model4 loaded successfully!".format(model_path4))
        log.info("{} model5 loaded successfully!".format(model_path5))
        log.info("{} model6 loaded successfully!".format(model_path6))

        log.info("Begin predicting...")
        # _, _, prob1 = predict_on_batch(model1, test_dirs, width=w1, height=h1, batch_size=bs1)
        _, _, prob2 = predict_on_batch(model2, test_dirs, width=w1, height=h1, batch_size=bs1, pad=True)
        # _, _, prob3 = predict_on_batch(model3, test_dirs, width=w1, height=h1, batch_size=bs1, pad=True)
        _, _, prob4 = predict_on_batch(model4, test_dirs, width=w1, height=h1, batch_size=bs1, pad=True)
        # _, _, prob5 = predict_on_batch(model5, test_dirs, width=w1, height=h1, batch_size=bs1, pad=True)
        _, _, prob6 = predict_on_batch(model6, test_dirs, width=w1, height=h1, batch_size=bs1, pad=True)
        log.info("Test sets predicted successfully!")

        # ***--------------------------------------------------***
        # ***--------------------------------------------------***
        # 预测的结果处理
        # prob_ensemble1 = np.add(np.add(prob1, prob2), np.add(prob3, prob4)) / 4.0
        # prob_ensemble2 = np.add(np.add(prob3, prob4), np.add(prob5, prob6)) / 4.0
        # prob_ensemble3 = np.add(np.add(prob1, prob2), np.add(prob4, prob6)) / 4.0
        # prob_ensemble4 = np.add(np.add(prob2, prob3), np.add(prob4, prob6)) / 4.0
        # prob_ensemble5 = np.add(np.add(np.add(prob1, prob2), np.add(prob3, prob4)), np.add(prob5, prob6)) / 6.0
        # prob_ensemble6 = np.add(np.add(prob1, prob2), np.add(np.add(prob3, prob4), prob6)) / 5.0
        prob_ensemble7 = np.add(np.add(prob2, prob4), prob6) / 3.0

        # get_result(prob_ensemble1, fix="1234")
        # get_result(prob_ensemble2, fix="3456")
        # get_result(prob_ensemble3, fix="1246")
        # get_result(prob_ensemble4, fix="2346")
        # get_result(prob_ensemble5, fix="123456")
        # get_result(prob_ensemble6, fix="12346")
        get_result(prob_ensemble7, fix="246")

        # low_confidence = list()
        # c = 0
        # for i, pred in enumerate(prob_ensemble):
        #     if np.max(pred) < 0.6:
        #         # print '低置信度图像', test_dirs[i], '\n', pred
        #         low_confidence.append(test_dirs[i])
        #         c += 1
        # print '低置信度的图片数量是', c
        # pred_cla = np.argmax(prob_ensemble, axis=1)
        #
        # results = list()
        # for i in range(len(test_dirs)):
        #     result = dict()
        #     result["image_id"] = test_dirs[i]
        #     result["disease_class"] = pred_cla[i]
        #     results.append(result)
        # log.info("成功预测的图片数量是 %s 张" % len(results))

    # ***--------------------------------------------------***
    # ***--------------------------------------------------***
    # 保存的文件，注意修改名称
    # with open("output/low_conf_605.json", "w") as fw:
    #     json.dump(low_confidence, fw, indent=4)  # indent设置tab键格式
    # log.info("low confidence image names saved successfully!")
    #
    # with open("output/results_605.json", "w") as fw:
    #     json.dump(results, fw, indent=4)  # indent设置tab键格式
    # log.info("Prediction results saved successfully!")
