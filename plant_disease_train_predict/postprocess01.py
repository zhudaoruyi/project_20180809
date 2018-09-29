#!/usr/bin/python
# coding=utf-8

import logging
import shutil
import json
import os

logging.basicConfig(format='%(asctime)s %(levelname)s [%(module)s] %(message)s', level=logging.INFO)
log = logging.getLogger()

test_path = "/home/pzw/hdd/dataset/ai_challenger/AgriculturalDisease_testA/images/"
test_dirs = os.listdir(test_path)
log.info("需要预测的图片数量是 %s 张" % len(test_dirs))

json_file = os.path.join("output", "low_confidence604.json")

with open(json_file, 'r') as fr:
    img_names = json.load(fr, 'utf-8')
log.info("低置信度的图片数量是 %s 张" % len(img_names))

dst = os.path.join("output", "low_conf02")
if not os.path.exists(dst):
    os.makedirs(dst)
for img_name in img_names:
    shutil.copy(os.path.join(test_path, img_name), dst)
log.info("{}低置信度图片成功移动到{}".format(len(os.listdir(dst)), dst))
