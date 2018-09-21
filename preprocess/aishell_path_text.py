#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/16/18 2:37 PM
# @Author  : renxiaoming@julive.com
# @Site    : 
# @File    : aishell_path_text.py.py
# @Software: PyCharm
import codecs
import os
import numpy as np
import re

#1.保存语音和文字映射关系的文件名
save_path = "aishell_data/train_dev_test.npz"
# train_dev_test = np.load(save_path)
# train_filepath = train_dev_test["train_filepath"][()]
# dev_filepath = train_dev_test["dev_filepath"][()]
# test_filepath = train_dev_test["test_filepath"][()]

#2.语音和文字的地址
main_path = "/mnt/steven/data/data_aishell"
text_path = os.path.join(main_path,"transcript","aishell_transcript_v0.8.txt")
wav_path = os.path.join(main_path,"wav")

train_filepath = {}
dev_filepath = {}
test_filepath = {}
#3. 设置train，dev，test三个映射关系，使用字典表示
#将train，dev，test三个文件夹中的文件地址作为value保存在各自的字典中，key是根据语音的文件名截取，用于和文字建立索引
for dir_filename in ["train","dev","test"]:
    for root,subdirs,_ in os.walk(os.path.join(wav_path,dir_filename)):
        for subdir in subdirs:
            for root2,_,filenames in os.walk(os.path.join(root,subdir)):
                for filename in filenames:
                    # 动态变量名
                    value = eval(dir_filename+"_filepath")
                    value[filename[:-4]] = os.path.join(root2,filename)
#4.读取所有文本
with codecs.open(text_path, encoding="utf-8") as f:
    all_texts = f.read().split("\n")

save_train_path_text = []
save_dev_path_text = []
save_test_path_text = []
#5.将文本中的空格去除，从语音的train，dev，test的字典的key连接语音的路径和文字的映射，保存到数组中
for index,text in enumerate(all_texts):
    #print "index = %d" % index
    split = text.split(" ",1)
    if split[0] in train_filepath.iterkeys():
        save_train_path_text.append([train_filepath[split[0]],re.sub(" ","",split[1])])
    elif split[0] in dev_filepath.iterkeys():
        save_dev_path_text.append([dev_filepath[split[0]],re.sub(" ","",split[1])])
    elif split[0] in test_filepath.iterkeys():
        save_test_path_text.append([test_filepath[split[0]],re.sub(" ","",split[1])])

np.savez(save_path,
         train_filepath=save_train_path_text,
         dev_filepath=save_dev_path_text,
         test_filepath=save_test_path_text
         )