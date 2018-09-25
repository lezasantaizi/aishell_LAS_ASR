#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/8 20:28
# @Author  : renxiaoming@julive.com
# @Site    : 
# @File    : preprocess_torch.py
# @Software: PyCharm

import torch.utils.data as data
import os
import os.path
import shutil
import torch
import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
from sklearn import preprocessing
import codecs
import re
import copy
import argparse


class Lang:
    def __init__(self):
        self.word2index = {"<PAD>":0,"<GO>":1,"<EOS>":2,"<UNK>":3}
        self.word2count = {}
        self.index2word = {0:"<PAD>", 1:"<GO>",2:"<EOS>",3:"<UNK>"}
        self.n_words = 4  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)
    def transSentence(self, sentence):
        sentence_list = []
        for word in sentence:
            if word in self.word2index:
                sentence_list.append(self.word2index[word])
            else:
                sentence_list.append(self.word2index["<UNK"])

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

class use_exist_lang:
    def __init__(self,lang_filename):
        self.word2index = {"<PAD>":0,"<GO>":1,"<EOS>":2,"<UNK>":3}
        self.index2word = {0:"<PAD>", 1:"<GO>",2:"<EOS>",3:"<UNK>"}
        self.n_words = 4  # Count SOS and EOS
        self.filename = lang_filename
    def build(self):
        with codecs.open(self.filename,encoding="utf-8") as f:
            alllines = f.readlines()
            for i in range(len(alllines)):
                item = alllines[i].split("*")[-2].strip()
                for word in item:
                    if word in self.word2index:
                        None
                    else:
                        self.word2index[word] = self.n_words
                        self.index2word[self.n_words] = word
                        self.n_words+=1

    def trans_sentence(self,sentence):
        list_ids = []
        for word in sentence:
            if word in self.word2index:
                list_ids.append(self.word2index[word])
            else:
                list_ids.append(self.word2index["<UNK"])
        return list_ids

def read_audio(fp):
    #sig, sr = torchaudio.load(fp)
    sr, sig = wav.read(fp)

    # 当前发现使用calcMFCC的语音识别准确率要高于使用logfbank，这个calcMFCC来自于core文件夹
    # inputs = calcMFCC(audio, samplerate=fs, feature_len=80, mode="fbank")
    inputs = logfbank(sig, samplerate=sr, nfilt=80)
    delta_input = delta(inputs, N=2)
    delta_delta_input = delta(delta_input, N=2)
    inputs = np.array(np.concatenate([inputs, delta_input, delta_delta_input], axis=1), dtype="float32")
    # 对单个语音进行了zscore归一化
    # 如果数据量太大，做不了对全量数据的归一化，只能先暂时对单个语音数据进行归一化
    inputs = preprocessing.scale(inputs, axis=0)
    return inputs, sr


class aishell(data.Dataset):

    def __init__(self, root, dataset_name="aishell",transform=None, target_transform=None, download=False, type="train"):
        self.root = os.path.expanduser(root)
        self.processed_folder = dataset_name+'/processed'
        self.transform = transform
        self.target_transform = target_transform
        self.type = type
        self.data = []
        self.labels = []
        self.chunk_size = 1000 if dataset_name=="aishell" else 10000
        self.num_samples = 0
        self.max_len = 0
        self.cached_pt = 0

        if download:
            self.download()

        if not self._check_exists(self.type):
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')
        self._read_info(self.type)
        self.lang = torch.load(os.path.join(self.root, self.processed_folder, "aishell_info.pt"))
        self.data, self.labels, self.sentences = torch.load(os.path.join(
            self.root, self.processed_folder,self.type, "aishell_{:04d}.pt".format(self.cached_pt)))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.cached_pt != index // self.chunk_size:
            self.cached_pt = int(index // self.chunk_size)
            self.data, self.labels, self.sentences = torch.load(os.path.join(
                self.root, self.processed_folder,self.type, "aishell_{:04d}.pt".format(self.cached_pt)))
        index = index % self.chunk_size
        audio, target,sentence = self.data[index], self.labels[index], self.sentences[index]

        if self.transform is not None:
            audio = self.transform(audio)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return audio, target, sentence

    def __len__(self):
        return self.num_samples

    def _check_exists(self,type):
        return os.path.exists(os.path.join(self.root, self.processed_folder,type, "%s_info.txt"%type))

    def _write_info(self, type,num_items):
        info_path = os.path.join(
            self.root, self.processed_folder,type, "%s_info.txt"%type)
        with open(info_path, "w") as f:
            f.write("num_samples,{}\n".format(num_items))
            f.write("max_len,{}\n".format(self.max_len))

    def _read_info(self,type):
        info_path = os.path.join(
            self.root, self.processed_folder,type, "%s_info.txt"%type)
        with open(info_path, "r") as f:
            self.num_samples = int(f.readline().split(",")[1])
            self.max_len = int(f.readline().split(",")[1])

    def download(self):
        """Download the VCTK data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import tarfile

        if self._check_exists(self.type):
            return

        # raw_abs_dir = os.path.join(self.root, self.raw_folder)
        processed_abs_dir = os.path.join(self.root, self.processed_folder)
        #实例化生成字典
        lang = Lang()

        # download files
        try:
            # os.makedirs(raw_abs_dir)
            os.makedirs(processed_abs_dir)
        except OSError as e:
            None

        #读取语音列表和对应的文本
        train_dev_test_filename = os.path.join(".", "train_dev_test.npz")
        train_dev_test = np.load(train_dev_test_filename)
        train_filepath = train_dev_test["train_filepath"]
        dev_filepath = train_dev_test["dev_filepath"]
        test_filepath = train_dev_test["test_filepath"]
        # train_dev_test_filename = os.path.join(".", "thchs30_train_dev_test.npz")
        # train_dev_test = np.load(train_dev_test_filename)
        # train_filepath = np.concatenate([train_filepath,train_dev_test["train_filepath"]],axis=0)
        # test_filepath = np.concatenate([test_filepath,train_dev_test["test_filepath"]],axis=0)
        np.random.shuffle(train_filepath)

        print ("Found [train,dev,test = %d,%d,%d ] audio files" %(len(train_filepath),len(dev_filepath),len(test_filepath)))

        for type_name in ["train","dev","test"]:
            # 使用了可以修改的变量名
            value = eval(type_name + "_filepath")
            self.max_len = 0
            os.makedirs(os.path.join(self.root, self.processed_folder,type_name))
            for n in range(len(value) // self.chunk_size + 1):
                tensors = []
                labels = []
                sentences = []
                lengths = []
                st_idx = n * self.chunk_size
                end_idx = st_idx + self.chunk_size
                for i, f in enumerate(value[st_idx:end_idx]):
                    #这里返回 sig和 samplerate ，只保存sig
                    sig = read_audio(f[0])[0]
                    tensors.append(sig)
                    lengths.append(sig.size)
                    sentences.append(f[1])
                    #添加句子用于生成字典
                    lang.addSentence(f[1])
                    labels.append([lang.word2index[index] for index in f[1]] + [lang.word2index["<EOS>"]])
                    self.max_len = sig.size if sig.size > self.max_len else self.max_len
                # # sort sigs/labels: longest -> shortest
                # tensors, labels, sentences = zip(*[(b, c, d) for (a, b, c, d) in sorted(
                #     zip(lengths, tensors, labels,sentences), key=lambda x: x[0], reverse=True)])
                data = (tensors, labels, sentences)
                torch.save(
                    data,
                    os.path.join(
                        self.root,
                        self.processed_folder,
                        type_name,
                        "aishell_{:04d}.pt".format(n)
                    )
                )
                # if n > 0:
                #     break
            #这里保存 type中的数据总数信息到 train_info.txt dev_info.txt, test_info.txt
            self._write_info(type_name,(n * self.chunk_size) + i + 1)

        #这里保存字典信息在aishell_info.txt
        torch.save(lang, os.path.join(
                self.root,
                self.processed_folder,
                "aishell_info.pt"
            )
        )
        print('Done!')

    def collate_fn(self,batch):
        # batch.sort(key=lambda x: len(x[1]), reverse=True)
        enc_inputs, dec_outputs, sentences = zip(*batch)

        def pad_sentence_batch(sentence_batch, pad_int):
            """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
            max_lens = max([len(sentence) for sentence in sentence_batch])
            real_lens = [len(seq) for seq in sentence_batch]
            if pad_int is None:
                #语音
                max_lens = max_lens if max_lens % 8 == 0 else (max_lens // 8 + 1) * 8
                temp = [np.concatenate([sentence, np.zeros(((max_lens - len(sentence)), sentence.shape[-1]))], axis=0)
                    for sentence in sentence_batch]

            else:
                #文字
                # padded_seqs = torch.zeros(len(sentence_batch), max(lens)).long()
                # for i, seq in enumerate(sentence_batch):
                #     end = lens[i]
                #     padded_seqs[i, :end] = torch.LongTensor(seq[:end])
                temp = [sentence + [pad_int] * (max_lens - len(sentence)) for sentence in sentence_batch]
            return temp, real_lens, max_lens

        batch_text_ids_pad, real_target_lens,maxlen_batch_text_ids_pad = pad_sentence_batch(dec_outputs,self.lang.word2index["<PAD>"])
        batch_inputs_pad, real_inputs_lens,maxlen_batch_inputs_pad = pad_sentence_batch(enc_inputs, None)

        return batch_inputs_pad, batch_text_ids_pad, sentences,maxlen_batch_inputs_pad,maxlen_batch_text_ids_pad,real_inputs_lens,real_target_lens



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/Users/comjia/Downloads/code/pytorch_seq2seq/pytorch_seq2seq',
                        help='输入aishell数据所在的主目录path')
    # parser.add_argument('--cuda', type=int, default=1, help='GPU vs CPU')
    opt = parser.parse_args()
    aishell_class = aishell(opt.root_dir,dataset_name="aishell",transform=None, target_transform=None, download=False)
    data_loader = torch.utils.data.DataLoader(aishell_class,
                                              batch_size=64,
                                              shuffle=False,
                                              num_workers=3,
                                              collate_fn=aishell_class.collate_fn)
    # for epoch in range(3):
    #     for index,data in enumerate(data_loader):
    #         print "index:%d,data=%s"%(index,len(data[0]))
