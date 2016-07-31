#!/usr/bin/env python
#coding=utf-8
import os
import sys
import codecs
import nltk
import cPickle
import network
import time
import math
import re
import jieba

def print_log(s):
    RED = '\033[91m'
    ENDC = '\033[0m'
    print RED + str(s) + ENDC

def normalize(x):
    l = list(x)
    modl = 0
    for i in l:
        modl += i*i
    if modl == 0:
        return x
    modl **= 0.5
    l = [x/modl for x in l]
    return l

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower().decode('utf-8')

class Scanner:
    
    def __init__(self):
        self.train_set = []
        self.test_set = []
        self.all_words = []
        self.freq_dist = []
        self.label_set = set()

    def scan(self, files):
        self.base_dir = os.getcwd()
        for file in files:
            self.__scan(file)
        os.chdir(self.base_dir)
        print_log('scan finished')

        self.label_set = list(self.label_set)

        self.labeled_words = {}

        for x, y in self.train_set:
            idx = self.label_set.index(y)
            if self.labeled_words.has_key(idx):
                self.labeled_words[idx] |= set(x)
            else:
                self.labeled_words[idx] = set(x)

        self.labeled_words_clean = {}

        for k in self.labeled_words:
            self.labeled_words_clean[k] = set(self.labeled_words[k])

        for k1 in self.labeled_words_clean:
            for k2 in self.labeled_words:
                if k1 != k2:
                    self.labeled_words_clean[k1] -= self.labeled_words[k2]

        for x, y in self.train_set:
            self.all_words.extend(x)

        self.all_words = list(self.all_words)

        self.freq_dist = nltk.FreqDist(w for w in self.all_words)
        
        self.dump()

    def __scan(self, file):
        if os.path.isdir(file):
            os.chdir(file)
            for obj in os.listdir(os.curdir):
                self.__scan(obj)
            os.chdir(os.pardir)
        else:
            file_path = os.getcwd() + os.sep + file
            f = open(file_path, 'r')
            text = clean_str(f.read())
            tokens = [x.lower() for x in jieba.cut(text, cut_all=True)]
            if len(tokens) == 0:
                return
            dir_split = os.getcwd().split('/')
            self.label_set.add(dir_split[-1])
            if dir_split[-2] == 'test':
                self.test_set.append((tokens, dir_split[-1]))
            elif dir_split[-2] == 'train':
                self.train_set.append((tokens, dir_split[-1]))
    
    def dump(self):
        try:
            f = open('tmp/dumped_data_main_zh', 'w+')
            cPickle.dump(self.__dict__, f)
            print_log('data saved')
            f.close()
        except:
            print_log('data saved failed')

    def load(self):
        try:
            f = open('tmp/dumped_data_main_zh', 'r')
            self.__dict__.update(cPickle.load(f))
            f.close()
            print_log('data loaded')
            return True
        except:
            print_log('data loaded failed')
            return False

class Classifier(Scanner):
        
    def featurify(self, doc):
        features = []

        for k in self.labeled_words_clean:
            cnt = 0
            for word in doc:
                if word in self.labeled_words_clean[k]:
                    cnt += 1
            features.append(cnt)

        for word, freq in self.top_freq_word:
            features.append(doc.count(word)*math.log(len(self.train_set)/self.df[word]))
        

        self.features_size = len(features)
        return normalize(features)

    def preprocess(self):    
        self.top_freq_word = self.freq_dist.most_common(300)
        
        self.df = {}

        for word, freq in self.top_freq_word:
            for x, y in self.train_set:
                if word in x:
                    if self.df.has_key(word):
                        self.df[word] += 1
                    else:
                        self.df[word] = 1

        self.train_set_featured = [(self.featurify(x), self.label_set.index(y)) for x, y in self.train_set]
        self.test_set_featured = [(self.featurify(x), self.label_set.index(y)) for x, y in self.test_set]

        print '####'
        print len(self.train_set_featured)
        print len(self.test_set_featured)
        print self.features_size
        print len(self.label_set)
        print '####'


    def classify(self, epochs, eta):
        nw = network.Network([self.features_size, 50, len(self.label_set)])
        nw.train(epochs, eta, 1, self.train_set_featured, self.test_set_featured)

if __name__ == '__main__':
    classifier = Classifier()

    if len(sys.argv) <= 1:
        classifier.load()
    else:
        classifier.scan(sys.argv[1:])

    classifier.preprocess()

    start_time = time.time()
    classifier.classify(20, 1)
    end_time = time.time()

    print_log('\ncost time(exclude preprocess time): {} seconds'.format(end_time-start_time))