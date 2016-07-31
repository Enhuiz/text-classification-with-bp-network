#!/usr/bin/env python
#coding=utf-8
import os
import sys
import codecs

def to_utf8(file):
    try:
        f = open(file, 'r')
        s = f.read()
        f.close()
        f = open(file, 'w')
        s = s.decode('gbk').encode('utf-8')
        f.write(s)
        f.close()
    except:
        print '{} is not coding in gbk'.format(file)

def scandir(file):
    if os.path.isdir(file):
        os.chdir(file)
        for obj in os.listdir(os.curdir):
            scandir(obj)
        os.chdir(os.pardir)
    else:
        file_path = os.getcwd() + os.sep + file
        to_utf8(file_path)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        for i in xrange(1, len(sys.argv)):
            scandir(sys.argv[i])
    else:
        print 'please give at least one file'
    
