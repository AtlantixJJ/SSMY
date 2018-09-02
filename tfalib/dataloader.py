from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import os
import scipy.misc as misc
from datetime import datetime
import random
import sys
import threading
import numpy as np

class FileNameFlow(object):
    def __init__(self,list_of_files):
        self.list_of_files = list_of_files
        self.files = [tf.train.string_input_producer([f]) for f in self.list_of_files]
        self.file_reader = [tf.TextLineReader() for f in self.list_of_files]
        self.lines = []
        for i,f in enumerate(self.file_reader):
            key, val = f.read(self.files[i])
            self.lines.append(val)
    
    def testrun(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        with tf.Session(config=config) as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            for i in range(10):
                lines = sess.run(self.lines)
                print(lines)
            coord.request_stop()
            coord.join(threads)

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_ss(imagefiles, labelfiles, name):
    num_examples = len(imagefiles)

    filename = os.path.join("./", name)
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        image_raw = misc.imread(imagefiles[index])
        label_raw = misc.imread(labelfiles[index])
        print(imagefiles[index],labelfiles[index])
        print(image_raw.shape,label_raw.shape)
        try:
            assert(image_raw.shape[:2] == label_raw.shape[:2])
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(image_raw.shape[0]),
                'width': _int64_feature(image_raw.shape[1]),
                'depth': _int64_feature(image_raw.shape[2]),
                'label': _bytes_feature(label_raw.tostring()),
                'image_raw': _bytes_feature(image_raw.tostring())}))
        except:
            print("skip")
        writer.write(example.SerializeToString())
    writer.close()

def build_mitplace(imagelist, labellist, name):
    filenames = open(imagelist).readlines()
    filenames = ["data/images/"+i.strip() for i in filenames]
    labelnames = open(labellist).readlines()
    labelnames = ["data/annotations/"+i.strip() for i in labelnames]
    convert_to_ss(filenames, labelnames, name)


if __name__ == "__main__":
    import sys
    print(sys.argv)
    build_mitplace(sys.argv[1], sys.argv[2], sys.argv[3])