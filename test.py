import dataloader
import tensorflow as tf
import tensorpack as tp

dataset = dataloader.FileNameFlow(["data/images/training.txt","data/annotations/training.txt"])
dataset.testrun()
