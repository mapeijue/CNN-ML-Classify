# -*- coding: UTF-8 -*-
"""
  @Author: mpj
  @Date  : 2022/2/16 20:20
  @version V1.0
"""
import itertools
import os
import random
import re
import time
from typing import List

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.python.platform import gfile

from log import log

# 用来固定seed，保证每次运行的结果一致
random.seed(1234)
np.random.seed(1234)
tf.random.set_seed(1234)

# 用来使用GPU
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)


# TensorFlow inception-v3 feature extraction
def create_graph(path='./imagenet/classify_image_graph_def.pb'):
	"""Create the CNN graph"""
	with gfile.FastGFile(path, 'rb') as f:
		graph_def = tf.compat.v1.GraphDef()
		graph_def.ParseFromString(f.read())
		_ = tf.import_graph_def(graph_def, name='')


def plot_features(feature_labels, t_sne_features, name, isTrain=True):
	"""feature plot"""
	plt.figure(figsize=(9, 9), dpi=100)

	colors = itertools.cycle(["r", "b", "g", "c", "m", "y",
	                          "slategray", "plum", "cornflowerblue",
	                          "hotpink", "darkorange", "forestgreen",
	                          "tan", "firebrick", "sandybrown"])

	label_feature_map = {}
	for label, feature in zip(feature_labels, t_sne_features):
		if label not in label_feature_map:
			label_feature_map[label] = [feature]
		else:
			label_feature_map[label].append(feature)

	for label, features in label_feature_map.items():
		features = np.array(features)
		plt.scatter(features[:, 0], features[:, 1], c=next(colors), s=10, edgecolors='none')
		plt.annotate(label, xy=(np.mean(features[:, 0]), np.mean(features[:, 1])))

	save_path = './result/{}/{}.png'.format('train' if isTrain else 'test', name)
	plt.savefig(save_path)
	plt.show()


def plot_confusion_matrix(y_true, y_pred, matrix_title, cmap=plt.cm.Blues, isTrain=True):
	"""confusion matrix computation and display"""
	plt.figure(figsize=(9, 9), dpi=100)

	# use sklearn confusion matrix
	cm_array = confusion_matrix(y_true, y_pred)

	plt.imshow(cm_array, interpolation='nearest', cmap=cmap)
	plt.title(matrix_title, fontsize=16)

	cbar = plt.colorbar(fraction=0.046, pad=0.04)
	cbar.set_label('Number of images', rotation=270, labelpad=30, fontsize=12)

	true_labels = np.unique(y_true)
	pred_labels = np.unique(y_pred)
	xtick_marks = np.arange(len(true_labels))
	ytick_marks = np.arange(len(pred_labels))

	plt.xticks(xtick_marks, true_labels, rotation=90)
	plt.yticks(ytick_marks, pred_labels)
	plt.tight_layout()
	plt.ylabel('True label', fontsize=14)
	plt.xlabel('Predicted label', fontsize=14)
	plt.tight_layout()

	save_path = './result/{}/{}.png'.format('train' if isTrain else 'test', matrix_title)
	plt.savefig(save_path)
	plt.show()


# 用来详细显示混淆矩阵的结果
def plot_confusion_matrix_detail(y_true, y_pred, matrix_title, normalize=False, cmap=plt.cm.Blues, isTrain=True):
	"""confusion matrix computation and display"""
	plt.figure(figsize=(9, 9), dpi=100)

	# use sklearn confusion matrix
	cm_array = confusion_matrix(y_true, y_pred)

	if normalize:
		cm_array = cm_array.astype('float') / cm_array.sum(axis=1)[:, np.newaxis]
		np.set_printoptions(formatter={'float': '{: 0.2f}'.format})

	plt.imshow(cm_array, interpolation='nearest', cmap=cmap)
	plt.title(matrix_title, fontsize=16)

	cbar = plt.colorbar(fraction=0.046, pad=0.04)
	cbar.set_label('Number of images', rotation=270, labelpad=30, fontsize=12)

	true_labels = np.unique(y_true)
	pred_labels = np.unique(y_pred)
	xtick_marks = np.arange(len(true_labels))
	ytick_marks = np.arange(len(pred_labels))

	plt.xticks(xtick_marks, true_labels, rotation=90)
	plt.yticks(ytick_marks, pred_labels)
	fmt = '.2f' if normalize else 'd'
	thresh = cm_array.max() / 2.
	for i, j in itertools.product(range(cm_array.shape[0]), range(cm_array.shape[1])):
		plt.text(j, i, format(cm_array[i, j], fmt),
		         horizontalalignment="center",
		         color="white" if cm_array[i, j] > thresh else "black")
	plt.tight_layout()
	plt.ylabel('True label', fontsize=14)
	plt.xlabel('Predicted label', fontsize=14)
	plt.tight_layout()

	save_path = './result/{}/{}.png'.format('train' if isTrain else 'test', matrix_title)
	plt.savefig(save_path)
	plt.show()


# Classifier performance
def run_classifier(clf, x_train_data, y_train_data, x_test_data, y_test_data, acc_str, matrix_header_str,
                   isTrain=True):
	"""run chosen classifier and display results"""
	start_time = time.time()
	if isTrain:
		clf.fit(x_train_data, y_train_data)
	y_pred = clf.predict(x_test_data)
	timeInterval = time.time() - start_time
	# print("%f seconds" % timeInterval)
	log.debug("%f seconds" % timeInterval)

	# confusion matrix computation and display
	# 混淆矩阵计算与显示
	score = accuracy_score(y_test_data, y_pred) * 100
	# print(acc_str.format(score))
	log.debug(acc_str.format(score))
	plot_confusion_matrix(y_test_data, y_pred, matrix_header_str, isTrain=isTrain)


def get_images_list(images_dir: str) -> List[str]:
	dir_list = [x[0] for x in os.walk(images_dir)]
	dir_list = dir_list[1:]
	list_images = []
	for image_sub_dir in dir_list:
		sub_dir_images = [image_sub_dir + '/' + f for f in os.listdir(image_sub_dir) if re.search('jpg|JPG', f)]
		list_images.extend(sub_dir_images)
	# 讲图片目录打乱
	random.shuffle(list_images)
	return list_images


def predict_feature(features, labels, list_images, next_to_last_tensor, sess):
	"""
	通过next_to_last_tensor获取特征，并将特征和标签存入features和labels中
	:param features: 特征存放list或者np.array
	:param labels: 标签存放list或者np.array
	:param list_images: 图片路径list
	:param next_to_last_tensor: 图片特征处理的tensor
	:param sess: myTensorflow session
	:return:
	"""
	for ind, image in enumerate(list_images):
		# 根据路径名获得图片的label
		im_label = image.split('/')[-2]

		# rough indication of progress
		if ind % 100 == 0:
			print('Processing', image, im_label)
		if not gfile.Exists(image):
			log.warning('File not found: %s', image)

		image_data = gfile.FastGFile(image, 'rb').read()
		predictions = sess.run(next_to_last_tensor, {'DecodeJpeg/contents:0': image_data})
		features[ind, :] = np.squeeze(predictions)
		labels.append(im_label)
