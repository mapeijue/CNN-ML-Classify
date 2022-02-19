# -*- coding: UTF-8 -*-
"""
  @Author: mpj
  @Date  : 2022/2/16 19:01
  @version V1.0
"""

import os
import pickle
import time

import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE

import common
from log import log, destroyLog

# what and where
model_dir = './imagenet/classify_image_graph_def.pb'
images_dir = './caltech_101_images/test/'

# 是否导入特征处理好的文件直接用来分类
isLoadModel = False

log.debug("*** test start ***")


# Classifier performance
def run_classifier(clf, x_test_data, y_test_data, acc_str, matrix_header_str):
	common.run_classifier(clf, None, None, x_test_data, y_test_data, acc_str, matrix_header_str, isTrain=False)


# TensorFlow inception-v3 feature extraction
def extract_features(list_images):
	"""Extract bottleneck features"""
	nb_features = 2048
	test_features = np.empty((len(list_images), nb_features))
	test_labels = []

	common.create_graph()

	# 'pool_3:0': A tensor containing the next-to-last layer containing 2048 float description of the image.
	# 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG encoding of the image.

	with tf.compat.v1.Session() as sess:
		next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')

		common.predict_feature(test_features, test_labels, list_images, next_to_last_tensor, sess)

	return test_features, test_labels


if __name__ == '__main__':
	global_start_time = time.time()
	# Read in images and extract features

	# get images - labels are from the subdirectory names
	if os.path.exists('model/test_features') and isLoadModel:
		log.debug('Pre-extracted test_features and labels found. Loading them ...')
		test_features = pickle.load(open('model/test_features', 'rb'))
		test_labels = pickle.load(open('model/test_labels', 'rb'))
	else:
		# get the images and the labels from the subdirectory names
		list_images = common.get_images_list(images_dir)

		# extract features
		test_features, test_labels = extract_features(list_images)

		# save, so they can be used without re-running the last step which can be quite long
		pickle.dump(test_features, open('model/test_features', 'wb'))
		pickle.dump(test_labels, open('model/test_labels', 'wb'))
		log.debug('CNN features obtained and saved.')

	# t-sne feature plot
	if os.path.exists('model/tsne_test_features.npz') and isLoadModel:
		log.debug('t-sne tsne_test_features found. Loading ...')
		tsne_test_features = np.load('model/tsne_test_features.npz')['tsne_features']
	else:
		log.debug('No t-sne tsne_test_features found. Obtaining ...')
		tsne_test_features = TSNE().fit_transform(test_features)
		np.savez('model/tsne_test_features', tsne_features=tsne_test_features)
		log.debug('t-sne tsne_test_features obtained and saved.')

	common.plot_features(test_labels, tsne_test_features, "tsne_test_features", isTrain=False)

	# Classification

	# prepare training and test datasets
	X_test, y_test = test_features, test_labels
	log.debug('test datasets prepared.')
	log.debug('Test dataset size: %d' % len(X_test))

	# classify the images with a Linear Support Vector Machine (SVM)
	log.debug('Support Vector Machine LinearSVC starting ...')
	clf = pickle.load(open('./model/LinearSVC.pkl', 'rb'))
	run_classifier(clf, X_test, y_test, "CNN-LinearSVC Accuracy: {0:0.1f}%", "LinearSVC Confusion matrix")

	log.debug('Support Vector Machine SVC finished.')
	clf = pickle.load(open('./model/SVC.pkl', 'rb'))
	run_classifier(clf, X_test, y_test, "CNN-SVC Accuracy: {0:0.1f}%", "SVC Confusion matrix")

	# classify the images with an Extra Trees Classifier
	log.debug('Extra Trees Classifier starting ...')
	clf = pickle.load(open('./model/ExtraTreesClassifier.pkl', 'rb'))
	run_classifier(clf, X_test, y_test, "CNN-ET Accuracy: {0:0.1f}%", "Extra Trees Confusion matrix")

	# classify the images with a Random Forest Classifier
	log.debug('Random Forest Classifier starting ...')
	clf = pickle.load(open('./model/RandomForestClassifier.pkl', 'rb'))
	run_classifier(clf, X_test, y_test, "CNN-RF Accuracy: {0:0.1f}%", "Random Forest Confusion matrix")

	# classify the images with a k-Nearest Neighbors Classifier
	log.debug('K-Nearest Neighbours Classifier starting ...')
	clf = pickle.load(open('./model/KNeighborsClassifier.pkl', 'rb'))
	run_classifier(clf, X_test, y_test, "CNN-KNN Accuracy: {0:0.1f}%", "K-Nearest Neighbor Confusion matrix")

	# classify the image with a Multi-layer Perceptron Classifier
	log.debug('Multi-layer Perceptron Classifier starting ...')
	clf = pickle.load(open('./model/MLPClassifier.pkl', 'rb'))
	run_classifier(clf, X_test, y_test, "CNN-MLP Accuracy: {0:0.1f}%", "Multi-layer Perceptron Confusion matrix")

	# classify the images with a Gaussian Naive Bayes Classifier
	log.debug('Gaussian Naive Bayes Classifier starting ...')
	clf = pickle.load(open('./model/GaussianNB.pkl', 'rb'))
	run_classifier(clf, X_test, y_test, "CNN-GNB Accuracy: {0:0.1f}%", "Gaussian Naive Bayes Confusion matrix")

	# classify the images with a Linear Discriminant Analysis Classifier
	log.debug('Linear Discriminant Analysis Classifier starting ...')
	clf = pickle.load(open('./model/LinearDiscriminantAnalysis.pkl', 'rb'))
	run_classifier(clf, X_test, y_test, "CNN-LDA Accuracy: {0:0.1f}%", "Linear Discriminant Analysis Confusion matrix")

	# classify the images with a Quadratic Discriminant Analysis Classifier
	log.debug('Quadratic Discriminant Analysis Classifier starting ...')
	clf = pickle.load(open('./model/QuadraticDiscriminantAnalysis.pkl', 'rb'))
	run_classifier(clf, X_test, y_test, "CNN-QDA Accuracy: {0:0.1f}%",
	               "Quadratic Discriminant Analysis Confusion matrix")

	log.debug(f'test classification finished total time: {time.time() - global_start_time}')

	destroyLog()
