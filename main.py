# -*- coding: utf-8 -*-

'''''''''''''''''''''''''''''''''
@Author : Vic P.
@Email  : vic4key@gmail.com
@Name   : MNIST - KNN
'''''''''''''''''''''''''''''''''

# Eg.
# python3 main.py train
# python3 main.py predict img_1.jpg
# python3 main.py web

from scipy.misc import imread
from sklearn import neighbors
import matplotlib.pyplot as plt
import numpy as np
import sys, os, glob, cv2, scipy, pickle

from PyVutils import Cv, Others

import web

data_dir = "data"
knn_clf  = None
clf_path = RF"{data_dir}\digits.clf"

def train():

	global knn_clf

	print("Training MINIST with KNN classification ...")

	Xs = []
	Ys = []

	for dir in glob.glob(RF"{data_dir}\train\*"):
		for img_path in glob.glob(dir + R"\*.*"):
			img = Cv.Load(img_path)
			feature = Cv.ExtractFeature(img)
			Xs.append(feature) # extract image feature
			Ys.append(dir[-1]) # folder name as label

	knn_clf = neighbors.KNeighborsClassifier(n_neighbors=3, weights="distance")
	knn_clf.fit(Xs, Ys)

	with open(clf_path, "wb") as f: pickle.dump(knn_clf, f)

	print("Training completed")

	return

def predict(name, plot=True):
	img_name = name if name.strip() else input(RF"{data_dir}\test\? ")
	img = Cv.Load(RF"{data_dir}\test\{img_name}")
	return predict_img(img, plot)

def predict_img(img, plot):

	global knn_clf

	if not knn_clf:
		with open(clf_path, "rb") as f:
			u = pickle._Unpickler(f)
			u.encoding = "latin1"
			knn_clf = u.load()
		print(F"Loaded the trained model '{clf_path}'")

	feature = Cv.ExtractFeature(img)
	results = knn_clf.predict([feature])

	result = results[0] if results else "-"

	if plot:
		fig, ax = plt.subplots()
		ax.axis("off")
		ax.imshow(img)
		ax.set_title(result)
		plt.show()

	return result

def main():

	nargs = len(sys.argv)
	if nargs < 2: sys.exit(0)

	act = sys.argv[1].lower()

	if act == "train": train()
	elif act == "predict": predict(sys.argv[2] if nargs == 3 else "", True)
	elif act == "web": web.run()

	return

if __name__ == "__main__":
	try: main()
	except (Exception, KeyboardInterrupt): Others.LogException(sys.exc_info())
	sys.exit()
