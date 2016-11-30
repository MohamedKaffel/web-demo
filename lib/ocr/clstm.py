import cv2
import numpy as np
from matplotlib import pyplot as plt
import pyclstm
from PIL import Image
import sys, getopt
import os

CACHE_FOLDER = '/tmp/caffe_demos_uploads/cache'
this_dir = os.path.dirname(__file__)


def convert_to_binary(img):
	if (img.shape >= 3):
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret, imgBinary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	height = 60
	r,c = img.shape[:2]
	res = cv2.resize(imgBinary,((int)(height*c)/r, height), interpolation = cv2.INTER_CUBIC)
	out_path = os.path.join(CACHE_FOLDER, "out.png")
	cv2.imwrite(out_path, res)
	return out_path


def extract_text(img_path, model_path):
	ocr = pyclstm.ClstmOcr()
	ocr.load(model_path)
	imgFile = Image.open(img_path)
	text = ocr.recognize(imgFile)
	text.encode('utf-8')
	return text


def clstm_ocr(img, islieu=False):
	if not os.path.exists(CACHE_FOLDER):
		os.makedirs(CACHE_FOLDER)
	model_path = os.path.join(this_dir, 'model-nomprenom2911-binary.clstm')
	if islieu:
		model_path = os.path.join(this_dir, 'model-lieu2911-binary.clstm')
	converted_image = convert_to_binary(img)
	return extract_text(converted_image, model_path)


if __name__ == '__main__':
	filename = os.path.join(this_dir, '..', 'nom1.png')
	clstm_ocr()