# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pyclstm
from PIL import Image
import os
import difflib
import sys
reload(sys)  
sys.setdefaultencoding('utf8')

CACHE_FOLDER = '/tmp/caffe_demos_uploads/cache'
this_dir = os.path.dirname(__file__)

def convert_to_binary(img):
	if (img.shape >= 3):
		img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret, imgBinary = cv2.threshold(img,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	height = np.size(img, 0)
	width = np.size(img, 1)
	height=60
	r,c=img.shape[:2]
	res = cv2.resize(imgBinary,((int)(height*c)/r, height), interpolation = cv2.INTER_CUBIC)
	res = cv2.fastNlMeansDenoising(res, 20, 7, 21)
	out_path = os.path.join(CACHE_FOLDER, "out.png")
	cv2.imwrite(out_path,res)
	return out_path, res


def extract_text(img_path, model_path):
	ocr = pyclstm.ClstmOcr()
	ocr.load(model_path)
	imgFile = Image.open(img_path)
	text = ocr.recognize(imgFile)
	text.encode('utf-8')
	chars = ocr.recognize_chars(imgFile)
	prob = 1
	index = 0
	# print text
	if text.find(u':') != -1 and text.index(u':') < 3:
		index = text.index(u':')+1
	if text.find(u' ') != -1 and (text.index(u' ') <= 3):
		if(len(text) > text.index(u' ') + 1):
			index = text.index(u' ') + 1
	for ind, j in enumerate(chars):
		if ind >= index:		
			prob *= j.confidence
	return text, prob


def crop_image(img,cropX=0, cropY=0, cropWidth=0, cropHeight=0):
	h = np.size(img, 0)
	w = np.size(img, 1)	
	res=img[cropY:h-cropHeight, cropX:w-cropWidth]
	out_path = os.path.join(CACHE_FOLDER, "croped.png")
	cv2.imwrite(out_path,res)
	return out_path

def clean(s):
	return s.strip().replace(',','').replace(':', '')

def similar(a, b):
	return difflib.SequenceMatcher(None, a, b).ratio()

def check_prenom(raw_string, seuil):
	print "Checking prenom... " + raw_string
	res = ""
	if ':' in raw_string:
		ind = raw_string.index(':')
		raw_string = raw_string[ind:]
	utiles = [clean(x.split()[-1]) for x in raw_string.split(',') if len(x.split()) > 0]
	dict_file = os.path.join(this_dir, 'prenoms.txt')
	with open(dict_file, 'r') as f:
		lines = f.readlines()
	words = [unicode(x.strip()) for x in lines]
	for utile in utiles:
		word_proposals = difflib.get_close_matches(unicode(utile), words, 1, 0.75)
		result = word_proposals[0] if len(word_proposals) > 0 else utile
		res += result + ', '
	return res[:-2], 0

def check_lieu(raw_string, seuil):
	print "Checking birth place... " + raw_string
	dict_file = os.path.join(this_dir, 'lieux.txt')
	with open(dict_file, 'r') as f:
		lines = f.readlines()
	words = [unicode(x.strip()) for x in lines]
	word_proposals = difflib.get_close_matches(unicode(raw_string), words, 1, seuil)
	if len(word_proposals) > 0:
		return word_proposals[0], similar(word_proposals[0], raw_string)
	else:
		return raw_string, 0

def check(s, islieu=False, seuil=0.6):
	return check_lieu(s, seuil) if islieu else check_prenom(s, seuil)

def clstm_ocr(img, islieu=False):
	if not os.path.exists(CACHE_FOLDER):
		os.makedirs(CACHE_FOLDER)
	model_path = os.path.join(this_dir, 'model-nomprenom2911-binary.clstm')
	if islieu:
		model_path = os.path.join(this_dir, 'model-lieu2911-binary.clstm')
	converted_image_path, image = convert_to_binary(img)
	maxPro = 0
	ocr_result = ""
	cropX, cropY, cropWidth, cropHeight = 1, 8, 1, 10
	if islieu:
		cropX, cropY, cropWidth, cropHeight = 3, 3, 3, 3
	for i in range (0,cropX):
		for j in range (0,cropY):
			for k in range (0,cropWidth):
				for h in range (0, cropHeight):
					img_path = crop_image(image, i, j, k, h)
					text, prob = extract_text(img_path, model_path)
					if(prob > maxPro) and (len(text)>=2):
						maxPro = prob
						ocr_result = text
					if (maxPro > 0.95) and (len(text) >= 2):
						break
	""" if result is not good enough, we do dictionary verification """
	if maxPro < 1 and islieu:
		ocr_result, prob = check(ocr_result, islieu)
		maxPro = max(prob, maxPro)
	return (ocr_result, maxPro)


def clstm_ocr_calib(img, islieu=False):
	if not os.path.exists(CACHE_FOLDER):
		os.makedirs(CACHE_FOLDER)
	model_path = os.path.join(this_dir, 'model-nomprenom2911-binary.clstm')
	if islieu:
		model_path = os.path.join(this_dir, 'model-lieu2911-binary.clstm')
		#model_path = os.path.join(this_dir, 'model-lieu-1212-binary.clstm')
	converted_image_path, image = convert_to_binary(img)
	#maxPro = 0
	#ocr_result = ""
	ocr_result, maxPro=extract_text(converted_image_path, model_path)
	if(islieu):
		if(maxPro<0.9):
			ocr_result = check(ocr_result, islieu, 0.5)
		else:
			ocr_result = check(ocr_result, islieu)
	return (ocr_result, maxPro)

if __name__ == '__main__':
	# filename = os.path.join(this_dir, 'demo', 'lieu0.png')
	# img = cv2.imread(filename)
	# s = clstm_ocr(img)
	s = ': VIGROIS)'
	print check_lieu(s)[0]
	s = 'AUROYE, JIIUIIIL'
	print check_prenom(s)[0]
	# print similar('MEINANDE', 'MEZIANE')
