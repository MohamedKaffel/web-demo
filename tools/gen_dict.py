import os

this_dir = os.path.dirname(__file__)

def remove_duplicate(dictitems):
	words_ensemble = {}
	for ditem in dictitems:
		if ditem not in words_ensemble:
			words_ensemble[ditem] = 1
			print ditem

def file_remove_duplicate(filename):
	with open(filename, 'r') as f:
		lines = f.readlines()
	dictitems = [x.strip() for x in lines]
	remove_duplicate(dictitems)

def split_comma(filename):
	with open(filename, 'r') as f:
		lines = f.readlines()
	dictitems = [x.split(':')[1].replace('.', '').split() for x in lines]
	flatten = [i.strip().replace(',', '') for item in dictitems for i in item]
	remove_duplicate(flatten)

if __name__ == '__main__':
	dictpath = os.path.join(this_dir, '../lib/ocr/annot')
	dictfile = os.path.join(dictpath, 'prenoms.txt')
	# file_remove_duplicate(dictfile)
	split_comma(dictfile)
