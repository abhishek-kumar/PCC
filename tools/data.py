import csv
import numpy
from numpy import *
import pdb
import copy
from copy import deepcopy

# For Scene, p=294, k=6

def readData(filename,p,k):
	f = csv.reader(open(filename, "rb"))
	#no error checking for now
	x = []
	y = []
	for data in f:
		#convert data to numeric type
		for i in range(len(data)):
			data[i] = float(data[i])
		x.append(data[0:p])
		y.append(data[p:p+k])
	return [x,y]

def normalize(x, xref=[]):
	m = [] # mean
	s = [] # standard devs
	p = len(x[0]) # number of features

	retval = deepcopy(x)
	if xref == []:
		xref = retval

	for i in range(p):
		featurei = array([xref[temp][i] for temp in range(len(xref))])
		m.append(featurei.mean())
		s.append(featurei.std())

	for i in range(len(x)):
		for j in range(p):
			retval[i][j] = ((1.0*x[i][j])-m[j])/s[j]
	return retval

def normalize_scale(x, xref = []):
	minval = [] # min of each predictor
	maxval = [] # max of each predictor
	p = len(x[0])

	retval = deepcopy(x)
	if xref == []:
		xref = retval
	
	# find the min and max values
	for i in range(p):
		featurei = array([xref[temp][i] for temp in range(len(xref))])
		minval.append(float(featurei.min()))
		maxval.append(float(featurei.max()))
		if minval[i] == maxval[i]:
			maxval[i] = minval[i]+1 # all values of this feature are 0, so this prevents div zero error
	
	for i in range(len(x)):
		for j in range(p):
			scaled_01 = (float(x[i][j]) - minval[j])/(maxval[j] - minval[j])
			retval[i][j] = (scaled_01 * 2) - 1.0
	return retval


