import data
from data import *
import FixedPCC
from FixedPCC import *
import PCC
from PCC import *
import pdb
import numpy
import util
from util import *
import copy
from copy import *

numpy.seterr(all='raise')
logging.basicConfig(filename="FixedPCCClassifier.log", level=logging.DEBUG)
print "NOTE: Numpy, Scipy and Scikit are needed to run this file."
print "      Installation instructions are in 'setup_commands.txt'"
print 

#Scene
[xorig,yorig] = readData("../../data/scene/scene-train.csv", 294, 6)
xorig = array(xorig)
y = reorder(yorig, [1,2,5,4,0,3]) 
[xtestorig,ytestorig] = readData("../../data/scene/scene-test.csv", 294, 6)
ytest = reorder(ytestorig, [1,2,5,4,0,3])

#validation set
skfold = StratifiedKFold(stratifier(y, 2), 2)
for train, validate in skfold:
	xtrain = normalize(xorig[train])
	ytrain = y[train]
	xvalidate = normalize(xorig[validate], xorig[train])
	yvalidate = y[validate]
	xtest = normalize(xtestorig, xorig[train])

	print "Fitting FixedPCC"
	clf = FixedPCCClassifier(b=-10)
	clf.fit(array(xtrain), array(ytrain))

	print "Predicting on validation set"
	clf.b = 15
	yp = clf.predict(array(xtrain))
	[tla, lla ] = computeMetrics(yp, ytrain)
	print "Training hamming loss: " + str(1-tla)
	print "Training 0-1 loss: " + str(1-lla)
	yp = clf.predict(array(xvalidate))
	[tla, lla ] = computeMetrics(yp, yvalidate)
	print "Validation set hamming loss: " + str(1-tla)
	print "Validation set 0-1 loss: " + str(1-lla)
	yp = clf.predict(array(xtest))
	[tla, lla ] = computeMetrics(yp, ytest)
	print "Test set hamming loss: " + str(1-tla)
	print "Test set validation loss: " + str(1-lla)

