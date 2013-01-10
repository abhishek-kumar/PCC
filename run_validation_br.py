import tools.data
from tools.data import *
import classifiers.FixedPCC
from classifiers.FixedPCC import *
import classifiers.PCC
from classifiers.PCC import *
import pdb
import numpy
import tools
import tools.util
from tools.util import *
import copy
from copy import *
import classifiers.BinaryRelevance
from classifiers.BinaryRelevance import *

numpy.seterr(all='raise')
logging.basicConfig(filename="FixedPCCClassifier.log", level=logging.DEBUG)
print "NOTE: Numpy, Scipy and Scikit are needed to run this file."
print "      Installation instructions are in 'setup_commands.txt'"
print 

#Scene
'''
[xorig,yorig] = readData("../../data/scene/scene-train.csv", 294, 6)
xorig = array(xorig)
y = reorder(yorig, [1,2,5,4,0,3]) 
[xtestorig,ytestorig] = readData("../../data/scene/scene-test.csv", 294, 6)
ytest = reorder(ytestorig, [1,2,5,4,0,3])
'''

#yeast
[xorig,yorig] = readData("../../data/yeast/yeast-train.csv", 103, 14)
xorig = array(xorig)
y = array(yorig) #no reordering yet
[xtestorig,ytestorig] = readData("../../data/yeast/yeast-test.csv", 103, 14)
ytest = array(ytestorig)
#validation set
#skfold = StratifiedKFold(stratifier(y, 2), 2)
skfold = cross_validation.KFold(len(y), 2)
for train, validate in skfold:
	#pdb.set_trace()
	xtrain = normalize(xorig[train])
	ytrain = y[train]
	xvalidate = normalize(xorig[validate], xorig[train])
	yvalidate = y[validate]
	xtest = normalize(xtestorig, xorig[train])

	print "Fitting BR"
	clf = BinaryRelevanceClassifier()
	clf.fit(array(xtrain), array(ytrain))

	print "Predicting on validation set"
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
	print "Test set 0-1 loss: " + str(1-lla)

