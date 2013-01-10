from numpy import *
import os
from scipy.linalg import eigh
import pdb
import copy
from copy import *
import traceback
import sys

# computes kta score
# X: m x n data matrix (m examples, n features)
# c: target alignment vector, m-dimensional vector
# Y: conditioned labels, m x d matrix for d conditioned labels, can be None
# (intuitively) the KTA score is a surrogate for p(c | x, y1, y2,...yd)
def kta(X, c, Y=None):
	m,n = X.shape
		
	K = dot(X,X.T) # only dot-product kernel
	#if Y!=None: K = K*dot(Y,Y.T) # this multiplication is element-wise ONLY if X is an nparray (not matrix)
	
	if c.min() == 0.0 and c.max() == 0.0: c = c + 0.00000000001 # take care of 0-values case
	c2 = outer(c,c)
	c4 = sum(c2*c2) # this is equivalent to the Trace of the product because c2 is symmetrical
	K2 = sum(K*K)   # --"--
	
	kta = sum(K*c2)*1./sqrt(K2*c4)
	
	return kta

# KTA score of X w.r.t. multilabel matrix Y
# X: m x n data matrix
# Y: target labels of x. m x k
def kta_label(X, Y):
	m,n = X.shape
	K1 = dot(X, X.T)
	K2 = dot(Y, Y.T)
	
	kta = sum(K1*K2) * 1./sqrt(sum(K1*K1)*sum(K2*K2))
	return kta

# computes AUC
# yp: prediction vector (real-valued)
# yt: target {0,1} / {-1,1} vector
# returns AUC score
def _auc(yp,yt):
	m = len(yt)
	pids=where(yt==1)[0]
	nids=where(yt!=1)[0]
	auc=sum([int(yp[i]>yp[j]) for i in pids for j in nids])
	ties=sum([0.5*int(yp[i]==yp[j]) for i in pids for j in nids])
	#print str(ties), " ", str(auc), " ", str(len(pids)), " ", str(len(nids))
	#pdb.set_trace()
	return (ties+auc)*1./(len(pids)*len(nids) + 0.0001) # numerical issues

def auc(yp, yt): # todo: merge this with above function after making sure there are no dependencies
	yt = [int(round(yt[i])) for i in range(len(yt))]
	return _auc(array(yp), array(yt)) # to take care of non-array lists

def negativeAuc(yp, yt):
  return (-1)*auc(yt, yp) #note: parameter positions swapped due to signature of loss_func

def zerooneloss(y1, y2):
	y1 = array(y1)
	y2 = array(y2)
	y11=[]
	for i in y1:
		if(i < 0.5):
			y11.append(0);
		else:
			y11.append(1)

	y22=[]
	for i in y2:
		if(i < 0.5):
			y22.append(0);
		else:
			y22.append(1)
	ids=where(array(y11)==array(y22))[0]
	accuracy=(len(ids)*1.0)/(len(y1)*1.0)
	return 1-accuracy

# computes rank loss 
# yp: prediction vector (real-valued)
# yt: target {0,1} / {-1,1} vector
# returns rank loss
def rloss(yp,yt):
	m = len(yt)
	pids=where(yt==1)[0]
	nids=where(yt!=1)[0]
	rl=sum([int(yp[i]<yp[j]) for i in pids for j in nids])
	ties=sum([0.5*int(yp[i]==yp[j]) for i in pids for j in nids])
	ties2=sum([int(yp[i]==yp[j]) for i in pids for j in nids])
	r = 0.0;
	for ii in range(m):
		if(yt[ii]>0):
			r += 1;
		
	return [(ties+rl), (rl+ties2)/(r*(m-r))]

#def rlossVector(yp, yt):
	

# write data in libsvm format
# X: m X n matrix (m examples, n features)
# y: m-dim label vector
# fname: output file name
def write_data_libsvm(X,y,fname='data'):
	m,n = X.shape
	f = open(fname,'w')
	for i in arange(m):		
		_str = str(y[i]).replace(" ", "").replace("[","").replace("]","") + " "
		#_str = str(y[i]) + " " 
		for j in arange(n):
			_str = _str + str(j+1) + ":" + str(X[i,j])  + " "
		_str += "\n"
		f.write(_str)
		f.flush()
	f.close()
	# As a sanity check, libsvm has a tool to check if data is in multilabel svm fmt
	#$> python libsvm-3.12/tools/checkdata.py ../../data/scene/scene-train.svm

# X: m X n data matrix
# cf: fraction of variance to retain
# returns top eigen values and eigen vectors
def pca(X,cf=0.9):
	m,n =X.shape
	_m = mean(X,0)
	X = X-_m
	C = dot(X.T,X)*1./(m-1)
	eigvals,eigvecs = eigh(C)
	eigvals=eigvals[::-1]
	eigvecs=eigvecs[:,::-1]
	s = sum(eigvals)
	ids=where(cumsum(eigvals)>=cf*sum(eigvals))[0]
	return eigvals[0:ids[0]+1], eigvecs[:,0:ids[0]+1]
	
# Compute tag level accuracy,
# label level accuracy 
# returns [hammingloss, 0-1 loss, rank loss]
def computeMetrics(yp, yp_p, yt):
		k = len(yt[0])*(1.0)
		n = len(yt)*(1.0)

		accurateTags = 0.0
		microAverageAccuracy = 0.0
		accurateLabels = 0.0
		for i in range(int(n)): 										#for each example
			labelEqual = True
			accurateTagsi = 0.0
			nlli = 0.0
			#print >>tempf, str(yt[i]) + "\t" + str(yp[i]) + "\t" + str(yp_p[i])
			for j in range(int(k)): 									#for each tag in label
				if int(round(yp[i][j])) == int(round(yt[i][j])):
					accurateTags += 1
					accurateTagsi += 1
				else:
					labelEqual = False
				#nlli += yt[i][j]*log(yp_p[i][j]) + (1-yt[i][j])*log(1-yp_p[i][j])
			if labelEqual == True:
				accurateLabels += 1

		# Ranking Loss, and other tag-level losses
		rl = 0.0; nrl = 0.0; oe = 0.0; avgprec = 0.0;
		for i in range(int(n)):
			currl = 0.0; curnrl = 0.0;
			[currl, curnrl] = rloss(array(yp_p[i]), array(yt[i]))
			rl += currl; nrl += curnrl;

			ymax=0.0; argmaxy=-1;
			for ii in range(int(k)):
				if(yp_p[i][ii]>ymax):
					ymax = yp_p[i][ii];
					argmaxy = ii;
			if(yt[i][argmaxy] < 0.5):
				oe += 1;

			ap=0.0; r = 0.0;
			for ii in range(int(k)):
				rankk = 0.0; 
				for iii in range(int(k)):
					if(yp_p[i][iii] >= yp_p[i][ii]):
						rankk += 1;
				countt = 0.0;
				for iii in range(int(k)):
					if(yp_p[i][iii] >= yp_p[i][ii]):
						countt += yt[i][iii];
				ap = ap + yt[i][ii]*(countt / rankk);
				r += yt[i][ii];
			avgprec += (ap / r);

		rl = rl/n; nrl = nrl/n; oe = oe/n; avgprec = avgprec / n;
		return [1 - ((accurateTags) / (n*k)), 1-((accurateLabels)/(n)), rl, nrl, oe, avgprec]

	
# Given X and Y, return LL
# This method is meant to be dynamically added to 
# a classifier that implements "predict_proba"
# such as logisticRegression in scikits
def logLikelihood(self, X, Y):
	LCL = 0
	for i in range(len(X)):
		p = self.predict_proba([X[i]])[0] # P(Y = 1 | X)
		p = pCorrect(p)
		LCL += (Y[i]*log(p[1])) + (1-Y[i])*log(1-p[1])
	#pdb.set_trace)
	return LCL

def pCorrect(pr):
  if len(pr) < 2: pr = append(pr, (1-pr[0]))
  if pr[0] > 0.99999: pr[0] = 0.99999
  if pr[0] < 1-0.99999: pr[0] = 1-0.99999
  if pr[1] > 0.99999: pr[1] = 0.99999
  if pr[1] < 1-0.99999: pr[1] = 1-0.99999
  return pr

def logLoss(yp, yt):
	LL=0
	try:
		for i in range(len(yp)):
			for j in range(len(yp[i])):
				LL -= (yt[i][j]*log(yp[i][j]) + (1-yt[i][j])*log(1-yp[i][j]))
	except:
		pdb.set_trace()
		traceback.print_exc(file=sys.stdout);
	return LL

def reorder(Y, reorder):
	# reorders the tags in Y according to reorder
	n = len(Y)
	k = len(Y[0])
	if reorder == []:
		return Y

	k = len(reorder)
	yt = []
	for i in range(n):
		orderIterator = iter(reorder)
		yi_new = [Y[i][orderIterator.next()] for numTags in range(k)]
		yt.append(yi_new)

	return array(yt)

# Given a vector yl, stratifier returns another
# vector which can be used to split yl into k folds
# in a stratified way. It removes classes that occur
# less than k times
def stratifier(ylabel, folds):
	n = len(ylabel)
	if len(shape(ylabel)) == 1:
		y = [[ylabel[i]] for i in range(n)]
	else:
		y = deepcopy(ylabel)
	k = len(y[0])
	ystrat = []
	strat = int(log2((1.0*n)/(folds))) # max tags possible for good stratification
	strat = min(strat, k)
	for i in range(n):
		yi_strat = 0
		for j in range(strat):
			yi_strat = yi_strat + (2**(strat-1-j))*y[i][j]
		#print "Label: " + str(y[i]) + " stratified value: " + str(yi_strat)
		ystrat.append(yi_strat)
	

	#make sure ystrat has no labels occurring just once or twice (stratification error otherwise)
	ystrat = array(ystrat)
	ystratu = unique(ystrat)
	for i in range(len(ystratu)):
		ids=where(ystrat==ystratu[i])[0]
		if len(ids) < folds:
			print "Rare label! " + str(ystratu[i]) + " count: " + str(len(ids))
			ystrat[ids]=0
	ystratu = unique(ystrat)
	ids=where(ystrat==0)[0]
	if (len(ids) < folds and len(ids) > 0): ystrat[ids]=ystratu[1]

	return ystrat

def mse(yp, yt):
	m = len(yp)
	k = len(yp[0])
	sqerror = 0.0;
	# mean squared error
	for i in range(m):
		for j in range(k):
			sqerror += (yp[i][j] - yt[i][j])**2
	return sqerror / (m*k)
