from numpy import *
from numpy.random import randn
from sklearn.linear_model.base import LinearModel
from sklearn.linear_model import *
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
import tools.util
from tools.util import *
import copy
from copy import *
import pdb 
import logging
logf = logging.getLogger(__name__)


class BinaryRelevanceClassifier:
	def __init__(self):
		self.classifiers = []
	
	def fit(self, X, Y):
		# if Y is one tag long only, correct it
		if len(shape(Y)) == 1:
			Y_corrected = [[i] for i in Y]
			Y = Y_corrected

		k = len(Y[0])
		n = len(Y)

		LogisticRegression.score = logLikelihood
		for l in range(k):
			yl = [Y[i][l] for i in range(n)]
			parameters = {'C':[2**i for i in range(-15,15,4)]}
			clf = LogisticRegression(C=1000, penalty='l2')
			clf.score = logLikelihood
			classifier_cv = GridSearchCV(clf, parameters, loss_func=None)
			classifier_cv.fit(array(X), array(yl), cv=StratifiedKFold(stratifier(yl,2), 2), refit=True, n_jobs=-1)
			self.classifiers.append(classifier_cv.best_estimator)
			if classifier_cv.best_estimator.C < 2**-13 or classifier_cv.best_estimator.C > 2**12:
				pdb.set_trace() # we are outside the range of regularization parameters.
			#clf.fit(array(X), array(yl))
			#self.classifiers.append(clf)

		logf.info("learnt " + str(k) + " logistic regression classifers, with regularization estimation done via 10-F CV")
	
	def predict(self, T):
		k = len(self.classifiers)
		n = len(T)
		predictions = []
		for t in range(n):
			prediction = []
			for l in range(k):
				prediction.append(self.classifiers[l].predict(T[t]))
			predictions.append(prediction)
		return predictions
	
	def predict_proba(self, T):
		k = len(self.classifiers)
		n = len(T)
		predictions = []
		for t in range(n):
			prediction = []
			for l in range(k):
				pr = self.classifiers[l].predict_proba([T[t]])[0]
				pr = pCorrect(pr)
				prediction.append(pr[1])
			predictions.append(prediction)
		return predictions

	def score(self, X, Y):
		n,k = shape(Y)
		LCL = 0
		for t in range(n):
			prediction = []
			for l in range(k):
				pr = self.classifiers[l].predict_proba([X[t]])[0]
				pr = pCorrect(pr)
				LCL += (Y[t]*log(pr[1])) + (1-Y[t])*log(1-pr[1])
		return LCL

