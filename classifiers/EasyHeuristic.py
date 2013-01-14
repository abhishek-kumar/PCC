import numpy
from numpy import *
import scipy
from scipy import *
from tools import util
from tools import Node
from tools.util import *
from tools.Node import Node
import FixedPCC
from FixedPCC import *
import BinaryRelevance
from BinaryRelevance import *
import sklearn
from sklearn import cross_validation
import copy
from copy import *

logf = logging.getLogger(__name__)

# A PCC Classifier which also determines a good ordering of tags
# implements beam search for ordering [reference here]

class EasyHeuristicClassifier:
	def __init__(self, b=3):
		self.b = b
		self.tagOrder = []
		self.clf = []
		self.binaryRelevance = []
	
	def fit(self, X, Y):
		b = self.b
		k = len(Y[0])
		p = len(X[0])
		n = len(Y)

		logf.info("Learning an EasyHeuristic classifier on " + str(n) + "examples")

		allNodes = []
		allScores = []
		for i in range(k): # add k nodes to the top level
			curNode = Node()
			curNode.tagsSoFar = [i]
			children  = []
			self.setPCCScore(X, Y, curNode) # Fourth argument = none
			allNodes.append(curNode)
			allScores.append(curNode.getScore())
		
		selectedNodes = argsort(allScores[:])[::-1]
		snIter = iter(selectedNodes)
		tagOrdering = [allNodes[snIter.next()].tagsSoFar[0] for count in range(len(allNodes))]

		# Pick the best
		self.tagOrder = tagOrdering
		yReordered = reorder(Y, self.tagOrder)
		self.clf = FixedPCCClassifier(b=64)
		self.clf.fit(X,yReordered)

	def predict(self, T):
		logf.debug("Predicting labels for dataset of size :" + str(len(T)))
		return [self.predict1(T[i]) for i in range(len(T))]

	# Predict label for one example
	def predict1(self, T):
		ytemp = self.clf.predict1(T)
		yp = deepcopy(ytemp)

		for i in range(len(self.tagOrder)):
			currentTag = self.tagOrder[i]
			yp[currentTag] = ytemp[i]
		return yp
	
	def predict_proba(self, T):
		ytemp = self.clf.predict_proba(T)
		yp = deepcopy(ytemp)
		for t in range(len(self.tagOrder)):
			currentTag = self.tagOrder[t]
			for i in range(len(T)):
				yp[i][currentTag] = ytemp[i][t]
		return yp

	def setPCCScore(self, X, Y, curNode, parentNode = Node()):
		updateKTAScore(self, X, Y, curNode, parentNode);

