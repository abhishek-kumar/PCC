import numpy
from numpy import *
import scipy
from scipy import *
from tools import util
from tools.util import *
import FixedPCC
from FixedPCC import *
import BinaryRelevance
from BinaryRelevance import *
import sklearn
from sklearn import cross_validation
import copy
from copy import *

logf = logging.getLogger(__name__)

class Node: # A node of the search tree; each leaf of the tree is a permutation of tags
	def __init__(self):
		self.candidateChildren = []  # list of tags that can follow this node
		self.tagsSoFar = []          # current partial ordering of tags
		self.score = 0               # higher is better
		self.brScore = 0               # higher is better
		self.accuracy = 0            # used to calculate score
		self.tagClassifiers = []     # cache of tag level classifiers so we don't re-learn the same weights as we traverse
	
	def getCandidateChildren(self):
		return self.candidateChildren
	
	def setCandidateChildren(self, newCandidateChildren):
		self.candidateChildren = newCandidateChildren

	def getTagsSoFar(self):
		return self.tagsSoFar
	
	def getScore(self):
		return self.score
	
	def makeChildOf(self, anotherNode, newTag):
		self.candidateChildren = deepcopy(anotherNode.getCandidateChildren())
		self.tagsSoFar = deepcopy(anotherNode.getTagsSoFar())
		self.tagsSoFar.append(newTag)
		# remove this current tag from possible children because the same tag cannot repeat
		self.candidateChildren.remove(newTag)
		self.tagClassifiers = [ anotherNode.tagClassifiers[i] for i in range(len(anotherNode.tagClassifiers)) ]

# A PCC Classifier which also determines a good ordering of tags
# implements beam search for ordering [reference here]

class PCCClassifier:
	def __init__(self, b=3):
		self.b = b
		self.tagOrder = []
		self.clf = []
		self.binaryRelevance = []
	
	def fit(self, X, Y):
		# We perform a BFS type traversal of the graph
		# making sure that we select b nodes at each 
		# level (or depth) before proceeding down

		b = self.b
		k = len(Y[0])
		p = len(X[0])
		n = len(Y)

		logf.info("Learning a PCC classifier on " + str(n) + "examples")

		#initialize beam for level=0
		allNodes = []
		allScores = []
		for i in range(k): # add k nodes to the top level
			curNode = Node()
			curNode.tagsSoFar = [i]
			children  = []
			for l in range(k):
				if l != i: children.append(l) 
			curNode.setCandidateChildren(children)
			self.setPCCScore(X, Y, curNode) # Fourth argument = none
			allNodes.append(curNode)
			allScores.append(curNode.getScore())
			#self.binaryRelevance.append(curNode.accuracy)
		
		#beam = allNodes
		#select the top b
		if len(allNodes) <= b:
			beam = allNodes
		else:
			selectedNodes = argsort(allScores[:])[::-1]
			snIter = iter(selectedNodes)
			beam = [allNodes[snIter.next()] for count in range(b)]

		logf.debug("At level 0, partial tags in beam:")
		logf.debug("\t"+str([beam[i].tagsSoFar for i in range(len(beam))]))
		logf.debug("Scores (in order):" + str([beam[i].score for i in range(len(beam))]))
		
		#traverse down to the leaves of the tree
		for depth in range(k-1): 
			# At this stage, beam contains b nodes
			# step 1: calculate scores for all possible nodes at current depth
			allNodes = []
			allScores = []
			for nodeIndex in range(len(beam)):
				possibleTags = beam[nodeIndex].getCandidateChildren()
				for tag in possibleTags:
					curNode = Node()
					curNode.makeChildOf(beam[nodeIndex], tag)
					self.setPCCScore(X, Y, curNode, beam[nodeIndex]) 
					allNodes.append(curNode)
					allScores.append(curNode.getScore())

			#step 2: select the top b
			if len(allNodes) <= b:
				beam = allNodes
			else:
				selectedNodes = argsort(allScores[:])[::-1]
				snIter = iter(selectedNodes)
				beam = [allNodes[snIter.next()] for count in range(b)]
			
			logf.debug("At level " + str(depth+1) + ", partial tags in beam:")
			logf.debug("\t"+str([beam[i].tagsSoFar for i in range(len(beam))]))
			logf.debug("Scores (in order):" + str([beam[i].score for i in range(len(beam))]))

		# Pick the best
		allScores = [beam[i].getScore() for i in range(len(beam))]
		bestNode = beam[ ((argsort(allScores[:]))[::-1])[0]  ]
		self.tagOrder = bestNode.getTagsSoFar()
		yReordered = reorder(Y, self.tagOrder)
		self.clf = FixedPCCClassifier(tagClassifiersOld=bestNode.tagClassifiers, b=64)
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
		#p = len(X[0]) # number of predictors, or input dimensionality
		#n = len(X)    # number of examples
		n,p = shape(X)
		#k = len(Y[0]) # number of tags in the label
		k = shape(Y)[1]

		#reorder the tags in a different sequence, if provided
		yt = reorder(Y, curNode.tagsSoFar)
		ystrat = stratifier(yt, 2)
		
		#cross validate and obtain accuracy
		logf.debug("Cross-validating the FixedPCC Classifier")
		skfold = StratifiedKFold(ystrat, 2)
		scores = []   # Fixed PCC scores
		#brScores = [] # binary relevance scores for latest tag
		t = len(curNode.tagsSoFar) - 1.0
		for train, test in skfold:
			clf = FixedPCCClassifier(tagClassifiersOld=parentNode.tagClassifiers, b=64)
			clf.fit(X[train], yt[train])
			#brclf = BinaryRelevanceClassifier()
			#y_tag = [ [ yt[j][t] ] for j in range(n)]
			#brclf.fit(array(X)[train], array(y_tag)[train])
			scores.append(clf.score(array(X)[test], array(yt)[test]))
			#brScores.append(brclf.score(array(X)[test], array(y_tag)[test]))
		curNode.accuracy = array(scores).mean() 
		
		# calculate score, scale by binary relevance
		#if len(curNode.tagsSoFar) == 1: # this is a top level node
		#	curNode.score = curNode.accuracy
		#else:
		#	curNode.brScore = (parentNode.brScore*t + array(brScores).mean()) / (t+1)
		#	curNode.score = curNode.accuracy / curNode.brScore
		curNode.score = curNode.accuracy

		# finally, train the model on entire training set
		clf = FixedPCCClassifier(tagClassifiersOld=parentNode.tagClassifiers, b=64)
		clf.fit(X,yt)
		#cache the tag-level-classifiers so we don't have to compute them again
		curNode.tagClassifiers = [ clf.tagClassifiers[i] for i in range(len(clf.tagClassifiers)) ] 
		print "Tag Order: ",str(curNode.tagsSoFar),"\tScore: ", curNode.score


		

