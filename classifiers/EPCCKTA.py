import classifiers.EPCC
from classifiers.EPCC import *
logf = logging.getLogger(__name__)

# override

def updateKTAScore(self, X, Y, curNode, parentNode = Node()):
	n,p = shape(X)
	k = shape(Y)[1]

	# create the feature space xplus = x \oplus y \oplus (x \otimes y)
	if parentNode.tagsSoFar == []:
		yt = None
		xplus = array(X)
	else:
		yt = reorder(Y, parentNode.tagsSoFar)
		#xplus = [hstack((X[i], yt[i], kron(yt[i], X[i]))) for i in range(n)]
		xplus = [hstack((X[i], yt[i])) for i in range(n)]
		xplus = array(xplus)
		#xplus = array(yt)
	l = curNode.tagsSoFar[-1]  # current tag position
	c = array([Y[i][l] for i in range(n)]) # current tag pos values extracted out

	ktaScore = kta(xplus, c) 
	ktaScore_withouty = kta(array(X), c)
	#curNode.score = parentNode.score + (ktaScore) # simple sum
	#curNode.score = parentNode.score + log(ktaScore*ktaScore_withouty)
	curNode.score = parentNode.score + ktaScore
	logf.debug("\tPartial Tag Order: "+str(curNode.tagsSoFar)+"\tTotal Score: "+str(curNode.score) + "\tCurrent tag's Contribution: " + str(curNode.score-parentNode.score) + "\tCurrent tag's Score without y: " + str(ktaScore_withouty))

def updateKTAScoreRatio(self, X, Y, curNode, parentNode = Node()):
	n,p = shape(X)
	k = shape(Y)[1]

	# create the feature space xplus = x \oplus y \oplus (x \otimes y)
	if parentNode.tagsSoFar == []:
		yt = None
		xplus = array(X)
	else:
		yt = reorder(Y, parentNode.tagsSoFar)
		xplus = [hstack((X[i], yt[i], kron(yt[i], X[i]))) for i in range(n)]
		xplus = array(xplus)
		#xplus = array(yt)
	l = curNode.tagsSoFar[-1]  # current tag position
	c = array([Y[i][l] for i in range(n)]) # current tag pos values extracted out

	ktaScore = kta(xplus, c) 
	ktaScore_withouty = kta(array(X), c)
	#curNode.score = parentNode.score + (ktaScore) # simple sum
	#curNode.score = parentNode.score + log(ktaScore*ktaScore_withouty)
	if parentNode.score == 0:
		curNode.score = 1.0
		curNode.scoreAccuracy = ktaScore
		curNode.scoreBr = ktaScore_withouty
	else:
		curNode.scoreAccuracy = parentNode.scoreAccuracy + ktaScore
		curNode.scoreBr = parentNode.scoreBr + ktaScore_withouty
		curNode.score = curNode.scoreAccuracy / curNode.scoreBr

	logf.debug("\tPartial Tag Order: "+str(curNode.tagsSoFar)+"\tTotal Score: "+str(curNode.score) + "\tCurrent tag's Contribution: " + str(curNode.score-parentNode.score) + "\tCurrent tag's Score without y: " + str(ktaScore_withouty))

def updateKTAScore2(self, X, Y, curNode, parentNode = Node()):
	n,p = shape(X)
	k = shape(Y)[1]

	# create the feature space xplus = x \oplus y \oplus (x \otimes y)
	yt = reorder(Y, curNode.tagsSoFar)
	l = curNode.tagsSoFar[-1]  # current tag position
	c = array([Y[i][l] for i in range(n)]) # current tag pos values extracted out

	ktaScore = kta_label(array(X), array(yt))
	ktaScore_withouty = kta(array(X), c)
	#curNode.score = parentNode.score + (ktaScore) # simple sum
	curNode.score = (ktaScore)
	logf.debug("\tPartial Tag Order: "+str(curNode.tagsSoFar)+"\tTotal Score: "+str(curNode.score) + "\tCurrent tag's Score: " + str(curNode.score-parentNode.score) + "\tCurrent tag's Score without y: " + str(ktaScore_withouty))

def updateKTAScore3(self, X, Y, curNode, parentNode = Node()):
	n,p = shape(X)
	k = shape(Y)[1]

	# create the feature space xplus = x \oplus y \oplus (x \otimes y)
	yt = reorder(Y, curNode.tagsSoFar)
	xplus = [hstack((X[i], yt[i], kron(yt[i], X[i]))) for i in range(n)]
	xplus = array(xplus)
	y_remaining = reorder(Y, curNode.candidateChildren)
	l = curNode.tagsSoFar[-1]  # current tag position
	c = array([Y[i][l] for i in range(n)]) # current tag pos values extracted out

	ktaScore = kta_label(xplus, y_remaining)
	ktaScore_withouty = kta(array(X), c)
	#curNode.score = parentNode.score + (ktaScore) # simple sum
	curNode.score = (-1.0)*(ktaScore)
	logf.debug("\tPartial Tag Order: "+str(curNode.tagsSoFar)+"\tTotal Score: "+str(curNode.score) + " Remaining tags " + str(curNode.candidateChildren) + "\tCurrent tag's Score without y: " + str(ktaScore_withouty))

# Replace the scoring function in PCC, and we're done!
EPCCClassifier.setPCCScore = updateKTAScore
