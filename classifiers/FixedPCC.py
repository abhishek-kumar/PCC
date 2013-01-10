from numpy import *
from numpy.random import randn
from sklearn.linear_model.base import LinearModel
from sklearn.linear_model import *
import pdb
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
import tools
from tools import util
from tools.util import *
import copy
from copy import *
import logging
import types
from collections import Counter
logf = logging.getLogger(__name__)

# A PCC classifier which does not
# determine the order of tags in labels

class FixedPCCClassifier(LinearModel):

    def __init__(self, tagClassifiersOld=[], b=3):
      # previously trained classifier at each tag
      self.tagClassifiersOld = [ tagClassifiersOld[i] for i in range(len(tagClassifiersOld)) ]  

      # classifers for each tag that we will use for prediction
      self.tagClassifiers = [tagClassifiersOld[i] for i in range(len(tagClassifiersOld))]     

      # maximum number of nodes to consider at each level. Higher b => greater accuracy
      self.b = b               

    #This algorithm learns weights for a PCC classifier 
    #for a fixed ordering of labels
    def fit(self, x, y):
      p = len(x[0]) # number of predictors, or input dimensionality
      n = len(x)    # number of examples
      k = len(y[0]) # number of tags in the label

      # This is to make sure that tagClassifiers is reset everytime we call fit()
      self.tagClassifiers = [self.tagClassifiersOld[i] for i in range(len(self.tagClassifiersOld)) ]
      
      # Our evaluation uses log likelihood rather than least squares
      LogisticRegression.score = logLikelihood

      logf.info("Fitting a FixedPCC model on dataset, with k = " + str(k))
      
      for l in arange(k):
        logf.info("\tLearning classifier for Tag " + str(l))
        
        #check if previously computed. If not, we need to learn a tag classifier
        if len(self.tagClassifiersOld) > l:
          logf.debug("\t\tTag classifier for '" + str(l) + "' was previously trained. Not re-training")
        else:
          #construct the cross product feature space
          yl = [y[i][l] for i in range(n)]
          if l == 0:
            xplus = x
          else:
            xplus = [hstack((x[i], y[i][0:l])) for i in arange(n)] # x \oplus y
            #xplus = [hstack((x[i], y[i][0:l], kron(y[i][0:l], x[i]))) for i in arange(n)]
          logf.info("\t\tCross product feature space has dimensionality " + str(len(xplus[0])))
          
          # Grid search for regularization weights using 10-fold stratified CV
          parameters = {'C':[2**i for i in range(-12,12,1)]} # Exhaustive search 
          ystrat = stratifier(yl, 5) #note: yl => stratify at tag level 
          clf = LogisticRegression(C=0.000001, penalty='l2')
          clf.score = logLikelihood
          classifier_cv = GridSearchCV(clf, parameters, loss_func=None, score_func=None, n_jobs=1) #n_jobs = -1 => parallel
          classifier_cv.fit(array(xplus), array(yl), cv=StratifiedKFold(ystrat, 5), refit=True)
          logf.info("\t\tGrid search complete. [regularizaton, score] values are: ")
          logf.info("\t\t " + str([[classifier_cv.grid_scores_[s][0]['C'], classifier_cv.grid_scores_[s][1]] for s in range(len(classifier_cv.grid_scores_))]))
          print "\t\tBest C for tag #" + str(l) + "\t" + str(classifier_cv.best_estimator.C)
          if classifier_cv.best_estimator.C > 2**10 or classifier_cv.best_estimator.C < 2**-10:
            cnt = Counter(yl)
            print "Warning: C went beyond regularization limits. C = " + str(classifier_cv.best_estimator.C) + " for tag " + str(l) + ". Count of different values of this tag: " + str(cnt)
          # clf = LogisticRegression(C=classifier_cv.best_estimator.C/10.0, penalty='l2')
          # clf.fit(array(xplus), array(yl))
          self.tagClassifiers.append(classifier_cv.best_estimator)
      logf.info("Learnt classifiers for all " + str(k) +  " tags successfully.")
      return self
    
    # predict labels for a test set
    # each label has multiple tags
    def predict(self, T):
      #print "Predicting. Beam size = " + str(self.b)
      return [self.predict1(T[i]) for i in range(len(T))]

    # Predict the leaf node of the inference tree
    # that contains the prediction for all tags
    # Essentially this is 'beam search for inference' as described in the paper
    # T must be a _single_ example
    def predictNode(self, T):
      p = len(T)
      b = self.b
      k = len(self.tagClassifiers)

      #initialize beam
      beam = []
      allNodes = []
      allScores = []
      pr = self.tagClassifiers[0].predict_proba([T])[0]
      pr = pCorrect(pr) # handle numerical issues
      for tag in range(2):
        curNode = InferenceNode()
        curNode.makeChildOf(InferenceNode(), tag, pr[1])
        curNode.score = log(pr[tag])
        allNodes.append(curNode)
        allScores.append(curNode.score)
      
      #select top b nodes
      if len(allNodes) <= b:
        beam = allNodes
      else:
        selectedNodes = argsort(allScores[:])[::-1]
        snIter = iter(selectedNodes)
        beam = [allNodes[snIter.next()] for count in range(b)]

      for depth in range(k-1):
        # We have a max of b nodes in beam
        # We consider 2*b nodes and select the best b
        allNodes = []
        allScores = []
        for nodeIndex in range(len(beam)):
          xplus = hstack((T, beam[nodeIndex].tagsSoFar))
          #xplus = hstack((T, beam[nodeIndex].tagsSoFar, kron(beam[nodeIndex].tagsSoFar, T)))
          pr = self.tagClassifiers[depth+1].predict_proba( [xplus] )[0]
          pr = pCorrect(pr)
          for tag in range(2):
            curNode = InferenceNode()
            curNode.makeChildOf(beam[nodeIndex], tag, pr[1])
            curNode.score += log(pr[tag])
            allNodes.append(curNode)
            allScores.append(curNode.score)

        #select the top b nodes
        if len(allNodes) <= b:
          beam = allNodes
        else:
          selectedNodes = argsort(allScores[:])[::-1]
          snIter = iter(selectedNodes)
          beam = [allNodes[snIter.next()] for count in range(b)]

      # Finally, select the best one
      allScores = [beam[i].score for i in range(len(beam))]
      return beam[(argsort(allScores[:])[::-1])[0]] 
      # note: many of our incorrect predictions often have the correct prediction at position 1 (not 0) above

    # Predict for a single example
    def predict1(self, T):
      print "\t\tDebug: Predicting"
      return self.predictNode(T).tagsSoFar

    # rather than 0/1 predict probability values for each tag P(tag = 1)
    def predict_proba(self, T):
      return [self.predictNode(T[i]).tagsSoFar_p for i in range(len(T))]
    
    # LCL of a test dataset  
    def score(self, X, Y):
      n = len(X)
      k = len(Y[0])
      p = len(X[0])

      # return Log likelihood as our scoring function
      LCL = 0
      for i in range(n):
        for j in range(k):
          if j == 0:
            xplus = [X[i]]
          else:
            xplus = [ hstack((X[i], Y[i][0:j])) ]
            #xplus = [ hstack((X[i], Y[i][0:j], kron(Y[i][0:j], X[i]))) ]
          # probability of tag j in example i being 1
          pr = self.tagClassifiers[j].predict_proba(xplus)[0]
          pr = pCorrect(pr)
          LCL += Y[i][j]*log(pr[1]) + (1-Y[i][j])*log(1-pr[1])

      return LCL

    def predictNodes(self, T, labelOrder):
      p = len(T)
      b = self.b
      k = len(self.tagClassifiers)
      allYScores = [-1000000000000000 for i in range(2**k)]
      allYLabels = [[] for i in range(2**k)]
      allYP = [[] for i in range(2**k)]

      #initialize beam
      beam = []
      allNodes = []
      allScores = []
      pr = self.tagClassifiers[0].predict_proba([T])[0]
      pr = pCorrect(pr) # handle numerical issues
      for tag in range(2):
        curNode = InferenceNode()
        curNode.makeChildOf(InferenceNode(), tag, pr[1])
        curNode.score = log(pr[tag])
        allNodes.append(curNode)
        allScores.append(curNode.score)
      
      #select top b nodes
      if len(allNodes) <= b:
        beam = allNodes
      else:
        selectedNodes = argsort(allScores[:])[::-1]
        snIter = iter(selectedNodes)
        beam = [allNodes[snIter.next()] for count in range(b)]

      for depth in range(k-1):
        # We have a max of b nodes in beam
        # We consider 2*b nodes and select the best b
        allNodes = []
        allScores = []
        for nodeIndex in range(len(beam)):
          xplus = hstack((T, beam[nodeIndex].tagsSoFar))
          #xplus = hstack((T, beam[nodeIndex].tagsSoFar, kron(beam[nodeIndex].tagsSoFar, T)))
          pr = self.tagClassifiers[depth+1].predict_proba( [xplus] )[0]
          pr = pCorrect(pr)
          for tag in range(2):
            curNode = InferenceNode()
            curNode.makeChildOf(beam[nodeIndex], tag, pr[1])
            curNode.score += log(pr[tag])
            allNodes.append(curNode)
            allScores.append(curNode.score)

        #select the top b nodes
        if len(allNodes) <= b:
          beam = allNodes
        else:
          selectedNodes = argsort(allScores[:])[::-1]
          snIter = iter(selectedNodes)
          beam = [allNodes[snIter.next()] for count in range(b)]

      # For each node in beam
      for b in range(len(beam)):
        # reorder the predicted tags
        tagsSoFar_reordered = deepcopy(beam[b].tagsSoFar)
        p_reordered = deepcopy(beam[b].tagsSoFar_p)
        for position in range(len(tagsSoFar_reordered)):
          currentTag = labelOrder[position]
          tagsSoFar_reordered[currentTag] = beam[b].tagsSoFar[position]
          p_reordered[currentTag] = beam[b].tagsSoFar_p[position]

        labelnumber = 0
        for i in range(k):
          labelnumber += int(round((2.0**i)*tagsSoFar_reordered[k-i-1]))
        allYScores[labelnumber] = beam[b].score
        allYLabels[labelnumber] = deepcopy(tagsSoFar_reordered)
        allYP[labelnumber] = deepcopy(p_reordered)

      return [allYScores, allYLabels, allYP]
      
class InferenceNode:
  
  def __init__(self):
    self.tagsSoFar = [] # partial classification done so far
    self.tagsSoFar_p = [] # partial classification done so far, in terms of probabilities
    self.score = 0      # score of the current tag prediction, higher => better

  def makeChildOf(self, anotherNode, tag, p1):
    self.tagsSoFar = deepcopy(anotherNode.tagsSoFar)
    self.tagsSoFar_p = deepcopy(anotherNode.tagsSoFar_p)
    self.score = deepcopy(anotherNode.score)
    self.tagsSoFar.append(tag)
    self.tagsSoFar_p.append(p1)

if __name__ == "__main__":
  # testing
  x = [[0,1,2], [1, 0, 2], [2, 1, 0], [0,0,0]]
  y = [[0,1,1],[1, 0, 1],[1, 1, 0], [0,0,0]]
  clf = FixedPCCClassifier(C = [1, 1, 1])
  clf.fit(array(x), array(y))
  print "x = " + str(x[0]) + "; y = " + str(y[0]) + "; prediction = " + str(clf.predict1(x[0]))
  print "x = " + str(x[1]) + "; y = " + str(y[1]) + "; prediction = " + str(clf.predict1(x[1]))
  print "x = " + str([1,1,1]) + "; prediction = " + str(clf.predict1([1,1,1]))
  print "x = " + str([0,0,0]) + "; prediction = " + str(clf.predict1([0,0,0]))

def zeroone(yp, y):
  #assumption: yp and y are both vectors, and not matrices
  yp=array(yp)
  y=array(y)
  ids=where(yp==y)[0]
  loss = 1-(len(ids)*1.0)/(len(y)*1.0)
  print "\t\t(Debug statement during FixedPCC parameter tuning) loss = " + str(loss)
  return loss 
