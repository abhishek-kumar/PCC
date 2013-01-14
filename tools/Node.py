
# A node of the search tree (can be used for both: tag ordering and inference); 
# Each leaf of the tree is a permutation of tags

class Node: 
	def __init__(self):
		self.candidateChildren = []  # list of tags that can follow this node
		self.tagsSoFar = []          # current partial ordering of tags
		self.score = 0               # higher is better
		self.brScore = 0             # higher is better
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

