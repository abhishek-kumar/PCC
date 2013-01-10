# kernel dependency estimation


from numpy import *
from numpy.random import randn
from scipy.linalg import eigh

def test():
	# create random data
	X = array(randn(500,75))
	W = array(randn(75,50))
	Y = array(dot(X,W))
	
	# train and test splits
	Xtr = X[0:300,:]; Ytr = Y[0:300,:];
	Xte = X[300:,:]; Yte = Y[300:,:];
	
	# estimate weight matrix in the transformed space
	W1,P,_meanY = kde(Xtr,Ytr,1)
	
	# do predictions in the transformed space
	Yte1 = dot(Xte,W1)
	
	# project labels back to the original label space
	Yte1 = dot(Yte1,P.T)
	Yte1 = Yte1 + _meanY # NOT SURE IF THIS IS THE RIGHT PLACE TO ADD THE MEAN	

# X: data matrix, m x n, (m examples, n features)
# Y: label matrx, m x k, (m examples, k tags)
# c: regularization parameter 
# returns weight matrix, W, of size n x k
# returne projection matrix, P, of size n x d (d < k)
def kde(X,Y,c=1):
	m,n=X.shape
	m,k=Y.shape

	eigvals,P, _meanY = pca(Y)
	Yt = dot(Y,P) # transformed Y of size m x d, with reduced dimension d < k
		
	W = rlsr(X,Yt,c) # weight matrix, size n x d
	
	return W, P, _meanY


# regularized least squares regression
# X: data matrix, m x n, (m examples, n features)
# Y: label matrx, m x d, (m examples, d outputs)
# c: regularization parameter 
# returns weight matrix W, of size n x d
def rlsr(X,Y,c=1):	
	m,n = X.shape
	W = dot(dot(linalg.inv(dot(X.T,X) + m*c*eye(n)),X.T),Y)
	return W

# principal component analysis
# Y: m x k
# cf: amount of variance to retain, determines the number of principal components
# returns top eigvals and eigvecs (projection matrix)
def pca(Y,cf=1.0):
	m,n =Y.shape
	_m = mean(Y,0)
	Y = Y-_m
	C = dot(Y.T,Y)*1./(m-1)
	eigvals,eigvecs = eigh(C)
	eigvals=eigvals[::-1]
	eigvecs=eigvecs[:,::-1]
	s = sum(eigvals)
	ids=where(cumsum(eigvals)>=cf*sum(eigvals))[0]
	return eigvals[0:ids[0]+1], eigvecs[:,0:ids[0]+1], _m

if __name__ == "__main__":
	test()
