import tools.data
from tools.data import *
import tools.util
import tools.kde
from tools.util import *
from tools.kde import *
import classifiers.PCC 
from classifiers.PCC import *
import pdb
import numpy
import traceback
import sys
import sklearn
from sklearn import cross_validation
from tools.StratifiedShuffleSplit import *

def predictkde(X, W, P, ytrain):
		Yte1 = dot(array(X),W)
		Yte1 = dot(Yte1,P.T)
		Yte1 = normalize_scale(Yte1, array(ytrain))
		
		# convert to binary
		yp = []
		for i in range(len(Yte1)):
			ypi = []
			for j in range(len(Yte1[i])):
				if Yte1[i][j] >= 0.0: 
					ypi.append(1.0)
				else:
					ypi.append(0.0)
			yp.append(ypi)
		yp = array(yp)
		Yte1 = (Yte1 + 1)/2.0
		return [Yte1, yp]

def processDataset(datafile_train, datafile_test, resultfile_train, resultfile_test, p, k):
		logf = logging.getLogger(__name__)
		[x_full,y_full] = readData(datafile_train, p, k)
		y=array(y_full)
		x = array(normalize_scale(x_full))

		[xtest_f,ytest_f] = readData(datafile_test, p, k)
		ytest=array(ytest_f)
		xtest = array(normalize_scale(xtest_f, x_full))

		outf_test  = open(resultfile_test, "w")

		logf.info("Training KDE...")
		ystrat = stratifier(y, 5)
		bestC = 2**-14
		bestMSE = 10000000000
		for C in [2**i for i in range(-14, 14, 1)]:
			sss = StratifiedShuffleSplit(ystrat, 5, test_size=0.2, random_state=16)
			squaredErrors = []
			for train_index, test_index in sss: # 5 times
				xtr = x[train_index]
				ytr = y[train_index]
				W,P,_meanY = kde(xtr,ytr,C)
				[yp_p, yp] = predictkde(x[test_index], W, P, ytr)
				squaredError = mse(yp_p, y[test_index])
				squaredErrors.append(squaredError)
			meanSquaredError = mean(squaredErrors)
			if meanSquaredError < bestMSE:
				bestMSE = meanSquaredError
				bestC = C
		#train based on bestC
		W,P,_meanY = kde(x,y,bestC)
		logf.info("Training complete. Best C: " + str(bestC) + "\tAverage MSE using CV: " + str(bestMSE))

		#predict on the final test set
		[yp_p, yp] = predictkde(xtest, W, P, y)

		[hl, sl, rl, nrl, oe, avprec] = computeMetrics(yp, yp_p, ytest)
		#ll = logLoss(yp_p, ytest)
		logf.info("KDE: Test Set Hamming Loss: " + str(hl))
		logf.info("KDE Test Set 0-1 Loss: " + str(sl))
		logf.info("KDE Test Set Rank Loss: " + str(rl))
		#logf.info("KDE Test Set Log Loss: " + str(ll))
		logf.info("KDE Test Set Normalized Rank Loss: " + str(nrl))
		logf.info("KDE Test Set One-Error: " + str(oe))
		logf.info("KDE Test Set Avg Prec: " + str(avprec))
		print >>outf_test, "KDE\t" + str(hl) + "\t" + str(sl) + "\t" + str(rl) 

		outf_test.close()

if __name__ == "__main__":
		numpy.seterr(all='raise')
		logging.basicConfig(filename="./output/KDEClassifier.log", level=logging.DEBUG)
		logf = logging.getLogger(__name__)
		
		if not os.path.exists("./output"): os.makedirs("./output")

		try:
			logf.info("Started processing the scene dataset")
			processDataset("../../data/scene/scene-train.csv", "../../data/scene/scene-test.csv", "./output/scene-train-kde.txt", "./output/scene-test-kde.txt", 294, 6)
			logf.info("Finished processing the scene dataset")
			
			logf.info("Started processing the yeast dataset")
			processDataset("../../data/yeast/yeast-train.csv", "../../data/yeast/yeast-test.csv", "./output/yeast-train-kde.txt", "./output/yeast-test-kde.txt", 103, 14)
			logf.info("Finished processing the yeast dataset")

			#logf.info("Started processing the medical dataset")
			#processDataset("../../data/medical/medical-train.csv", "../../data/medical/medical-test.csv", "./output/medical-train-kde.txt", "./output/medical-test-kde.txt", 1449, 45)
			#logf.info("Finished processing the medical dataset")

			#logf.info("Started processing the genbase dataset")
			#processDataset("../../data/genbase/genbase-train.csv", "../../data/genbase/genbase-test.csv", "./output/genbase-train-kde.txt", "./output/genbase-test-kde.txt", 1186,26)
			#logf.info("Finished processing the genbase dataset")

			logf.info("Started processing the enron dataset")
			processDataset("../../data/enron/enron-train.csv", "../../data/enron/enron-test.csv", "./output/enron-train-kde.txt", "./output/enron-test-kde.txt", 1001,53)
			logf.info("Finished processing the enron dataset")

			logf.info("Started processing the emotions dataset")
			processDataset("../../data/emotions/emotions-train.csv", "../../data/emotions/emotions-test.csv", "./output/emotions-train-kde.txt", "./output/emotions-test-kde.txt", 72,6)
			logf.info("Finished processing the emotion dataset")
		except Exception:
			traceback.print_exc(file=sys.stdout)
			pdb.set_trace()

