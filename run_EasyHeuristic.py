import tools.data
from tools.data import *
import tools.util
from tools.util import *
import pdb
import numpy
import traceback
import sys
import sklearn
from sklearn import cross_validation
import classifiers.EasyHeuristic
from classifiers.EasyHeuristic import *

print "NOTE: Numpy, Scipy and Scikit are needed to run this file."
print "      Installation instructions are in 'setup_commands.txt'"
print 

def processDataset(datafile_train, datafile_test, resultfile_train, resultfile_test, p, k):
		logf = logging.getLogger(__name__)
		[x_full,y_full] = readData(datafile_train, p, k)
		y=y_full
		x = normalize_scale(x_full)

		[xtest_f,ytest_f] = readData(datafile_test, p, k)
		ytest=ytest_f
		xtest = normalize_scale(xtest_f, x_full)

		outf_train = open(resultfile_train, "w")
		outf_test  = open(resultfile_test, "w")

		logf.info("Training EasyHeuristic")
		clf = EasyHeuristicClassifier()
		clf.fit(array(x), array(y))
		logf.info("Predicting...")

		#yp = clf.predict(array(x))
		#yp_p = clf.predict_proba(array(x))
		#[hl, sl, rl] = computeMetrics(yp, yp_p, y)
		#logf.info(" Training Set Hamming Loss: "+ str(hl))
		#logf.info(" Training Set 0-1 Loss: "+ str(sl))
		#logf.info(" Training Set Rank Loss: "+ str(rl))
		#print >>outf_train, str(hl) + "\t" + str(sl) + "\t" + str(rl)

		yp = clf.predict(array(xtest))
		yp_p = clf.predict_proba(array(xtest))
		[hl, sl, rl, nrl, oe, avprec] = computeMetrics(yp, yp_p, ytest)
		#ll = logLoss(yp_p, ytest)
		logf.info("EH: Test Set Hamming Loss: " + str(hl))
		logf.info("EH Test Set 0-1 Loss: " + str(sl))
		logf.info("EH Test Set Rank Loss: " + str(rl))
		#logf.info("EH Test Set Log Loss: " + str(ll))
		logf.info("EH Test Set Normalized Rank Loss: " + str(nrl))
		logf.info("EH Test Set One-Error: " + str(oe))
		logf.info("EH Test Set Avg Prec: " + str(avprec))
		print >>outf_test, str(hl) + "\t" + str(sl) + "\t" + str(rl)

		outf_train.close()
		outf_test.close()

#Scene
if __name__ == "__main__":
		numpy.seterr(all='raise')
		logging.basicConfig(filename="./output/EasyHeuristic.log", level=logging.DEBUG)
		logf = logging.getLogger(__name__)
		
		if not os.path.exists("./output"): os.makedirs("./output")

		# Scene dataset
		try:
			#logf.info("Started processing the scene dataset")
			#processDataset("../../data/scene/scene-train.csv", "../../data/scene/scene-test.csv", "./output/scene-train-br.txt", "./output/scene-test-br.txt", 294, 6)
			#logf.info("Finished processing the scene dataset")
			
			#logf.info("Started processing the yeast dataset")
			#processDataset("../../data/yeast/yeast-train.csv", "../../data/yeast/yeast-test.csv", "./output/yeast-train-br.txt", "./output/yeast-test-br.txt", 103, 14)
			#logf.info("Finished processing the yeast dataset")

			#logf.info("Started processing the medical dataset")
			#processDataset("../../data/medical/medical-train.csv", "../../data/medical/medical-test.csv", "./output/medical-train-br.txt", "./output/medical-test-br.txt", 1449, 45)
			#logf.info("Finished processing the medical dataset")

			logf.info("Started processing the genbase dataset")
			processDataset("../../data/genbase/genbase-train.csv", "../../data/genbase/genbase-test.csv", "./output/genbase-train-br.txt", "./output/genbase-test-br.txt", 1186,26)
			logf.info("Finished processing the genbase dataset")

			logf.info("Started processing the enron dataset")
			processDataset("../../data/enron/enron-train.csv", "../../data/enron/enron-test.csv", "./output/enron-train-br.txt", "./output/enron-test-br.txt", 1001,53)
			logf.info("Finished processing the enron dataset")

			#logf.info("Started processing the emotions dataset")
			#processDataset("../../data/emotions/emotions-train.csv", "../../data/emotions/emotions-test.csv", "./output/emotions-train-br.txt", "./output/emotions-test-br.txt", 72,6)
			#logf.info("Finished processing the emotion dataset")

			#logf.info("Started processing the cal500 dataset")
			#processDataset("../../data/CAL500/CAL500.csv", "../../data/CAL500/CAL500.csv", "./output/cal500-train-br.txt", "./output/cal500-test-br.txt", 68,174)
			#logf.info("Finished processing the enron dataset")
		except Exception:
			traceback.print_exc(file=sys.stdout)
			pdb.set_trace()

