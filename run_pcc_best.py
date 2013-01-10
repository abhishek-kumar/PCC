import tools.data
from tools.data import *
import tools.util
from tools.util import *
import classifiers.PCC 
from classifiers.PCC import *
import pdb
import numpy
import traceback
import sys
import sklearn
from sklearn import cross_validation


print "NOTE: Numpy, Scipy and Scikit are needed to run this file."
print "      Installation instructions are in 'setup_commands.txt'"
print 

def processDataset(datafile_train, datafile_test, resultfile_train, resultfile_test, p, k):
		logf = logging.getLogger(__name__)
		[x_full,y_full] = readData(datafile_train, p, k)
		#x, x_te, y, y_te = cross_validation.train_test_split(x_full, y_full, test_fraction=0.8, random_state=0)
		#x=x_full
		y=y_full
		x = normalize_scale(x_full)

		[xtest_f,ytest_f] = readData(datafile_test, p, k)
		#xtest, x_te, ytest, y_te = cross_validation.train_test_split(xtest_f, ytest_f, test_fraction=0.8, random_state=0)
		#xtest=xtest_f
		ytest=ytest_f
		xtest = normalize_scale(xtest_f, x_full)

		outf_train = open(resultfile_train, "w")
		outf_test  = open(resultfile_test, "w")

		b = 15
		logf.info("Training PCC with beam size = " + str(b))
		clf = PCCClassifier(b=b)
		clf.fit(array(x), array(y))
		logf.info("\tBest tag order : " + str(clf.tagOrder))
			
		inferenceb = 15
		clf.clf.b = inferenceb

		yp = clf.predict(array(x))
		yp_p = clf.predict_proba(array(x))
		[hl, sl, rl] = computeMetrics(yp, yp_p, y)
		logf.info("\tSearch Beam size " + str(b) + "; Inference Beam size " + str(inferenceb) + " Training Set Hamming Loss: " + str(hl))
		logf.info("\tSearch Beam size " + str(b) + "; Inference Beam Size " + str(inferenceb) + " Training Set 0-1 Loss: " + str(sl))
		logf.info("\tSearch Beam size " + str(b) + "; Inference Beam Size " + str(inferenceb) + " Training Set Rank Loss: " + str(rl))
		print >>outf_train, str(b) + "\t" + str(inferenceb) + "\t" + str(hl) + "\t" + str(sl) + "\t" + str(rl)

		yp = clf.predict(array(xtest))
		yp_p = clf.predict_proba(array(xtest))
		[hl, sl, rl] = computeMetrics(yp, yp_p, ytest)
		logf.info("\tSearch Beam size " + str(b) + "; Inference Beam Size " + str(inferenceb) + " Test Set Hamming Loss: " + str(hl))
		logf.info("\tSearch Beam size " + str(b) + "; Inference Beam Size " + str(inferenceb) + " Test Set 0-1 Loss: " + str(sl))
		logf.info("\tSearch Beam size " + str(b) + "; Inference Beam Size " + str(inferenceb) + " Test Set Rank Loss: " + str(rl))
		print >>outf_test, str(b) + "\t" + str(inferenceb) + "\t" + str(hl) + "\t" + str(sl) + "\t" + str(rl)

		outf_train.close()
		outf_test.close()

if __name__ == "__main__":
		numpy.seterr(all='raise')
		logging.basicConfig(filename="./output/PCCBestClassifier.log", level=logging.DEBUG)
		logf = logging.getLogger(__name__)
		
		if not os.path.exists("./output"): os.makedirs("./output")

		try:
			logf.info("Started processing the scene dataset")
			processDataset("../../data/scene/scene-train.csv", "../../data/scene/scene-test.csv", "./output/scene-train-pcc-best.txt", "./output/scene-test-pcc-best.txt", 294, 6)
			logf.info("Finished processing the scene dataset")
			
			logf.info("Started processing the yeast dataset")
			processDataset("../../data/yeast/yeast-train.csv", "../../data/yeast/yeast-test.csv", "./output/yeast-train-pcc-best.txt", "./output/yeast-test-pcc-best.txt", 103, 14)
			logf.info("Finished processing the yeast dataset")

			logf.info("Started processing the medical dataset")
			processDataset("../../data/medical/medical-train.csv", "../../data/medical/medical-test.csv", "./output/medical-train-pcc-best.txt", "./output/medical-test-pcc-best.txt", 1449, 45)
			logf.info("Finished processing the medical dataset")

			logf.info("Started processing the genbase dataset")
			processDataset("../../data/genbase/genbase-train.csv", "../../data/genbase/genbase-test.csv", "./output/genbase-train-pcc-best.txt", "./output/genbase-test-pcc-best.txt", 1186,27)
			logf.info("Finished processing the genbase dataset")

			logf.info("Started processing the enron dataset")
			processDataset("../../data/enron/enron-train.csv", "../../data/enron/enron-test.csv", "./output/enron-train-pcc-best.txt", "./output/enron-test-pcc-best.txt", 1001,53)
			logf.info("Finished processing the enron dataset")

			logf.info("Started processing the emotions dataset")
			processDataset("../../data/emotions/emotions-train.csv", "../../data/emotions/emotions-test.csv", "./output/emotions-train-pcc-best.txt", "./output/emotions-test-pcc-best.txt", 72,6)
			logf.info("Finished processing the emotion dataset")
		except Exception:
			traceback.print_exc(file=sys.stdout)
			pdb.set_trace()

