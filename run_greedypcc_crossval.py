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
		yt=array(y_full)
		y = reorder(yt, [0, 6, 4, 9, 1, 7, 2, 11, 8, 5, 3, 10])
		x = array(normalize_scale(x_full))

		#[xtest_f,ytest_f] = readData(datafile_test, p, k)
		#ytest=ytest_f
		#xtest = normalize_scale(xtest_f, x_full)

		#outf_train = open(resultfile_train, "w")
		#outf_test  = open(resultfile_test, "w")

		ystrat = stratifier(y, 5)
		skfold = StratifiedKFold(ystrat, 5)
		lossHl = []; lossSl = []; lossRl = []; lossNrl = []; lossOe = []; lossAvprec = [];
		for train, test in skfold:
			#clf = FixedPCCClassifier()
			clf = BinaryRelevanceClassifier()
			clf.fit(x[train], y[train])
			# predict
			#clf.b = 1
			yp = clf.predict(x[test])
			yp_p = clf.predict_proba(x[test])
			[hl, sl, rl, nrl, oe, avprec] = computeMetrics(yp, yp_p, y[test])
			lossHl.append(hl); lossSl.append(sl); lossRl.append(rl); 
			lossNrl.append(nrl); lossOe.append(oe); lossAvprec.append(avprec);
		print "After training, average performance over 5 folds:"
		print "\tHL: ",array(lossHl).mean(), " +- ", array(lossHl).std()
		print "\tSL: ",array(lossSl).mean(), " +- ", array(lossSl).std()
		print "\tRL: ",array(lossRl).mean(), " +- ", array(lossRl).std()
		print "\tNRL: ",array(lossNrl).mean(), " +- ", array(lossNrl).std()
		print "\tOE: ",array(lossOe).mean(), " +- ", array(lossOe).std()
		print "\tAVPREC: ",array(lossAvprec).mean(), " +- ", array(lossAvprec).std()
		
		'''
		for inferenceb in [1]:
		#for inferenceb in [2**len(ytest[0])]:
			clf.b = inferenceb
			#yp = clf.predict(array(x))
			#yp_p = clf.predict_proba(array(x))

			#[hl, sl, rl] = computeMetrics(yp, yp_p, y)
			#ll = logLoss(yp_p, y)
			#logf.info("\tInference Beam size " + str(inferenceb) + " Training Set Hamming Loss: " + str(hl))
			#logf.info("\tInference Beam Size " + str(inferenceb) + " Training Set 0-1 Loss: " + str(sl))
			#logf.info("\tInference Beam Size " + str(inferenceb) + " Training Set Rank Loss: " + str(rl))
			#logf.info("\tInference Beam Size " + str(inferenceb) + " Training Set Log Loss: " + str(ll))
			#print >>outf_train, str(inferenceb) + "\t" + str(hl) + "\t" + str(sl) + "\t" + str(rl) + "\t" + str(ll)

			yp = clf.predict(array(xtest))
			yp_p = clf.predict_proba(array(xtest))
			[hl, sl, rl, nrl, oe, avprec] = computeMetrics(yp, yp_p, ytest)
			#ll = logLoss(yp_p, ytest)
			logf.info("\tInference Beam Size " + str(inferenceb) + " Test Set Hamming Loss: " + str(hl))
			logf.info("\tInference Beam Size " + str(inferenceb) + " Test Set 0-1 Loss: " + str(sl))
			logf.info("\tInference Beam Size " + str(inferenceb) + " Test Set Rank Loss: " + str(rl))
			#logf.info("\tInference Beam Size " + str(inferenceb) + " Test Set Log Loss: " + str(ll))
			logf.info("\tInference Beam Size " + str(inferenceb) + " Test Set Normalized Rank Loss: " + str(nrl))
			logf.info("\tInference Beam Size " + str(inferenceb) + " Test Set One-Error: " + str(oe))
			logf.info("\tInference Beam Size " + str(inferenceb) + " Test Set Avg Prec: " + str(avprec))
			print >>outf_test, str(inferenceb) + "\t" + str(hl) + "\t" + str(sl) + "\t" + str(rl) + "\t" + str(nrl) + "\t" + str(oe) + "\t" + str(avprec)
			#pdb.set_trace()

		outf_train.close()
		outf_test.close()
		'''

if __name__ == "__main__":
		numpy.seterr(all='raise')
		logging.basicConfig(filename="./output/GreedyPccCv.log", level=logging.DEBUG)
		logf = logging.getLogger(__name__)
		
		if not os.path.exists("./output"): os.makedirs("./output")

		try:
			#logf.info("Started processing the scene dataset")
			#processDataset("../../data/scene/scene-train.csv", "../../data/scene/scene-test.csv", "./output/scene-train-fixedpcc.txt", "./output/scene-test-fixedpcc.txt", 294, 6)
			#logf.info("Finished processing the scene dataset")
			
			#logf.info("Started processing the yeast dataset")
			#processDataset("../../data/yeast/yeast-train.csv", "../../data/yeast/yeast-test.csv", "./output/yeast-train-fixedpcc.txt", "./output/yeast-test-fixedpcc.txt", 103, 14)
			#logf.info("Finished processing the yeast dataset")

			#logf.info("Started processing the medical dataset")
			#processDataset("../../data/medical/medical-train.csv", "../../data/medical/medical-test.csv", "./output/medical-train-fixedpcc.txt", "./output/medical-test-fixedpcc.txt", 1449, 45)
			#logf.info("Finished processing the medical dataset")

			#logf.info("Started processing the genbase dataset")
			#processDataset("../../data/genbase/genbase-train.csv", "../../data/genbase/genbase-test.csv", "./output/genbase-train-fixedpcc.txt", "./output/genbase-test-fixedpcc.txt", 1186,26)
			#logf.info("Finished processing the genbase dataset")

			#logf.info("Started processing the enron dataset")
			#processDataset("../../data/enron/enron-train.csv", "../../data/enron/enron-test.csv", "./output/enron-train-fixedpcc.txt", "./output/enron-test-fixedpcc.txt", 1001,53)
			#logf.info("Finished processing the enron dataset")

			logf.info("Started processing the movie dataset")
			processDataset("../../data/moviegenre/moviegenre-all.csv", "../../data/moviegenre/moviegenre-all.csv", "./output/movie-train-fixedpcc.txt", "./output/movie-test-fixedpcc.txt", 4904,12)
			logf.info("Finished processing the movie dataset")

			#logf.info("Started processing the emotions dataset")
			#processDataset("../../data/emotions/emotions-train.csv", "../../data/emotions/emotions-test.csv", "./output/emotions-train-fixedpcc.txt", "./output/emotions-test-fixedpcc.txt", 72,6)
			#logf.info("Finished processing the emotion dataset")
		except Exception:
			traceback.print_exc(file=sys.stdout)
			pdb.set_trace()

