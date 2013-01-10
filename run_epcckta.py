import run_epcc
from run_epcc import *
import classifiers.EPCCKTA
from classifiers.EPCCKTA import *

if __name__ == "__main__":
		numpy.seterr(all='raise')
		logging.basicConfig(filename="./output/EPCCKTAClassifier.log", level=logging.DEBUG)
		logf = logging.getLogger(__name__)
		
		if not os.path.exists("./output"): os.makedirs("./output")

		try:
			logf.info("Started processing the scene dataset")
			processDataset("../../data/scene/scene-train.csv", "../../data/scene/scene-test.csv", "./output/scene-train-epcckta.txt", "./output/scene-test-epcckta.txt", 294, 6)
			logf.info("Finished processing the scene dataset")
			
			logf.info("Started processing the yeast dataset")
			processDataset("../../data/yeast/yeast-train.csv", "../../data/yeast/yeast-test.csv", "./output/yeast-train-epcckta.txt", "./output/yeast-test-epcckta.txt", 103, 14)
			logf.info("Finished processing the yeast dataset")

			#logf.info("Started processing the medical dataset")
			#processDataset("../../data/medical/medical-train.csv", "../../data/medical/medical-test.csv", "./output/medical-train-epcckta.txt", "./output/medical-test-epcckta.txt", 1449, 45)
			#logf.info("Finished processing the medical dataset")

			#logf.info("Started processing the genbase dataset")
			#processDataset("../../data/genbase/genbase-train.csv", "../../data/genbase/genbase-test.csv", "./output/genbase-train-epcckta.txt", "./output/genbase-test-epcckta.txt", 1186,26)
			#logf.info("Finished processing the genbase dataset")

			#logf.info("Started processing the enron dataset")
			#processDataset("../../data/enron/enron-train.csv", "../../data/enron/enron-test.csv", "./output/enron-train-epcckta.txt", "./output/enron-test-epcckta.txt", 1001,53)
			#logf.info("Finished processing the enron dataset")

			logf.info("Started processing the emotions dataset")
			processDataset("../../data/emotions/emotions-train.csv", "../../data/emotions/emotions-test.csv", "./output/emotions-train-epcckta.txt", "./output/emotions-test-epcckta.txt", 72,6)
			logf.info("Finished processing the emotion dataset")
		except Exception:
			traceback.print_exc(file=sys.stdout)
			pdb.set_trace()

