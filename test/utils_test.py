import unittest
from utils.utils import CTFIDFVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

class CTFIDFVectorizerTest(unittest.TestCase):
	def setUp(self):
		self.text = ["This is a test on this system", "This is not a test just random", "Here is another test sentence"]
		self.count_matrix = CountVectorizer().fit_transform(self.text)
	def test_csr_output(self):
		ctfidf = CTFIDFVectorizer().fit_transform(self.count_matrix, n_samples = len(self.text)).toarray()
		assert ctfidf.shape == (3,11)

