import unittest
from utils.utils import CTFIDFVectorizer
from utils.utils import Translator
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


class CTFIDFVectorizerTest(unittest.TestCase):
	def setUp(self):
		self.text = ["This is a test on this system", "This is not a test just random", "Here is another test sentence"]
		self.count_matrix = CountVectorizer().fit_transform(self.text)

	def test_csr_output(self):
		ctfidf = CTFIDFVectorizer().fit_transform(self.count_matrix, n_samples = len(self.text)).toarray()
		assert ctfidf.shape == (3,11)


class TranslatorTest(unittest.TestCase):
	def test_translation(self):
		t = Translator(auth_key="c1f62eb8-649b-514f-1f73-b3dc19e1c339:fx", source_lang="EN", target_lang="DE")
		ttext = t.translate_text("Hello, world!")
		assert ttext == "Hallo, Welt!"
