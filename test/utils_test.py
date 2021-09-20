import unittest
import copy
from utils.utils import CTFIDFVectorizer
from utils.utils import Translator
from utils.utils import MatchCounter
from utils.utils import KWIC
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

class MatchCounterTest(unittest.TestCase):
	def setUp(self):
		self.mc = MatchCounter()

		self.listed = {"letter_cat": ["consonants", "consonants", "consonants", "consonants", "vowels", "vowels", "vowels", "vowels"],
				"letter_order": ["first_two_c", "first_two_c", "second_two_c", "second_two_c", "first_two_v", "first_two_v", "second_two_v", "second_two_v"],
				"regex": ["b", "c", "d", "f", "a", "e", "i", "o"],
				"case-sensitive": [False, False, False, False, True, True, True, True]}

		self.listed_w_none = {"letter_cat": [None, "consonants", "consonants", "consonants", "vowels", "vowels", "vowels", "vowels"],
				"letter_order": ["first_two_c", "first_two_c", None, "second_two_c", "first_two_v", "first_two_v", "second_two_v", "second_two_v"],
				"regex": ["b", "c", "d", "f", "a", "e", "i", "o"],
				"case-sensitive": [False, False, False, False, True, True, True, True]}

		self.nested = {"consonants": 
					{"first_two_c": {"regex": ["b", "c"], "case-sensitive": [False, False]}, 
					"second_two_c": {"regex" : ["d", "f"], "case-sensitive": [False, False]}
					}, 
				"vowels": 
					{"first_two_v": {"regex": ["a", "e"], "case-sensitive": [True, True]},
					"second_two_v": {"regex": ["i", "o"], "case-sensitive": [True, True]}
					}
				}

		self.nested_w_counts = {"consonants": 
					{"first_two_c": {"regex": ["b", "c"], "case-sensitive": [False, False], "agg": {"sum": 7}}, 
					"second_two_c": {"regex" : ["d", "f"], "case-sensitive": [False, False], "agg": {"sum": 7}}
					}, 
				"vowels": 
					{"first_two_v": {"regex": ["a", "e"], "case-sensitive": [True, True], "agg": {"sum": 2}},
					"second_two_v": {"regex": ["i", "o"], "case-sensitive": [True, True], "agg": {"sum": 3}}
					}
				}

		self.nested_w_counts_processed = {"consonants": 
					{"first_two_c": {"regex": ["b", "c"], "case-sensitive": [False, False], "agg": {"sum": 7}}, 
					"second_two_c": {"regex" : ["d", "f"], "case-sensitive": [False, False], "agg": {"sum": 7}},
					"agg": {"sum": 14}
					}, 
				"vowels": 
					{"first_two_v": {"regex": ["a", "e"], "case-sensitive": [True, True], "agg": {"sum": 2}},
					"second_two_v": {"regex": ["i", "o"], "case-sensitive": [True, True], "agg": {"sum": 3}},
					"agg": {"sum": 5}
					},
				"agg": {"sum": 19}
				}

		self.flattened = {"consonants": 14, "first_two_c": 7, "second_two_c": 7, "vowels": 5, "first_two_v": 2, "second_two_v": 3}

	def test_nestify(self):
		ret = self.mc.nestify(self.listed, col_order=["letter_cat", "letter_order"], inner_cols=["regex", "case-sensitive"])
		assert ret == self.nested

	def test_nestify_w_nones(self):
		self.assertRaises(AssertionError, self.mc.nestify, self.listed_w_none, col_order=["letter_cat", "letter_order"], inner_cols=["regex", "case-sensitive"])

	def test_process_leafs(self):
		ns = copy.deepcopy(self.nested)

		self.mc._process_leafs(ns, "AaaBbbCcCcDdDdEEEFFFiIiIJjJo")

		assert ns == self.nested_w_counts

	def test_process_inner_nodes(self):
		ns = copy.deepcopy(self.nested_w_counts)

		self.mc._process_inner_nodes(ns)

		assert ns == self.nested_w_counts_processed

	def test_flatten_by(self):
		nwcp = copy.deepcopy(self.nested_w_counts_processed)

		ret = self.mc.flatten_by(nwcp, "sum")

		assert ret == self.flattened

	def test_count_matches(self):
		nested = copy.deepcopy(self.nested)

		ret = self.mc.count_matches(nested, "AaaBbbCcCcDdDdEEEFFFiIiIJjJo")

		assert ret == self.nested_w_counts_processed

	def test_incoherent(self):
		listed = copy.deepcopy(self.listed)

		ret = self.mc.nestify(self.listed, col_order=["letter_order", "letter_cat"], inner_cols=["regex", "case-sensitive"])

		ret = self.mc.count_matches(ret, "AaaBbbCcCcDdDdEEEFFFiIiIJjJo")

		ret = self.mc.flatten_by(ret, "sum")

		assert ret == self.flattened

	def test_depth_control_warning(self):
		inp = {"A": {"A": {"regex": 1, "agg": {"sum": 5}}, "agg": {"sum": 5}}, "B": {"A": {"regex": 1, "agg": {"sum": 3}}, "agg": {"sum": {3}}}, "C": {"X": {"regex": 1, "agg": {"sum": 5}}, "agg": {"sum": 5}}}

		self.assertRaises(AssertionError, self.mc.flatten_by, inp, "sum")






