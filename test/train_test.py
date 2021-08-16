import unittest
import os
from train.train import Learner


class ActiveLearningTest(unittest.TestCase):
	def setUp(self):
		self.X = {"train": [[1, 1], [2, 2], [1, 0], [2, 0]], "unlabelled": [[3, 3], [3, 1.5], [4, 2]]}
		self.y = [1, 1, 0, 0]
		self.learner = Learner("testLearner", n_suggest=1, X=self.X, y=self.y)

	def test_returns_predictions_and_probabilities(self):
		learner = self.learner
		predicts, probas = learner.get_predicts()
		assert len(predicts) == len(probas)

	def test_returns_oracle_sample_indices(self):
		learner = self.learner
		queryset = learner.get_queryset()
		total_element_count = 0
		for k in queryset.keys():
			total_element_count += len(queryset[k])
		assert len(queryset.keys()) == 2 and total_element_count <= 2
		# print(queryset.items())

	def test_save_model(self):
		self.learner.save()
		assert os.path.exists("models/testLearner.pickle")
		os.remove("models/testLearner.pickle")

	def test_retrieve_model(self):
		pass

