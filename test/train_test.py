import unittest
import os
import numpy as np
from train.train import Learner


class ActiveLearningTestMultiClass(unittest.TestCase):
    def setUp(self):
        self.X = {"train": [[1, 1], [2, 2], [1, 0], [2, 0]],
                  "unlabelled": [[3, 3], [3, 1.5], [4, 2]]}
        self.y = [1, 1, 0, 0]
        self.y_ml = [[1, 1], [1, 1], [0, 0], [0, 0]]
        self.learner = Learner("testLearner", n_suggest=1, X=self.X, y=self.y)
        self.learner_ml = Learner(
            "testLearnerML", n_suggest=1, X=self.X, y=self.y_ml, multilabel=True)

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

    @unittest.skip("Skipped, fix when needed!")
    def test_multilabel_returns_predictions_and_probabilities(self):
        learner = self.learner_ml
        predicts, probas = learner.get_predicts()
        print(predicts)
        print(probas)
        assert np.shape(predicts) == np.shape(probas)

    @unittest.skip("Skipped, fix when needed!")
    def test_multilabel_oracle_sample_indices(self):
        learner = self.learner_ml
        queryset = learner.get_queryset()
        total_element_count = 0
        for k in queryset.keys():
            total_element_count += len(queryset[k])
        assert len(queryset.keys()) == 2 and total_element_count <= 2

    def test_save_model(self):
        self.learner.save()
        assert os.path.exists("models/testLearner.pickle")
        os.remove("models/testLearner.pickle")

    def test_getting_n_nearest_per_col(self):
        matrix = np.array([[1, 4, 8], [2, 5, 9], [13, 3, 6]])
        res = self.learner._idx_columnwise_nearest(
            matrix, nearestto=5, n_nearest=1)
        print(res)
        assert res == [[1], [1], [2]]

    def test_retrieve_model(self):
        pass
