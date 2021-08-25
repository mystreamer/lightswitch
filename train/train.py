import pickle
import os
from modAL.models import ActiveLearner
# imports for multilabel case
from modAL.multilabel import max_score, min_confidence
# imports for multiclass case
from modAL.uncertainty import uncertainty_sampling, margin_sampling
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.svm import SVC

class Learner(object):
	""" Manage the active learning process for teaching a classifier """
	def __init__(self, learner_name: str, n_suggest: int, X: dict, y):
		""" Initialize the active learner
			Takes:
			learner_name: give it a name, such that when it is called later on a update it can be recalled
			n_suggest: how many samples per criteria should the learner return?
			X: a dict with lists of samples with keys "train" and "unlabelled"
			y: a list of labels corresponding to the "train" part of X
		"""
		self.learner_name = learner_name
		self.n_suggest = n_suggest
		self.X = X
		self.y = y
		self.learners = []
		self.__load()

	def __load(self):
		# check if a model under the given name already exists, if yes: load it
		if os.path.exists(f"models/{self.learner_name}.pickle"):
			with open(f"models/{self.learner_name}.pickle", "rb") as f:
				self.clf = pickle.load(f)
		else:
			self.clf = OneVsRestClassifier(SVC(probability=True))


		# initialize learners
		self.learners.append(
			ActiveLearner(
				estimator=self.clf,
				query_strategy=uncertainty_sampling,
				X_training=self.X["train"], y_training=self.y
				))

		self.learners.append(
			ActiveLearner(
				estimator=self.clf,
				query_strategy=margin_sampling,
				X_training=self.X["train"], y_training=self.y
				))


	def update(self):
		""" Currently not implemented """
		pass

	def drop(self):
		""" Currently not implemented """
		pass

	def save(self):
		with open(f"models/{self.learner_name}.pickle", "wb") as f:
			pickle.dump(self.learners[0].estimator, f)

	def get_predicts(self):
		""" Get predictions from the inread unlabelled samples, returned in-sequence to the supplies unlabelled samples """
		predicts = self.learners[0].predict(self.X["unlabelled"]).tolist()
		probas = self.learners[0].predict_proba(self.X["unlabelled"]).tolist()

		return predicts, probas

	def get_queryset(self):
		""" Return a dict of indices related to unlabelled instanced selected on the criteria given to learners """
		indexset = {}
		for l in self.learners:
			x, _ = l.query(self.X["unlabelled"], n_instances=self.n_suggest)
			x = set(x)
			i = self.learners.index(l) - 1
			while i >= 0 and len(indexset.keys()) > 0:
				x -= indexset[i]
				i -= 1
			indexset.update({self.learners.index(l) : x})
		return indexset
