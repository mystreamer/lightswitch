from sentence_transformers import SentenceTransformer
from tqdm import tqdm

class SBERTEncoder(object):
	"""
	Encode text segments into BERT-based numeric encodings
	params: model name
	"""
	def __init__(self, model_name: str):
		self.model = SentenceTransformer(model_name)

	def encode(self, text: list):
		"""Provides a method to encode text-strings into BE-representations
		params: text: a list of strings
		"""
		with tqdm(total=len(text), leave=True, position=0):
			ret = []
			for a in tqdm(text, position=0, leave=True):
				ret.append(self.model.encode(a))
			return ret