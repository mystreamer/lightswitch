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

	def chunks(self, lst, n):
	    """Yield successive n-sized chunks from lst."""
	    for i in range(0, len(lst), n):
	        yield lst[i:i + n]

	def encode_multiprocessed(self, text: list, chunk_size=1000):
		"""Provides a method to encode text-strings into BE-representations multiprocessed
		params: text: a list of strings
		"""
		pool = self.model.start_multi_process_pool()

		chunx = list(self.chunks(text, chunk_size))

		with tqdm(total=len(chunx), leave=True, position=0):
			ret = []
			for a in tqdm(chunx, position=0, leave=True):
				ret +=  self.model.encode_multi_process(a, pool).tolist()
			self.model.stop_multi_process_pool(pool)
			return ret