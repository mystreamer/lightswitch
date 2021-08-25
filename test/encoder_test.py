import unittest
from encoder.encoder import SBERTEncoder


class SBert_Encoder_Test(unittest.TestCase):
	def setUp(self):
		self.encoder_instance = SBERTEncoder('T-Systems-onsite/cross-en-de-roberta-sentence-transformer')

	def test_validate_embeddings(self):
		texts = ["I like python", "Diese Welt ist schön."]
		res = self.encoder_instance.encode(texts)
		print("Test")
		assert len(res) == 2