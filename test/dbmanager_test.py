import unittest
import mongomock
from db.db_manager import DBManager


class TestDBManager(unittest.TestCase):
	def setUp(self):
		self.mock = mongomock.MongoClient()
		self.dbm = DBManager("myDB")
		self.dbm.client = self.mock
		self.dbm.db = self.mock.db

	def test_get_collection(self):
		coll = self.dbm.get_collection("myColl")

	def test_import_tsv(self):
		res = self.dbm.import_tsv("test_files/mock_file.tsv")
		words = ["strange", "social", "flag", "frequently", "rise"]
		for elem in self.mock.db["corpus"].find({}):
			assert elem["element"] in words

	def test_provision_db(self):
		# self.dbm.provision_db()
		assert True
