import json
import pandas as pd
from collections import OrderedDict
from pymongo.mongo_client import MongoClient


class DBManager(object):
	def __init__(self, dbname: str, **kwargs):
		""" Get access to a MongoDB Database instance """
		self.dbname = dbname
		self.client = MongoClient(**kwargs)
		self.db = self.client[dbname]

	def get_collection(self, collection: str):
		return self.db[collection]

	def import_tsv(self, filepath: list, rename: dict = None):
		table = pd.read_csv(filepath, sep="\t", header=0, index_col=0)
		if rename:
			table = table.rename(rename)
		self.db["corpus"].insert_many(table.to_dict('records'))

	def provision_db(self):
		""" Equip the given database with validators """
		with open("db/schemas.json", "r", encoding="utf-8") as f:
			schemas = json.load(f)
		for schema_name in schemas.keys():
			# print(schemas[schema_name])
			cmd = OrderedDict([('collMod', schema_name), ('validator', schemas[schema_name]), ('validationAction', 'error'), ('validationLevel', 'strict')])
			self.db.command(cmd)
