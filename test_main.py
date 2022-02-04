# QUICKTEST: python -m pytest -o -s -k "optional_method" log_cli=true test_main.py
import os
import click
import shutil
import unittest
import pandas as pd
from click.testing import CliRunner

from main import ( extract, 
                utils )

class ExtractTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.runner = CliRunner()

    def setUp(self):
        self.data = "views/data.csv"
        self.newdata = "views/newdata.csv"
        shutil.copyfile("test_files/MOCK_DATA_SENTIMENT.csv", self.data)

    # def tearDown(self):
    #     os.remove("views/data.csv")
    #     try:
    #         os.remove("views/newdata.csv")
    #     except Exception as e:
    #         pass

    def test_kwic(self):

        result = self.runner.invoke(extract, ['data', 
            'newdata',
            'kwic', 
            'test_files/test_keywords.txt', 
            'text'])


        assert result.exit_code == 0

        d = pd.read_csv(self.newdata)

        assert all([colname in d.columns.tolist() for colname in ['sent_ranges', 'context', 'keyword', 'parent_id']])

        assert len(d) == 31


    def test_ctfidf(self):
        
        result = self.runner.invoke(extract, ['data',
            'newdata',
            'ctfidf',
            'topic',
            'text'])

        d = pd.read_csv(self.newdata)

        assert result.exit_code == 0

        assert all([colname in d.columns.tolist() for colname in ['group_label', 'word', 'rank', 'tfidf']])

        assert len(d) == 1620

        mask = (d["word"] == "bad") & (d["group_label"] == '"booz allen"')

        assert float(d.loc[mask, ["tfidf"]]["tfidf"]) == 0.0878212304837641


    def test_similarity(self):

        result = self.runner.invoke(extract, ['data',
            'newdata',
            'similarity',
            '--lang',
            'english',
            'topic',
            'text'])

        d = pd.read_csv(self.newdata)

        assert result.exit_code == 0

        assert len(d) == 81
        
        assert d.loc[d["Unnamed: 0"] == "g2", '"booz allen"'].item() == 0.0009484395652787
        

class UtilsTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.runner = CliRunner()

    def setUp(self):
        self.data = "views/data.csv"
        shutil.copyfile("test_files/MOCK_DATA_SENTIMENT.csv", self.data)
        
    # def tearDown(self):
    #     os.remove("views/data.csv")
    #     try:
    #         os.remove("views/newdata.csv")
    #     except Exception as e:
    #         pass
    
    def test_sentiment(self):

        result = self.runner.invoke(utils, ['text',
            'data',
            'sentiment'])

        print(result.output)

        assert result.exit_code == 0

        d = pd.read_csv(self.data)

        assert "sentiment_score" in d.columns.tolist()

        assert d["sentiment_score"].between(-1, 1).all()

    def test_matchcounter(self):
        pass





    # def test_segment(self):

    #     result = self.runner.invoke(segment, ['someview', 'world', 'no2'])

    #     print(result.output)

        # assert result.exit_code == 0

# if __name__ == "__main__":
    # test_kwic()
    # test_segment()