# essentially wrappers for classes / functions that require a single call
import re
import json
import copy
import nltk
import requests
import functools
import numpy as np
import pandas as pd
import scipy.sparse as sp
from nltk.corpus import stopwords
from sklearn.preprocessing import normalize
from nltk.tokenize.punkt import PunktSentenceTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer


class CTFIDFVectorizer(TfidfTransformer):
    def __init__(self, *args, **kwargs):
        super(CTFIDFVectorizer, self).__init__(*args, **kwargs)

    def fit(self, X: sp.csr_matrix, n_samples: int):
        """Learn the idf vector (global term weights) """
        _, n_features = X.shape
        df = np.squeeze(np.asarray(X.sum(axis=0)))
        idf = np.log(n_samples / df)
        self._idf_diag = sp.diags(idf, offsets=0,
                                  shape=(n_features, n_features),
                                  format='csr',
                                  dtype=np.float64)
        return self

    def transform(self, X: sp.csr_matrix) -> sp.csr_matrix:
            """Transform a count-based matrix to c-TF-IDF """
            X = X * self._idf_diag
            X = normalize(X, axis=1, norm='l1', copy=False)
            return X

    def __TFIDF_on_group(self, df, groupby, text_column_name, stopword_lang):
        text_per_class = df.groupby(groupby, as_index=False).agg({text_column_name: ' '.join}).reset_index().set_index(groupby)
        nltk.download('stopwords')
        count_vectorizer = CountVectorizer(stop_words=stopwords.words(stopword_lang)).fit(text_per_class[text_column_name])
        count = count_vectorizer.transform(text_per_class[text_column_name])
        words = count_vectorizer.get_feature_names()
        ctfidf = self.fit_transform(count, n_samples=len(df)).toarray()
        return ctfidf, words, text_per_class.index.tolist()

    def get_most_prominent_words(self, dc, groupby, text_column_name, nr_of_ranks, stopword_lang):
        """ Generate TFIDF DataFrame with the n_most prominent words per group
        Takes:
            dc: A dict to use a the basis for the calculation.
            groupby[list|str]: List or string of criteria on which the documents will be based.
            text_column_name[str]: Underlying column to use as text in documents
            nr_of_ranks[int]: How many ranks do you wish to generate?

        Returns:
            A list-oriented dict with Document, Word, Rank
        """
        df = pd.DataFrame.from_dict(dc)

        ctfidf, words, doc_names = self.__TFIDF_on_group(df, groupby, text_column_name, stopword_lang)

        words_per_class = {doc_name: list([words[index] for index in ctfidf[i].argsort()[-nr_of_ranks:]]) for i, doc_name in enumerate(doc_names)}

        ctftid_dict = {'group_label': [d for d in doc_names for i in range(0, len(words_per_class[d]))],
            'word': [words_per_class[d][i] for d in doc_names for i in range(0, len(words_per_class[d]))],
            'rank': [i for d in doc_names for i in range(1, nr_of_ranks + 1)],
            'tfidf': [ctfidf[i][score] for i, _ in enumerate(doc_names) for score in ctfidf[i].argsort()[-nr_of_ranks:][::-1]]}

        return pd.DataFrame.from_dict(ctftid_dict).to_dict(orient="list")

    def get_similarity_matrix(self, dc, groupby, text_column_name, stopword_lang):
        """ Generate Matrix Dataframe to see semantic similarities between different groupings of text
        Takes:
            df: A dict to use a the basis for the calculation.
            groupby[list|str]: List or string of criteria on which the documents will be based.
            text_column_name[str]: Underlying column to use as text in documents

        Returns:
            A DataFrame matrix denoting the similarity between different "groups"
        """
        df = pd.DataFrame.from_dict(dc)

        ctfidf, _, doc_names = self.__TFIDF_on_group(df, groupby, text_column_name, stopword_lang)

        return pd.DataFrame(cosine_similarity(ctfidf), index=doc_names, columns=doc_names)

class Translator(object):
    """ Translate any strings given a source and target language using the DeepL API """
    api_root = "https://api-free.deepl.com/v2/translate"

    def __init__(self, auth_key, source_lang, target_lang):
        """ Initialize the Translator object.
        Takes:
        auth_key: The API authentication key for requests to the DeepL API
        source_lang: Abbreviation of the source language, e.g. "DE"
        target_lang: Abbreviation of the target language, e.g. "EN"
        """
        self.auth_key = auth_key
        self.source_lang = source_lang
        self.target_lang = target_lang

    def translate_text(self, text):
        """ Pass a string to this function which is to be translated """
        params = {
            "auth_key": self.auth_key,
            "source_lang": self.source_lang,
            "text": text,
            "target_lang": self.target_lang
        }

        try:
            res = requests.get(f"https://api-free.deepl.com/v2/translate", params=params)
        except requests.exceptions.RequestException as e:
            raise SystemExit(e)

        # TODO: Cases for multiple translations, if even possible by the API.

        return res.json()["translations"][0]["text"]

class KWIC(object):
    def __init__(self):
        pass

    def generate_sent_ranges(self, dc: dict, text_col: str):
        return list(map(self.__text_to_sent_range, dc[text_col]))

    def __text_to_sent_range(self, text):
        ranges = []
        for start, end in PunktSentenceTokenizer().span_tokenize(text):
            ranges.append((start, end))
        return ranges

    @staticmethod
    def get_index_of_range_list(range_list, x):
        i = 0
        for r in range_list:
            if x in range(*r):
                return i
            i += 1
        return -1

    @staticmethod
    def get_keywords(filepath):
        with open(filepath, "r+", encoding="UTF-8") as f:
            keywords = []
            for line in f.readlines():
                keywords.append(line.replace("\n", "").strip())
            return keywords


class MatchCounter(object):
    def __init__(self):
        """ Use hierarchical matchcounting to extract the number of matches"""
        pass

    def __nestify_rec(self, col_order: list, inner_cols:list):
        # base case
        if col_order == []:
            return lambda x: x.to_dict(orient="list")
        else:
            col = col_order.pop(0)
            f = self.__nestify_rec(copy.deepcopy(col_order), inner_cols)
            return lambda x: x.groupby(col)[col_order + inner_cols].apply(f).to_dict()

    def nestify(self, dc, col_order: list, inner_cols: list) -> dict:
        """ Entry call to creating a nested data structure from a dict of lists """
        df = pd.DataFrame.from_dict(dc)
        col = col_order.pop(0)
        f = self.__nestify_rec(copy.deepcopy(col_order), inner_cols)
        return df.groupby(col)[col_order + inner_cols].apply(f).to_dict()

    def _iter_leaf_item(self, ds, func, leaf_criterion):
        """ Apply a transform or aggregation to a leaf-item and persist it to that leaf
        TAKES:
        func: A function to perform the transform, the leaf dict itself will be passed as arg
        leaf_criterion: a criterion that identifies the leaf, must return True of leaf

        """
        for key in ds.keys():
            # base
            if leaf_criterion(ds[key]):
                func(ds[key])
            else:
                self._iter_leaf_item(ds[key], func, leaf_criterion)

    def _agg_on_nested(self, ds, agg_func, leaf_criterion):
        """ Apply an aggregation function on a nested dict 
        TAKES:
        agg_func: The aggregation function to be applied to inner nodes (non-immediate ancestor of a leaf)

        RETURNS: The same data structure populated with an additional attribute, as specified by the functions
        """
        for key in copy.deepcopy(list(ds.keys())):
            if leaf_criterion(ds[key]):
                break
            else:
                self._agg_on_nested(ds[key], agg_func, leaf_criterion)
        agg_func(ds)

    def _process_leafs(self, ds, text: str):
        f = lambda t, x: x.update({"agg": {"sum": sum([len(re.findall(re.compile(term) if c else re.compile(term, re.IGNORECASE), t)) for term, c in zip(x["regex"], x["case-sensitive"])])}})
        partial = functools.partial(f, text)
        lc = lambda x: True if any([k in ["regex"] for k in x.keys()]) else False
        self._iter_leaf_item(ds, partial, lc)

    def _process_inner_nodes(self, ds):
        agg = lambda x: x.update({"agg": {"sum": sum([x[y]["agg"]["sum"] for y in x.keys()])}})
        lc = lambda x: True if any([k in ["regex"] for k in x.keys()]) else False
        self._agg_on_nested(ds, agg, lc)

    def count_matches(self, ds: dict, text: str):
        self._process_leafs(ds, text)
        self._process_inner_nodes(ds)
        return ds

    def __rec_flatten_by(self, ds, on, to_fill, leaf_criterion):
        for key in list(set(ds.keys()) - set(["agg"])):
            if leaf_criterion(ds[key]):
                # print(ds[key]["agg"][on])
                to_fill.update({key: ds[key]["agg"][on] if key not in to_fill.keys() else ds[key]["agg"][on] + to_fill[key]})
            else:
                self.__rec_flatten_by(ds[key], "sum", to_fill, leaf_criterion)
                # print(ds[key]["agg"][on])
                to_fill.update({key: ds[key]["agg"][on] if key not in to_fill.keys() else ds[key]["agg"][on] + to_fill[key]})

    def flatten_by(self, ds, on):
        """ Flatten and transpose a nested dict into linear records with respect to an item key and a key key. """
        to_fill = {}
        lc = lambda x: True if any([k in ["regex"] for k in x.keys()]) else False
        self.__rec_flatten_by(ds, on, to_fill, lc)
        return to_fill
        


