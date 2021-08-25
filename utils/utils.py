# essentially wrappers for classes / functions that require a single call
import json
import requests
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize
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


class Translator(object):
    api_root = "https://api-free.deepl.com/v2/translate"

    def __init__(self, auth_key, source_lang, target_lang):
        self.auth_key = auth_key
        self.source_lang = source_lang
        self.target_lang = target_lang

    def translate_text(self, text):
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
