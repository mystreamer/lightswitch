import json
import numpy as np
import pandas as pd
from typing import Union
from functools import partial


class ViewBuilder(object):
    """ Encapsulates some high-level behaviour on dicts and lists using other libraries """

    def __init__(self, viewname: str):
        self.viewname = viewname

    def load(self, filepath=None, clip=None, index_col=0):
        # TODO: ensure type consistency
        df = pd.read_csv("views/" + self.viewname +
                         ".csv" if not filepath else filepath, index_col=index_col)
        df = df[:clip]
        df = df.replace({np.nan: None})
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        return df.to_dict(orient='list')

    def save(self, dc, filepath=None):
        df = pd.DataFrame.from_dict(dc)
        df.to_csv("views/" + self.viewname +
                  ".csv" if not filepath else filepath)

    @staticmethod
    def filter(dc, predicate):
        """ Method to filter dicts based on a predicate (lambda)
        returns: 
        mapper: reversemapper filtered_index -> previous_index
        filtered result: dict where predicate holds true
        """
        # sort keys
        sorted_keys = sorted(dc.keys())
        # constraint check
        assert all([len(dc[x]) == len(dc[sorted_keys[0]])
                   for x in sorted_keys[1:len(sorted_keys)]])
        # convert dict elements into tuples -> rowwise
        tuples = map(lambda x: tuple(dc[k][x] for k in sorted_keys), range(
            len(list(dc.values())[0])))
        # filter tuple-wise, still preserving attribute names
        filtered = list(filter(lambda x: predicate(x[1]), enumerate(
            map(lambda x: {y[0]: y[1] for y in zip(sorted_keys, x)}, tuples))))
        # flatten over individual dicts
        ret_filtered = {k: v for k, v in zip(
            sorted_keys, [[s[1][k] for s in filtered] for k in sorted_keys])}
        # return mapper and filtered result
        mapper = {k: v for k, v in enumerate([x[0] for x in filtered])}

        return mapper, ret_filtered

    @staticmethod
    def join_on(dc_a, dc_b, label, join_type="inner", default=None):
        """ a generic join on a specified column """
        df_a = pd.DataFrame.from_dict(dc_a)
        df_b = pd.DataFrame.from_dict(dc_b)
        merged = pd.merge(df_a, df_b, how=join_type, on=label)
        merged = merged.replace({np.nan: None})
        return merged.to_dict(orient='list')

    @staticmethod
    def change_orientation(data, to: str = "list"):
        return pd.DataFrame(data).to_dict(orient=to)

    @staticmethod
    def stringify(elements: list):
        """ Serializes a list objects and make them CSV compatible """
        return list(map(json.dumps, elements))

    @staticmethod
    def unstringify(elements: list):
        """ Reads serialized list of python objects """
        return list(map(json.loads, elements))

    @staticmethod
    def aggregate_text_on_label(dc, label_col: Union[list, str], text_col: Union[list, str], delim=" "):
        """ Buckets all text together that belongs to the same label, Nones are ignored"""
        label_col = label_col if isinstance(label_col, list) else [label_col]
        text_col = text_col if isinstance(text_col, list) else [text_col]
        tpl = pd.DataFrame.from_dict(dc)
        j_func = partial(str.join, delim)
        f_func = partial(filter, None)
        aggregations = dict({col: lambda x: j_func(f_func(x))
                            for col in text_col})
        tpl = tpl.groupby(label_col, as_index=False).agg(aggregations)
        return tpl.to_dict(orient='list')

    @staticmethod
    def aggregate_text_on_columns(dc, cols: list, delim=""):
        """ Combine different text cols into one """
        df = pd.DataFrame.from_dict(dc)
        df["ret"] = df[cols].apply(
            lambda row: delim.join(row.values.astype(str)), axis=1)
        return df["ret"].tolist()

    @staticmethod
    def size_of_groups(dc, on, horizontal=False):
        """ Get the size of different groups """
        df = pd.DataFrame.from_dict(dc)
        if horizontal:
            return df[on].sum().reset_index(name="count").rename(columns={"index": "cols"}).to_dict(orient="list")
        return df.groupby(on).size().reset_index(name="count").to_dict(orient='list')

    @staticmethod
    def combine(*args):
        """ combines iterables and constant into a single flattened list """
        ret = []
        for a in zip(*args):
            base = []
            for e in a:
                if isinstance(e, list):
                    base += a[0]
                else:
                    base.append(e)
            ret.append(base)
        return ret
