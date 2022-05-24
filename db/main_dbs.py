import re
import ast
import click
import umap as mp
import hdbscan as hdb
import numpy as np
import pandas as pd
from train.train import Learner
from db.db_manager import DBManager
from encoder.encoder import SBERTEncoder
from utils.utils import CTFIDFVectorizer
from viewbuilder.viewbuilder import ViewBuilder as vb
from sklearn.preprocessing import MultiLabelBinarizer


class PythonLiteralOption(click.Option):
    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except Exception as e:
            raise click.BadParameter(value)


class ArgHolder(object):
    pass


@click.group()
def entrypoint():
    pass

# insights from text


@click.group()
@click.option("--newview", default=None, help="Name of the new view, else column is appended / or view overwritten.")
@click.argument("feature")
@click.argument("viewname")
# ADD ARGUMENTS: "corpus_name"
# ADD OPTIONS "--overwrite", in case attributes exist.
# REMOVE NEWVIEW, VIEWNAME
@click.pass_context
def utils(ctx, newview, feature, viewname):
    """ Perform a utility on a feature column to generate new features

    FEATURE : The name of the column the util will be performed on.

    VIEWNAME : The name of the view which is used.
    """
    # persist common attributes to click
    ctx.obj = (newview, feature, viewname)

# TODO: add fine tuning parameters


@utils.command()
@click.option("--includep", is_flag=True, help="Raise flag to include probability scores.")
@click.pass_context
def hdbscan(ctx, includep):
    """ Perform hdbscan on a desired text-vector representation """
    newview, feature, viewname = ctx.obj

    data = vb(viewname).load()

    data[feature] = vb.unstringify(data[feature])

    clustered = hdb.HDBSCAN(
        min_cluster_size=15, prediction_data=True, min_samples=6).fit(data[feature])

    data['hdbscan'] = clustered.labels_

    if includep:
        data['hdbscan_p'] = clustered.probabilities_

    vb(newview if newview else viewname).save(data)


@utils.command()
@click.pass_context
def umap(ctx):
    """ Perform umap on a desired text-vector representation """
    newview, feature, viewname = ctx.obj

    data = vb(viewname).load()

    data[feature] = vb.unstringify(data[feature])

    reduced = mp.UMAP(n_components=10, n_neighbors=18, random_state=42)

    data['umap'] = vb.stringify(reduced.fit_transform(data[feature]).tolist())

    vb(newview if newview else viewname).save(data)


def ctfidf(ctx):
    pass

# encode text into vector representations


@click.group()
@click.option("--newview", default=None, help="Name of the new view, else column is appended / or view overwritten.")
@click.option("--filepath", default=None, help="Specify a filepath if a view doesn't exist yet")
@click.argument("textcol")
@click.argument("viewname")
@click.pass_context
# ADD ARGUMENTS: "corpus_name"
# ADD OPTIONS "--overwrite", in case attributes exist.
# REMOVE NEWVIEW, VIEWNAME
def encoder(ctx, newview, filepath, textcol, viewname):
    """ Encoders for encoding natural language text into vector-representation

    FILEPATH : Filepath to a file wanting to be loaded.

    TEXTCOL : The name of the csv column that holds text.

    VIEWNAME : The name of the view, to be created if from filebath, or to be used.
    """
    ctx.obj = (newview, filepath, textcol, viewname)


@encoder.command()
@click.option("--modelname", default=None, help="SBERT model to be used")
@click.option("--clip", default=None, help="Optional clipping parameter")
@click.pass_context
def sbert(ctx, modelname, clip):
    newview, filepath, textcol, viewname = ctx.obj

    if not newview and clip:
        click.confirm(
            'Clip is active, without a new viewname. This may overwrite your current view. Continue?', abort=True)

    data = vb(viewname).load(filepath, clip=int(clip))

    enc = SBERTEncoder(
        'T-Systems-onsite/cross-en-de-roberta-sentence-transformer' if not modelname else modelname)

    embeds = enc.encode(data[textcol])

    data['sbert'] = vb.stringify(map(lambda x: x.tolist(), embeds))

    vb(newview if newview else viewname).save(data)

# multilabel vs. multiclass
# TODO: specify featureset (columns)
# TODO: specify label (if new, add new... if old, specify a NULL value)


@click.command()
@click.option("--annotatorfile", default="oracle.csv", help="Choose the file which will query you for annotation")
# @click.option("--newlabelcolumn")
# @click.option("--criticalvalue", default=-1, )
@click.option("--nsuggest", default=5)
@click.option("--learnername", default="mylearner")
@click.option("--multilabel", is_flag=True, default=False)
@click.argument("features")
@click.argument("label")
@click.argument("viewname")
def train(annotatorfile, nsuggest, learnername, multilabel, features, label, viewname):
    """ Initiate a training process on a chosen view.

    FEATURES: comma separated feature names / columns of the views (will be combined)

    LABEL: the label the model will be trained on
    """
    dc = vb(viewname).load()

    feature_cols = list([f.strip() for f in features.split(",")])

    dc["dim"] = vb.unstringify(dc["dim"])

    dc["reduced_dim"] = vb.unstringify(dc["reduced_dim"])

    dc["feature_combination"] = vb.combine(*(dc[f] for f in feature_cols))

    X = {}

    mapper_unl, unlabelled = vb.filter(dc, lambda x: x[label] is None)

    X["unlabelled"] = unlabelled["feature_combination"]

    mapper_l, labelled = vb.filter(dc, lambda x: x[label] is not None)

    X["train"] = labelled["feature_combination"]

    MLB = MultiLabelBinarizer()

    if multilabel:
        print([list(set([f.strip() for l in labelled[label]
              for f in re.split("[,;]", l.lower())]))])
        MLB.fit([list(set([f.strip() for l in labelled[label]
                for f in re.split("[,;]", l.lower())]))])
        y = MLB.transform(list([list(map(lambda x: x.strip(), re.split(
            "[,;]", e.lower()))) for e in labelled[label]]))
    else:
        # if there a multiple labels present under the nML setting, we just use the first
        print([list(set([f.strip() for l in labelled[label]
              for f in [re.split("[,;]", l.lower())[0]]]))])
        MLB.fit([list(set([f.strip() for l in labelled[label]
                for f in [re.split("[,;]", l.lower())[0]]]))])
        y = MLB.transform(list([list(map(lambda x: x.strip(), [
                          re.split("[,;]", e.lower())[0]])) for e in labelled[label]]))

    learner = Learner(learner_name=learnername,
                      n_suggest=nsuggest, X=X, y=y, multilabel=multilabel)

    predicts, probas = learner.get_predicts()

    rand_encounters = 0

    # TODO: Wrap in function (dangerous var leak)
    for i, x, z in zip(range(0, len(predicts)), predicts, probas):
        if np.sum(x) == 0:
            if np.sum(z) != 0:
                x[np.argmax(z)] = 1
                predicts[i] = x
            else:
                rand_encounters += 1
                x[np.random.choice(range(0, len(x)))] = 1
                predicts[i] = x

    print(f"Experienced {rand_encounters} random encounters.")

    for c in MLB.classes_.tolist():
        dc[c] = list([0 for i in range(0, len(dc[list(dc.keys())[0]]))])

    for c, col_l, col_unl in zip(MLB.classes_.tolist(), np.array(y).T, np.array(predicts).T):
        for i, x in enumerate(col_l.tolist()):
            dc[c][mapper_l[i]] = x
        for i, x in enumerate(col_unl.tolist()):
            dc[c][mapper_unl[i]] = x

    dc["dim"] = vb.stringify(dc["dim"])

    dc["reduced_dim"] = vb.stringify(dc["reduced_dim"])

    dc.pop("feature_combination")

    vb(f"{viewname}_classified").save(dc)
    # qs = learner.get_queryset()


# multilabel vs. multiclass
@click.group()
def validate():
    """ Validate your sklearn model using features and a label

    FEATURES : comma separated feature names / columns (will be combined).

    LABEL : the label the model will be validated on.
    """
    pass


# add different sub entrypoints
entrypoint.add_command(utils)
entrypoint.add_command(encoder)
entrypoint.add_command(train)
entrypoint.add_command(validate)

if __name__ == "__main__":
    entrypoint()
