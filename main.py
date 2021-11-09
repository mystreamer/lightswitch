import re
import ast
import copy
import click
import umap as mp
import numpy as np
import pandas as pd
import hdbscan as hdb
from tqdm import tqdm
from train.train import Learner
from encoder.encoder import SBERTEncoder
from viewbuilder.viewbuilder import ViewBuilder as vb
from sklearn.preprocessing import MultiLabelBinarizer
from utils.utils import KWIC, CTFIDFVectorizer, MatchCounter, Translator


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


@click.command()
@click.option("--type", default="inner")
@click.argument("left")
@click.argument("right")
@click.argument("newview")
@click.argument("leftcols")
@click.argument("rightcols")
@click.pass_context
def join(ctx, type, left, right, newview, leftcols, rightcols):
	""" Join two files based on a single common attribute

	LEFT: The name of the first view

	RIGHT: The name of the second view

	LEFTCOLS: Cols from the left to use the join on (left will always be the dominant naming)

	RIGHTCOLS: Cols from the right to use the join on

	TYPE: e.g. inner

	"""

	data1 = vb(left).load()

	data2 = vb(right).load()

	leftcols = list([f.strip() for f in leftcols.split(",")])

	rightcols = list([f.strip() for f in rightcols.split(",")])

	assert len(leftcols) == len(rightcols), "The left and right column number must match"

	# normalize attributes of second columns
	for attr1, attr2 in zip(leftcols, rightcols):
		data2[attr1] = data2.pop(attr2)

	data = vb.join_on(data1, data2, leftcols, join_type=type)

	vb(newview).save(data)

# extracts from existing view
@click.group()
@click.argument("viewname")
@click.argument("newview")
@click.pass_context
def extract(ctx, viewname, newview):
	""" Extract text from an existing corpus

	VIEWNAME : The name of the view which is used.

	NEWVIEW : The name of the view which will be generated from the extract

	"""
	# persist common attributes to click
	ctx.obj = (viewname, newview)


@extract.command()
@click.option("--nooverlap", is_flag=True, default=False, help="Kwics will not overlap in their contexts, the first match only will count.")
@click.option("--keepdata", is_flag=True, default=True, help="Whether the kwics will be joined with all previous data.")
@click.option("--masterexpr", default=None, help="An optional master regex, where the keywords will be interpolated in, with <KEYWORD>")
@click.argument("keywords")
@click.argument("cols")
# TODO: add a window size argument
@click.pass_context
def kwic(ctx, nooverlap, keepdata, masterexpr, keywords, cols):
	""" Generates a set of matching keywords with surrounding contexts

	KEYWORDS: Filepath to a file of keywords (can be a list of regex expressions)

	COLS: If multiple columns are presented (comma separated in ""), they will be combined into a single text_dump

	"""
	viewname, newview = ctx.obj

	data = vb(viewname).load()

	data.update({"id": list(range(0, len(data[list(data.keys())[0]])))})

	keywords = KWIC().get_keywords(keywords)

	# pre_validate
	for regex in keywords:
		try:
			re.findall(re.compile(regex), "some stringg")
		except Exception as e:
			raise Exception(f"failed at regex {regex}")

	cols = list([f.strip() for f in cols.split(",")])

	if len(cols) > 1:
		data["combined_text"] = vb.aggregate_text_on_columns(data, cols, delim=". ")
		cols = ["combined_text"]

	if masterexpr:
		compiled_contexts = [(re.compile(r'%s' % (masterexpr.replace("<KEYWORD>", kw)), flags=re.IGNORECASE), kw) for kw in keywords]
	else:
		compiled_contexts = [(re.compile(r'%s' % (kw), flags=re.IGNORECASE), kw) for kw in keywords]

	matches = []

	print("Getting matches")
	with tqdm(total=len(data[cols[0]]), leave=True, position=0):
		for id, text in enumerate(tqdm(data[cols[0]], leave=True, position=0)):
			matches += [{"match": list(re.finditer(expr[0], text)), "keyword": expr[1], "id": id} for expr in compiled_contexts]

	# print(matches)

	matches = list(filter(lambda x: x["match"] if x["match"] != [] else False, matches))

	DS = {"id": [], "context": [], "keyword": []}

	non_overlapping_set = set()

	data["sent_ranges"] = KWIC().generate_sent_ranges(data, text_col=cols[0])

	print("Mapping counts")
	with tqdm(total=len(matches), leave=True, position=0):
		for match in tqdm(matches, position=0, leave=True):
			lst = []
			match["match"] = [(m.start(), m.end()) for m in match["match"]]
			for m in match["match"]:
				try:
					i = KWIC.get_index_of_range_list(data["sent_ranges"][match["id"]], m[0])

					if nooverlap:
						if any([x in non_overlapping_set for x in [(match["id"], i), (match["id"], i + 1), (match["id"], i - 1)]]):
							continue
						else:
							non_overlapping_set = non_overlapping_set.union({(match["id"], i), (match["id"], i + 1), (match["id"], i - 1)})

					# TODO: Add window size variable
					l_sent_start, l_sent_end = data["sent_ranges"][match["id"]][max(i - 1, 0)] if (i != 0) else (0, 0)
					m_sent_start, m_sent_end = data["sent_ranges"][match["id"]][i]
					r_sent_start, r_sent_end = data["sent_ranges"][match["id"]][i + 1:i + 2][0] if (data["sent_ranges"][match["id"]][i + 1:i + 2] != []) else (0, 0)
					DS["context"].append(f'{data[cols[0]][match["id"]][l_sent_start:l_sent_end]} {data[cols[0]][match["id"]][m_sent_start:m_sent_end]} {data[cols[0]][match["id"]][r_sent_start:r_sent_end]}')
					# if corpus_dict["article_text"][match["id"]][m[1]:m[1]+350] == "" or corpus_dict["article_text"][match["id"]][m[1]:m[1]+350] == " ":
					# 	print(corpus_dict["article_text"][match["id"]][m[0]: m[1]].upper())
					# 	print("____________")
					# 	print(corpus_dict["article_text"][match["id"]])
				except TypeError as e:
					DS["context"].append("n.A.")
				DS["id"].append(match["id"])
				DS["keyword"].append(match["keyword"])

			match.update({"kwic": lst})

	if keepdata:
		joined_res = vb.join_on(data, DS, "id")
		joined_res["parent_id"] = joined_res.pop("id")
		vb(newview).save(joined_res)
	else:
		DS["parent_id"] = DS.pop("id")
		vb(newview).save(DS)


@extract.command()
@click.option("--includetf", is_flag=True, default=False, help="Should word_tf (simple term frequency) be included?")
@click.option("--ranks", default=20, help="How many ranks should be generated?")
@click.option("--lang", default="german", help="Language to use for stopword removal")
@click.argument("groupby")
@click.argument("cols")
@click.pass_context
def ctfidf(ctx, includetf, ranks, lang, groupby, cols):
	viewname, newview = ctx.obj

	data = vb(viewname).load()

	cols = list([f.strip() for f in cols.split(",")])

	if len(cols) > 1:
		data["combined_text"] = vb.aggregate_text_on_columns(data, cols, delim=". ")
		cols = ["combined_text"]

	records = CTFIDFVectorizer().get_most_prominent_words(data, groupby, cols[0], int(ranks), lang)

	if not includetf:
		records.pop("word_tf")

	vb(newview).save(records)


@extract.command()
@click.option("--lang", default="de", help="Language to use for stopword removal")
@click.argument("groupby")
@click.argument("cols")
@click.pass_context
def similarity(ctx, lang, groupby, cols):
	viewname, newview = ctx.obj
	
	data = vb(viewname).load()

	cols = list([f.strip() for f in cols.split(",")])

	if len(cols) > 1:
		data["combined_text"] = vb.aggregate_text_on_columns(data, cols, delim=". ")
		cols = ["combined_text"]

	records = CTFIDFVectorizer().get_similarity_matrix(data, groupby, cols[0], lang)

	vb(newview).save(records)

@extract.command()
@click.argument("lbd")
@click.pass_context
def filterby(ctx, lbd):
	viewname, newview = ctx.obj

	lbd = eval(lbd)

	assert callable(lbd) and lbd.__name__ == "<lambda>"

	data = vb(viewname).load()

	_, filtered = vb.filter(data, lbd)

	vb(newview).save(filtered)


@extract.command()
@click.option("--printoutput", is_flag=True, default=False, help="Whether to print the output of the count.")
@click.option("--horizontal", is_flag=True, default=False, help="Only works for col<int>.")
@click.argument("cols")
@click.pass_context
def groupbycount(ctx, printoutput, horizontal, cols):
	viewname, newview = ctx.obj

	cols = list([f.strip() for f in cols.split(",")])

	data = vb(viewname).load()

	counts = vb.size_of_groups(data, cols, horizontal=True if horizontal else False)

	if horizontal:
		cols = ["cols"]

	str_tmpl = "\t".join(["%s" for x in range(0, len(cols) + 1)])

	if printoutput:
		print(str_tmpl % tuple(x for x in (cols + ["count"])))

		for elems in zip(*tuple(counts[x] for x in cols), counts["count"]):
			print(str_tmpl % elems)

	vb(newview).save(counts)


@extract.command()
@click.option("--delim", default=", ", help="Only works for col<int>.")
@click.argument("groupbycols")
@click.argument("condensecols")
@click.pass_context
def groupbycondense(ctx, delim, groupbycols, condensecols):
	""" Condense text column(s) by grouping on col(s) """
	viewname, newview = ctx.obj

	groupbycols = list([f.strip() for f in groupbycols.split(",")])

	condensecols = list([f.strip() for f in condensecols.split(",")])

	data = vb(viewname).load()

	data = vb.aggregate_text_on_label(data, label_col=groupbycols, text_col=condensecols, delim=delim)

	vb(newview).save(data)

# insights from text
@click.group()
@click.option("--newview", default=None, help="Name of the new view, else column is appended / or view overwritten.")
@click.argument("feature")
@click.argument("viewname")
@click.pass_context
def utils(ctx, newview, feature, viewname):
	""" Perform a utility on a feature column to generate new features

	FEATURE : The name of the column the util will be performed on.

	VIEWNAME : The name of the view which is used.
	"""
	# persist common attributes to click
	ctx.obj = (newview, feature, viewname)

@utils.command()
@click.option("--icol", default=None)
@click.option("--ival", default=None)
@click.option("--newcol", default=None)
@click.argument("source")
@click.argument("target")
@click.pass_context
def translate(ctx, icol, ival, newcol, source, target):
	""" Translate texts """
	newview, feature, viewname = ctx.obj

	print(f"Translating all values where {icol} = {ival} on {feature} from {source} to {target}")

	t = Translator(auth_key="c1f62eb8-649b-514f-1f73-b3dc19e1c339:fx", source_lang=source, target_lang=target)

	data = vb(viewname).load()

	col = newcol if newcol else feature

	if icol and ival:
		data[col] = map(lambda x: t.translate_text(x[0]) if x[1] == ival else x[0], zip(data[feature], data[icol]))
	else:
		data[col] = map(lambda x: t.translate_text(x), data[feature])

	vb(newview if newview else viewname).save(data)


@utils.command()
@click.option("--cscol", default=None, help="Name of an inner column that indicated case sensitivity of the match. Default, case insensitive.")
@click.argument("nestedcolorder")
@click.argument("regexcol")
@click.argument("keywordsfile")
@click.pass_context
def matchcounter(ctx, cscol, nestedcolorder, regexcol, keywordsfile):
	""" Perform a matchcount operation """
	newview, feature, viewname = ctx.obj

	data = vb(viewname).load()

	keywords = vb(keywordsfile).load(index_col=None)

	nestedcolorder = list([f.strip() for f in nestedcolorder.split(",")])

	if not cscol:
		keywords["case-sensitive"] = list([False for x in range(list(0, data.keys())[0])])
	else:
		keywords["case-sensitive"] = keywords.pop(cscol)

	keywords["regex"] = keywords.pop(regexcol)

	# pre_validate
	for regex in keywords["regex"]:
		try:
			re.findall(re.compile(regex), "some stringg")
		except Exception as e:
			raise Exception(f"failed at regex {regex}")

	mc = MatchCounter()

	nest = mc.nestify(keywords, nestedcolorder, inner_cols=["regex", "case-sensitive"])

	# pre_validate
	preval = mc.count_matches(copy.deepcopy(nest), "My String")
	preval_flattened = mc.flatten_by(preval, "sum")
	print(preval_flattened)


	with tqdm(total=len(data[feature]), leave=True, position=0):
		prepared_ds = list([mc.count_matches(copy.deepcopy(nest), text) for text in tqdm(data[feature], position=0, leave=True)])

	print("Flattening:")
	with tqdm(total=len(prepared_ds), leave=True, position=0):
		match_records = list([mc.flatten_by(ds, "sum") for ds in tqdm(prepared_ds, position=0, leave=True)])

	tbl = pd.DataFrame.from_dict(match_records).to_dict(orient="list")

	for word in copy.deepcopy(list(tbl.keys())):
		data[word] = tbl.pop(word)

	vb(newview if newview else viewname).save(data)

# TODO: add fine tuning parameters
@utils.command()
@click.option("--includep", is_flag=True, help="Raise flag to include probability scores.")
@click.option("--clusterlb", default=15, help="min_cluster_size (HDB) (how small should the clusters be minimally?)")
@click.option("--samplelb", default=6, help="min_samples (how conservative should the clustering be?) (larger, more conservative)")
@click.pass_context
def hdbscan(ctx, includep, clusterlb, samplelb):
	""" Perform hdbscan on a desired text-vector representation """
	newview, feature, viewname = ctx.obj

	data = vb(viewname).load()

	data[feature] = vb.unstringify(data[feature])

	clustered = hdb.HDBSCAN(min_cluster_size=int(clusterlb), prediction_data=True, min_samples=int(samplelb)).fit(data[feature])

	data['hdbscan'] = clustered.labels_

	if includep:
		data['hdbscan_p'] = clustered.probabilities_

	print(f"In total {len(set(clustered.labels_))} clusters have been generated.")

	sog = vb.size_of_groups(data, on="hdbscan")

	print("clusterID\tcount")

	for x, y in zip(sog["hdbscan"], sog["count"]):
		print(f"{str(x)}\t{str(y)}")

	vb(newview if newview else viewname).save(data)


@utils.command()
@click.option("--components", default=10, help="UMAP (number of dimension)")
@click.option("--neighbors", default=18, help="UMAP (low neightbors: focus on local structure)")
@click.option("--seed", default=42, help="A random seed for controlling consistency.")
@click.option("--dist",  default=0.1, help="UMAP (larger value: allow for broader topological structure / less clumps)")
@click.pass_context
def umap(ctx, components, neighbors, seed, dist):
	""" Perform umap on a desired text-vector representation """
	newview, feature, viewname = ctx.obj

	data = vb(viewname).load()

	data[feature] = vb.unstringify(data[feature])

	reduced = mp.UMAP(n_components=int(components), n_neighbors=int(neighbors), random_state=int(seed), min_dist=float(dist))

	data['umap'] = vb.stringify(reduced.fit_transform(data[feature]).tolist())

	vb(newview if newview else viewname).save(data)

# encode text into vector representations
@click.group()
@click.option("--newview", default=None, help="Name of the new view, else column is appended / or view overwritten.")
@click.option("--filepath", default=None, help="Specify a filepath if a view doesn't exist yet")
@click.argument("textcol")
@click.argument("viewname")
@click.pass_context
def encoder(ctx, newview, filepath, textcol, viewname):
	""" Encoders for encoding natural language text into vector-representation

	FILEPATH : Filepath to a file wanting to be loaded.

	TEXTCOL : The name of the csv column that holds text.

	VIEWNAME : The name of the view, to be created if from filebath, or to be used.
	"""
	ctx.obj = (newview, filepath, textcol, viewname)


@encoder.command()
@click.option("--modelname", default=None, help="SBERT model to be used")
@click.option("--multiprocessing", is_flag=True, default=False, help="Use multiprocessing?")
@click.option("--chunksize", default=None)
@click.option("--clip", default=None, help="Optional clipping parameter")
@click.pass_context
def sbert(ctx, modelname, multiprocessing, chunksize, clip):
	newview, filepath, textcol, viewname = ctx.obj

	if not newview and clip:
		click.confirm('Clip is active, without a new viewname. This may overwrite your current view. Continue?', abort=True)

	data = vb(viewname).load(filepath if filepath else None, clip=int(clip) if clip else None)

	enc = SBERTEncoder('T-Systems-onsite/cross-en-de-roberta-sentence-transformer' if not modelname else modelname)

	if multiprocessing:
		embeds = enc.encode_multiprocessed(data[textcol], chunk_size=int(chunksize) if chunksize else None)
		data['sbert'] = embeds
	else:
		embeds = enc.encode(data[textcol])
		data['sbert'] = vb.stringify(map(lambda x: x.tolist(), embeds))

	vb(newview if newview else viewname).save(data)

# multilabel vs. multiclass
# TODO: specify featureset (columns)
# TODO: specify label (if new, add new... if old, specify a NULL value)
@click.command()
@click.option("--newview", default=None, help="Name of the new view, else column is appended / or view overwritten.")
@click.option("--annotatorfile", default="oracle.csv", help="Choose the file which will query you for annotation")
# @click.option("--newlabelcolumn")
# @click.option("--criticalvalue", default=-1, )
@click.option("--nsuggest", default=5)
@click.option("--learnername", default="mylearner")
@click.option("--multilabel", is_flag=True, default=False)
@click.option("--binarize", is_flag=True, default=False)
@click.argument("features")
@click.argument("label")
@click.argument("viewname")
def train(newview, annotatorfile, nsuggest, learnername, multilabel, binarize, features, label, viewname):
	""" Initiate a training process on a chosen view.

	FEATURES: comma separated feature names / columns of the views (will be combined)

	LABEL: the label the model will be trained on
	"""
	dc = vb(viewname).load()

	assert multilabel == binarize, "Non binarized, multilabel currently not supported."

	feature_cols = list([f.strip() for f in features.split(",")])

	dc["sbert"] = vb.unstringify(dc["sbert"])

	dc["umap"] = vb.unstringify(dc["umap"])

	dc["feature_combination"] = vb.combine(*(dc[f] for f in feature_cols))

	X = {}

	mapper_unl, unlabelled = vb.filter(dc, lambda x: x[label] is None)

	X["unlabelled"] = unlabelled["feature_combination"]

	mapper_l, labelled = vb.filter(dc, lambda x: x[label] is not None)

	X["train"] = labelled["feature_combination"]

	MLB = MultiLabelBinarizer()

	if multilabel:
		print([list(set([f.strip() for l in labelled[label] for f in re.split("[,;]", str(l).lower())]))])
		MLB.fit([list(set([f.strip() for l in labelled[label] for f in re.split("[,;]", str(l).lower())]))])
		y = MLB.transform(list([list(map(lambda x: x.strip(), re.split("[,;]", str(e).lower()))) for e in labelled[label]]))
	else:
		# if there a multiple labels present under the nML setting, we just use the first
		print([list(set([f.strip() for l in labelled[label] for f in [re.split("[,;]", str(l).lower())[0]]]))])
		MLB.fit([list(set([f.strip() for l in labelled[label] for f in [re.split("[,;]", str(l).lower())[0]]]))])
		y = MLB.transform(list([list(map(lambda x: x.strip(), [re.split("[,;]", str(e).lower())[0]])) for e in labelled[label]]))

	learner = Learner(learner_name=learnername, n_suggest=nsuggest, X=X, y=y, multilabel=multilabel)

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

	if binarize:
		for c in MLB.classes_.tolist():
			dc[c] = list([0 for i in range(0, len(dc[list(dc.keys())[0]]))])

		for c, col_l, col_unl in zip(MLB.classes_.tolist(), np.array(y).T, np.array(predicts).T):
			for i, x in enumerate(col_l.tolist()):
				dc[c][mapper_l[i]] = x
			for i, x in enumerate(col_unl.tolist()):
				dc[c][mapper_unl[i]] = x
	else:
		dc["train"] = list([0 for i in range(0, len(dc[list(dc.keys())[0]]))])

		ivt_l = MLB.inverse_transform(y)
		for i, row_l in enumerate(ivt_l):
			dc["train"][mapper_l[i]] = row_l[0]

		ivt_unl = MLB.inverse_transform(np.array(predicts))
		for i, row_unl in enumerate(ivt_unl):
			dc["train"][mapper_unl[i]] = row_unl[0]

	dc["sbert"] = vb.stringify(dc["sbert"])

	dc["umap"] = vb.stringify(dc["umap"])

	dc.pop("feature_combination")

	vb(newview if newview else viewname).save(dc)
	# qs = learner.get_queryset()



# multilabel vs. multiclass
@click.group()
def validate():
	""" Validate your sklearn model using features and a label

	FEATURES : comma separated feature names / columns (will be combined).

	LABEL : the label the model will be validated on.
	"""
	pass

@click.group()
@click.argument("view1")
@click.argument("view2")
@click.pass_context
def append(ctx, view1, view2):
	"""Append a column to another view"""
	ctx.obj = (view1, view2)

@append.command()
@click.option("--newcolname", default=None, help="Should the column be renamed before being appended?")
@click.argument("col")
# TODO: add a window size argument
@click.pass_context
def column(ctx, newcolname, col):
	""" Append column from view1 to view 2 """
	view1, view2 = ctx.obj

	data1 = vb(view1).load()
	data2 = vb(view2).load()

	newcolname = newcolname if newcolname else col

	if newcolname in data2.keys():
		click.confirm('You are about to overwrite an initial column from view2. Continue?', abort=True)

	data2[newcolname] = data1[col]

	vb(view2).save(data2)

# add different sub entrypoints
entrypoint.add_command(join)
entrypoint.add_command(extract)
entrypoint.add_command(utils)
entrypoint.add_command(encoder)
entrypoint.add_command(train)
entrypoint.add_command(validate)
entrypoint.add_command(append)

if __name__ == "__main__":
	entrypoint()