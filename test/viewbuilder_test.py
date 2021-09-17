import unittest
unittest.TestLoader.sortTestMethodsUsing = None
from viewbuilder.viewbuilder import ViewBuilder as vb

class ViewBuilderTest(unittest.TestCase):
	def setUp(self):
		self.nested_list = [[0.555, 0.26, 5], [1, 2, 3]]
		self.string_nested_list = ['[0.555, 0.26, 5]', '[1, 2, 3]']
		self.test_dict = {'label': [1,1,2,2], 'text': ["Hallo", "Welt", "Hello", "World"]}

	def test_save_and_load(self):
		vb("helloworld").save(self.test_dict)
		vb("helloworld").save(self.test_dict, "views/myview.csv")
		a = vb("helloworld").load()
		b = vb("helloworld").load("views/myview.csv")
		assert self.test_dict == a and self.test_dict == b

	def test_aggregate_text_on_label(self):
		test_dict = {'label': [1,1,2,2], 'text': ["Hallo", "Welt", "Hello", "World"]}
		ret = vb.aggregate_text_on_label(test_dict, 'label', 'text')
		assert ret == {'label': [1,2], 'text': ["Hallo Welt", "Hello World"]}

	def test_stringify(self):
		ret = vb.stringify(self.nested_list)
		assert ret == self.string_nested_list

	def test_unstringify(self):
		ret = vb.unstringify(self.string_nested_list)
		assert ret == self.nested_list

	def test_join_on_inner(self):
		dict_a = {'x': ["A", "A", "C"], 'z': [9,4,2]}
		dict_b = {'x': ["A", "B", "C"], 'y': [1,2,3]}
		ret = vb.join_on(dict_a, dict_b, "x")
		assert ret == {'x': ["A", "A", "C"], 'y': [1,1,3], 'z':[9,4,2]}

	def test_outer_left_join(self):
		dict_a = {'x': ["A", "A", "C"], 'z': [9,4,2]}
		dict_b = {'x': ["A", "B", "C"], 'y': [1,2,3]}
		ret = vb.join_on(dict_a, dict_b, "x", join_type="outer")
		assert ret == {'x': ["A", "A", "C", "B"], 'y': [1,1,3,2], 'z':[9,4,2,None]}
		# assert ret["z"][ret["x"].index("B")] == None

	def test_filter_select(self):
		dict_a = {'x': ["A", "A", "C"], 'z': [9,4,2]}
		r1, r2 = vb.filter(dict_a, lambda x: x['x'] == "A")
		assert r1 == {0: 0, 1:1} and r2 == {'x': ["A", "A"], "z": [9, 4]}

	def test_combine(self):
		x = [[1, 2], [1,3]]
		y = [1,2]
		z = [3,4]
		r1 = vb.combine(x, y)
		r2 = vb.combine(y, z)
		assert r1 == [[1,2,1], [1,3,2]] and r2 == [[1,3], [2,4]]

	def test_combine_single_attribute(self):
		x = [[1, 2], [1,3]]
		r1 = vb.combine(x)
		assert r1 == [[1, 2], [1,3]]

	def test_change_orientation_to_list(self):
		records = [{"a": 1, "b": 1}, {"a": 2, "b": 2}]
		res = vb.change_orientation(records, to="list")
		assert res == {"a": [1, 2], "b": [1, 2]}

	def test_change_orientation_to_records(self):
		ls = {"a": [1, 2], "b": [1, 2]}
		res = vb.change_orientation(ls, to="records")
		assert res == [{"a": 1, "b": 1}, {"a": 2, "b": 2}]

	def test_size_of_groups(self):
		gs = {"A": [1,1,1,4,4,3,3,3,3]}
		solution = {1: 3, 4: 2, 3: 4}
		res = vb.size_of_groups(gs, on="A")
		for x, y in zip(res["A"], res["count"]):
			assert solution[x] == y

	def test_aggregate_text_on_columns_no_delim(self):
		tx = {"A": ["Hello1", "Hello2", "Hello3"], "B": ["World1", "World2", "World3"], "C": ["!1", "!2", "!3"]}
		res = vb.aggregate_text_on_columns(tx, cols=["A", "B", "C"])
		assert res == ["Hello1World1!1", "Hello2World2!2", "Hello3World3!3"]

	def test_aggregate_text_on_columns_delim(self):
		tx = {"A": ["Hello1", "Hello2", "Hello3"], "B": ["World1", "World2", "World3"], "C": ["!1", "!2", "!3"]}
		res = vb.aggregate_text_on_columns(tx, cols=["A", "B", "C"], delim=". ")
		assert res == ["Hello1. World1. !1", "Hello2. World2. !2", "Hello3. World3. !3"]

