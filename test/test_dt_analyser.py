import pytest
from .sample import get_iris_data
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from decty import DecisionTreeAnalyser


class TestDecisionTreeAnalyser:
    @pytest.fixture(autouse=True)  # allows load spark fixture in the setup_method
    def setup_method(self, spark):
        """
        Load the iris dataset, fit with a spark DecisionTreeModel, and create a DecisionTreeAnalyser
        """
        data_pd = get_iris_data()
        data = spark.createDataFrame(data_pd)
        feat_cols = ["sepal_lenght", "sepal_width", "petal_lenght", "petal_width"]
        assembler = VectorAssembler(
            inputCols=feat_cols,
            outputCol="features")
        self.data = assembler.transform(data)
        dt = DecisionTreeClassifier(labelCol="class", featuresCol="features", maxDepth=5)
        model = dt.fit(self.data)
        self.dta = DecisionTreeAnalyser.from_spark(model)

    def test_reading(self):
        """Check whether the input was loaded as expected"""
        assert self.data.columns == ["sepal_lenght", "sepal_width", "petal_lenght",
                                     "petal_width", "class", "features"]
        assert self.data.count() == 112
        assert self.data.select("sepal_lenght").rdd.flatMap(lambda x: x).collect()[0] == 5.9

    def test_tree_properties(self):
        assert len(self.dta.nodes) == 13, "Total number of nodes should be 13"
        assert len(self.dta.path_properties()) == 7, "Total number of leaves should be 7"
        assert self.dta.decision_path_as_list(4) == [None, 'feature 2 > 2.35', 'feature 2 > 4.95'], \
            "Wrong path for node n.4"
