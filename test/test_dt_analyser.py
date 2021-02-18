import pytest
from .sample import get_iris_data
from pyspark.ml.feature import VectorAssembler

pytestmark = pytest.mark.usefixtures("spark_context")


def test_reading(spark):
    data_pd = get_iris_data()
    data = spark.createDataFrame(data_pd)
    feat_cols = ["sepal_lenght", "sepal_width", "petal_lenght", "petal_width"]
    assembler = VectorAssembler(
        inputCols=feat_cols,
        outputCol="features")
    data = assembler.transform(data)
    assert data.columns == feat_cols + ["class", "features"]
    assert data.count() == 112
    assert data.select("sepal_lenght").rdd.flatMap(lambda x: x).collect()[0] == 5.9
