from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


def get_iris_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, _, y_train, _ = train_test_split(X, y, random_state=0)
    return pd.DataFrame(
        np.concatenate((X_train, y_train.reshape(len(y_train), 1)), axis=1),
        columns=[
            "sepal_lenght",
            "sepal_width",
            "petal_lenght",
            "petal_width",
            "class"
        ]
    )
