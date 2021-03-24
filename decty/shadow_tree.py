import numpy as np
import pandas as pd
from typing import List, Mapping

from pyspark.ml.classification import DecisionTreeClassificationModel

from dtreeviz.models.spark_decision_tree import ShadowSparkTree


class ShadowDectyTree(ShadowSparkTree):
    """Represents a shadow tree that can be visualised using the dtreeviz package."""
    def __init__(self, tree_model: DecisionTreeClassificationModel,
                 x_data,
                 y_data,
                 feature_names: List[str] = None,
                 target_name: str = None,
                 class_names: (List[str], Mapping[int, str]) = None):

        self.nodes_info = None  # lazy initialization
        super().__init__(tree_model, x_data, y_data, feature_names, target_name, class_names)

    def get_nodes_info(self):
        """Returns a frame representation of all the nodes with their features"""
        if self.nodes_info is not None:
            return self.nodes_info

        df = pd.DataFrame(
            list(zip(self.children_left, self.children_right)),
            columns=['left_child', 'right_child']
        )

        node_ids = df.index.to_series()
        df['feat_id'] = self.get_features()
        df['is_leaf'] = df['feat_id'] == -1
        df['parent_id'] = [-1] + [df[(df['left_child'] == i) | (df["right_child"] == i)].index[0] for i in node_ids[1:]]
        df['is_left'] = node_ids.isin(df['left_child'])
        df['is_right'] = node_ids.isin(df['right_child'])
        df['split_thresh'] = self.get_thresholds()
        df['impurity'] = node_ids.apply(self.get_node_criterion)
        df['prediction'] = node_ids.apply(lambda i: self.tree_nodes[i].prediction())
        # get number of samples in each node
        n_samples = node_ids.apply(self.get_node_nsamples_by_class)
        for n in self.classes():
            df[f'n_samples_in_{n}'] = n_samples.apply(lambda x: x[n])
        df['n_samples'] = node_ids.apply(self.get_node_nsamples)
        self.nodes_info = df
        return self.nodes_info

    def decision_path(self, index):
        """Returns a frame representation of the nodes that led to the node with id `index`"""
        path = []
        found_path = False
        parent_index = index
        while not found_path:
            if self.get_nodes_info().loc[parent_index, "parent_id"] == -1:
                found_path = True
                continue
            parent_index = self.get_nodes_info().loc[parent_index, "parent_id"]
            path.append(parent_index)
        path = ([index] + path)[::-1]
        return self.get_nodes_info().loc[path].copy()

    def subtree(self, index):
        """Returns a frame representation of the nodes that descend from the node with id `index`"""
        def get_children(i):
            node = self.get_nodes_info().loc[i]
            if node["is_leaf"]:
                return []
            else:
                return self.get_nodes_info().loc[[node["left_child"], node["right_child"]]]

        subtree = []
        children = get_children(index)
        while len(children) > 0:
            # select a child
            child = children.iloc[0]
            # save its name
            subtree.append(child.name)
            # get its children
            new_children = get_children(child.name)
            # drop the child
            children = children.drop(child.name)
            # append its children
            children = children.append(new_children)
        return self.get_nodes_info().loc[subtree].copy()

    def decision_path_as_list(self, index) -> List[str]:
        """Returns the decision path that lead to the node with id `index`"""
        def decision_as_str(i, feat_mapping=None):
            # return the string with the last decision that led to the node
            if i == 0:
                return

            parent = self.get_nodes_info().loc[self.get_nodes_info().loc[i, "parent_id"]]
            feat_name = f'feature {parent["feat_id"]}' if feat_mapping is None else feat_mapping[
                parent["feat_id"]]
            symbol = '<=' if self.get_nodes_info().loc[i, "is_left"] else '>'
            return f'{feat_name} {symbol} {parent["split_thresh"]}'

        return [decision_as_str(i) for i in self.decision_path(index).index]

    def path_properties(self):
        """Returns a data frame with relevant properties of each leaf"""
        # select leaves only
        mask = self.get_nodes_info()["is_leaf"]
        df = self.get_nodes_info()[mask].copy()

        def get_depth(index):
            """
            As a proxy of the depth, count the features that have been tested
            in the the decision path that led to leaf n. `index`
            """
            x = self.decision_path(index)["feat_id"]
            return len(x) - 1

        def count_unique_features(index):
            """
            Count unique features that have been tested
            in the decision path that led to leaf n. `index`
            """
            x = self.decision_path(index)["feat_id"]
            msk = x != -1
            return x[msk].nunique()

        def count_unique_predictions(index, level=1):
            """
            Count unique predictions obtained in the subtree that
            originates from parent distant `level` from leaf n. `index`
            """
            # select grand parent of leaf based on `level`
            grand_parent = self.get_nodes_info().loc[index]
            for i in range(level):
                grand_parent = self.get_nodes_info().loc[grand_parent["parent_id"]]

            # get the subtree and count unique predictions
            subtree = self.subtree(grand_parent.name)
            return subtree.loc[subtree["is_leaf"], "prediction"].nunique()

        df['depth'] = df.index.to_series().apply(get_depth)
        df['nunique_features'] = df.index.to_series().apply(count_unique_features)
        df['nunique_predictions'] = df.index.to_series().apply(count_unique_predictions)
        return df[["depth", "nunique_features", "nunique_predictions", "prediction"]]

    def predict_array(self, X: np.ndarray):
        """Rreturns an array with `prediction` and `leaf id` for each observation in `X`"""
        if len(X.shape) == 1:
            pred, path = self.predict(X)
            return np.array((pred, path[-1].id))
        elif len(X.shape) == 2:
            preds = []
            for row in X:
                pred, path = self.predict(row)
                preds.append((pred, path[-1].id))
            return np.array(preds)
        else:
            raise ValueError("X should have shape (x,) or (x, y)")

    def get_leaf_subsample_counts(self, X: np.ndarray, min_samples=0, max_samples=None):
        """Returns an array with `leaf id` and `sample counts` for each observation in`X`"""
        preds = self.predict_array(X)
        leaf_id = preds[:, 1]
        unique, counts = np.unique(leaf_id, return_counts=True)

        max_samples = max_samples if max_samples else max(counts)
        leaf_samples = [(i, n) for i, n in zip(unique, counts) if
                        min_samples <= n <= max_samples]
        x, y = zip(*leaf_samples)
        return np.array(x), np.array(y)
