import pandas as pd
import numpy as np


class DecisionTreeAnalyser:
    def __init__(self, nodes):
        self.nodes = nodes

    @classmethod
    def from_spark(cls, tree):
        # prepare arrays to be filled with nodes' info
        n_nodes = tree.numNodes
        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        feature_id = np.zeros(shape=n_nodes, dtype=np.int64)
        split_thresh = np.zeros(shape=n_nodes, dtype=np.float)
        node_imp = np.zeros(shape=n_nodes, dtype=np.float)
        node_gain = np.zeros(shape=n_nodes, dtype=np.float)
        node_pred = np.zeros(shape=n_nodes, dtype=np.float)
        is_leaf = np.zeros(shape=n_nodes, dtype=bool)
        walking_index = np.zeros(shape=n_nodes, dtype=np.int64)
        is_left = np.zeros(shape=n_nodes, dtype=bool)
        is_right = np.zeros(shape=n_nodes, dtype=bool)
        left_child_index = -np.ones(shape=n_nodes, dtype=np.int64)
        right_child_index = -np.ones(shape=n_nodes, dtype=np.int64)
        parent_index = -np.ones(shape=n_nodes, dtype=np.int64)

        # set initial conditions for the while loop
        stack = [(tree._call_java('rootNode'), 0)]  # start with the root node id (0) and its depth (0)
        n = 0
        while len(stack) > 0:  # cycle over the nodes, starting from the root node
            # `pop` ensures each node is only visited once
            node, depth = stack.pop(0)
            if node.numDescendants() != 0:  # split node

                # stack children nodes and increase depth
                offset = len(stack)
                stack.append((node.leftChild(), depth + 1))
                stack.append((node.rightChild(), depth + 1))

                # fill info for this node and its children
                walking_index[n] = n
                left_child_index[n] = offset + n + 1
                right_child_index[n] = offset + n + 2
                is_left[offset + n + 1] = True
                is_right[offset + n + 1] = False
                is_left[offset + n + 2] = False
                is_right[offset + n + 2] = True
                parent_index[offset + n + 1] = n
                parent_index[offset + n + 2] = n
                feature_id[n] = node.split().featureIndex()
                split_thresh[n] = node.split().threshold()
                node_imp[n] = node.impurity()
                node_gain[n] = node.gain()
                node_pred[n] = -1
                is_leaf[n] = False
            else:  # leaf node
                # fill info for this node
                walking_index[n] = n
                feature_id[n] = -1
                split_thresh[n] = -1
                node_imp[n] = -1
                node_pred[n] = node.prediction()
                is_leaf[n] = True

            node_depth[n] = depth
            n = n + 1

        nodes = pd.DataFrame(
            list(zip(node_depth, feature_id, split_thresh, node_gain,
                     node_imp, node_pred, is_leaf, is_left, is_right, parent_index,
                     left_child_index, right_child_index)),
            columns=["depth", "feature_index", "threshold", "gain",
                     "impurity", "prediction", "is_leaf", "is_left", "is_right", "parent_index",
                     "left_child_index", "right_child_index"]
        )
        nodes.index.name = "node_id"
        return cls(nodes)

    def decision_path(self, index):
        path = []
        found_path = False
        parent_index = index
        while not found_path:
            if self.nodes.loc[parent_index, "parent_index"] == -1:
                found_path = True
                continue
            parent_index = self.nodes.loc[parent_index, "parent_index"]
            path.append(parent_index)
        path = ([index] + path)[::-1]
        return self.nodes.loc[path].copy()

    def subtree(self, index):
        def get_children(i):
            node = self.nodes.loc[i]
            if node["is_leaf"]:
                return []
            else:
                return self.nodes.loc[[node["left_child_index"], node["right_child_index"]]]

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
        return self.nodes.loc[subtree].copy()

    def decision_path_as_list(self, index):
        def decision_as_str(i, feat_mapping=None):
            # return the string with the last decision that led to the node
            if i == 0:
                return

            parent = self.nodes.loc[self.nodes.loc[i, "parent_index"]]
            feat_name = f'feature {parent["feature_index"]}' if feat_mapping is None else feat_mapping[
                parent["feature_index"]]
            symbol = '<=' if self.nodes.loc[i, "is_left"] else '>'
            return f'{feat_name} {symbol} {parent["threshold"]}'

        return [decision_as_str(i) for i in self.decision_path(index).index]

    def path_properties(self):
        # select leaves only
        mask = self.nodes["is_leaf"]
        df = self.nodes[mask].copy()

        def count_unique_features(index):
            """Count unique features that have been tested in the decision path that led to leaf n. `index`"""
            x = self.decision_path(index)["feature_index"]
            mask = x != -1
            return x[mask].nunique()

        def count_unique_predictions(index, level=1):
            """Count unique predictions obtained in the subtree that originates from parent distant `level` from leaf n. `index`"""
            # select grand parent of leaf based on `level`
            grand_parent = self.nodes.loc[index]
            for i in range(level):
                grand_parent = self.nodes.loc[grand_parent["parent_index"]]

            # get the subtree and count unique predictions
            subtree = self.subtree(grand_parent.name)
            return subtree.loc[subtree["is_leaf"], "prediction"].nunique()

        df['nunique_features'] = df.index.to_series().apply(count_unique_features)
        df['nunique_predictions'] = df.index.to_series().apply(count_unique_predictions)
        return df[['depth', "nunique_features", "nunique_predictions", "prediction"]]
