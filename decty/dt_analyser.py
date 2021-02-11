from pyspark.sql import functions as F
import os
import numpy as np
import pandas as pd
import re
from pyspark.ml.classification import DecisionTreeClassificationModel
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql import SparkSession


class DecisionTreeAnalyser:
    def __init__(self, model, feat_mapping):
        self.model = model
        self.feat_mapping = feat_mapping
        self._set_nodes()

    def _set_nodes(self):
        # get tree from string
        decisions_report = self.model.toDebugString.split('\n')[1:]
        tree_as_list = [dec for dec in decisions_report][:-1]
        self._check_tree_consistency(tree_as_list)

        # sort tree as list to assign node ids
        indentations = list(map(self.indentation_level, tree_as_list))
        index = list(range(len(tree_as_list)))
        df = pd.DataFrame(list(zip(tree_as_list, indentations, index)), columns=['tree', 'level', 'index'])
        df.sort_values(by=['level', 'index'], inplace=True)
        tree_as_list = df['tree'].tolist()
        indentations = df['level'].tolist()
        index = df['index'].tolist()

        # create pandas dataframe with node info from tree
        # assign node ids and leaves' predictions
        node_id = 0
        ids = []
        preds = []
        for n, (row, level) in enumerate(zip(tree_as_list, indentations)):
            if row.strip().startswith('If'):
                ids.append(node_id)
                preds.append(None)
                node_id += 1
            elif row.strip().startswith('Else'):
                for j, l in enumerate(indentations[:n]):
                    if l == level:
                        try:
                            ids[n] = ids[j]
                        except IndexError:
                            ids.append(ids[j])
                            preds.append(None)
            else:  # this is a leaf
                ids.append(node_id)
                preds.append(float(row.split('Predict:')[1]))
                node_id += 1

        # sort back again
        df = pd.DataFrame(list(zip(tree_as_list, ids, indentations, index, preds)),
                          columns=['tree', 'ids', 'level', 'index', 'preds'])
        df.sort_values(by='index', inplace=True)
        tree_as_list = df['tree'].tolist()
        indentations = df['level'].tolist()
        ids = df['ids'].tolist()
        preds = df['preds'].tolist()

        # assign children of nodes
        children = []
        for k, l1 in enumerate(indentations):
            found_left = False
            for j, l2 in enumerate(indentations[k:]):
                if l2 == (l1 + 1):
                    children.append(ids[k + j])
                    found_left = True
                    break
            if not found_left:
                children.append(-1)

        # create data frame
        df = pd.DataFrame(
            {'row': tree_as_list, 'level': indentations, 'node_id': ids, 'children': children, 'leaf_pred': preds})
        # small correction to compensate the for-loop used to calculate children
        df.loc[df['row'].str.strip().str.startswith('Predict'), 'children'] = -1

        # calculate left and right children of nodes
        df.sort_values(['node_id', 'level'], inplace=True)
        df['l_child'] = df.groupby('node_id')['children'].transform("first")
        df['r_child'] = df.groupby('node_id')['children'].transform("last")
        df.drop('children', inplace=True, axis=1)

        # separate if...else expression
        df['if'] = df.groupby('node_id')['row'].transform("first")
        df['else'] = df.groupby('node_id')['row'].transform("last")
        df.drop('row', inplace=True, axis=1)

        # reset index to be a unique node id
        df = df.set_index('node_id')
        df = df[~df.index.duplicated(keep='first')]

        # label leaves
        df['leaf'] = False
        df.loc[df['if'].str.strip().str.startswith('Predict'), 'leaf'] = True

        # calculate the id of the feature used as condition for the node
        df['feature_id'] = -1
        mask = ~df['leaf']
        df.loc[mask, 'feature_id'] = df.loc[mask, 'if'].str.extract(r'(\d+)')[0].astype(int)

        self.nodes = df

        self.leaves = self.nodes[self.nodes['leaf']].drop(['leaf', 'feature_id'], axis=1)
        self._set_leaves_paths()
        self._leaves_analysis()

    def _set_leaves_paths(self, max_iterations=100):
        for i, leaf in self.leaves.iterrows():
            path = []
            found_path = False
            counter = 0
            parent = self.nodes[self.nodes.index == leaf.name].copy()
            while not found_path:
                p1 = self.nodes[self.nodes["l_child"] == parent.index[0]].copy()
                p2 = self.nodes[self.nodes["r_child"] == parent.index[0]].copy()
                if len(p2) > len(p1):
                    parent = p2
                else:
                    parent = p1
                path.append(parent.iloc[0])

                if parent.index[0] == 0:
                    found_path = True

                counter += 1
                if counter == max_iterations:
                    print("max iterations reached")
                    break
            path = path[::-1]
            self.leaves.loc[leaf.name, 'parent_nodes'] = '//'.join([str(node.name) for node in path])
            self.leaves.loc[leaf.name, 'parent_id'] = path[-1].name

            conditions = []
            for n, node in enumerate(path[:-1]):
                if path[n + 1].name == node["l_child"]:
                    conditions.append(self.statement_to_query(node["if"]))
                elif path[n + 1].name == node["r_child"]:
                    conditions.append(self.statement_to_query(node["else"]))
            self.leaves.loc[leaf.name, 'decision_path'] = '//'.join(conditions)

    def _leaves_analysis(self):
        """Leaf analysis:
        n_dn_features : number of dinamo features used for the decision
        heterogeneity : number of unique features used for the decision
        importance : for all the features used for the decision, sum of the feat. importance coefficients
        polarity : absolute difference of the prediction of sibling leaves
        """
        imps = self.model.featureImportances
        for i, leaf in self.leaves.iterrows():
            node_ids = list(map(int, leaf["parent_nodes"].split('//')))
            feat_ids = self.nodes.loc[node_ids, 'feature_id']
            self.leaves.loc[i, "n_dn_features"] = sum([1 for i in feat_ids if "_dn_" in self.feat_mapping[i]])
            self.leaves.loc[i, "heterogeneity"] = feat_ids.nunique()
            self.leaves.loc[i, "depth"] = len(node_ids)
            self.leaves.loc[i, "importance"] = sum([imps[i] for i in feat_ids.tolist()])
            siblings = self.leaves[self.leaves["parent_nodes"] == leaf["parent_nodes"]]
            if len(siblings) == 2:
                self.leaves.loc[i, "polarity"] = abs(siblings["leaf_pred"].iloc[0] - siblings["leaf_pred"].iloc[1])

    def get_leaf_counts(self, master, ids=None):
        if ids is None:
            ids = list(self.leaves.index)
        mask = self.leaves.index.isin(ids)
        relevant_nodes = [int(n) for n in set('//'.join(self.leaves.loc[mask, 'parent_nodes'].tolist()).split('//'))]

        # create sub data column with subset of data based on the node decision
        self.nodes.loc[0, 'sub_df'] = master
        for i, node in self.nodes.iterrows():
            if i in relevant_nodes:
                if pd.isna(self.nodes.loc[node['l_child'], 'sub_df']):
                    cond_if = self.statement_to_query(node['if'])
                    cond_else = self.statement_to_query(node['else'])
                    self.nodes.loc[node['l_child'], 'sub_df'] = self.nodes.loc[i, 'sub_df'].filter(cond_if)
                    self.nodes.loc[node['r_child'], 'sub_df'] = self.nodes.loc[i, 'sub_df'].filter(cond_else)

        leaf_counts = {}
        for i, leaf in self.leaves.iterrows():
            if i in ids:
                leaf_counts[i] = self.nodes.loc[i, 'sub_df'].count()
        return leaf_counts

    def get_leaf_precision(self, master, label, ids=None):
        if ids is None:
            ids = list(self.leaves.index)
        mask = self.leaves.index.isin(ids)
        relevant_nodes = [int(n) for n in set('//'.join(self.leaves.loc[mask, 'parent_nodes'].tolist()).split('//'))]

        # create sub data column with subset of data based on the node decision
        self.nodes.loc[0, 'sub_df'] = master
        for i, node in self.nodes.iterrows():
            if i in relevant_nodes:
                if pd.isna(self.nodes.loc[node['l_child'], 'sub_df']):
                    cond_if = self.statement_to_query(node['if'])
                    cond_else = self.statement_to_query(node['else'])
                    self.nodes.loc[node['l_child'], 'sub_df'] = self.nodes.loc[i, 'sub_df'].filter(cond_if)
                    self.nodes.loc[node['r_child'], 'sub_df'] = self.nodes.loc[i, 'sub_df'].filter(cond_else)

        leaf_precision = {}
        for i, leaf in self.leaves.iterrows():
            if i in ids:
                df = self.nodes.loc[i, 'sub_df']
                metrics = MulticlassMetrics(df.select("label", "prediction").rdd.map(tuple))
                try:
                    leaf_precision[i] = metrics.precision(label)
                except:
                    leaf_precision[i] = None

        return leaf_precision

    def statement_to_query(self, st):
        """ map if...else condition to a sql statement """
        feat = re.findall(r'feature \d+', st)[0]
        feat_name = self.feat_mapping[int(feat.replace('feature ', ''))]
        if 'If' in st:
            cond = st.strip().replace('If (', '').replace(')', '')
            cond = re.sub(r'feature \d+', feat_name, cond)
        elif 'Else' in st:
            cond = st.strip().replace('Else (', '').replace(')', '')
            cond = re.sub(r'feature \d+', feat_name, cond)
        return cond

    @staticmethod
    def indentation_level(st, offset=2):
        """Return leading spaces in a string"""
        return len(st) - len(st.lstrip()) - offset

    def _check_tree_consistency(self, tree):
        # check if in the tree ifs are as many as elses
        n_ifs = 0
        n_els = 0
        for row in tree:
            if row.strip().startswith('If'):
                n_ifs += 1
            elif row.strip().startswith('Else'):
                n_els += 1
            elif row.strip().startswith('Predict'):
                pass
            else:
                raise ValueError(row)
        assert (n_ifs == n_els)

    @property
    def max_depth(self):
        return self.nodes['level'].max()

    @property
    def num_features(self):
        return self.model.numFeatures