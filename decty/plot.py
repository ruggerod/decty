import numpy as np

from dtreeviz.colors import adjust_colors

import matplotlib.pyplot as plt

from decty import ShadowDectyTree


def viz_leaf_subsamples(
        tree_model,
        X: np.ndarray,
        figsize: tuple = (10, 5),
        display_type: str = "plot",
        colors: dict = None,
        fontsize: int = 14,
        fontname: str = "Arial",
        grid: bool = False,
        bins: int = 10,
        min_samples: int = 0,
        max_samples: int = None
):
    """Visualize the number of data samples in the array of observations `X` from each leaf.

        Interpreting leaf samples can help us to see how the data is spread over the tree:
        - if we have a leaf with many samples and a good impurity, it means that we can be pretty confident
        on its prediction.
        - if we have a leaf with few samples and a good impurity, we cannot be very confident on its predicion and
        it could be a sign of overfitting.
        - by visualizing leaf samples, we can easily discover important leaves . Using describe_node_sample() function we
        can take all its samples and discover common patterns between leaf samples.
        - if the tree contains a lot of leaves and we want a general overview about leaves samples, we can use the
        parameter display_type='hist' to display the histogram of leaf samples.

        There is the option to filter the leaves with samples between 'min_samples' and 'max_samples'. This is helpful
        especially when you want to investigate leaves with number of samples from a specific range.


        We can call this function by using shadow tree
            ex. viz_leaf_samples(shadow_dtree)
            - we need to initialize shadow_tree before this call
                - ex. shadow_dtree = ShadowSKDTree(tree_model, dataset[features], features)
            - the main advantage is that we can use the shadow_tree for other visualisations methods as well

        This method contains three types of visualizations:
        - If display_type = 'plot' it will show leaf samples using a plot.
        - If display_type = 'text' it will show leaf samples as plain text. This method is preferred if number
        of leaves is very large and the plot become very big and hard to interpret.
        - If display_type = 'hist' it will show leaf sample histogram. Useful when you want to easily see the general
        distribution of leaf samples.

        :param tree_model: decty.ShadowDectyTree
            The dtreeviz shadow tree model to interpret
        :param X: np.ndarray
            The dataset based on which we want to make this visualisation.
        :param figsize: tuple of int
            The plot size
        :param display_type: str, optional
           'plot', 'text'. 'hist'
        :param colors: dict
            The set of colors used for plotting
        :param fontsize: int
            Plot labels font size
        :param fontname: str
            Plot labels font name
        :param grid: bool
            True if we want to display the grid lines on the visualization
        :param bins: int
            Number of histogram bins
        :param min_samples: int
            Min number of samples for a leaf
        :param max_samples: int
            Max number of samples for a leaf
        """
    if isinstance(tree_model, ShadowDectyTree):
        shadow_tree = tree_model
    else:
        raise ValueError("Function implemented only for ShadowDectyTree")

    leaf_id, leaf_samples = shadow_tree.get_leaf_subsample_counts(X, min_samples, max_samples)

    if display_type == "plot":
        colors = adjust_colors(colors)

        fig, ax = plt.subplots(figsize=figsize)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(.3)
        ax.spines['bottom'].set_linewidth(.3)
        ax.set_xticks(range(0, len(leaf_id)))
        ax.set_xticklabels(leaf_id)
        barcontainers = ax.bar(range(0, len(leaf_id)), leaf_samples, color=colors["hist_bar"], lw=.3, align='center',
                               width=1)
        for rect in barcontainers.patches:
            rect.set_linewidth(.5)
            rect.set_edgecolor(colors['rect_edge'])
        ax.set_xlabel("leaf ids", fontsize=fontsize, fontname=fontname, color=colors['axis_label'])
        ax.set_ylabel("samples count", fontsize=fontsize, fontname=fontname, color=colors['axis_label'])
        ax.grid(b=grid)
    elif display_type == "text":
        for leaf, samples in zip(leaf_id, leaf_samples):
            print(f"leaf {leaf} has {samples} samples")
    elif display_type == "hist":
        colors = adjust_colors(colors)

        fig, ax = plt.subplots(figsize=figsize)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(.3)
        ax.spines['bottom'].set_linewidth(.3)
        n, bins, patches = ax.hist(leaf_samples, bins=bins, color=colors["hist_bar"])
        for rect in patches:
            rect.set_linewidth(.5)
            rect.set_edgecolor(colors['rect_edge'])
        ax.set_xlabel("leaf sample", fontsize=fontsize, fontname=fontname, color=colors['axis_label'])
        ax.set_ylabel("leaf count", fontsize=fontsize, fontname=fontname, color=colors['axis_label'])
        ax.grid(b=grid)
