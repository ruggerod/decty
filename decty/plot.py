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
