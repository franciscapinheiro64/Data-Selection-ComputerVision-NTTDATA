import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap  # type: ignore[import-untyped]

from src.config import (
    CABINETS_EMBEDDINGS_PATH,
    CABINETS_HDBSCAN_METADATA_PATH,
    FIGURE_SIZE,
    LEGEND_FONT_SIZE,
    RANDOM_STATE,
    SAVE_BBOX,
    SAVE_DPI,
    SCATTER_POINT_SIZE,
    UMAP_N_COMPONENTS,
    SAMPLE_SIZE,
)
from src.sampling import sample_embeddings_stratified  # type: ignore[import-untyped]
from src.embeddings_io import load_embeddings  # type: ignore[import-untyped]
from src.output_utils import get_output_dir

def reduce_dimensions(
    embeddings: np.ndarray,
    n_components: int = UMAP_N_COMPONENTS,
) -> np.ndarray:
    """
    Reduces embeddings to a lower-dimensional representation using UMAP.

    Args:
        embeddings (np.ndarray): Array containing the image embeddings.
        n_components (int): Number of dimensions to reduce to.

    Returns:
        np.ndarray: Two-dimensional representation of the embeddings.

    Raises:
        ValueError:
            If embeddings is not a 2D array.
        ValueError:
            If embeddings contains fewer than two samples.
        ValueError:
            If n_components is not a positive integer.
        ValueError:
            If embeddings does not contain any features.
    """

    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D array")

    if len(embeddings) < 2:
        raise ValueError("embeddings must contain at least two samples")

    if not isinstance(n_components, int) or isinstance(n_components, bool) or n_components <= 0:
        raise ValueError("n_components must be a positive integer")

    if embeddings.shape[1] == 0:
        raise ValueError(
            "embeddings must contain at least one feature."
        )

    reducer = umap.UMAP(
        n_components=n_components,
        random_state=RANDOM_STATE,
    )

    return reducer.fit_transform(embeddings)


def plot_embeddings(
    embeddings_2d: np.ndarray,
    groups: pd.Series | np.ndarray,
    title: str,
    output_name: str,
) -> None:
    """
    Plots a two-dimensional embeddings to a file.

    Args:
        embeddings_2d (np.ndarray): Two-dimensional embeddings produced by UMAP.
        groups (pd.Series | np.ndarray): Labels or cluster assignments used to color the points.
        title (str): Title of the plot.
        output_name (str): Name of the output file without the extension.

    Returns:
        None: The plot is saved to disk.

    Raises:
        TypeError:
            If embeddings_2d is not a NumPy array.
        ValueError:
            If embeddings_2d is not a 2D array.
        ValueError:
            If the number of embeddings does not match the number of groups.
    """
    if not isinstance(embeddings_2d, np.ndarray):
        raise TypeError("embeddings_2d must be a numpy array")

    if embeddings_2d.ndim != 2:
        raise ValueError("embeddings_2d must be a 2D array")

    if len(embeddings_2d) != len(groups):
        raise ValueError(
            "The number of embeddings must match the number of groups."
        )

    plt.figure(figsize=FIGURE_SIZE)

    unique_groups = np.unique(groups)

    for i, group in enumerate(unique_groups):
        mask = groups == group

        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            s=SCATTER_POINT_SIZE,
            color = plt.cm.tab20(i % 20),  # Use a colormap for better color distinction
            label=group,
        )

    plt.title(title)

    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    if len(unique_groups) > 1:
        plt.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=LEGEND_FONT_SIZE,
        )

    output_dir = get_output_dir()

    plt.tight_layout()

    plt.savefig(
        output_dir / f"{output_name}.png",
        dpi=SAVE_DPI,
        bbox_inches=SAVE_BBOX,
    )

    plt.close()


def plot_selected_embeddings(
    embeddings_2d: np.ndarray,
    labels: pd.Series | np.ndarray,
    selected_labels: list[str] | np.ndarray,
    title: str,
    output_name: str,
) -> None:
    """
    Plots a selected subset of embedding classes.

    Args:
        embeddings_2d (np.ndarray): Two-dimensional embeddings produced by UMAP.
        labels (pd.Series | np.ndarray): Labels associated with each embedding.
        selected_labels (list[str] | np.ndarray): Labels to include in the visualization.
        title (str): Title of the plot.
        output_name (str): Name of the output file without the extension.

    Returns:
        None: The plot is saved to disk.

    Raises:
        TypeError:
            If embeddings_2d is not a NumPy array.
        ValueError:
            If embeddings_2d is not a 2D array.
        ValueError:
            If selected_labels is empty.
        ValueError:
            If the number of embeddings does not match the number of labels.
        ValueError:
            If none of the selected labels are found.
    """

    if not isinstance(embeddings_2d, np.ndarray):
        raise TypeError("embeddings_2d must be a numpy array")

    if embeddings_2d.ndim != 2:
        raise ValueError("embeddings_2d must be a 2D array")

    if len(selected_labels) == 0:
        raise ValueError("selected_labels must not be empty")

    if len(embeddings_2d) != len(labels):
        raise ValueError("The number of embeddings must match the number of labels.")

    labels = pd.Series(labels)

    mask = labels.isin(selected_labels)

    embeddings_subset = embeddings_2d[mask]

    if len(embeddings_subset) == 0:
        raise ValueError(
            "No embeddings found for the selected labels."
        )
    labels_subset = labels[mask]

    plt.figure(figsize=FIGURE_SIZE)

    unique_labels = np.unique(labels_subset)

    for label in unique_labels:
        class_mask = labels_subset == label

        plt.scatter(
            embeddings_subset[class_mask, 0],
            embeddings_subset[class_mask, 1],
            s=SCATTER_POINT_SIZE,
            label=label,
        )


    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title(title)
    plt.legend()


    output_dir = get_output_dir()

    plt.tight_layout()

    plt.savefig(
        output_dir / f"{output_name}.png",
        dpi=SAVE_DPI,
        bbox_inches=SAVE_BBOX,
    )

    plt.close()


if __name__ == "__main__":

    embeddings, metadata = load_embeddings(
        CABINETS_EMBEDDINGS_PATH,
        # CABINETS_METADATA_PATH,
        CABINETS_HDBSCAN_METADATA_PATH,
    )


    #mask = metadata["cluster"] == CABINET_CLUSTER
    #embeddings = embeddings[mask]
    #metadata = metadata[mask].copy()


    USE_SAMPLE = False

    if USE_SAMPLE:
        embeddings, metadata = sample_embeddings_stratified(
            embeddings,
            metadata,
            SAMPLE_SIZE,
        )

    embeddings_2d = reduce_dimensions(embeddings)

    # Assign all points to the same group to visualize the overall embedding distribution
    # groups = np.zeros(len(metadata))
    # groups = metadata["subcluster"]
    groups = metadata["cluster"]


    plot_embeddings(
    # plot_selected_embeddings(
        embeddings_2d,

        groups,

        # metadata["label"].unique()[:7],

        "Visualization of HDBSCAN Clusters with UMAP",

        output_name="umap_hdbscan_cabinets_clusters",
    )
