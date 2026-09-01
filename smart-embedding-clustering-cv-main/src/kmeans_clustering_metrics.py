import logging
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans # type: ignore[import-untyped]
from sklearn.metrics import silhouette_score # type: ignore[import-untyped]

from src.config import (
    RANDOM_STATE,
    FIGURE_SIZE,
    SAVE_DPI,
    SAVE_BBOX,
    CABINETS_EMBEDDINGS_PATH,
    CABINETS_METADATA_PATH,
    SAMPLE_SIZE,
)
from src.embeddings_io import load_embeddings
from src.output_utils import get_output_dir
from src.sampling import sample_embeddings

logger = logging.getLogger(__name__)

def _validate_calculation_inputs(
    embeddings: np.ndarray,
    k_values: range | Sequence[int],
) -> None:
    """
    Validates the inputs used to calculate clustering evaluation metrics.

    Args:
        embeddings (np.ndarray): Embedding array to evaluate.
        k_values (range | Sequence[int]): Collection of cluster values to evaluate.

    Raises:
        ValueError:
            If embeddings is not a 2D array.
        ValueError:
            If embeddings is empty.
        TypeError:
            If k_values is not a sequence of integers.
        ValueError:
            If k_values is empty.
        TypeError:
            If any value in k_values is not an integer.
        ValueError:
            If any k value is not positive.
        ValueError:
            If any k value is greater than the number of samples.

    Returns:
        None.
    """

    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D array")

    if len(embeddings) == 0:
        raise ValueError(
            "The embedding array is empty."
        )

    if not isinstance(k_values, Sequence) or isinstance(k_values, (str, bytes)):
        raise TypeError("k_values must be a sequence of integers")

    if len(k_values) == 0:
        raise ValueError(
            "k_values must contain at least one value."
        )

    for k in k_values:
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
            raise TypeError("Each k value must be an integer")

        if k <= 0:
            raise ValueError("Each k value must be positive")

        if k > len(embeddings):
            raise ValueError("Each k value must be less than or equal to the number of samples")

def _validate_plot_inputs(
    k_values: range | Sequence[int],
    values: Sequence[float],
    metric_name: str,
) -> None:
    """
    Validates the inputs used to plot clustering evaluation metrics.

    Args:
        k_values (range | Sequence[int]): Collection of evaluated cluster values.
        values (Sequence[float]): Metric values corresponding to each cluster value.
        metric_name (str): Name of the evaluation metric.

    Raises:
        ValueError:
            If k_values is empty.
        ValueError:
            If there are no metric values to plot.
        ValueError:
            If k_values and values do not have the same length.

    Returns:
        None.
    """

    if len(k_values) == 0:
        raise ValueError(
            "k_values must contain at least one value."
        )

    if len(values) == 0:
        raise ValueError(
            f"There are no {metric_name} values to plot."
        )

    if len(k_values) != len(values):
        raise ValueError(
            "k_values and metric values must have the same length."
        )


def get_cluster_label(
    k_values: range | Sequence[int],
) -> str:
    """
    Creates a label describing the evaluated cluster range.

    Args:
        k_values (range | Sequence[int]): Collection of evaluated cluster values.

    Returns:
        str: Label representing the cluster range.
    """
    values = sorted(int(k) for k in k_values)
    return f"k{values[0]}-{values[-1]}"


def calculate_elbow(
    embeddings: np.ndarray,
    k_values: range | Sequence[int],
) -> list[float]:
    """
    Calculates the inertia for different values of k using the Elbow method.

    Args:
        embeddings (np.ndarray): Array containing the image embeddings.
        k_values (range | Sequence[int]): Collection of k values to evaluate.

    Raises:
        ValueError:
            If embeddings is not a valid 2D array.
        ValueError:
            If k_values contains invalid values.
        TypeError:
            If k_values is not a valid sequence of integers.

    Returns:
        list[float]: Inertia value computed for each k.
    """
    _validate_calculation_inputs(
        embeddings,
        k_values,
    )

    inertias = []

    logger.info("Calculating Elbow...")

    for k in k_values:

        logger.info(f"  k = {k}")

        kmeans = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
        )

        kmeans.fit(embeddings)

        inertias.append(kmeans.inertia_)

    logger.info("Elbow calculation completed.\n")

    return inertias


def plot_elbow(
    k_values: range | Sequence[int],
    inertias: list[float],
    title: str | None = None,
) -> None:
    """
    Plots the Elbow Method graph.

    Args:
        k_values (range | Sequence[int]): Collection of k values evaluated.
        inertias (list[float]): Inertia values corresponding to each k.
        title (str | None): Plot title. If None, a default title is used.

    Raises:
        ValueError:
            If k_values is empty.
        ValueError:
            If there are no inertia values to plot.
        ValueError:
            If k_values and inertias do not have the same length.

    Returns:
        None: The plot is saved to disk.
    """
    _validate_plot_inputs(
        k_values,
        inertias,
        "inertia",
    )
    logger.info("Showing Elbow Method graph...")

    plt.figure(figsize=FIGURE_SIZE)

    plt.plot(
        k_values,
        inertias,
        marker="o",
    )

    cluster_label = get_cluster_label(k_values)
    plt.title(
    title or f"Elbow Method ({cluster_label})"
    )
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")

    plt.grid(True)

    output_dir = get_output_dir()
    plt.savefig(
        output_dir / f"elbow_method_{cluster_label}.png",
        dpi=SAVE_DPI,
        bbox_inches=SAVE_BBOX,
    )
    plt.close()

    logger.info(f"Elbow Method graph saved to {output_dir}.\n")


def calculate_silhouette(
    embeddings: np.ndarray,
    k_values: range | Sequence[int],
) -> list[float]:
    """
    Calculates the Silhouette Score for different values of k.

    Args:
        embeddings (np.ndarray): Array containing the image embeddings.
        k_values (range | Sequence[int]): Collection of k values to evaluate.

    Raises:
        ValueError:
            If embeddings is not a valid 2D array.
        ValueError:
            If k_values contains invalid values.
        TypeError:
            If k_values is not a valid sequence of integers.

    Returns:
        list[float]: Silhouette Score computed for each k.
    """
    _validate_calculation_inputs(
        embeddings,
        k_values,
    )

    scores = []

    logger.info("Calculating Silhouette Score...")

    for k in k_values:

        logger.info(f"  k = {k}")

        kmeans = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
        )

        clusters = kmeans.fit_predict(
            embeddings,
        )

        score = silhouette_score(
            embeddings,
            clusters,
        )

        scores.append(score)

    logger.info("Silhouette calculation completed.\n")

    return scores


def plot_silhouette(
    k_values: range | Sequence[int],
    scores: list[float],
    title: str | None = None,
) -> None:
    """
    Plots the Silhouette Score graph.

    Args:
        k_values (range | Sequence[int]): Collection of k values evaluated.
        scores (list[float]): Silhouette Score values corresponding to each k.
        title (str | None): Plot title. If None, a default title is used.

    Raises:
        ValueError:
            If k_values is empty.
        ValueError:
            If there are no silhouette score values to plot.
        ValueError:
            If k_values and scores do not have the same length.

    Returns:
        None: The plot is saved to disk.
    """
    _validate_plot_inputs(
        k_values,
        scores,
        "silhouette score",
    )

    logger.info("Showing Silhouette Score graph...")

    plt.figure(figsize=FIGURE_SIZE)

    plt.plot(
        k_values,
        scores,
        marker="o",
    )

    cluster_label = get_cluster_label(k_values)

    plt.title(
        title or f"Silhouette Score ({cluster_label})"
    )
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette Score")

    plt.grid(True)

    output_dir = get_output_dir()
    plt.savefig(
        output_dir / f"silhouette_score_{cluster_label}.png",
        dpi=SAVE_DPI,
        bbox_inches=SAVE_BBOX,
    )
    plt.close()

    logger.info(f"Silhouette Score graph saved to {output_dir}.\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    logger.info("Loading embeddings...")

    # Fruits
    # embeddings, metadata = load_embeddings(
    #     FRUITS_EMBEDDINGS_PATH,
    #     FRUITS_METADATA_PATH,
    # )

    # Cabinets
    embeddings, metadata = load_embeddings(
        CABINETS_EMBEDDINGS_PATH,
        CABINETS_METADATA_PATH,
    )

    USE_SAMPLE = False

    if USE_SAMPLE:
        # Use stratified sampling if label distribution must be preserved.
        embeddings, metadata = sample_embeddings(
            embeddings,
            metadata,
            SAMPLE_SIZE,
        )

    logger.info("Embeddings loaded.")

    k_values = range(2, 21)

    inertias = calculate_elbow(
        embeddings,
        k_values,
    )

    DATASET_NAME = "Cabinets"

    plot_elbow(
        k_values,
        inertias,
        title=f"Elbow Method ({DATASET_NAME})"
    )

    scores = calculate_silhouette(
        embeddings,
        k_values,
    )

    plot_silhouette(
        k_values,
        scores,
        title=f"Silhouette Score ({DATASET_NAME})"
    )

    logger.info("Calculation completed successfully.")
