import logging
import hdbscan # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans # type: ignore[import-untyped]
from sklearn.metrics import ( # type: ignore[import-untyped]
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.metrics.cluster import contingency_matrix # type: ignore[import-untyped]

from src.embeddings_io import load_embeddings # type: ignore[import-untyped]

from src.config import (
    CABINETS_CLUSTERS_METADATA_PATH,
    CABINETS_EMBEDDINGS_PATH,
    DEFAULT_NUMBER_OF_CLUSTERS,
    MIN_CLUSTER_SIZE,
    MIN_SAMPLES,
    RANDOM_STATE,
)

logger = logging.getLogger(__name__)

def _validate_labels(
    true_labels: pd.Series | np.ndarray,
    predicted_labels: np.ndarray,
) -> None:
    """
    Validates the ground-truth and predicted labels.

    Args:
        true_labels (pd.Series | np.ndarray):
            Ground-truth labels.
        predicted_labels (np.ndarray):
            Cluster labels assigned by the clustering algorithm.

    Raises:
        ValueError:
            If the label arrays are empty.
        ValueError:
            If the label arrays do not have the same length.

    Returns:
        None.
    """

    if len(true_labels) == 0:
        raise ValueError(
            "Labels must not be empty."
        )

    if len(true_labels) != len(predicted_labels):
        raise ValueError(
            "true_labels and predicted_labels must have the same length."
        )

def purity_score(
    true_labels: pd.Series | np.ndarray,
    predicted_labels: np.ndarray,
) -> float:
    """
    Calculates the purity score of a clustering result.

    Purity measures the extent to which each cluster contains samples
    belonging to a single ground-truth class.

    Args:
        true_labels (pd.Series | np.ndarray):
            Ground-truth labels.
        predicted_labels (np.ndarray):
            Cluster labels assigned by the clustering algorithm.

    Returns:
        float:
            Purity score in the range [0, 1], where higher values indicate
            better agreement between clusters and ground-truth labels.
    """

    _validate_labels(
        true_labels,
        predicted_labels,
    )

    contingency = contingency_matrix(
        true_labels,
        predicted_labels,
    )

    return (
        np.sum(np.max(contingency, axis=0))
        / np.sum(contingency)
    )


def evaluate_clustering(
    true_labels: pd.Series | np.ndarray,
    predicted_labels: np.ndarray,
) -> tuple[float, float, float]:
    """
    Evaluates a clustering result using external clustering metrics.

    The evaluated metrics are:
        - Adjusted Rand Index (ARI)
        - Normalized Mutual Information (NMI)
        - Purity

    Args:
        true_labels (pd.Series | np.ndarray):
            Ground-truth labels.
        predicted_labels (np.ndarray):
            Cluster labels assigned by the clustering algorithm.

    Returns:
        tuple[float, float, float]:
            Tuple containing:
                - Adjusted Rand Index (ARI)
                - Normalized Mutual Information (NMI)
                - Purity
    """
    _validate_labels(
        true_labels,
        predicted_labels,
    )

    ari = adjusted_rand_score(
        true_labels,
        predicted_labels,
    )

    nmi = normalized_mutual_info_score(
        true_labels,
        predicted_labels,
    )

    purity = purity_score(
        true_labels,
        predicted_labels,
    )

    return ari, nmi, purity


def _print_results(
    algorithm: str,
    ari: float,
    nmi: float,
    purity: float,
) -> None:
    """
    Prints the clustering evaluation results.

    Args:
        algorithm (str):
            Name of the clustering algorithm.
        ari (float):
            Adjusted Rand Index.
        nmi (float):
            Normalized Mutual Information.
        purity (float):
            Purity score.

    Returns:
        None.
    """

    print(f"\n{algorithm}")
    print(f"ARI: {ari:.4f}")
    print(f"NMI: {nmi:.4f}")
    print(f"Purity: {purity:.4f}")

def evaluate_algorithm(
    algorithm: str,
    true_labels: pd.Series,
    predicted_labels: np.ndarray,
) -> None:
    """
    Evaluates a clustering algorithm and prints the evaluation metrics.

    Args:
        algorithm (str):
            Name of the clustering algorithm.
        true_labels (pd.Series):
            Ground-truth labels.
        predicted_labels (np.ndarray):
            Cluster labels assigned by the clustering algorithm.

    Returns:
        None.
    """

    ari, nmi, purity = evaluate_clustering(
        true_labels,
        predicted_labels,
    )

    _print_results(
        algorithm,
        ari,
        nmi,
        purity,
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    logger.info("Loading embeddings...")

    embeddings, metadata = load_embeddings(
        CABINETS_EMBEDDINGS_PATH,
        CABINETS_CLUSTERS_METADATA_PATH,
    )

    reference_labels = metadata["label"]

    logger.info("Embeddings loaded.\n")

    logger.info("Running K-Means...")

    kmeans = KMeans(
        n_clusters=DEFAULT_NUMBER_OF_CLUSTERS,
        random_state=RANDOM_STATE,
    )

    kmeans_labels = kmeans.fit_predict(
        embeddings,
    )

    evaluate_algorithm(
        "K-Means",
        reference_labels,
        kmeans_labels,
    )

    logger.info("\nRunning HDBSCAN...")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
    )

    hdbscan_labels = clusterer.fit_predict(
        embeddings,
    )

    evaluate_algorithm(
        "HDBSCAN",
        reference_labels,
        hdbscan_labels,
    )

    # Number of clusters found (excluding noise)
    clusters = len(set(hdbscan_labels)) - (
        1 if -1 in hdbscan_labels else 0
    )

    # Number of noise points
    noise = np.sum(hdbscan_labels == -1)

    logger.info(f"\nClusters found: {clusters}")
    logger.info(f"Noise points: {noise}")

    logger.info("\nEvaluation completed successfully.")
