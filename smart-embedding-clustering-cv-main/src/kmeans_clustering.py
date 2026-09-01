import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans # type: ignore[import-untyped]
from src.config import (
    DEFAULT_NUMBER_OF_CLUSTERS,
    RANDOM_STATE,
    EMBEDDINGS_PATH,
    METADATA_PATH,
    CLUSTERS_METADATA_PATH,
)
from src.embeddings_io import load_embeddings

logger = logging.getLogger(__name__)

def _validate_clustering_inputs(
    embeddings: np.ndarray,
    k: int,
) -> None:
    """
    Validates the inputs used for K-Means clustering.

    Args:
        embeddings (np.ndarray): Array containing the input embeddings.
        k (int): Number of clusters to generate.

    Raises:
        ValueError:
            If embeddings is not a 2D array.
        ValueError:
            If embeddings contains fewer than two samples.
        ValueError:
            If k is not a positive integer.
        ValueError:
            If k is greater than the number of samples.

    Returns:
        None: The inputs are validated in place.
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D array")

    if len(embeddings) < 2:
        raise ValueError(
            "embeddings must contain at least two samples"
        )

    if k <= 0:
        raise ValueError("k must be a positive integer")

    if k > len(embeddings):
        raise ValueError("k must be less than or equal to the number of samples")


def cluster_embeddings(
    embeddings: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Cluster embeddings using the K-Means algorithm.

    Args:
        embeddings (np.ndarray): Array containing the image embeddings.
        k (int): Number of clusters to generate.

    Raises:
        ValueError:
            Propagated from _validate_clustering_inputs if the inputs are invalid.

    Returns:
        np.ndarray: Array containing the cluster assigned to each embedding.
    """
    _validate_clustering_inputs(embeddings, k)

    kmeans = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
    )

    clusters = kmeans.fit_predict(embeddings)

    return clusters


def save_clusters(
    metadata: pd.DataFrame,
    clusters: np.ndarray,
    output_path: str | Path,
) -> None:
    """
    Saves cluster assignments to a metadata file.

    Args:
        metadata (pd.DataFrame): Metadata associated with each embedding.
        clusters (np.ndarray): Array containing the cluster assigned to each embedding.
        output_path (str | pathlib.Path): Output path for the metadata file.

    Raises:
        ValueError:
            If metadata and clusters do not have the same length.

    Returns:
        None: The cluster assignments are written to disk.
    """

    if len(metadata) != len(clusters):
        raise ValueError(
            "metadata and clusters must have the same length."
        )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = metadata.copy()

    metadata["cluster"] = clusters

    metadata.to_csv(
        output_path,
        index=False,
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    logger.info("Loading embeddings...")

    embeddings, metadata = load_embeddings(
        EMBEDDINGS_PATH,
        METADATA_PATH,
    )

    logger.info(f"Embeddings: {embeddings.shape}")
    logger.info(f"Metadata: {metadata.shape}")

    logger.info("\nRunning K-Means...")

    clusters = cluster_embeddings(
        embeddings,
        DEFAULT_NUMBER_OF_CLUSTERS,
    )

    logger.info("Saving clustering results...")

    save_clusters(
        metadata,
        clusters,
        CLUSTERS_METADATA_PATH,
    )

    logger.info("K-Means clustering completed.")
