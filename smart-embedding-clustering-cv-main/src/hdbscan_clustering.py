import logging
import hdbscan  # type: ignore[import-untyped]
import numpy as np

from src.config import (
    CABINETS_EMBEDDINGS_PATH,
    CABINETS_HDBSCAN_METADATA_PATH,
    CABINETS_METADATA_PATH,
    MIN_CLUSTER_SIZE,
    MIN_SAMPLES,
)
from src.embeddings_io import load_embeddings

logger = logging.getLogger(__name__)

def _validate_hdbscan_inputs(
    embeddings: np.ndarray,
    min_cluster_size: int,
    min_samples: int | None,
) -> int | None:
    """
    Validate the input arguments for HDBSCAN clustering.

    Args:
        embeddings (np.ndarray): Embedding array to cluster.
        min_cluster_size (int): Minimum number of samples required to form a cluster.
        min_samples (int | None): Minimum number of neighboring samples.

    Raises:
        TypeError:
            If embeddings is not a NumPy array.
        ValueError:
            If embeddings is not a 2D array.
        ValueError:
            If embeddings contains fewer than two samples.
        ValueError:
            If min_cluster_size is not positive.
        ValueError:
            If min_samples is not positive when provided.

    Returns:
        int | None: The validated value to use for min_samples.
    """
    if not isinstance(embeddings, np.ndarray):
        raise TypeError("embeddings must be a numpy array")

    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D array")

    if len(embeddings) < 2:
        raise ValueError(
            "embeddings must contain at least two samples."
        )

    if (
        not isinstance(min_cluster_size, int)
        or isinstance(min_cluster_size, bool)
        or min_cluster_size <= 0
    ):
        raise ValueError(
            "min_cluster_size must be a positive integer"
        )

    if (
        min_samples is not None
        and (
            not isinstance(min_samples, int)
            or isinstance(min_samples, bool)
            or min_samples <= 0
        )
    ):
        raise ValueError(
            "min_samples must be a positive integer or None"
        )

    return min_samples if min_samples is not None else min_cluster_size


def cluster_embeddings_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int,
    min_samples: int | None = None,
) -> np.ndarray:
    """
    Cluster image embeddings using HDBSCAN.

    Args:
        embeddings (np.ndarray): Array containing the image embeddings.
        min_cluster_size (int): Minimum number of samples required to form a cluster.
        min_samples (int | None): Minimum number of neighboring samples required for a point to be considered a core point. If None, the value of min_cluster_size is used.

    Returns:
        np.ndarray: One cluster label per embedding. Noise samples are assigned the label -1.
    """
    min_samples = _validate_hdbscan_inputs(
        embeddings,
        min_cluster_size,
        min_samples,
    )
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )

    return clusterer.fit_predict(embeddings)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    logger.info("Loading embeddings...")

    embeddings, metadata = load_embeddings(
        CABINETS_EMBEDDINGS_PATH,
        CABINETS_METADATA_PATH,
    )

    logger.info("Embeddings loaded.")
    logger.info("Running HDBSCAN...")

    labels = cluster_embeddings_hdbscan(
        embeddings,
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
    )

    metadata["cluster"] = labels

    metadata.to_csv(
        CABINETS_HDBSCAN_METADATA_PATH,
        index=False,
    )
    #mask = metadata["cluster"] == CABINET_CLUSTER

    #embeddings_subset = embeddings[mask]
    #metadata_subset = metadata[mask].copy()

    #subclusters = cluster_embeddings_hdbscan(
        #embeddings_subset,
        #min_cluster_size=MIN_CLUSTER_SIZE,
        #min_samples=MIN_SAMPLES,
    #)

    #metadata_subset["subcluster"] = subclusters


    #metadata_subset.to_csv(
        #CABINETS_HDBSCAN_METADATA_PATH,
        #index=False,
    #)

    #unique, counts = np.unique(
        #subclusters,
        #return_counts=True,
    #)

    # Exclude the noise cluster (-1)
    #n_clusters = len(unique) - (
        #1 if -1 in unique else 0
    #)

    #logger.info(f"\nClusters found: {n_clusters}")

    #for cluster, count in zip(unique, counts):
        #logger.info(f"Cluster {cluster}: {count} images")

    logger.info("HDBSCAN clustering completed.")
