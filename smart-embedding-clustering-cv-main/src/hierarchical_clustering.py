import logging
from pathlib import Path

import pandas as pd
import numpy as np

from src.config import (
    CABINETS_CLUSTERS_METADATA_PATH,
    CABINETS_EMBEDDINGS_PATH,
    CABINET_CLUSTER,
    CABINETS_SUBCLUSTERS_METADATA_PATH,
    NUMBER_OF_SUBCLUSTERS,
)
from src.embeddings_io import load_embeddings
from src.kmeans_clustering import cluster_embeddings

logger = logging.getLogger(__name__)


def run_subclustering(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    cluster_id: int = CABINET_CLUSTER,
    n_clusters: int = NUMBER_OF_SUBCLUSTERS,
    output_path: str | Path = CABINETS_SUBCLUSTERS_METADATA_PATH,
) -> pd.DataFrame:
    """
    Run a second-stage clustering on a selected cluster.

    Args:
        embeddings (np.ndarray): Full embedding matrix.
        metadata (pd.DataFrame): Metadata associated with the embeddings.
        cluster_id (int): Cluster identifier to subset before subclustering.
        n_clusters (int): Number of subclusters to generate.
        output_path (str | pathlib.Path): Path where the subcluster metadata will be saved.

    Raises:
        ValueError:
            If the metadata is empty.
        ValueError:
            If the metadata does not contain a 'cluster' column.
        ValueError:
            If the number of embeddings does not match the number of metadata rows.

    Returns:
        pd.DataFrame: Metadata corresponding only to the selected cluster, with an
        additional `subcluster` column containing the K-Means assignments.
    """

    mask = metadata["cluster"] == cluster_id

    if "cluster" not in metadata.columns:
        raise ValueError(
            "metadata must contain a 'cluster' column"
        )

    if metadata.empty:
        raise ValueError(
            "metadata must not be empty"
        )

    if len(embeddings) != len(metadata):
        raise ValueError(
            "embeddings and metadata must have the same length"
        )

    embeddings_subset = embeddings[mask]
    metadata_subset = metadata[mask].copy()

    subclusters = cluster_embeddings(
        embeddings_subset,
        k=n_clusters,
    )

    metadata_subset["subcluster"] = subclusters

    metadata_subset.to_csv(output_path, index=False)

    return metadata_subset


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    logger.info("Loading embeddings...")

    embeddings, metadata = load_embeddings(
        CABINETS_EMBEDDINGS_PATH,
        CABINETS_CLUSTERS_METADATA_PATH,
    )

    logger.info("Running second-stage K-Means...")

    metadata_subset = run_subclustering(
        embeddings,
        metadata,
    )

    logger.info("\nSubclustering completed.")

    logger.info(metadata_subset["subcluster"].value_counts())

    logger.info(
        f"\nSaved to: {CABINETS_SUBCLUSTERS_METADATA_PATH}"
    )
