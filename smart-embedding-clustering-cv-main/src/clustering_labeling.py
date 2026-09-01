from pathlib import Path
import logging
import pandas as pd

from src.config import (
    CABINETS_CLUSTERS_METADATA_PATH,
    CABINETS_LABELLED_METADATA_PATH,
)

logger = logging.getLogger(__name__)


def label_clusters(
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assign semantic labels to the clustering results.

    Args:
        metadata (pd.DataFrame): Metadata containing the clustering assignments.

    Raises:
        TypeError:
            If metadata is not a pandas DataFrame.
        ValueError:
            If metadata is empty.
        ValueError:
            If metadata does not contain a 'cluster' column.

    Returns:
        pd.DataFrame: A copy of the input metadata with a new
        predicted_label column mapping cluster IDs to semantic labels.
    """

    if not isinstance(metadata, pd.DataFrame):
        raise TypeError("metadata must be a pandas DataFrame")

    if metadata.empty:
        raise ValueError("metadata must not be empty")

    if "cluster" not in metadata.columns:
        raise ValueError(
            "Metadata must contain a 'cluster' column."
        )

    cluster_mapping = {
        0: "no_cabinet",
        1: "cabinet",
    }

    if not metadata["cluster"].isin(cluster_mapping).all():
        unknown_clusters = sorted(
            metadata.loc[
                ~metadata["cluster"].isin(cluster_mapping),
                "cluster",
            ].unique()
        )

        raise ValueError(
            f"Unknown cluster IDs found: {unknown_clusters}"
        )

    metadata = metadata.copy()

    metadata["predicted_label"] = metadata["cluster"].map(
        cluster_mapping
    )

    return metadata


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    metadata = pd.read_csv(
        CABINETS_CLUSTERS_METADATA_PATH,
    )

    metadata = label_clusters(
        metadata,
    )

    Path(
        CABINETS_LABELLED_METADATA_PATH,
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata.to_csv(
        CABINETS_LABELLED_METADATA_PATH,
        index=False,
    )

    logger.info(
        f"Saved labelled metadata to {CABINETS_LABELLED_METADATA_PATH}"
    )
