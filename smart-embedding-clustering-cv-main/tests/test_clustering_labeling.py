from __future__ import annotations

from typing import cast

import pandas as pd
import pytest

from src.clustering_labeling import label_clusters


def test_label_clusters_assigns_predicted_label() -> None:
    # Verifies that the predicted label is assigned correctly based on the cluster.
    metadata: pd.DataFrame = pd.DataFrame(
        {
            "cluster": [0, 1, 1, 0],
            "filename": ["img_0.jpg", "img_1.jpg", "img_2.jpg", "img_3.jpg"],
        }
    )

    result: pd.DataFrame = label_clusters(metadata)

    assert "predicted_label" in result.columns
    assert result.loc[0, "predicted_label"] == "no_cabinet"
    assert result.loc[1, "predicted_label"] == "cabinet"
    assert result.loc[2, "predicted_label"] == "cabinet"
    assert result.loc[3, "predicted_label"] == "no_cabinet"


def test_label_clusters_raises_type_error_for_non_dataframe() -> None:
    # Verifies that passing a non-DataFrame value raises TypeError.
    with pytest.raises(TypeError, match="metadata must be a pandas DataFrame"):
        label_clusters(metadata=cast(pd.DataFrame, [{"cluster": 0}]))


def test_label_clusters_raises_value_error_for_empty_dataframe() -> None:
    # Verifies that an empty DataFrame raises ValueError.
    metadata: pd.DataFrame = pd.DataFrame(columns=["cluster"])

    with pytest.raises(ValueError, match="metadata must not be empty"):
        label_clusters(metadata)


def test_label_clusters_raises_value_error_when_cluster_column_missing() -> None:
    # Verifies that missing the `cluster` column raises ValueError.
    metadata: pd.DataFrame = pd.DataFrame(
        {
            "label": ["apple", "banana"],
        }
    )

    with pytest.raises(ValueError, match="Metadata must contain a 'cluster' column"):
        label_clusters(metadata)
