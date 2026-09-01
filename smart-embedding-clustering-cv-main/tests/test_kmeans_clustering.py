from __future__ import annotations

import numpy as np
import pytest

from src.kmeans_clustering import cluster_embeddings


def test_cluster_embeddings_returns_numpy_array() -> None:
    # Verifies that clustering returns a label array.
    embeddings: np.ndarray = np.random.rand(10, 4)

    clusters: np.ndarray = cluster_embeddings(
        embeddings,
        k=3,
    )

    assert isinstance(clusters, np.ndarray)


def test_cluster_embeddings_returns_one_cluster_per_embedding() -> None:
    # Verifies that there is one cluster label per embedding.
    embeddings: np.ndarray = np.random.rand(10, 4)

    clusters: np.ndarray = cluster_embeddings(
        embeddings,
        k=3,
    )

    assert len(clusters) == len(embeddings)


def test_cluster_labels_are_within_expected_range() -> None:
    # Verifies that cluster labels are within [0, k-1].
    embeddings: np.ndarray = np.random.rand(10, 4)

    k: int = 3

    clusters: np.ndarray = cluster_embeddings(
        embeddings,
        k=k,
    )

    assert clusters.min() >= 0
    assert clusters.max() < k


def test_cluster_embeddings_raises_for_non_2d_embeddings() -> None:
    # Verifies that non-2D embeddings raise a ValueError.
    embeddings: np.ndarray = np.random.rand(10)

    with pytest.raises(
        ValueError,
        match="embeddings must be a 2D array",
    ):
        cluster_embeddings(
            embeddings,
            k=2,
        )


def test_cluster_embeddings_raises_for_too_few_embeddings() -> None:
    # Verifies that at least two embeddings are required.
    embeddings: np.ndarray = np.random.rand(1, 4)

    with pytest.raises(
        ValueError,
        match="embeddings must contain at least two samples",
    ):
        cluster_embeddings(
            embeddings,
            k=1,
        )


def test_cluster_embeddings_raises_for_invalid_k() -> None:
    # Verifies that k must be positive.
    embeddings: np.ndarray = np.random.rand(10, 4)

    with pytest.raises(
        ValueError,
        match="k must be a positive integer",
    ):
        cluster_embeddings(
            embeddings,
            k=0,
        )


def test_cluster_embeddings_raises_when_k_is_larger_than_dataset() -> None:
    # Verifies that k cannot exceed the number of embeddings.
    embeddings: np.ndarray = np.random.rand(5, 4)

    with pytest.raises(
        ValueError,
        match="k must be less than or equal to the number of samples",
    ):
        cluster_embeddings(
            embeddings,
            k=6,
        )
