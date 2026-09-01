from __future__ import annotations

import numpy as np
import pytest

from src.hdbscan_clustering import (
    _validate_hdbscan_inputs,
    cluster_embeddings_hdbscan,
)


def test_validate_hdbscan_inputs_returns_default_min_samples() -> None:
    # Verifies that min_cluster_size is used when min_samples is None.
    embeddings: np.ndarray = np.random.rand(20, 5)

    result: int | None = _validate_hdbscan_inputs(
        embeddings,
        min_cluster_size=10,
        min_samples=None,
    )

    assert result == 10


def test_validate_hdbscan_inputs_raises_for_invalid_dimensions() -> None:
    # Verifies that a ValueError is raised for non-2D embeddings.
    embeddings: np.ndarray = np.random.rand(20)

    with pytest.raises(
        ValueError,
        match="embeddings must be a 2D array",
    ):
        _validate_hdbscan_inputs(
            embeddings,
            min_cluster_size=10,
            min_samples=None,
        )


def test_validate_hdbscan_inputs_raises_for_too_few_embeddings() -> None:
    # Verifies that at least two embeddings are required.
    embeddings: np.ndarray = np.random.rand(1, 5)

    with pytest.raises(
        ValueError,
        match="embeddings must contain at least two samples",
    ):
        _validate_hdbscan_inputs(
            embeddings,
            min_cluster_size=5,
            min_samples=None,
        )


def test_validate_hdbscan_inputs_raises_for_invalid_min_cluster_size() -> None:
    # Verifies that a ValueError is raised when min_cluster_size is not positive.
    embeddings: np.ndarray = np.random.rand(20, 5)

    with pytest.raises(
        ValueError,
        match="min_cluster_size must be a positive integer",
    ):
        _validate_hdbscan_inputs(
            embeddings,
            min_cluster_size=0,
            min_samples=None,
        )


def test_validate_hdbscan_inputs_raises_for_invalid_min_samples() -> None:
    # Verifies that a ValueError is raised when min_samples is not positive.
    embeddings: np.ndarray = np.random.rand(20, 5)

    with pytest.raises(
        ValueError,
        match="min_samples must be a positive integer or None",
    ):
        _validate_hdbscan_inputs(
            embeddings,
            min_cluster_size=5,
            min_samples=0,
        )


def test_cluster_embeddings_hdbscan_returns_one_label_per_embedding() -> None:
    # Verifies that one cluster label is returned for each embedding.
    embeddings: np.ndarray = np.random.rand(50, 8)

    clusters: np.ndarray = cluster_embeddings_hdbscan(
        embeddings,
        min_cluster_size=5,
    )

    assert isinstance(clusters, np.ndarray)
    assert len(clusters) == len(embeddings)


def test_cluster_embeddings_hdbscan_raises_for_empty_embeddings() -> None:
    # Verifies that a ValueError is raised for an empty embedding array.
    embeddings: np.ndarray = np.empty((0, 8))

    with pytest.raises(
        ValueError,
        match="embeddings must contain at least two samples",
    ):
        cluster_embeddings_hdbscan(
            embeddings,
            min_cluster_size=5,
        )
