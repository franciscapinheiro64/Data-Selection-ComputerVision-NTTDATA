from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from src.kmeans_clustering_metrics import (
    calculate_elbow,
    calculate_silhouette,
    get_cluster_label,
    get_output_dir,
    plot_elbow,
    plot_silhouette,
)


def test_get_output_dir_creates_directory(tmp_path: Path) -> None:
    # Verifies that the output directory is created correctly.
    output_dir: Path = get_output_dir(
        base_dir=tmp_path,
        date=datetime(2026, 7, 20),
    )

    assert output_dir.exists()
    assert output_dir.is_dir()


def test_get_cluster_label_with_range() -> None:
    # Verifies that the cluster range label is generated correctly.
    label: str = get_cluster_label(
        range(2, 21),
    )

    assert label == "k2-20"


def test_calculate_elbow_returns_one_value_per_k() -> None:
    # Verifies that one inertia value is returned for each k value.
    embeddings: np.ndarray = np.random.rand(30, 5)
    k_values: range = range(2, 6)

    inertias: list[float] = calculate_elbow(
        embeddings,
        k_values,
    )

    assert len(inertias) == len(k_values)
    assert all(
        isinstance(value, float)
        for value in inertias
    )


def test_calculate_elbow_raises_for_empty_embeddings() -> None:
    # Verifies that empty embeddings raise a ValueError.
    embeddings: np.ndarray = np.empty((0, 5))

    with pytest.raises(
        ValueError,
        match="The embedding array is empty.",
    ):
        calculate_elbow(
            embeddings,
            range(2, 6),
        )


def test_calculate_elbow_raises_for_empty_k_values() -> None:
    # Verifies that at least one k value is required.
    embeddings: np.ndarray = np.random.rand(30, 5)

    with pytest.raises(
        ValueError,
        match="k_values must contain at least one value.",
    ):
        calculate_elbow(
            embeddings,
            [],
        )


def test_plot_elbow_saves_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Verifies that the Elbow plot is saved successfully.
    monkeypatch.setattr(
        "src.kmeans_clustering_metrics.get_output_dir",
        lambda: tmp_path,
    )

    k_values: range = range(2, 5)
    inertias: list[float] = [100.0, 80.0, 65.0]

    plot_elbow(
        k_values,
        inertias,
    )

    assert (
        tmp_path / "elbow_method_k2-4.png"
    ).exists()


def test_plot_elbow_raises_for_different_lengths() -> None:
    # Verifies that k values and inertia values must have the same length.
    with pytest.raises(
        ValueError,
        match="k_values and metric values must have the same length.",
    ):
        plot_elbow(
            range(2, 5),
            [10.0, 8.0],
        )


def test_calculate_silhouette_returns_one_value_per_k() -> None:
    # Verifies that one silhouette score is returned for each k value.
    embeddings: np.ndarray = np.random.rand(30, 5)
    k_values: range = range(2, 6)

    scores: list[float] = calculate_silhouette(
        embeddings,
        k_values,
    )

    assert len(scores) == len(k_values)
    assert all(
        isinstance(score, float)
        for score in scores
    )


def test_calculate_silhouette_raises_for_empty_embeddings() -> None:
    # Verifies that empty embeddings raise a ValueError.
    embeddings: np.ndarray = np.empty((0, 5))

    with pytest.raises(
        ValueError,
        match="The embedding array is empty.",
    ):
        calculate_silhouette(
            embeddings,
            range(2, 6),
        )


def test_calculate_silhouette_raises_for_empty_k_values() -> None:
    # Verifies that at least one k value is required.
    embeddings: np.ndarray = np.random.rand(30, 5)

    with pytest.raises(
        ValueError,
        match="k_values must contain at least one value.",
    ):
        calculate_silhouette(
            embeddings,
            [],
        )


def test_plot_silhouette_saves_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Verifies that the Silhouette plot is saved successfully.
    monkeypatch.setattr(
        "src.kmeans_clustering_metrics.get_output_dir",
        lambda: tmp_path,
    )

    k_values: range = range(2, 5)
    scores: list[float] = [0.42, 0.51, 0.48]

    plot_silhouette(
        k_values,
        scores,
    )

    assert (
        tmp_path / "silhouette_score_k2-4.png"
    ).exists()


def test_plot_silhouette_raises_for_different_lengths() -> None:
    # Verifies that k values and silhouette scores must have the same length.
    with pytest.raises(
        ValueError,
        match="k_values and metric values must have the same length.",
    ):
        plot_silhouette(
            range(2, 5),
            [0.4, 0.5],
        )
