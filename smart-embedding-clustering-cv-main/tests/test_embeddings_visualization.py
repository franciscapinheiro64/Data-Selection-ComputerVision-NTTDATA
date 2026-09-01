from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import EMBEDDING_DIMENSION
from src.embeddings_visualization import (
    get_output_dir,
    load_embeddings,
    plot_embeddings,
    plot_selected_embeddings,
    reduce_dimensions,
)


def test_reduce_dimensions_returns_numpy_array() -> None:
    # Verifies that dimensionality reduction returns a NumPy array
    embeddings: np.ndarray = np.random.rand(
        100,
        EMBEDDING_DIMENSION,
    )

    embeddings_2d: np.ndarray = reduce_dimensions(embeddings)

    assert isinstance(embeddings_2d, np.ndarray)


def test_reduce_dimensions_shape() -> None:
    # Verifies that the output shape is (n, 2)
    embeddings: np.ndarray = np.random.rand(
        100,
        EMBEDDING_DIMENSION,
    )

    embeddings_2d: np.ndarray = reduce_dimensions(embeddings)

    assert embeddings_2d.shape == (100, 2)


def test_get_output_dir_uses_date_folder(tmp_path: Path) -> None:
    # Verifies that the output directory is created using the provided date
    output_dir: Path = get_output_dir(
        base_dir=tmp_path,
        date=datetime(2026, 7, 20),
    )

    assert output_dir == tmp_path / "20-07"
    assert output_dir.exists()


def test_load_embeddings(tmp_path: Path) -> None:
    # Verifies that embeddings and metadata are loaded correctly
    embeddings: np.ndarray = np.random.rand(
        5,
        EMBEDDING_DIMENSION,
    )

    metadata: pd.DataFrame = pd.DataFrame(
        {
            "path": [
                "image1.jpg",
                "image2.jpg",
                "image3.jpg",
                "image4.jpg",
                "image5.jpg",
            ],
            "filename": [
                "image1.jpg",
                "image2.jpg",
                "image3.jpg",
                "image4.jpg",
                "image5.jpg",
            ],
            "label": [
                "apple",
                "banana",
                "orange",
                "pear",
                "grape",
            ],
        }
    )

    embeddings_file: Path = tmp_path / "embeddings.npy"
    metadata_file: Path = tmp_path / "metadata.csv"

    np.save(embeddings_file, embeddings)
    metadata.to_csv(metadata_file, index=False)

    loaded_embeddings: np.ndarray
    loaded_metadata: pd.DataFrame
    loaded_embeddings, loaded_metadata = load_embeddings(
        embeddings_file,
        metadata_file,
    )

    assert loaded_embeddings.shape == embeddings.shape
    assert loaded_metadata.equals(metadata)


def test_load_embeddings_raises_if_embeddings_file_does_not_exist(
    tmp_path: Path,
) -> None:
    # Verifies that a missing embeddings file raises FileNotFoundError
    metadata: pd.DataFrame = pd.DataFrame(
        {
            "path": ["image.jpg"],
            "filename": ["image.jpg"],
            "label": ["apple"],
        }
    )

    metadata_file: Path = tmp_path / "metadata.csv"
    metadata.to_csv(metadata_file, index=False)

    with pytest.raises(FileNotFoundError):
        load_embeddings(
            tmp_path / "missing.npy",
            metadata_file,
        )


def test_load_embeddings_raises_if_metadata_file_does_not_exist(
    tmp_path: Path,
) -> None:
    # Verifies that a missing metadata file raises FileNotFoundError
    embeddings: np.ndarray = np.random.rand(
        1,
        EMBEDDING_DIMENSION,
    )

    embeddings_file: Path = tmp_path / "embeddings.npy"
    np.save(embeddings_file, embeddings)

    with pytest.raises(FileNotFoundError):
        load_embeddings(
            embeddings_file,
            tmp_path / "missing.csv",
        )


def test_load_embeddings_raises_if_lengths_do_not_match(
    tmp_path: Path,
) -> None:
    # Verifies that mismatched embeddings and metadata raise ValueError
    embeddings: np.ndarray = np.random.rand(
        5,
        EMBEDDING_DIMENSION,
    )

    metadata: pd.DataFrame = pd.DataFrame(
        {
            "path": [
                "image1.jpg",
                "image2.jpg",
            ],
            "filename": [
                "image1.jpg",
                "image2.jpg",
            ],
            "label": [
                "apple",
                "banana",
            ],
        }
    )

    embeddings_file: Path = tmp_path / "embeddings.npy"
    metadata_file: Path = tmp_path / "metadata.csv"

    np.save(embeddings_file, embeddings)
    metadata.to_csv(metadata_file, index=False)

    with pytest.raises(ValueError):
        load_embeddings(
            embeddings_file,
            metadata_file,
        )


def test_plot_embeddings_uses_custom_output_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Verifies that the plot is saved with the specified output name
    monkeypatch.setattr(
        "src.embeddings_visualization.get_output_dir",
        lambda: tmp_path,
    )

    embeddings_2d: np.ndarray = np.random.rand(10, 2)
    groups: np.ndarray = np.array(["A"] * 5 + ["B"] * 5)

    plot_embeddings(
        embeddings_2d,
        groups,
        "Title",
        output_name="custom_plot",
    )

    assert (tmp_path / "custom_plot.png").exists()


def test_plot_selected_embeddings_uses_custom_output_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Verifies that the selected embeddings plot is saved successfully
    monkeypatch.setattr(
        "src.embeddings_visualization.get_output_dir",
        lambda: tmp_path,
    )

    embeddings_2d: np.ndarray = np.random.rand(10, 2)

    labels: np.ndarray = np.array(
        [
            "apple",
            "apple",
            "apple",
            "banana",
            "banana",
            "orange",
            "orange",
            "orange",
            "orange",
            "orange",
        ]
    )

    plot_selected_embeddings(
        embeddings_2d,
        labels,
        ["apple", "banana"],
        "Title",
        output_name="selected_plot",
    )

    assert (tmp_path / "selected_plot.png").exists()
