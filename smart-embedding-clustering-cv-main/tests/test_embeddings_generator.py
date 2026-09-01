from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from unittest.mock import patch

from src.clusters_visualization import visualize_cluster
from src.config import EMBEDDING_DIMENSION
from src.embeddings_generator import (
    MetadataSample,
    generate_embedding,
    generate_embeddings,
    load_model,
    save_embeddings,
)
from src.dataset_loader import (
    TrainSample,
    UnlabeledSample,
    load_image,
    load_train_dataset,
)

def test_generate_embedding() -> None:
    # Verifies that the function generates a NumPy embedding with the correct dimension
    processor, model = load_model()

    dataset: list[TrainSample] = load_train_dataset()
    image = load_image(dataset[0]["path"])

    embedding: np.ndarray = generate_embedding(
        image,
        processor,
        model,
    )

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (EMBEDDING_DIMENSION,)


@patch("src.embeddings_generator.generate_embedding")
@patch("src.embeddings_generator.load_image")
def test_generate_embeddings(
    mock_load_image: Any,
    mock_generate_embedding: Any,
) -> None:
    # Verifies embeddings and metadata generation for labeled samples
    mock_load_image.return_value = "dummy_image"
    mock_generate_embedding.return_value = np.zeros(
        EMBEDDING_DIMENSION
    )

    dataset: list[TrainSample] = [
        {
            "path": Path("image1.jpg"),
            "filename": "image1.jpg",
            "label": "apple",
        },
        {
            "path": Path("image2.jpg"),
            "filename": "image2.jpg",
            "label": "banana",
        },
    ]

    embeddings: list[np.ndarray]
    metadata: list[MetadataSample]
    embeddings, metadata = generate_embeddings(
        dataset,
        processor=None,
        model=None,
    )

    assert len(embeddings) == 2
    assert len(metadata) == 2

    assert embeddings[0].shape == (EMBEDDING_DIMENSION,)
    assert embeddings[1].shape == (EMBEDDING_DIMENSION,)

    assert metadata[0]["label"] == "apple"
    assert metadata[1]["label"] == "banana"


@patch("src.embeddings_generator.generate_embedding")
@patch("src.embeddings_generator.load_image")
def test_generate_embeddings_unlabeled_dataset(
    mock_load_image: Any,
    mock_generate_embedding: Any,
) -> None:
    # Verifies metadata generation for unlabeled samples
    mock_load_image.return_value = "dummy_image"
    mock_generate_embedding.return_value = np.zeros(
        EMBEDDING_DIMENSION
    )

    dataset: list[UnlabeledSample] = [
        {
            "path": Path("image1.jpg"),
            "filename": "image1.jpg",
        }
    ]

    embeddings: list[np.ndarray]
    metadata: list[MetadataSample]
    embeddings, metadata = generate_embeddings(
        dataset,
        processor=None,
        model=None,
    )

    assert len(embeddings) == 1
    assert len(metadata) == 1

    assert metadata[0]["label"] is None


def test_generate_embeddings_requires_non_empty_dataset() -> None:
    with pytest.raises(ValueError):
        generate_embeddings([], processor=None, model=None)


def test_visualize_cluster_requires_positive_images_per_cluster(tmp_path: Path) -> None:
    metadata: pd.DataFrame = pd.DataFrame(
        {
            "path": [tmp_path / "img1.jpg"],
            "cluster": [0],
        }
    )

    with pytest.raises(ValueError):
        visualize_cluster(metadata, cluster=0, images_per_cluster=0)


def test_save_embeddings(tmp_path: Path) -> None:
    # Verifies embeddings and metadata are persisted to disk
    embeddings: list[np.ndarray] = [
        np.zeros(EMBEDDING_DIMENSION),
        np.ones(EMBEDDING_DIMENSION),
    ]

    metadata: list[MetadataSample] = [
        {
            "path": "image1.jpg",
            "filename": "image1.jpg",
            "label": "apple",
        },
        {
            "path": "image2.jpg",
            "filename": "image2.jpg",
            "label": "banana",
        },
    ]

    save_embeddings(
        embeddings,
        metadata,
        tmp_path,
    )

    embeddings_file: Path = tmp_path / "embeddings.npy"
    metadata_file: Path = tmp_path / "metadata.csv"

    assert embeddings_file.exists()
    assert metadata_file.exists()

    loaded_embeddings: np.ndarray = np.load(embeddings_file)
    loaded_metadata: pd.DataFrame = pd.read_csv(metadata_file)

    assert loaded_embeddings.shape == (
        2,
        EMBEDDING_DIMENSION,
    )

    np.testing.assert_array_equal(
        loaded_embeddings[0],
        np.zeros(EMBEDDING_DIMENSION),
    )

    np.testing.assert_array_equal(
        loaded_embeddings[1],
        np.ones(EMBEDDING_DIMENSION),
    )

    assert list(loaded_metadata.columns) == [
        "path",
        "filename",
        "label",
    ]

    assert loaded_metadata.iloc[0]["label"] == "apple"
    assert loaded_metadata.iloc[1]["label"] == "banana"


def test_save_embeddings_accepts_numpy_array(tmp_path: Path) -> None:
    # Verifies that embeddings can be saved from a NumPy array
    embeddings: np.ndarray = np.zeros(
        (2, EMBEDDING_DIMENSION)
    )

    metadata: list[MetadataSample] = [
        {
            "path": "image1.jpg",
            "filename": "image1.jpg",
            "label": "apple",
        },
        {
            "path": "image2.jpg",
            "filename": "image2.jpg",
            "label": "banana",
        },
    ]

    save_embeddings(
        embeddings,
        metadata,
        tmp_path,
    )

    assert (tmp_path / "embeddings.npy").exists()
