
import numpy as np
import pandas as pd
import pytest


from src.sampling import (
    sample_embeddings,
    sample_embeddings_stratified,
)


def test_sample_embeddings_returns_requested_number_of_rows() -> None:
    # Verifies that the sampled embeddings and metadata have the requested size.
    embeddings: np.ndarray = np.arange(20).reshape(10, 2)

    metadata: pd.DataFrame = pd.DataFrame(
        {
            "label": ["apple", "banana"] * 5,
            "filename": [f"img_{i}" for i in range(10)],
        }
    )

    sampled_embeddings: np.ndarray
    sampled_metadata: pd.DataFrame
    sampled_embeddings, sampled_metadata = sample_embeddings(
        embeddings,
        metadata,
        sample_size=4,
        random_state=42,
    )

    assert sampled_embeddings.shape == (4, 2)
    assert len(sampled_metadata) == 4


def test_sample_embeddings_raises_for_invalid_sample_size() -> None:
    # Verifies that the sample size must be positive.
    embeddings: np.ndarray = np.arange(10).reshape(5, 2)

    metadata: pd.DataFrame = pd.DataFrame(
        {
            "label": ["a", "b", "c", "d", "e"],
        }
    )

    with pytest.raises(
        ValueError,
        match="sample_size must be a positive integer",
    ):
        sample_embeddings(
            embeddings,
            metadata,
            sample_size=0,
        )


def test_sample_embeddings_raises_when_sample_is_larger_than_dataset() -> None:
    # Verifies that the sample size cannot exceed the dataset size.
    embeddings: np.ndarray = np.arange(10).reshape(5, 2)

    metadata: pd.DataFrame = pd.DataFrame(
        {
            "label": ["a", "b", "c", "d", "e"],
        }
    )

    with pytest.raises(
        ValueError,
        match="sample_size cannot be larger than the dataset",
    ):
        sample_embeddings(
            embeddings,
            metadata,
            sample_size=6,
        )


def test_sample_embeddings_raises_when_dataset_sizes_do_not_match() -> None:
    # Verifies that embeddings and metadata must have the same size.
    embeddings: np.ndarray = np.arange(20).reshape(10, 2)

    metadata: pd.DataFrame = pd.DataFrame(
        {
            "label": ["a"] * 9,
        }
    )

    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        sample_embeddings(
            embeddings,
            metadata,
            sample_size=5,
        )


def test_sample_embeddings_raises_when_embeddings_are_not_two_dimensional() -> None:
    # Verifies that embeddings must be a two-dimensional array.
    embeddings: np.ndarray = np.arange(10)

    metadata: pd.DataFrame = pd.DataFrame(
        {
            "label": [f"class_{i}" for i in range(10)],
        }
    )

    with pytest.raises(
        ValueError,
        match="two-dimensional",
    ):
        sample_embeddings(
            embeddings,
            metadata,
            sample_size=5,
        )


def test_sample_embeddings_stratified_returns_expected_size() -> None:
    # Verifies that stratified sampling returns embeddings and metadata.
    embeddings: np.ndarray = np.arange(20).reshape(10, 2)

    metadata: pd.DataFrame = pd.DataFrame(
        {
            "label": ["apple"] * 4 + ["banana"] * 6,
            "filename": [f"img_{i}" for i in range(10)],
        }
    )

    sampled_embeddings: np.ndarray
    sampled_metadata: pd.DataFrame
    sampled_embeddings, sampled_metadata = sample_embeddings_stratified(
        embeddings,
        metadata,
        sample_size=5,
        random_state=42,
    )

    assert len(sampled_embeddings) == len(sampled_metadata)
    assert "label" in sampled_metadata.columns


def test_sample_embeddings_stratified_raises_when_label_column_is_missing() -> None:
    # Verifies that stratified sampling requires a label column.
    embeddings: np.ndarray = np.arange(10).reshape(5, 2)

    metadata: pd.DataFrame = pd.DataFrame(
        {
            "filename": [f"img_{i}" for i in range(5)],
        }
    )

    with pytest.raises(
        ValueError,
        match="label",
    ):
        sample_embeddings_stratified(
            embeddings,
            metadata,
            sample_size=3,
        )


def test_sample_embeddings_stratified_raises_when_dataset_sizes_do_not_match() -> None:
    # Verifies that embeddings and metadata must have the same size.
    embeddings: np.ndarray = np.arange(20).reshape(10, 2)

    metadata: pd.DataFrame = pd.DataFrame(
        {
            "label": ["apple"] * 9,
        }
    )

    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        sample_embeddings_stratified(
            embeddings,
            metadata,
            sample_size=5,
        )


def test_sample_embeddings_stratified_raises_when_embeddings_are_not_two_dimensional() -> None:
    # Verifies that embeddings must be a two-dimensional array.
    embeddings: np.ndarray = np.arange(10)

    metadata: pd.DataFrame = pd.DataFrame(
        {
            "label": [f"class_{i}" for i in range(10)],
        }
    )

    with pytest.raises(
        ValueError,
        match="two-dimensional",
    ):
        sample_embeddings_stratified(
            embeddings,
            metadata,
            sample_size=5,
        )
