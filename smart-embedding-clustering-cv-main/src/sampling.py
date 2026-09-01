
import numpy as np
import pandas as pd

from src.config import RANDOM_STATE


def sample_embeddings(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    sample_size: int,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Returns a random sample of embeddings and metadata.

    Args:
        embeddings (np.ndarray): Embedding array.
        metadata (pd.DataFrame): Metadata associated with the embeddings.
        sample_size (int): Number of embeddings to sample.
        random_state (int): Random seed.

    Raises:
        ValueError:
            If the embedding array is empty.
        ValueError:
            If embeddings is not a 2D array.
        ValueError:
            If the number of embeddings and metadata rows differ.
        ValueError:
            If sample_size is not a positive integer.
        ValueError:
            If sample_size is larger than the dataset.

    Returns:
        tuple[np.ndarray, pd.DataFrame]:
            Sampled embeddings and metadata.
    """

    if len(embeddings) == 0:
        raise ValueError(
            "The dataset is empty."
        )

    if embeddings.ndim != 2:
        raise ValueError(
            "Embeddings must be a two-dimensional array."
        )

    if len(embeddings) != len(metadata):
        raise ValueError(
            "The number of embeddings does not match the number of metadata entries."
        )

    if sample_size <= 0:
        raise ValueError(
            "sample_size must be a positive integer."
        )

    if sample_size > len(embeddings):
        raise ValueError(
            "sample_size cannot be larger than the dataset."
        )

    rng = np.random.default_rng(random_state)

    indices = rng.choice(
        len(embeddings),
        size=sample_size,
        replace=False,
    )

    return embeddings[indices], metadata.iloc[indices].reset_index(drop=True)


def sample_embeddings_stratified(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    sample_size: int,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Returns a stratified sample of embeddings and metadata.

    Args:
        embeddings (np.ndarray): Embedding array.
        metadata (pd.DataFrame): Metadata associated with the embeddings.
        sample_size (int): Number of samples to return.
        random_state (int): Random seed.

    Raises:
        ValueError:
            If embeddings and metadata have different lengths.
        ValueError:
            If embeddings is not a 2D array.
        ValueError:
            If the dataset is empty.
        ValueError:
            If sample_size is not a positive integer.
        ValueError:
            If sample_size is larger than the dataset.
        ValueError:
            If metadata does not contain a 'label' column.

    Returns:
        tuple[np.ndarray, pd.DataFrame]:
            Sampled embeddings and metadata.
    """
    if len(embeddings) == 0:
        raise ValueError(
            "The dataset is empty."
        )

    if len(embeddings) != len(metadata):
        raise ValueError(
            "The number of embeddings does not match the number of metadata entries."
        )

    if embeddings.ndim != 2:
        raise ValueError(
            "Embeddings must be a two-dimensional array"
        )

    if sample_size <= 0:
        raise ValueError("sample_size must be a positive integer")

    if sample_size > len(metadata):
        raise ValueError("sample_size cannot be larger than the dataset")

    if "label" not in metadata.columns:
        raise ValueError("metadata must contain a 'label' column")


    fraction = sample_size / len(metadata)

    metadata_sample = (
        metadata
        .groupby("label", group_keys=False)
        .sample(
            frac=fraction,
            random_state=random_state,
        )
        .sort_index()
    )

    embeddings_sample = embeddings[metadata_sample.index]

    metadata_sample = metadata_sample.reset_index(drop=True)

    return embeddings_sample, metadata_sample
