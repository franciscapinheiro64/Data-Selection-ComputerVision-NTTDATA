from pathlib import Path

import numpy as np
import pandas as pd


def load_embeddings(
    embeddings_path: str | Path,
    metadata_path: str | Path,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Load embeddings and their associated metadata from disk.

    Args:
        embeddings_path (str | pathlib.Path): Path to the NumPy file
            containing the embeddings.
        metadata_path (str | pathlib.Path): Path to the CSV file
            containing the metadata.

    Raises:
        FileNotFoundError:
            If the embeddings file does not exist.
        FileNotFoundError:
            If the metadata file does not exist.
        ValueError:
            If the embeddings array is not two-dimensional.
        ValueError:
            If the number of embeddings does not match the number of
            metadata entries.

    Returns:
        tuple[np.ndarray, pd.DataFrame]: A tuple containing the loaded
        embeddings array and the corresponding metadata table.
    """
    embeddings_path = Path(embeddings_path)
    metadata_path = Path(metadata_path)

    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Embeddings file not found: {embeddings_path}"
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}"
        )

    embeddings = np.load(embeddings_path)
    metadata = pd.read_csv(metadata_path)

    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D array")

    if len(embeddings) != len(metadata):
        raise ValueError(
            "The number of embeddings does not match "
            "the number of metadata entries."
        )

    return embeddings, metadata
