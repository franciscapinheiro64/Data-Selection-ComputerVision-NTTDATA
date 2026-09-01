import logging
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore[import-untyped]

from src.config import EMBEDDINGS_PATH, FIGURE_SIZE, METADATA_PATH
from src.embeddings_io import load_embeddings
from src.output_utils import get_output_dir  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

def _validate_similarity_inputs(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    image_index: int,
    k: int,
) -> int:
    """
    Validates the inputs used for similarity search.

    Args:
        embeddings (np.ndarray): Array containing the image embeddings.
        metadata (pd.DataFrame): Metadata associated with the embeddings.
        image_index (int): Index of the query image.
        k (int): Number of similar images to return.

    Raises:
        ValueError:
            If embeddings is not a 2D array.
        ValueError:
            If the number of embeddings and metadata rows differ.
        ValueError:
            If fewer than two embeddings are provided.
        IndexError:
            If image_index is outside the valid range.
        ValueError:
            If k is not a positive integer.

    Returns:
        int: The validated number of neighbors to retrieve.
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D array")

    if len(metadata) != len(embeddings):
        raise ValueError("metadata and embeddings must have the same number of rows")

    if len(embeddings) < 2:
        raise ValueError("at least two embeddings are required to find similar images")

    if not 0 <= image_index < len(embeddings):
        raise IndexError("image_index is out of range")

    if k <= 0:
        raise ValueError("k must be a positive integer")

    return min(k, len(embeddings) - 1)


def find_similar_images(
    image_index: int,
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    k: int,
) -> pd.DataFrame:
    """
    Finds the most similar images using cosine similarity.

    Args:
        image_index (int): Index of the query image.
        embeddings (np.ndarray): Array containing the image embeddings.
        metadata (pd.DataFrame): Metadata associated with each embedding.
        k (int): Number of similar images to return.

    Raises:
        ValueError:
            If the inputs fail validation.
        IndexError:
            If image_index is outside the valid range.

    Returns:
        pd.DataFrame:
            A table containing the paths, labels, and similarity scores of the
            most similar images.
    """
    k = _validate_similarity_inputs(
        embeddings,
        metadata,
        image_index,
        k,
    )

    query_embedding = embeddings[image_index].reshape(1, -1)

    similarities = cosine_similarity(
        query_embedding,
        embeddings,
    )[0]

    similarities[image_index] = -1

    top_indices = np.argsort(similarities)[::-1][:k]

    return pd.DataFrame(
        {
            "path": metadata.iloc[top_indices]["path"].values,
            "label": metadata.iloc[top_indices]["label"].values,
            "similarity": similarities[top_indices],
        }
    )

def show_similar_images(
    query_path: str | Path,
    similar_images: pd.DataFrame,
) -> None:
    """
    Displays the query image and its most similar images.

    Args:
        query_path (str | pathlib.Path): Path to the query image.
        similar_images (pd.DataFrame): DataFrame returned by find_similar_images.

    Raises:
        ValueError:
            If similar_images is empty.
        FileNotFoundError:
            If the query image does not exist.

    Returns:
        None: The images are displayed in a figure.
    """
    if similar_images.empty:
        raise ValueError("similar_images must contain at least one row")

    query_path = Path(str(query_path).replace("data/raw/train", "data/raw/fruits/train"))

    if not query_path.exists():
        raise FileNotFoundError(f"Query image path does not exist: {query_path}")


    plt.figure(figsize=FIGURE_SIZE)

    # Query image
    plt.subplot(1, len(similar_images) + 1, 1)

    image = Image.open(query_path)

    plt.imshow(image)
    plt.title("Query")
    plt.axis("off")

    # Similar images
    for i, row in enumerate(similar_images.itertuples(), start=2):

        plt.subplot(1, len(similar_images) + 1, i)
        image = Image.open(Path(cast(str, row.path)))
        plt.imshow(image)

        plt.title(
            f"{row.label}\n{row.similarity:.3f}",
            fontsize=8,
        )

        plt.axis("off")

    plt.tight_layout()

    output_dir = get_output_dir()

    plt.savefig(
        output_dir / "similarity_search.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    embeddings, metadata = load_embeddings(
        EMBEDDINGS_PATH,
        METADATA_PATH,
    )

    # Query image index to find similar images
    image_index = 1000
    # Choose the number of similar images to retrieve
    k = 5

    logger.info("Query image:")
    logger.info(metadata.iloc[image_index]["path"])

    logger.info(f"\nTop {k} similar images:\n")

    similar = find_similar_images(
        image_index,
        embeddings,
        metadata,
        k=k
    )

    # Display full image paths in the DataFrame
    pd.set_option("display.max_colwidth", None)

    logger.info(similar)

    # Display the similar images
    show_similar_images(
        metadata.iloc[image_index]["path"],
        similar,
    )
