import numpy as np
import pandas as pd
import pytest

from src.similarity_search import find_similar_images


def test_find_similar_images_returns_dataframe() -> None:
    # Verifies that the search returns a DataFrame
    embeddings: np.ndarray = np.eye(5)

    metadata: pd.DataFrame = pd.DataFrame(
        {
            "path": [f"image_{i}.jpg" for i in range(5)],
            "label": [f"class_{i}" for i in range(5)],
        }
    )

    result: pd.DataFrame = find_similar_images(
        image_index=0,
        embeddings=embeddings,
        metadata=metadata,
        k=3,
    )

    assert isinstance(result, pd.DataFrame)


def test_find_similar_images_returns_k_results() -> None:
    # Verifies that the number of results returned equals k
    embeddings: np.ndarray = np.eye(5)

    metadata: pd.DataFrame = pd.DataFrame(
        {
            "path": [f"image_{i}.jpg" for i in range(5)],
            "label": [f"class_{i}" for i in range(5)],
        }
    )

    result: pd.DataFrame = find_similar_images(
        image_index=0,
        embeddings=embeddings,
        metadata=metadata,
        k=3,
    )

    assert len(result) == 3


def test_query_image_is_not_returned() -> None:
    # Verifies that the query image is not included in the results
    embeddings: np.ndarray = np.eye(5)

    metadata: pd.DataFrame = pd.DataFrame(
        {
            "path": [f"image_{i}.jpg" for i in range(5)],
            "label": [f"class_{i}" for i in range(5)],
        }
    )

    result: pd.DataFrame = find_similar_images(
        image_index=0,
        embeddings=embeddings,
        metadata=metadata,
        k=4,
    )

    assert "image_0.jpg" not in result["path"].values


def test_similarities_are_sorted() -> None:
    # Verifies that results are sorted by similarity in descending order
    embeddings: np.ndarray = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],
            [0.7, 0.3],
            [0.0, 1.0],
        ]
    )

    metadata: pd.DataFrame = pd.DataFrame(
        {
            "path": [f"image_{i}.jpg" for i in range(5)],
            "label": [f"class_{i}" for i in range(5)],
        }
    )

    result: pd.DataFrame = find_similar_images(
        image_index=0,
        embeddings=embeddings,
        metadata=metadata,
        k=4,
    )

    similarities: np.ndarray = result["similarity"].to_numpy()

    assert np.all(similarities[:-1] >= similarities[1:])


def test_k_is_limited_to_maximum_number_of_results() -> None:
    # Verifies that k is capped to the maximum possible number of results
    embeddings: np.ndarray = np.eye(5)

    metadata: pd.DataFrame = pd.DataFrame(
        {
            "path": [f"image_{i}.jpg" for i in range(5)],
            "label": [f"class_{i}" for i in range(5)],
        }
    )

    result: pd.DataFrame = find_similar_images(
        image_index=0,
        embeddings=embeddings,
        metadata=metadata,
        k=100,
    )

    assert len(result) == 4


def test_raises_if_image_index_is_out_of_range() -> None:
    # Verifies that an invalid image index raises IndexError
    embeddings: np.ndarray = np.eye(5)

    metadata: pd.DataFrame = pd.DataFrame(
        {
            "path": [f"image_{i}.jpg" for i in range(5)],
            "label": [f"class_{i}" for i in range(5)],
        }
    )

    with pytest.raises(IndexError):
        find_similar_images(
            image_index=10,
            embeddings=embeddings,
            metadata=metadata,
            k=3,
        )


def test_raises_if_k_is_not_positive() -> None:
    # Verifies that a non-positive k raises ValueError
    embeddings: np.ndarray = np.eye(5)

    metadata: pd.DataFrame = pd.DataFrame(
        {
            "path": [f"image_{i}.jpg" for i in range(5)],
            "label": [f"class_{i}" for i in range(5)],
        }
    )

    with pytest.raises(ValueError):
        find_similar_images(
            image_index=0,
            embeddings=embeddings,
            metadata=metadata,
            k=0,
        )


def test_raises_if_embeddings_and_metadata_have_different_lengths() -> None:
    # Verifies that mismatched embeddings and metadata raise ValueError
    embeddings: np.ndarray = np.eye(5)

    metadata: pd.DataFrame = pd.DataFrame(
        {
            "path": ["image_1.jpg", "image_2.jpg"],
            "label": ["class_1", "class_2"],
        }
    )

    with pytest.raises(ValueError):
        find_similar_images(
            image_index=0,
            embeddings=embeddings,
            metadata=metadata,
            k=1,
        )


def test_raises_if_embeddings_has_only_one_sample() -> None:
    # Verifies that at least two embeddings are required
    embeddings: np.ndarray = np.array([[1.0, 0.0]])

    metadata: pd.DataFrame = pd.DataFrame(
        {
            "path": ["image.jpg"],
            "label": ["class"],
        }
    )

    with pytest.raises(ValueError):
        find_similar_images(
            image_index=0,
            embeddings=embeddings,
            metadata=metadata,
            k=1,
        )


def test_raises_if_embeddings_is_not_two_dimensional() -> None:
    # Verifies that embeddings must be a 2D array
    embeddings: np.ndarray = np.array([1.0, 2.0, 3.0])

    metadata: pd.DataFrame = pd.DataFrame(
        {
            "path": [
                "image_1.jpg",
                "image_2.jpg",
                "image_3.jpg",
            ],
            "label": [
                "class_1",
                "class_2",
                "class_3",
            ],
        }
    )

    with pytest.raises(ValueError):
        find_similar_images(
            image_index=0,
            embeddings=embeddings,
            metadata=metadata,
            k=1,
        )
