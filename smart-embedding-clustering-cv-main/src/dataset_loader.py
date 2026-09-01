import logging
from pathlib import Path
from typing import TypedDict
from PIL import Image

from src.config import (
    TRAIN_PATH,
    TEST_PATH,
    IMAGE_EXTENSIONS,
    CABINETS_PATH
)

logger = logging.getLogger(__name__)

class TrainSample(TypedDict):
    path: Path
    filename: str
    label: str

class TestSample(TypedDict):
    path: Path
    filename: str

class UnlabeledSample(TypedDict):
    path: Path
    filename: str

def load_train_dataset(
    dataset_path: Path | str = TRAIN_PATH,
) -> list[TrainSample]:
    """
    Loads the training dataset from disk.

    Args:
        dataset_path (pathlib.Path): Path to the dataset directory.

    Returns:
        list[TrainSample]: A list of samples, where each sample contains the
        image path, filename, and class label.

    Raises:
        TypeError:
            If dataset_path is not a string or Path object.
        FileNotFoundError:
            If the dataset directory does not exist.
        NotADirectoryError:
            If dataset_path is not a directory.
        ValueError:
            If no supported image files are found.
    """
    if not isinstance(dataset_path, (str, Path)):
        raise TypeError(
            "dataset_path must be a string or Path object"
        )


    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if not dataset_path.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {dataset_path}")

    dataset: list[TrainSample] = []

    for class_folder in sorted(dataset_path.iterdir()):

        if not class_folder.is_dir():
            continue

        label = class_folder.name

        for image_path in sorted(class_folder.iterdir()):

            if image_path.suffix.lower() in IMAGE_EXTENSIONS:

                dataset.append(
                    {
                        "path": image_path,
                        "filename": image_path.name,
                        "label": label,
                    }
                )

    if not dataset:
        raise ValueError(f"No images found in {dataset_path}")

    return dataset


def load_test_dataset(
    dataset_path: Path | str = TEST_PATH,
) -> list[TestSample]:
    """
    Loads the test dataset from disk.

    Args:
        dataset_path (pathlib.Path): Path to the dataset directory.

    Returns:
        list[TestSample]: A list of samples containing the image path and
        filename for each test image.

    Raises:
        TypeError:
            If dataset_path is not a string or Path object.
        FileNotFoundError:
            If the dataset directory does not exist.
        NotADirectoryError:
            If dataset_path is not a directory.
        ValueError:
            If no supported image files are found.
    """
    if not isinstance(dataset_path, (str, Path)):
        raise TypeError(
            "dataset_path must be a string or Path object"
        )

    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if not dataset_path.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {dataset_path}")

    dataset: list[TestSample] = []

    for image_path in sorted(dataset_path.iterdir()):

        if image_path.suffix.lower() in IMAGE_EXTENSIONS:

            dataset.append(
                {
                    "path": image_path,
                    "filename": image_path.name,
                }
            )

    if not dataset:
        raise ValueError(f"No images found in {dataset_path}")

    return dataset


def load_image(
    image_path: Path | str,
) -> Image.Image:
    """
    Loads an image file and converts it to RGB.

    Args:
    image_path (str | pathlib.Path):
        Path to the image file.

    Returns:
        PIL.Image.Image: The loaded image converted to RGB mode.

    Raises:
        TypeError:
            If image_path is not a string or Path object.
        FileNotFoundError:
            If the image file does not exist.
        ValueError:
            If image_path does not refer to a file.
    """
    if not isinstance(image_path, (str, Path)):
        raise TypeError("image_path must be a string or Path object")

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not image_path.is_file():
        raise ValueError(f"Image path is not a file: {image_path}")

    with Image.open(image_path) as image:
        return image.convert("RGB")


def load_unlabeled_dataset(
    dataset_path: Path | str = CABINETS_PATH,
) -> list[UnlabeledSample]:
    """
    Loads an unlabeled image dataset from disk.

    Args:
        dataset_path (pathlib.Path): Path to the dataset directory.

    Returns:
        list[UnlabeledSample]: A list of samples containing the image path and
        filename for each unlabeled image.

    Raises:
        TypeError:
            If dataset_path is not a string or Path object.
        FileNotFoundError:
            If the dataset directory does not exist.
        NotADirectoryError:
            If dataset_path is not a directory.
        ValueError:
            If no supported image files are found.
    """

    if not isinstance(dataset_path, (str, Path)):
        raise TypeError("dataset_path must be a string or Path object")

    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if not dataset_path.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {dataset_path}")

    dataset: list[UnlabeledSample] = []

    for image_path in sorted(dataset_path.iterdir()):

        if image_path.suffix.lower() in IMAGE_EXTENSIONS:

            dataset.append(
                {
                    "path": image_path,
                    "filename": image_path.name,
                }
            )

    if not dataset:
        raise ValueError(f"No images found in {dataset_path}")

    return dataset

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    dataset = load_unlabeled_dataset()
    # dataset = load_train_dataset()
    # dataset = load_test_dataset()

    logger.info(f"Loaded {len(dataset)} cabinet images.")
    logger.info(dataset[:5])

    image = load_image(dataset[0]["path"])
    logger.info(f"First image size: {image.size}")

    sizes = {
        load_image(sample["path"]).size
        for sample in dataset
    }

    logger.info(f"Found {len(sizes)} different image sizes.")
    logger.info(sizes)
