from pathlib import Path

import pytest

from src.config import IMAGE_EXTENSIONS
from src.dataset_loader import (
    TestSample,
    TrainSample,
    load_test_dataset,
    load_train_dataset,
)


def test_train_dataset_not_empty() -> None:
    # Verifies that the training dataset is not empty
    train: list[TrainSample] = load_train_dataset()

    assert len(train) > 0


def test_test_dataset_not_empty() -> None:
    # Verifies that the test dataset is not empty
    test: list[TestSample] = load_test_dataset()

    assert len(test) > 0


# TESTES DO DATASET (ESPECÍFICOS)
def test_train_dataset_has_expected_size() -> None:
    # Verifies the expected number of training images
    train: list[TrainSample] = load_train_dataset()

    assert len(train) == 16854


def test_test_dataset_has_expected_size() -> None:
    # Verifies the expected number of test images
    test: list[TestSample] = load_test_dataset()

    assert len(test) == 5641


def test_train_dataset_has_33_classes() -> None:
    # Verifies that the training dataset contains 33 distinct classes
    train: list[TrainSample] = load_train_dataset()

    labels: set[str] = {sample["label"] for sample in train}

    assert len(labels) == 33


def test_train_dataset_has_no_duplicate_paths() -> None:
    # Verifies that training image paths are unique
    train: list[TrainSample] = load_train_dataset()

    paths: list[Path] = [sample["path"] for sample in train]

    assert len(paths) == len(set(paths))


def test_test_dataset_has_no_duplicate_paths() -> None:
    # Verifies that test image paths are unique
    test: list[TestSample] = load_test_dataset()

    paths: list[Path] = [sample["path"] for sample in test]

    assert len(paths) == len(set(paths))


def test_train_samples_have_expected_fields() -> None:
    # Verifies that training samples contain the expected fields
    train: list[TrainSample] = load_train_dataset()

    sample: TrainSample = train[0]

    assert "path" in sample
    assert "filename" in sample
    assert "label" in sample


def test_test_samples_have_expected_fields() -> None:
    # Verifies that test samples contain the expected fields
    test: list[TestSample] = load_test_dataset()

    sample: TestSample = test[0]

    assert "path" in sample
    assert "filename" in sample


def test_train_sample_field_types() -> None:
    # Verifies the data types of training sample fields
    train: list[TrainSample] = load_train_dataset()

    sample: TrainSample = train[0]

    assert isinstance(sample["path"], Path)
    assert isinstance(sample["filename"], str)
    assert isinstance(sample["label"], str)


def test_test_sample_field_types() -> None:
    # Verifies the data types of test sample fields
    test: list[TestSample] = load_test_dataset()

    sample: TestSample = test[0]

    assert isinstance(sample["path"], Path)
    assert isinstance(sample["filename"], str)


def test_train_image_paths_exist() -> None:
    # Verifies that all training image paths exist
    train: list[TrainSample] = load_train_dataset()

    assert all(sample["path"].exists() for sample in train)


def test_test_image_paths_exist() -> None:
    # Verifies that all test image paths exist
    test: list[TestSample] = load_test_dataset()

    assert all(sample["path"].exists() for sample in test)


def test_train_images_have_valid_extensions() -> None:
    # Verifies that all training images have valid file extensions
    train: list[TrainSample] = load_train_dataset()

    assert all(
        sample["path"].suffix.lower() in IMAGE_EXTENSIONS
        for sample in train
    )


def test_test_images_have_valid_extensions() -> None:
    # Verifies that all test images have valid file extensions
    test: list[TestSample] = load_test_dataset()

    assert all(
        sample["path"].suffix.lower() in IMAGE_EXTENSIONS
        for sample in test
    )


def test_train_dataset_raises_if_directory_does_not_exist(tmp_path: Path) -> None:
    # Verifies that a missing training dataset raises FileNotFoundError
    missing_directory: Path = tmp_path / "missing"

    with pytest.raises(FileNotFoundError):
        load_train_dataset(missing_directory)


def test_test_dataset_raises_if_directory_does_not_exist(tmp_path: Path) -> None:
    # Verifies that a missing test dataset raises FileNotFoundError
    missing_directory: Path = tmp_path / "missing"

    with pytest.raises(FileNotFoundError):
        load_test_dataset(missing_directory)
