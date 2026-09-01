from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image
from pytest import MonkeyPatch

import src.clusters_visualization as clusters_visualization
from src.output_utils import RESULTS_DATE_FORMAT



def test_get_output_dir_creates_directory_for_date(tmp_path: Path):
    # Verifies the output directory is created for the given date.
    target_date: datetime = datetime(2026, 8, 3)

    output_dir: Path = clusters_visualization.get_output_dir(
        base_dir=tmp_path,
        date=target_date,
    )

    assert output_dir.exists()
    assert output_dir == tmp_path / target_date.strftime(
        RESULTS_DATE_FORMAT,
    )


def test_load_cluster_metadata_raises_file_not_found(tmp_path: Path):
    # Verifies that loading missing metadata raises FileNotFoundError.
    missing_file: Path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        clusters_visualization.load_cluster_metadata(missing_file)


def test_load_cluster_metadata_raises_for_empty_file(tmp_path: Path):
    # Verifies that an empty metadata file raises ValueError.
    empty_file: Path = tmp_path / "metadata.csv"
    empty_file.write_text("path,cluster\n")

    with pytest.raises(ValueError, match="The metadata file is empty"):
        clusters_visualization.load_cluster_metadata(empty_file)


def test_load_image_raises_file_not_found(tmp_path: Path):
    # Verifies that loading a missing image raises FileNotFoundError.
    missing_image: Path = tmp_path / "missing.jpg"

    with pytest.raises(FileNotFoundError):
        clusters_visualization.load_image(missing_image)


def test_load_image_loads_rgb_image(tmp_path: Path):
    # Verifies that load_image returns a valid RGB image.
    image_path: Path = tmp_path / "img.png"
    image: Image.Image = Image.new("RGB", (1, 1), color=(255, 0, 0))
    image.save(image_path)

    result: Image.Image = clusters_visualization.load_image(image_path)

    assert result.mode == "RGB"
    assert result.size == (1, 1)


def test_visualize_cluster_requires_positive_images_per_cluster(tmp_path: Path):
    # Verifies that an invalid images_per_cluster value raises ValueError.
    metadata = pd.DataFrame(
        {
            "path": [tmp_path / "img1.png"],
            "cluster": [0],
        }
    )

    with pytest.raises(ValueError):
        clusters_visualization.visualize_cluster(
            metadata,
            cluster=0,
            images_per_cluster=0,
        )


def test_visualize_cluster_saves_image_file(tmp_path: Path, monkeypatch: MonkeyPatch):
    # Verifies that visualize_cluster writes the cluster image file.
    image_path: Path = tmp_path / "img1.png"
    image: Image.Image = Image.new("RGB", (1, 1), color=(255, 0, 0))
    image.save(image_path)

    metadata = pd.DataFrame(
        {
            "path": [image_path],
            "cluster": [0],
        }
    )

    monkeypatch.setattr(
        clusters_visualization,
        "get_output_dir",
        lambda *args, **kwargs: tmp_path,
    )

    clusters_visualization.visualize_cluster(
        metadata,
        cluster=0,
        images_per_cluster=1,
    )

    expected_file: Path = tmp_path / "cluster_0.png"
    assert expected_file.exists()


def test_visualize_all_clusters_calls_visualize_cluster_for_each_cluster(monkeypatch: MonkeyPatch):
    # Verifies that visualize_all_clusters iterates over every unique cluster.
    metadata = pd.DataFrame(
        {
            "path": ["img1.png", "img2.png"],
            "cluster": [0, 1],
        }
    )

    calls: list[int] = []

    def fake_visualize_cluster(
        metadata_arg: pd.DataFrame,
        cluster: int,
        cluster_column: str = "cluster",
        **kwargs,
    ) -> None:
        calls.append(cluster)

    monkeypatch.setattr(
        clusters_visualization,
        "visualize_cluster",
        fake_visualize_cluster,
    )

    clusters_visualization.visualize_all_clusters(metadata)

    assert calls == [0, 1]
