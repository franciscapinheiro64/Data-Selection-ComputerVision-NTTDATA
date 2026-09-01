import logging
import matplotlib.pyplot as plt
import numbers
import math
from pathlib import Path
import pandas as pd
from PIL import Image
from src.config import (
    CABINETS_HDBSCAN_METADATA_PATH,
    FIGURE_SIZE,
    IMAGES_PER_CLUSTER,
    GRID_COLUMNS,
    SAVE_BBOX,
    SAVE_DPI,
)
from src.output_utils import get_output_dir
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def load_cluster_metadata(
    metadata_path: str | Path,
) -> pd.DataFrame:
    """
    Loads clustering metadata from disk.

    Args:
        metadata_path (str | pathlib.Path): Path to the metadata file.

    Returns:
        pd.DataFrame: Metadata containing image paths, labels, and cluster
        assignments.

    Raises:
        FileNotFoundError:
            If the metadata file does not exist.
        ValueError:
            If the metadata file is empty.
    """
    metadata_path = Path(metadata_path)

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}"
        )

    metadata = pd.read_csv(metadata_path)

    if metadata.empty:
        raise ValueError(
            "The metadata file is empty."
        )

    return metadata


def load_image(
    image_path: str | Path,
) -> Image.Image:
    """
    Loads an image from disk.

    Args:
        image_path (str | pathlib.Path): Path to the image.

    Returns:
        PIL.Image.Image: The loaded image converted to RGB mode.

    Raises:
        FileNotFoundError:
            If the image file does not exist.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    with Image.open(image_path) as image:
        return image.convert("RGB")


def visualize_cluster(
    metadata: pd.DataFrame,
    cluster: int,
    cluster_column: str = "cluster",
    images_per_cluster: int = IMAGES_PER_CLUSTER,
) -> None:
    """
    Creates and saves a grid with representative images for a cluster.

    Args:
        metadata (pd.DataFrame): Metadata containing image paths and cluster labels.
        cluster (int): Cluster identifier to visualize.
        cluster_column (str): Metadata column containing the cluster assignments.
        images_per_cluster (int): Maximum number of images to display.

    Returns:
        None: The visualization is saved to disk.

    Raises:
        TypeError:
            If metadata is not a pandas DataFrame.
        TypeError:
            If cluster is not an integer.
        ValueError:
            If images_per_cluster is not a positive integer.
        ValueError:
            If metadata does not contain the specified cluster column.
        ValueError:
            If metadata does not contain a 'path' column.
        ValueError:
            If metadata is empty.
    """

    if not isinstance(metadata, pd.DataFrame):
        raise TypeError("metadata must be a pandas DataFrame")

    if not isinstance(cluster, numbers.Integral) or isinstance(cluster, bool):
        raise TypeError("cluster must be an integer.")

    if not isinstance(images_per_cluster, int) or isinstance(images_per_cluster, bool) or images_per_cluster <= 0:
        raise ValueError("images_per_cluster must be a positive integer")

    if cluster_column not in metadata.columns:
        raise ValueError(
            f"Metadata must contain a '{cluster_column}' column."
        )

    if "path" not in metadata.columns:
        raise ValueError(
            "Metadata must contain a 'path' column."
        )

    if metadata.empty:
        raise ValueError(
            "Metadata is empty."
        )

    columns = GRID_COLUMNS
    cluster_metadata = metadata[
        metadata[cluster_column] == cluster
    ].head(images_per_cluster)

    if cluster_metadata.empty:
        return

    rows = math.ceil(
        len(cluster_metadata) / columns
    )

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=FIGURE_SIZE,
    )

    axes = axes.flatten()

    for axis, (_, row) in zip(
        axes,
        cluster_metadata.iterrows(),
    ):

        image = load_image(
            row["path"],
        )

        axis.imshow(image)

        if "label" in metadata.columns:
            axis.set_title(
                row["label"],
                fontsize=8,
            )

        axis.axis("off")

    for axis in axes[len(cluster_metadata):]:
        axis.axis("off")

    plt.suptitle(
        f"Cluster {cluster}",
        fontsize=14,
    )

    plt.tight_layout()

    output_dir = get_output_dir()

    if cluster == -1:
        filename = f"{cluster_column}_noise.png"
    else:
        filename = f"{cluster_column}_{cluster}.png"

    plt.savefig(
        output_dir / filename,
        dpi=SAVE_DPI,
        bbox_inches=SAVE_BBOX,
    )

    plt.close()


def visualize_all_clusters(
    metadata: pd.DataFrame,
    cluster_column: str = "cluster",
) -> None:
    """
    Creates and saves one visualization for each cluster.

    Args:
        metadata (pd.DataFrame): Metadata containing image paths and cluster labels.
        cluster_column (str): Metadata column containing the cluster assignments.

    Returns:
        None.

    Raises:
        TypeError:
            If metadata is not a pandas DataFrame.
        ValueError:
            If metadata is empty.
        ValueError:
            If metadata does not contain the specified cluster column.
    """
    if not isinstance(metadata, pd.DataFrame):
        raise TypeError(
            "metadata must be a pandas DataFrame."
        )

    if metadata.empty:
        raise ValueError(
            "Metadata is empty."
        )

    if cluster_column not in metadata.columns:
        raise ValueError(
            f"Metadata must contain a '{cluster_column}' column."
        )

    clusters = sorted(
        metadata[cluster_column].unique(),
    )


    for cluster in clusters:

        logger.info(
            f"Cluster {cluster}"
        )

        visualize_cluster(
            metadata,
            cluster,
            cluster_column=cluster_column,
        )



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    metadata = load_cluster_metadata(
        CABINETS_HDBSCAN_METADATA_PATH,
    )

    logger.info(metadata.shape)
    logger.info(metadata.columns)

    logger.info("Generating cluster visualizations...\n")

    visualize_all_clusters(
        metadata,
        cluster_column="cluster",
    )

    # For HDBSCAN
    # noise = metadata[metadata["cluster"] == -1]
    # logger.info(noise["label"].value_counts().head(10))

    logger.info("\nVisualizations saved successfully.")
