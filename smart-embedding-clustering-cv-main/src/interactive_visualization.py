from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px  # type: ignore[import-untyped]

from src.config import (
    CABINETS_EMBEDDINGS_PATH,
    CABINETS_CLUSTERS_METADATA_PATH,
    SAMPLE_SIZE,
)
from src.embeddings_io import (load_embeddings)
from src.embeddings_visualization import reduce_dimensions

from src.sampling import sample_embeddings_stratified


def interactive_plot(
    embeddings_2d: np.ndarray,
    metadata: pd.DataFrame,
    color_column: str | None = None,
    title: str = "Interactive UMAP Visualization",
    output_path: str | Path = Path("data/results/interactive_umap.html"),
    marker_size: int = 8,
) -> None:
    """
    Creates an interactive UMAP visualization using Plotly.

    Args:
        embeddings_2d (np.ndarray): Two-dimensional embeddings produced by UMAP.
        metadata (pd.DataFrame): Metadata associated with each embedding.
        color_column (str | None): Metadata column used to color the points.
        title (str): Plot title.
        output_path (str | Path): Path where the HTML visualization will be saved.
        marker_size (int): Marker size used in the scatter plot.

    Raises:
        ValueError:
            If the number of embeddings does not match the number of metadata rows.
        ValueError:
            If embeddings_2d does not contain exactly two columns.
        ValueError:
            If color_column is provided but does not exist in the metadata.

    Returns:
        None: The interactive visualization is displayed and saved as an HTML file.
    """
    if len(embeddings_2d) != len(metadata):
        raise ValueError(
            "The number of embeddings must match the number of metadata entries."
        )

    if embeddings_2d.ndim != 2:
        raise ValueError("embeddings_2d must be a 2D array")

    if embeddings_2d.shape[1] != 2:
        raise ValueError(
            "embeddings_2d must have exactly two columns."
        )

    if (
        not isinstance(marker_size, int)
        or isinstance(marker_size, bool)
        or marker_size <= 0
    ):
        raise ValueError(
            "marker_size must be a positive integer."
        )

    if (
        color_column is not None
        and color_column not in metadata.columns
    ):
        raise ValueError(
            f"Column '{color_column}' not found in metadata."
        )

    df = metadata.copy()

    df["x"] = embeddings_2d[:, 0]
    df["y"] = embeddings_2d[:, 1]


    fig = px.scatter(
        df,
        x="x",
        y="y",
        color=color_column,
        hover_data=[
            # color_column,
            # "label",
            "filename",
        ],
        title=title,
    )

    fig.update_traces(marker=dict(size=marker_size))

    fig.show()

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.write_html(output_path)

if __name__ == "__main__":

    embeddings, metadata = load_embeddings(
        CABINETS_EMBEDDINGS_PATH,
        CABINETS_CLUSTERS_METADATA_PATH,
    )

    USE_SAMPLE = False

    if USE_SAMPLE:
        embeddings, metadata = sample_embeddings_stratified(
            embeddings,
            metadata,
            SAMPLE_SIZE,
        )

    embeddings_2d = reduce_dimensions(embeddings)

    interactive_plot(
        embeddings_2d,
        metadata,
        color_column="cluster",
        title="Interactive Clusters Visualization - DINOv2",
    )
