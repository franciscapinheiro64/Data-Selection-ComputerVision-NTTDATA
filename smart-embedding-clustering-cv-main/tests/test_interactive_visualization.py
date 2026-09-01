from __future__ import annotations

import numpy as np
import pandas as pd

from typing import Any

import pytest
from src.interactive_visualization import interactive_plot


def test_interactive_plot_builds_expected_dataframe(monkeypatch: pytest.MonkeyPatch) -> None:
    # Verifies that the interactive Plotly figure is created correctly
    embeddings_2d: np.ndarray = np.array(
        [
            [0.1, 0.2],
            [0.3, 0.4],
        ]
    )
    metadata: pd.DataFrame = pd.DataFrame(
        {
            "label": ["Apple", "Banana"],
            "filename": ["a.jpg", "b.jpg"],
            "cluster": [0, 1],
        }
    )

    captured: dict[str, Any] = {}

    class DummyFigure:
        def update_traces(self, **kwargs):
            captured["update_traces"] = kwargs

        def show(self):
            captured["show"] = True

        def write_html(self, path):
            captured["html_path"] = path

    def fake_scatter(*args: Any, **kwargs: Any) -> DummyFigure:
        captured["dataframe"] = args[0]
        captured["kwargs"] = kwargs
        return DummyFigure()

    monkeypatch.setattr(
        "src.interactive_visualization.px.scatter",
        fake_scatter,
    )

    interactive_plot(
        embeddings_2d,
        metadata,
        color_column="cluster",
    )

    df = captured["dataframe"]

    assert list(df["x"]) == [0.1, 0.3]
    assert list(df["y"]) == [0.2, 0.4]
    assert list(df["cluster"]) == [0, 1]

    assert captured["kwargs"]["color"] == "cluster"
    assert captured["kwargs"]["hover_data"] == [
        "filename",
    ]

    assert captured["show"] is True
    assert "html_path" in captured


def test_interactive_plot_raises_if_lengths_do_not_match() -> None:
    # Verifies that embeddings and metadata must have the same length
    embeddings_2d: np.ndarray = np.random.rand(2, 2)

    metadata: pd.DataFrame = pd.DataFrame(
        {
            "filename": ["a.jpg"],
        }
    )

    with pytest.raises(ValueError):
        interactive_plot(
            embeddings_2d,
            metadata,
        )


def test_interactive_plot_raises_if_embeddings_are_not_two_dimensional() -> None:
    # Verifies that embeddings must have exactly two columns
    embeddings_2d: np.ndarray = np.random.rand(5, 3)

    metadata: pd.DataFrame = pd.DataFrame(
        {
            "filename": [f"{i}.jpg" for i in range(5)],
        }
    )

    with pytest.raises(ValueError):
        interactive_plot(
            embeddings_2d,
            metadata,
        )


def test_interactive_plot_raises_if_color_column_does_not_exist() -> None:
    # Verifies that the selected color column exists
    embeddings_2d: np.ndarray = np.random.rand(5, 2)

    metadata: pd.DataFrame = pd.DataFrame(
        {
            "filename": [f"{i}.jpg" for i in range(5)],
        }
    )

    with pytest.raises(ValueError):
        interactive_plot(
            embeddings_2d,
            metadata,
            color_column="cluster",
        )
