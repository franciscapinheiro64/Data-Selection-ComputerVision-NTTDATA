import numpy as np
import pytest

from src.clustering_external_metrics import (
    evaluate_clustering,
    _print_results,
    purity_score,
)


def test_purity_score_returns_expected_value() -> None:
    # Verifies that purity is computed correctly for a perfect clustering.
    true_labels: np.ndarray = np.array(
        ["apple", "apple", "banana", "banana"]
    )

    predicted_labels: np.ndarray = np.array(
        [0, 0, 1, 1]
    )

    purity: float = purity_score(
        true_labels,
        predicted_labels,
    )

    assert purity == 1.0


def test_evaluate_clustering_returns_three_metrics() -> None:
    # Verifies that ARI, NMI and purity are correctly computed.
    true_labels: np.ndarray = np.array(
        ["apple", "apple", "banana", "banana"]
    )

    predicted_labels: np.ndarray = np.array(
        [0, 0, 1, 1]
    )

    ari: float
    nmi: float
    purity: float
    ari, nmi, purity = evaluate_clustering(
        true_labels,
        predicted_labels,
    )

    assert ari == 1.0
    assert nmi == 1.0
    assert purity == 1.0


def test_print_results_outputs_summary(capsys: pytest.CaptureFixture[str]) -> None:
    # Verifies that the evaluation summary is printed correctly.
    _print_results(
        "K-Means",
        0.5,
        0.6,
        0.7,
    )

    captured = capsys.readouterr()

    assert "K-Means" in captured.out
    assert "ARI: 0.5000" in captured.out
    assert "NMI: 0.6000" in captured.out
    assert "Purity: 0.7000" in captured.out


def test_purity_score_raises_for_empty_labels() -> None:
    # Verifies that empty label arrays raise a ValueError.
    true_labels: np.ndarray = np.array([])
    predicted_labels: np.ndarray = np.array([])

    with pytest.raises(
        ValueError,
        match="Labels must not be empty.",
    ):
        purity_score(
            true_labels,
            predicted_labels,
        )


def test_purity_score_raises_for_different_lengths() -> None:
    # Verifies that both label arrays must have the same length.
    true_labels: np.ndarray = np.array(
        ["apple", "banana"]
    )

    predicted_labels: np.ndarray = np.array([0])

    with pytest.raises(
        ValueError,
        match="true_labels and predicted_labels must have the same length.",
    ):
        purity_score(
            true_labels,
            predicted_labels,
        )


def test_evaluate_clustering_raises_for_empty_labels() -> None:
    # Verifies that empty label arrays raise a ValueError.
    true_labels: np.ndarray = np.array([])
    predicted_labels: np.ndarray = np.array([])

    with pytest.raises(
        ValueError,
        match="Labels must not be empty.",
    ):
        evaluate_clustering(
            true_labels,
            predicted_labels,
        )


def test_evaluate_clustering_raises_for_different_lengths() -> None:
    # Verifies that both label arrays must have the same length.
    true_labels: np.ndarray = np.array(
        ["apple", "banana"]
    )

    predicted_labels: np.ndarray = np.array([0])

    with pytest.raises(
        ValueError,
        match="true_labels and predicted_labels must have the same length.",
    ):
        evaluate_clustering(
            true_labels,
            predicted_labels,
        )
