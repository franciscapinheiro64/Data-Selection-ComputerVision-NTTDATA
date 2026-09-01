import numpy as np

from src.clustering_external_metrics import evaluate_clustering


def test_merge_training_conflict_main_version():
    labels_true = np.array(["a", "a", "b", "b"])
    labels_pred = np.array([0, 0, 1, 1])

    ari, nmi, purity = evaluate_clustering(labels_true, labels_pred)

    assert ari == 1.0
    assert nmi == 1.0
    assert purity == 1.0
