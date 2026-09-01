# Smart Embedding Clustering for Computer Vision

## Overview

This repository demonstrates an end-to-end image clustering pipeline based on pretrained vision embeddings.

The workflow includes:
- loading fruit and cabinet image datasets,
- generating image embeddings with pretrained vision models,
- visualizing embeddings with UMAP,
- interactive visualization of embeddings,
- searching for visually similar images,
- clustering embeddings with K-Means and HDBSCAN,
- evaluating clustering quality (ARI, NMI and Purity),
- visualizing HDBSCAN clusters,
- saving processed outputs for later analysis.

## Key Features

### Image dataset loading

- Reads labeled fruit images from `data/raw/fruits/train/` and `data/raw/fruits/test/`, and unlabeled cabinet images from `data/raw/cabinets/images/`.
- Supports training data organized in class subfolders and unlabeled cabinet image collections.
- Loads images as `PIL.Image` objects and preserves metadata for downstream processing.

### Embedding extraction

- Uses pretrained vision models from Hugging Face.
- Generates vector embeddings for each image.
- Saves outputs to dataset-specific folders such as `data/processed/fruits/` and `data/processed/cabinets/`.
- Supports both CLIP and DINOv2 model families via `src.config.MODEL_NAME`.
- Stores embedding metadata alongside vectors in CSV.

### Embedding visualization

- Reduces embeddings to 2D with UMAP.
- Generates scatter plots for labels and cluster assignments.
- Saves visual results to the configured results directory.

### Similarity search

- Computes cosine similarity over image embeddings.
- Finds top-k visually similar images for a query image.
- Supports similarity search using the pretrained embedding space.

### Clustering

- Supports K-Means clustering of embeddings.
- Includes HDBSCAN-based analysis in `src/hdbscan_clustering.py`.
- Saves cluster assignments to metadata CSV files.
- External clustering evaluation using Adjusted Rand Index (ARI), Normalized Mutual Information (NMI) and Purity.

## Repository structure

```
smart-embedding-clustering-cv/
├── data/
│   ├── processed/
│   │   ├── cabinets/
│   │   │   ├── embeddings.npy
│   │   │   └── metadata.csv
│   │   ├── cabinets_clip/
│   │   │   ├── embeddings.npy
│   │   │   ├── labelled_metadata.csv
│   │   │   ├── metadata.csv
│   │   │   ├── metadata_clusters.csv
│   │   │   ├── metadata_hdbscan.csv
│   │   │   ├── metadata_subclusters_hdbscan.csv
│   │   │   └── metadata_subclusters_kmeans.csv
│   │   ├── cabinets_dinov2/
│   │   │   ├── embeddings.npy
│   │   │   ├── metadata.csv
│   │   │   └── metadata_clusters.csv
│   │   ├── fruits/
│   │   │   ├── embeddings.npy
│   │   │   ├── metadata.csv
│   │   │   ├── metadata_clusters.csv
│   │   │   └── metadata_hdbscan.csv
│   └── raw/
│       ├── sampleSubmission.csv
│       ├── cabinets/
│       │   └── images/
│       └── fruits/
│           ├── test/
│           └── train/
│               ├── Apple Braeburn/
│               ├── Apple Granny Smith/
│               └── ...
├── docs/
├── data/results/
│   ├── HDBSCAN_interactive_fruits_umap.html
│   ├── KMEANS_clip_interactive_cabinets.html
│   ├── KMEANS_dinov2_interactive_cabinets.html
│   ├── interactive_clip_cabinets_umap.html
│   ├── interactive_dinov2_cabinets_umap.html
│   ├── interactive_fruits_umap.html
│   ├── interactive_labeled_cabinets_umap.html
│   ├── 16-07/
│   ├── 20-07/
│   ├── 21-07/
│   ├── 22-07/
│   ├── 23-07/
│   ├── 24-07/
│   ├── 27-07/
│   └── 29-07/
├── src/
│   ├── clustering_external_metrics.py
│   ├── clustering_labeling.py
│   ├── clusters_visualization.py
│   ├── config.py
│   ├── dataset_loader.py
│   ├── embeddings_generator.py
│   ├── embeddings_io.py
│   ├── embeddings_visualization.py
│   ├── hdbscan_clustering.py
│   ├── hierarchical_clustering.py
│   ├── interactive_visualization.py
│   ├── kmeans_clustering.py
│   ├── kmeans_clustering_metrics.py
│   ├── main.py
│   ├── output_utils.py
│   ├── sampling.py
│   ├── similarity_search.py
│   └── __init__.py
├── tests/
│   ├── test_clustering_external_metrics.py
│   ├── test_clustering_labeling.py
│   ├── test_clusters_visualization.py
│   ├── test_dataset_loader.py
│   ├── test_embeddings_generator.py
│   ├── test_embeddings_visualization.py
│   ├── test_hdbscan_clustering.py
│   ├── test_interactive_visualization.py
│   ├── test_kmeans_clustering.py
│   ├── test_kmeans_clustering_metrics.py
│   ├── test_merge_training_conflict.py
│   ├── test_sampling.py
│   └── test_similarity_search.py
├── pyproject.toml
├── README.md
└── CONTRIBUTING.MD
```

## Dependencies

This project is built for Python 3.12.

Core dependencies are declared in `pyproject.toml`:

- `hdbscan`
- `matplotlib`
- `numpy`
- `pandas`
- `pillow`
- `plotly`
- `scikit-learn`
- `torch`
- `torchvision`
- `transformers`
- `umap-learn`

Dev dependencies:

- `mypy`
- `pandas-stubs`
- `pre-commit`
- `pytest`
- `ruff`

## Installation

```bash
git clone https://github.com/<username>/smart-embedding-clustering-cv.git
cd smart-embedding-clustering-cv
uv sync
```

## Usage

The repository does not include a production CLI yet, but the main modules are available for development and experimentation.

### Generate embeddings

There are two dataset flows:

- Fruits dataset
  - Labeled images are stored under `data/raw/fruits/train/` and `data/raw/fruits/test/`.
  - Use `src.dataset_loader.load_train_dataset()` to load the fruit training set.
  - Save outputs to `data/processed/fruits/`.

- Cabinets dataset
  - Unlabeled images are stored under `data/raw/cabinets/images/`.
  - Use `src.dataset_loader.load_unlabeled_dataset()` to load the cabinet dataset.
  - Save outputs to `data/processed/cabinets/`.

Run:

```bash
python -m src.embeddings_generator
```

The current `src/embeddings_generator.py` entry point defaults to loading cabinet images from `data/raw/cabinets/images/`. To process the fruit dataset instead, update the `__main__` block to load fruit training data and write outputs under `data/processed/fruits/`.

This creates `data/processed/<dataset>/embeddings.npy` and `data/processed/<dataset>/metadata.csv`.

### Visualize embeddings

```bash
python -m src.embeddings_visualization
```

### Run K-Means clustering

The current `src/kmeans_clustering.py` entry point is configured to load cabinet embeddings from `data/processed/cabinets/` by default.

```bash
python -m src.kmeans_clustering
```

### Run HDBSCAN clustering

```bash
python -m src.hdbscan_clustering
```

### Evaluate clustering

```bash
python -m src.clustering_external_metrics
```

### Visualize HDBSCAN clusters

```bash
python -m src.clusters_visualization
```

### Interactive visualization

```bash
python -m src.interactive_visualization
```

> Note: `src/main.py` currently prints a placeholder greeting and is not the primary pipeline entrypoint.

## Testing

Run the test suite with:

```bash
uv run pytest
```

## Notes

- The project uses pretrained vision embeddings to represent image content in a high-dimensional vector space.
- UMAP is used for 2D projection and visual analysis.
- The dataset is organized by dataset and class labels inside `data/raw/fruits/` and `data/raw/cabinets/`.
- Processed outputs and visualization results are stored in dataset-specific subfolders under `data/processed/` and `results/`.
- Clustering quality is evaluated using ARI, NMI and Purity.
- Interactive and static visualizations are provided for analysing labels and HDBSCAN clusters.


# Development Workflow

The project follows a Git workflow based on Pull Requests.

## Branch naming

Branches follow:

```
<type>/<short-description>
```

Examples:

```
feat/clip-embeddings
feat/similarity-search
fix/import-error
docs/update-readme
```

## Commit messages

Commits follow an imperative style:

Examples:

```
Add CLIP embedding extraction

Implement similarity search using cosine distance

Add K-Means clustering evaluation
```

All changes are reviewed through Pull Requests before being merged.

---

# Documentation

Main technologies documentation:

* Hugging Face Transformers
* CLIP model
* Scikit-learn
* UMAP
* PyTorch
* Pytest

---

# Contributors

Developed as part of a summer internship project.

---

# Developer

Francisca Sousa Coelho Nunes Pinheiro

---

# License

License information to be defined.
