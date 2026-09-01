from pathlib import Path

# =============================================================================
# Model configuration
# =============================================================================

# CLIP
MODEL_NAME = "openai/clip-vit-base-patch32"
EMBEDDING_DIMENSION = 512

#DINOv2
# MODEL_NAME = "facebook/dinov2-base"
# EMBEDDING_DIMENSION = 768

# =============================================================================
# Directories
# =============================================================================

DATA_DIR = Path("data")

RAW_DATA_DIR = DATA_DIR / "raw"

if "dinov2" in MODEL_NAME.lower():
    PROCESSED_DATA_DIR = (
        DATA_DIR / "processed" / "cabinets_dinov2"
    )
elif "clip" in MODEL_NAME.lower():
    PROCESSED_DATA_DIR = (
        DATA_DIR / "processed" / "cabinets_clip"
    )
else:
    raise ValueError(
        f"Unsupported model: {MODEL_NAME}"
    )

# Frutas
# PROCESSED_DATA_DIR = DATA_DIR / "processed" / "fruits"

RESULTS_DIR = DATA_DIR / "results"


# =============================================================================
# Raw datasets
# =============================================================================

# Fruits
TRAIN_PATH = RAW_DATA_DIR / "fruits" / "train"
TEST_PATH = RAW_DATA_DIR / "fruits" / "test"

# Cabinets
CABINETS_PATH = RAW_DATA_DIR / "cabinets" / "images"


# =============================================================================
# Processed datasets - Fruits
# =============================================================================

FRUITS_PROCESSED_DATA_DIR = (
    DATA_DIR / "processed" / "fruits"
)

EMBEDDINGS_PATH = (
    FRUITS_PROCESSED_DATA_DIR / "embeddings.npy"
)

METADATA_PATH = (
    FRUITS_PROCESSED_DATA_DIR / "metadata.csv"
)

CLUSTERS_METADATA_PATH = (
    FRUITS_PROCESSED_DATA_DIR / "metadata_clusters.csv"
)

HDBSCAN_METADATA_PATH = (
    FRUITS_PROCESSED_DATA_DIR / "metadata_hdbscan.csv"
)


# =============================================================================
# Processed datasets - Cabinets
# =============================================================================

CABINETS_EMBEDDINGS_PATH = (
    PROCESSED_DATA_DIR / "embeddings.npy"
)

CABINETS_METADATA_PATH = (
    PROCESSED_DATA_DIR / "metadata.csv"
)

CABINETS_CLUSTERS_METADATA_PATH = (
    PROCESSED_DATA_DIR / "metadata_clusters.csv"
)

CABINETS_HDBSCAN_METADATA_PATH = (
    PROCESSED_DATA_DIR / "metadata_hdbscan.csv"
)

CABINETS_SUBCLUSTERS_METADATA_PATH = (
    PROCESSED_DATA_DIR / "metadata_subclusters.csv"
)

CABINETS_LABELLED_METADATA_PATH = (
    PROCESSED_DATA_DIR / "labelled_metadata.csv"
)

# =============================================================================
# K-Means configuration
# =============================================================================

DEFAULT_NUMBER_OF_CLUSTERS = 2
RANDOM_STATE = 42


# =============================================================================
# HDBSCAN configuration
# =============================================================================

MIN_CLUSTER_SIZE = 250
MIN_SAMPLES = 5


# =============================================================================
# Subclustering configuration
# =============================================================================

# Cluster obtained in the first clustering stage that contains the cabinets.
CABINET_CLUSTER = 1

# Number of subclusters to generate.
NUMBER_OF_SUBCLUSTERS = 2


# =============================================================================
# Image configuration
# =============================================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}


# =============================================================================
# UMAP visualization
# =============================================================================

UMAP_N_COMPONENTS = 2

FIGURE_SIZE = (15, 8)
SCATTER_POINT_SIZE = 10
COLORMAP = "gist_ncar"
LEGEND_FONT_SIZE = 6

SAVE_DPI = 300
SAVE_BBOX = "tight"

RESULTS_DATE_FORMAT = "%d-%m"



# =============================================================================
# Sampling
# =============================================================================


SAMPLE_SIZE = 300


# =============================================================================
# Cluster visualization
# =============================================================================

IMAGES_PER_CLUSTER = 50
GRID_COLUMNS = 10
