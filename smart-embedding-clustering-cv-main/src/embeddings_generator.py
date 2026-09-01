import logging
from pathlib import Path
from typing import Any, Sequence, TypedDict, cast

import numpy as np
import pandas as pd
import torch
from PIL.Image import Image
from transformers import (
    CLIPProcessor,
    CLIPVisionModelWithProjection,
    AutoImageProcessor,
    AutoModel,
)

from src.config import FRUITS_PROCESSED_DATA_DIR, MODEL_NAME
from src.dataset_loader import (
    TrainSample,
    UnlabeledSample,
    load_image,
    load_train_dataset,
)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Processor = Any
VisionModel = Any

class MetadataSample(TypedDict):
    path: str
    filename: str
    label: str | None


# Use GPU if available; otherwise use CPU.
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)



def load_model() -> tuple[Processor, VisionModel]:
    """
    Loads the image processor and pretrained vision model.
    Currently supported models:
        - OpenAI CLIP
        - DINOv2

    Returns:
        tuple[Processor, VisionModel]: A tuple containing the processor and the loaded model
        for the configured model name.

    Raises:
        ValueError: If the configured model is not supported.
    """

    if MODEL_NAME.startswith("openai/clip"):
        processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        model = CLIPVisionModelWithProjection.from_pretrained(
            MODEL_NAME
        )

    elif "dinov2" in MODEL_NAME.lower():
        processor = AutoImageProcessor.from_pretrained(
            MODEL_NAME
        )
        model = AutoModel.from_pretrained(
            MODEL_NAME
        )

    else:
        raise ValueError(
            f"Unsupported model: {MODEL_NAME}"
        )

    model.to(DEVICE)  # type: ignore[arg-type]
    model.eval()

    return processor, model


def generate_embedding(
    image: Image,
    processor: Processor,
    model: VisionModel,
) -> np.ndarray:
    """
    Generates a feature embedding for a single image.

    Args:
        image (PIL.Image.Image): Input image.
        processor (Processor): Image processor associated with the configured model.
        model (VisionModel): Pretrained vision model used to generate embeddings.

    Returns:
        np.ndarray: A one-dimensional embedding vector for the input image.

    Raises:
        ValueError: If the configured model is not supported.
    """

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        if MODEL_NAME.startswith("openai/clip"):
            outputs = model(
                pixel_values=inputs["pixel_values"],
            )
            embedding = outputs.image_embeds

        elif "dinov2" in MODEL_NAME.lower():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0]

        else:
            raise ValueError(
                f"Unsupported model: {MODEL_NAME}"
            )

    return embedding.squeeze().cpu().numpy()


def generate_embeddings(
    dataset: Sequence[TrainSample | UnlabeledSample],
    processor: Processor,
    model: VisionModel,
) -> tuple[list[np.ndarray], list[MetadataSample]]:
    """
    Generates embeddings for all images in a dataset.

    Args:
        dataset (list[TrainSample | UnlabeledSample]): List of image metadata.
        processor (Processor): Image processor associated with the configured model.
        model (VisionModel): Pretrained vision model used to generate embeddings.

    Returns:
        tuple[list[np.ndarray], list[MetadataSample]]: A tuple containing the
        list of generated embeddings and the metadata associated with each one.

    Raises:
        ValueError:
            If dataset is empty.
        ValueError:
            If a dataset sample does not contain both "path" and "filename".
    """

    if not dataset:
        raise ValueError("dataset must contain at least one sample")

    embeddings: list[np.ndarray] = []
    metadata: list[MetadataSample] = []

    for index, sample in enumerate(dataset):
        if "path" not in sample or "filename" not in sample:
            raise ValueError("Each dataset sample must contain 'path' and 'filename'")
        if (index + 1) % 100 == 0:
            logger.info(f"{index + 1}/{len(dataset)} processed images")

        image = load_image(sample["path"])

        embedding = generate_embedding(
            image,
            processor,
            model,
        )

        embeddings.append(embedding)

        if "label" in sample:
            label = cast(TrainSample, sample)["label"]
        else:
            label = None

        metadata_sample: MetadataSample = {
            "path": str(sample["path"]),
            "filename": sample["filename"],
            "label": label,
        }

        metadata.append(metadata_sample)

    return embeddings, metadata


def save_embeddings(
    embeddings: list[np.ndarray] | np.ndarray,
    metadata: list[MetadataSample],
    output_dir: str | Path,
) -> None:
    """
    Saves generated embeddings and metadata to disk.

    Args:
        embeddings (list[np.ndarray] | np.ndarray): Image embeddings.
        metadata (list[MetadataSample]): Metadata associated with each embedding.
        output_dir (str | pathlib.Path): Directory where the files will be saved.

    Returns:
        None: The embeddings are written to embeddings.npy and metadata to metadata.csv.

    Raises:
        ValueError: If embeddings or metadata are empty, or if their lengths do not match.
    """
    if len(embeddings) == 0:
        raise ValueError("embeddings and metadata must not be empty")

    if len(embeddings) != len(metadata):
        raise ValueError("embeddings and metadata must have the same length")

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    embeddings_array = np.array(embeddings)

    np.save(
        output_dir / "embeddings.npy",
        embeddings_array,
    )

    metadata_df = pd.DataFrame(metadata)

    metadata_df.to_csv(
        output_dir / "metadata.csv",
        index=False,
    )


if __name__ == "__main__":

    processor, model = load_model()

    # dataset = load_test_dataset()
    dataset = load_train_dataset()
    # dataset = load_unlabeled_dataset()

    logger.info(f"Loaded {len(dataset)} images.")
    logger.info(f"First image: {dataset[0]['path']}")

    embeddings, metadata_list = generate_embeddings(
        dataset,
        processor,
        model,
    )

    save_embeddings(
        embeddings,
        metadata_list,
        # PROCESSED_DATA_DIR,
        FRUITS_PROCESSED_DATA_DIR,
    )

    logger.info("Embeddings saved!")
