from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from reference_builder.crop_store import CropStore
from reference_builder.embedding_store import save_embeddings
from reference_builder.reference_config import EmbeddingConfig, ReferencePaths
from reference_builder.torch_device import resolve_device
from sharedkernel.logger import get_logger

LOGGER = get_logger("reference_builder.embedding_extractor")

_CLASS_TOKEN_INDEX = 0
_PROGRESS_INTERVAL = 2000


@dataclass(frozen=True)
class EmbeddingReport:
    crops_embedded: int
    crops_unreadable: int
    dimensions: int


def _to_pil(crop_path: Path) -> Image.Image | None:
    image = cv2.imread(str(crop_path))
    if image is None or image.size == 0:
        return None
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def _l2_normalized(embeddings: numpy.ndarray) -> numpy.ndarray:
    norms = numpy.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / numpy.maximum(norms, 1e-12)


class CropEmbedder:
    def __init__(self, config: EmbeddingConfig) -> None:
        self._device = resolve_device(config.device)
        self._processor = AutoImageProcessor.from_pretrained(config.model_id)
        self._model = AutoModel.from_pretrained(config.model_id).to(self._device).eval()
        LOGGER.info("embedder loaded: model=%s device=%s", config.model_id, self._device)

    @torch.inference_mode()
    def embed_batch(self, images: Sequence[Image.Image]) -> numpy.ndarray:
        inputs = self._processor(images=list(images), return_tensors="pt").to(self._device)
        hidden_states = self._model(**inputs).last_hidden_state
        return hidden_states[:, _CLASS_TOKEN_INDEX].float().cpu().numpy()


def build_embeddings(paths: ReferencePaths, config: EmbeddingConfig) -> EmbeddingReport:
    store = CropStore(paths.crops_dir)
    crop_ids = tuple(record.crop_id for record in store.load_all())
    if not crop_ids:
        raise ValueError(f"no crops in {paths.crops_dir}: run 'references detect' first")

    embedder = CropEmbedder(config)
    embedded_ids: list[str] = []
    vectors: list[numpy.ndarray] = []
    unreadable = 0
    for start in range(0, len(crop_ids), config.batch_size):
        batch_ids = crop_ids[start:start + config.batch_size]
        images = [(crop_id, _to_pil(store.crop_path(crop_id))) for crop_id in batch_ids]
        readable = [(crop_id, image) for crop_id, image in images if image is not None]
        unreadable += len(images) - len(readable)
        if not readable:
            continue
        vectors.append(embedder.embed_batch([image for _, image in readable]))
        embedded_ids.extend(crop_id for crop_id, _ in readable)
        if len(embedded_ids) % _PROGRESS_INTERVAL < config.batch_size:
            LOGGER.info("embedding progress: done=%d total=%d", len(embedded_ids), len(crop_ids))

    embeddings = _l2_normalized(numpy.vstack(vectors).astype(numpy.float32))
    save_embeddings(paths.embeddings_path, tuple(embedded_ids), embeddings)
    return EmbeddingReport(
            crops_embedded=len(embedded_ids),
            crops_unreadable=unreadable,
            dimensions=int(embeddings.shape[1]),
    )
