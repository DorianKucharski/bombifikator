from __future__ import annotations

import numpy
from sklearn.cluster import AgglomerativeClustering


def _pairwise_distances(embeddings: numpy.ndarray) -> numpy.ndarray:
    squared_norms = numpy.einsum("ij,ij->i", embeddings, embeddings)
    squared_distances = squared_norms[:, None] + squared_norms[None, :] - 2.0 * (embeddings @ embeddings.T)
    return numpy.sqrt(numpy.maximum(squared_distances, 0.0))


def medoid_row(embeddings: numpy.ndarray) -> int:
    if len(embeddings) == 1:
        return 0
    return int(numpy.argmin(_pairwise_distances(embeddings).sum(axis=1)))


def dominant_look_rows(embeddings: numpy.ndarray, look_distance_threshold: float) -> tuple[int, ...]:
    every_row = tuple(range(len(embeddings)))
    if len(embeddings) <= 2:
        return every_row
    looks = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=look_distance_threshold,
            metric="cosine",
            linkage="average",
    ).fit_predict(embeddings)
    dominant_look = numpy.argmax(numpy.bincount(looks))
    return tuple(int(row) for row in numpy.flatnonzero(looks == dominant_look))


def select_diverse_rows(embeddings: numpy.ndarray, count: int, seed_row: int) -> tuple[int, ...]:
    if count <= 0 or len(embeddings) == 0:
        return ()
    selected = [seed_row]
    distances_to_selected = numpy.linalg.norm(embeddings - embeddings[seed_row], axis=1)
    while len(selected) < min(count, len(embeddings)):
        distances_to_selected[selected] = -1.0
        farthest_row = int(numpy.argmax(distances_to_selected))
        selected.append(farthest_row)
        distances_to_selected = numpy.minimum(
                distances_to_selected,
                numpy.linalg.norm(embeddings - embeddings[farthest_row], axis=1),
        )
    return tuple(selected)
