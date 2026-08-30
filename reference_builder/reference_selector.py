from __future__ import annotations

import numpy


def _pairwise_distances(embeddings: numpy.ndarray) -> numpy.ndarray:
    return numpy.linalg.norm(embeddings[:, None, :] - embeddings[None, :, :], axis=2)


def medoid_row(embeddings: numpy.ndarray) -> int:
    if len(embeddings) == 1:
        return 0
    return int(numpy.argmin(_pairwise_distances(embeddings).sum(axis=1)))


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
