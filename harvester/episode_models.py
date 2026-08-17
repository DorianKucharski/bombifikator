from __future__ import annotations

from dataclasses import dataclass

from sharedkernel.utils.paths import slugify


@dataclass(frozen=True)
class Episode:
    video_id: str
    title: str
    url: str
    source_name: str

    @property
    def slug(self) -> str:
        return f"{slugify(self.title)}-{self.video_id}"


@dataclass(frozen=True)
class Shot:
    index: int
    start_seconds: float
    end_seconds: float

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds

    def sample_timestamps(self, stride_seconds: float) -> tuple[float, ...]:
        sample_count = max(1, int(self.duration_seconds // stride_seconds))
        step_seconds = self.duration_seconds / (sample_count + 1)
        return tuple(self.start_seconds + step_seconds * (position + 1) for position in range(sample_count))
