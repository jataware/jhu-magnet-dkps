"""Read response embeddings exported from a trusted local DKPS cache."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np


def response_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


class CachedResponseEmbeddings:
    """Reuse embeddings only; scores and fitted predictions are never stored here."""

    def __init__(self, directory, provider, model):
        self.directory = Path(directory).resolve()
        self.index_path = self.directory / "index.json"
        self.index = json.loads(self.index_path.read_text())
        if self.index["provider"] != provider or self.index["model"] != model:
            raise ValueError(
                "Cached embedding provider/model does not match the evaluation"
            )
        if self.index["format_version"] != 1:
            raise ValueError("Unsupported response embedding cache format")
        self._chunk = lru_cache(maxsize=128)(self._load_chunk)

    def _load_chunk(self, name):
        path = (self.directory / "chunks" / name).resolve()
        if not path.is_relative_to(self.directory):
            raise ValueError("Embedding chunk path escapes the cache directory")
        return np.load(path, mmap_mode="r", allow_pickle=False)

    def _record(self, panel, model, item_index):
        item_id = json.loads(panel.item_ids[item_index])[1]
        try:
            record = self.index["datasets"][panel.dataset][model][item_id]
        except KeyError as error:
            raise ValueError(
                f"Missing cached embedding: {panel.dataset}/{model}/{item_id}"
            ) from error
        if record["response_sha256"] != response_hash(
            panel.responses[model][item_index]
        ):
            raise ValueError(
                f"Cached response differs from HELM: {panel.dataset}/{model}/{item_id}"
            )
        return record

    def validate(self, panels):
        """Check coverage and chunk integrity before starting any DKPS fits."""
        from dkps.helm import uses_onehot

        chunks = set()
        for panel in panels:
            if uses_onehot(panel.dataset):
                continue
            for model in panel.models:
                for index in range(panel.pool_size):
                    record = self._record(panel, model, index)
                    chunks.add(record["chunk"])
                    chunk = self._chunk(record["chunk"])
                    if chunk.ndim != 2 or chunk.shape[1] != self.index["dimensions"]:
                        raise ValueError("Invalid cached embedding shape")
                    if not 0 <= record["row"] < len(chunk):
                        raise ValueError("Invalid cached embedding row")
        for name in sorted(chunks):
            chunk = self._chunk(name)
            if not np.isfinite(chunk).all():
                raise ValueError(f"Non-finite embeddings in {name}")
            path = self.directory / "chunks" / name
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != self.index["chunk_sha256"][name]:
                raise ValueError(f"Embedding chunk checksum mismatch: {name}")

    def embed_queries(self, panel, indices):
        result = {}
        for model in panel.models:
            vectors = []
            for index in indices:
                record = self._record(panel, model, index)
                vectors.append(self._chunk(record["chunk"])[record["row"]])
            result[model] = np.stack(vectors)[:, None, :]
        return result
