import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from jhu_ta1.magnet._embedding_cache import CachedResponseEmbeddings
from jhu_ta1.magnet._helm_pair_data import Panel
from jhu_ta1.magnet.precompute_embeddings import (
    DEFAULT_MODEL,
    PROVIDER,
    precompute,
)

DIMENSIONS = 4


class FakeEmbedder:
    """Deterministic vectors that depend only on the text, like a real embedder."""

    def __init__(self, fail_after=None):
        self.calls = []
        self.fail_after = fail_after

    def embed(self, texts, model):
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise RuntimeError("interrupted")
        self.calls.append(list(texts))
        return np.array(
            [
                np.frombuffer(hashlib.sha256(t.encode()).digest()[:DIMENSIONS], "u1")
                for t in texts
            ],
            dtype=np.float32,
        ).reshape(len(texts), DIMENSIONS)


def make_panel(dataset, *, models=4, items=3, tag="a"):
    names = [f"provider{i}/model" for i in range(models)]
    return Panel(
        dataset=dataset,
        metric="accuracy",
        models=names,
        item_ids=[json.dumps(["test", f"id{i}"]) for i in range(items)],
        responses={m: [f"{tag} {m} response {i}" for i in range(items)] for m in names},
        references=[[] for _ in range(items)],
        scores={m: np.zeros(items) for m in names},
        sources=[],
        original_pool_sizes=dict.fromkeys(names, items),
    )


class PrecomputeTests(unittest.TestCase):
    def run_precompute(self, cache, panels, embedder, **kwargs):
        manifest = {
            "datasets": [{"dataset": p, "metric": "accuracy"} for p in panels]
        }

        def load(suite, entry):
            return panels[entry["dataset"]]

        with patch("jhu_ta1.magnet.precompute_embeddings.load_panel", load):
            return precompute(
                "unused",
                manifest,
                cache,
                lambda: embedder,
                lambda name: name.startswith("legalbench"),
                chunk_size=5,
                **kwargs,
            )

    def test_cache_validates_and_returns_the_embedded_vectors(self):
        with tempfile.TemporaryDirectory() as tmp:
            panel = make_panel("math:subject=a")
            embedder = FakeEmbedder()
            summary = self.run_precompute(tmp, {panel.dataset: panel}, embedder)
            self.assertEqual(summary["embedded"], [panel.dataset])

            cache = CachedResponseEmbeddings(tmp, PROVIDER, DEFAULT_MODEL)
            cache.validate([panel])
            actual = cache.embed_queries(panel, [2, 0])
            for model in panel.models:
                expected = embedder.embed(
                    [panel.responses[model][2], panel.responses[model][0]], "x"
                )
                np.testing.assert_array_equal(actual[model][:, 0, :], expected)
                self.assertEqual(actual[model].shape, (2, 1, DIMENSIONS))

    def test_rerun_skips_and_new_datasets_extend_the_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = make_panel("math:subject=a")
            second = make_panel("math:subject=b", tag="b")
            panels = {first.dataset: first, second.dataset: second}

            self.run_precompute(tmp, panels, FakeEmbedder(), datasets=[first.dataset])
            again = FakeEmbedder()
            summary = self.run_precompute(
                tmp, panels, again, datasets=[first.dataset]
            )
            self.assertEqual(summary["cached"], [first.dataset])
            self.assertEqual(again.calls, [])

            extend = FakeEmbedder()
            summary = self.run_precompute(tmp, panels, extend)
            self.assertEqual(summary["cached"], [first.dataset])
            self.assertEqual(summary["embedded"], [second.dataset])
            self.assertEqual(sum(len(c) for c in extend.calls), 12)

            cache = CachedResponseEmbeddings(tmp, PROVIDER, DEFAULT_MODEL)
            cache.validate([first, second])

            forced = FakeEmbedder()
            self.run_precompute(tmp, panels, forced, datasets=[first.dataset], force=True)
            # Chunk files are named by their inputs, so a forced rebuild reuses them.
            self.assertEqual(forced.calls, [])

    def test_interrupted_run_resumes_from_finished_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            panel = make_panel("math:subject=a")  # 12 responses -> chunks of 5, 5, 2
            panels = {panel.dataset: panel}
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                self.run_precompute(tmp, panels, FakeEmbedder(fail_after=2))
            self.assertFalse((Path(tmp) / "index.json").exists())

            resumed = FakeEmbedder()
            self.run_precompute(tmp, panels, resumed)
            self.assertEqual([len(c) for c in resumed.calls], [2])
            CachedResponseEmbeddings(tmp, PROVIDER, DEFAULT_MODEL).validate([panel])

    def test_changed_responses_are_recomputed(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = make_panel("math:subject=a")
            self.run_precompute(tmp, {old.dataset: old}, FakeEmbedder())
            new = make_panel("math:subject=a", tag="changed")
            embedder = FakeEmbedder()
            summary = self.run_precompute(tmp, {new.dataset: new}, embedder)
            self.assertEqual(summary["embedded"], [new.dataset])
            CachedResponseEmbeddings(tmp, PROVIDER, DEFAULT_MODEL).validate([new])

    def test_onehot_datasets_are_not_embedded(self):
        with tempfile.TemporaryDirectory() as tmp:
            panel = make_panel("legalbench:subset=x")
            embedder = FakeEmbedder()
            summary = self.run_precompute(tmp, {panel.dataset: panel}, embedder)
            self.assertEqual(summary["onehot"], [panel.dataset])
            self.assertEqual(embedder.calls, [])

    def test_refuses_to_mix_models_and_unknown_datasets(self):
        with tempfile.TemporaryDirectory() as tmp:
            panel = make_panel("math:subject=a")
            panels = {panel.dataset: panel}
            self.run_precompute(tmp, panels, FakeEmbedder())
            with self.assertRaisesRegex(ValueError, "refusing to mix"):
                self.run_precompute(tmp, panels, FakeEmbedder(), model="other-model")
            with self.assertRaisesRegex(ValueError, "Not in the manifest"):
                self.run_precompute(
                    tmp, panels, FakeEmbedder(), datasets=["math:subject=missing"]
                )

    def test_repeated_item_ids_across_splits_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            panel = make_panel("math:subject=a")
            panel.item_ids[1] = json.dumps(["valid", "id0"])
            with self.assertRaisesRegex(ValueError, "repeat across splits"):
                self.run_precompute(tmp, {panel.dataset: panel}, FakeEmbedder())


if __name__ == "__main__":
    unittest.main()
