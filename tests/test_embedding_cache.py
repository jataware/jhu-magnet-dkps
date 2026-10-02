import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from jhu_ta1.magnet._embedding_cache import CachedResponseEmbeddings, response_hash
from jhu_ta1.magnet._helm_pair_data import Panel
from jhu_ta1.magnet.pair_coverage import evaluate, main
from scripts.prepare_pair_coverage_18 import select_pool


class EmbeddingCacheTests(unittest.TestCase):
    def test_unset_kwdagger_cache_path_is_optional(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            manifest.write_text("{}")
            with patch("jhu_ta1.magnet.pair_coverage.evaluate") as evaluator:
                evaluator.return_value = {"protocol": {}, "result": {"metrics": {}}}
                main(
                    [
                        "--helm_suite_path",
                        str(root),
                        "--dataset_manifest",
                        str(manifest),
                        "--embedding_cache_path=None",
                        "--out_fpath",
                        str(root / "result.json"),
                    ]
                )
            self.assertIsNone(evaluator.call_args.kwargs["embedding_cache_path"])

    def fixture(self, directory):
        models = [f"provider{i}/model" for i in range(4)]
        panel = Panel(
            dataset="math:subject=fixture",
            metric="accuracy",
            models=models,
            item_ids=[json.dumps(["test", f"id{i}"]) for i in range(3)],
            responses={
                model: [f"response {model} {i}" for i in range(3)] for model in models
            },
            references=[[] for _ in range(3)],
            scores={
                model: np.array([0.0, 1.0, i % 2]) for i, model in enumerate(models)
            },
            sources=[],
            original_pool_sizes=dict.fromkeys(models, 3),
        )
        chunks = directory / "chunks"
        chunks.mkdir()
        vectors = np.arange(24, dtype=float).reshape(12, 2)
        path = chunks / "batch.npy"
        np.save(path, vectors)
        records = {}
        for model_index, model in enumerate(models):
            records[model] = {
                f"id{i}": {
                    "response_sha256": response_hash(panel.responses[model][i]),
                    "chunk": "batch.npy",
                    "row": model_index * 3 + i,
                }
                for i in range(3)
            }
        index = {
            "format_version": 1,
            "provider": "google",
            "model": "gemini-embedding-001",
            "dimensions": 2,
            "datasets": {panel.dataset: records},
            "chunk_sha256": {
                "batch.npy": hashlib.sha256(path.read_bytes()).hexdigest()
            },
        }
        (directory / "index.json").write_text(json.dumps(index))
        return panel, vectors

    def test_cached_vectors_preserve_query_order_and_run_new_fits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel, vectors = self.fixture(root)
            cache = CachedResponseEmbeddings(root, "google", "gemini-embedding-001")
            cache.validate([panel])
            actual = cache.embed_queries(panel, [2, 0])
            np.testing.assert_array_equal(
                actual[panel.models[0]], vectors[[2, 0], None, :]
            )
            with (
                patch(
                    "jhu_ta1.magnet.pair_coverage.load_panels",
                    return_value=([panel], panel.models),
                ),
                patch(
                    "jhu_ta1.magnet.pair_coverage.embed_queries",
                    side_effect=AssertionError("Must not call an embedding provider"),
                ),
            ):
                result = evaluate(
                    root,
                    {},
                    query_budgets=[1, 2],
                    num_replicates=2,
                    dimensions=2,
                    embed_provider="google",
                    embed_model="gemini-embedding-001",
                    embedding_cache_path=root,
                )
            self.assertEqual(result["result"]["metrics"]["dkps_fits"], 16)
            self.assertEqual(result["result"]["metrics"]["cached_embedding_batches"], 4)

    def test_missing_changed_or_corrupt_cache_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel, _ = self.fixture(root)
            with self.assertRaisesRegex(ValueError, "provider/model"):
                CachedResponseEmbeddings(root, "google", "another-model")
            cache = CachedResponseEmbeddings(root, "google", "gemini-embedding-001")
            original = panel.responses[panel.models[0]][0]
            panel.responses[panel.models[0]][0] = "changed response"
            with self.assertRaisesRegex(ValueError, "response differs"):
                cache.validate([panel])
            panel.responses[panel.models[0]][0] = original
            cache.index["chunk_sha256"]["batch.npy"] = "wrong checksum"
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                cache.validate([panel])
            del cache.index["datasets"][panel.dataset][panel.models[0]]["id0"]
            with self.assertRaisesRegex(ValueError, "Missing cached embedding"):
                cache.validate([panel])

    def test_wmt_pool_matches_original_seeded_selection(self):
        import pandas as pd

        frame = pd.DataFrame(
            [
                {
                    "dataset": "wmt_14:language_pair=fixture",
                    "model": model,
                    "instance_id": f"wmt_14:language_pair=fixture--id{i}",
                }
                for model in ["a", "b"]
                for i in range(20)
            ]
        )
        selected = select_pool(frame, "wmt_14:language_pair=fixture")
        expected = np.random.default_rng(123).choice(
            frame.instance_id.unique(), 4, replace=False
        )
        self.assertEqual(set(selected.instance_id), set(expected))
        self.assertEqual(len(selected), 8)


if __name__ == "__main__":
    unittest.main()
