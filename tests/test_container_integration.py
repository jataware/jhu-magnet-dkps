import importlib.util
import json
import os
import shlex
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from scripts.run_pair_coverage import read_verdict


class RunnerTests(unittest.TestCase):
    def test_accepts_verified_and_falsified_verdicts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            run.mkdir()
            for result in ("VERIFIED", "FALSIFIED"):
                with self.subTest(result=result):
                    path = run / "verdict.json"
                    path.write_text(
                        json.dumps(
                            {
                                "evidence": {"available": 1},
                                "result": result,
                            }
                        )
                    )
                    self.assertEqual(read_verdict(root), (path, result))

    def test_rejects_absent_or_incomplete_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "Expected one MAGNET verdict"):
                read_verdict(root)
            run = root / "run"
            run.mkdir()
            for count, result in ((0, "VERIFIED"), (1, "INCONCLUSIVE")):
                with self.subTest(count=count, result=result):
                    (run / "verdict.json").write_text(
                        json.dumps(
                            {
                                "evidence": {"available": count},
                                "result": result,
                            }
                        )
                    )
                    with self.assertRaises(RuntimeError):
                        read_verdict(root)


@unittest.skipUnless(importlib.util.find_spec("magnet"), "MAGNET runtime not installed")
class ContainerIntegrationTests(unittest.TestCase):
    def test_pair_coverage_card_passes_real_magnet_schema(self):
        from magnet.schema import NewEvaluationRecipeSchema

        root = Path(__file__).resolve().parents[1]
        card = root / "jhu_ta1/cards/jhu_run_predict_pair_coverage_kwdagger.yaml"
        NewEvaluationRecipeSchema.model_validate(yaml.safe_load(card.read_text()))

    def test_offline_command_limits_mounts_and_does_not_forward_credentials(self):
        from jhu_ta1.magnet.offline_helm_node import OfflineHelmProcessNode

        with tempfile.TemporaryDirectory() as directory:
            node = OfflineHelmProcessNode(
                name="test", executable="python -m test", node_dpath=directory
            )
            node.container_image = "test-image"
            node.container_mounts = [directory, "/tmp/a path with spaces"]
            node.container_env = {
                "OPENAI_API_KEY": "must-not-forward",
                "HF_TOKEN": "must-not-forward",
            }
            with patch.dict(
                os.environ,
                {"OPENAI_API_KEY": "must-not-forward", "HF_TOKEN": "must-not-forward"},
            ):
                prefix = node._container_command_prefix()
            parts = shlex.split(prefix)
            self.assertEqual(parts.count("--network"), 1)
            self.assertEqual(parts[parts.index("--network") + 1], "none")
            self.assertNotIn("must-not-forward", prefix)
            self.assertNotIn("OPENAI_API_KEY", prefix)
            self.assertNotIn("HF_TOKEN", prefix)
            mounts = [parts[i + 1] for i, part in enumerate(parts) if part == "-v"]
            self.assertTrue(all(mount.endswith(":ro") for mount in mounts[:-1]))
            expected = Path(node.final_node_dpath).resolve()
            self.assertEqual(mounts[-1], f"{expected}:{expected}:rw")
            self.assertNotIn("docker.sock", prefix)
            self.assertIn('-w "$PWD"', prefix)


if __name__ == "__main__":
    unittest.main()
