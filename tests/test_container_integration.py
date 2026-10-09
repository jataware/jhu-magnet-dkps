import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

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

    def test_standard_pipeline_connects_materialization_to_evaluation(self):
        from kwdagger.yaml_pipeline import load_yaml_pipeline
        from magnet.process_node import MagnetProcessNode

        root = Path(__file__).resolve().parents[1]
        pipeline_path = root / "jhu_ta1/magnet/pair_coverage_pipeline.yaml"
        pipeline = load_yaml_pipeline(pipeline_path)
        self.assertEqual(set(pipeline.node_dict), {"materialize_lite", "pair_coverage"})
        for node in pipeline.node_dict.values():
            self.assertIs(type(node), MagnetProcessNode)
        evaluator = pipeline.node_dict["pair_coverage"]
        self.assertTrue(evaluator.inputs["helm_suite_path"].pred)

        # Supplying a prepared suite must not change the evaluation protocol.
        supplied_path = root / "jhu_ta1/magnet/pair_coverage_supplied_pipeline.yaml"
        standard = yaml.safe_load(pipeline_path.read_text())
        supplied = yaml.safe_load(supplied_path.read_text())
        self.assertEqual(
            standard["nodes"]["pair_coverage"], supplied["nodes"]["pair_coverage"]
        )


if __name__ == "__main__":
    unittest.main()
