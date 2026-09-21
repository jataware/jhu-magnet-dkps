import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import yaml

from jhu_ta1.magnet._helm_pair_data import load_panels
from jhu_ta1.magnet.pair_coverage import (
    evaluate,
    predict,
    summarize_errors,
)


class LivePairTests(unittest.TestCase):
    def run_claim(self, values):
        root = Path(__file__).resolve().parents[1]
        card = yaml.safe_load(
            (
                root / "jhu_ta1/cards/jhu_run_predict_pair_coverage_kwdagger.yaml"
            ).read_text()
        )
        exec(
            card["claim"]["python"],
            dict(
                coverage_threshold=0.95,
                metrics=SimpleNamespace(pair_coverage=SimpleNamespace(**values)),
            ),
        )

    def test_pooled_claim_can_pass_with_a_failing_budget(self):
        errors = np.zeros((1, 4, 3, 20))
        errors[:, 0, :, :3] = 0.8
        result = summarize_errors(
            errors,
            np.full_like(errors, 0.1),
            ["dataset"],
            [f"model{i}" for i in range(20)],
            [1, 2, 4, 8],
        )
        metrics = result["result"]["metrics"]
        self.assertEqual(metrics["num_settings"], 80)
        self.assertEqual(metrics["settings_improved"], 77)
        self.assertEqual(metrics["setting_improvement_fraction"], 77 / 80)
        self.assertEqual(result["per_budget"][0]["pair_improvement_fraction"], 17 / 20)
        self.assertEqual(len(result["per_setting"]), 80)
        # Averaging errors across budgets would leave only 17/20 models
        # improved. Counting combinations must still pass the card.
        metrics.update(alpha=0.8, n_components_cmds=8, dkps_fits=240)
        self.run_claim(metrics)

    def test_invalid_budgets_fail_before_loading(self):
        for budgets in ([], [1, 1], [0], [-1]):
            with self.subTest(budgets=budgets):
                with self.assertRaisesRegex(ValueError, "distinct positive integers"):
                    evaluate("unused", {}, query_budgets=budgets)

    def fixture(self, root):
        datasets = ["med_qa", "legalbench:subset=fixture"]
        for d, dataset in enumerate(datasets):
            for m in range(12):
                model = f"family{m // 4}/model{m}"
                directory = root / f"{dataset},model={model.replace('/', '_')}"
                directory.mkdir()
                requests, scores = [], []
                for i in range(8):
                    correct = i % 2
                    response = (i + m) % 2
                    labels = ["A", "B"] if d == 0 else ["yes", "no"]
                    instance = dict(
                        id=f"id{i}",
                        split="test",
                        input={"text": f"question {i}"},
                        references=[
                            {"output": {"text": labels[correct]}, "tags": ["correct"]}
                        ],
                    )
                    requests.append(
                        dict(
                            instance=instance,
                            train_trial_index=0,
                            result={"completions": [{"text": labels[response]}]},
                        )
                    )
                    score = float(response == correct)
                    stats = (
                        [
                            {
                                "name": {"name": "accuracy", "split": "test"},
                                "mean": score,
                                "count": 1,
                            }
                        ]
                        if d == 0
                        else {"accuracy": score}
                    )
                    scores.append(
                        dict(instance_id=f"id{i}", train_trial_index=0, stats=stats)
                    )
                (directory / "scenario_state.json").write_text(
                    json.dumps(
                        dict(adapter_spec={"model": model}, request_states=requests)
                    )
                )
                name = (
                    "per_instance_stats.json" if d == 0 else "display_predictions.json"
                )
                (directory / name).write_text(json.dumps(scores))
        return dict(
            datasets=[dict(dataset=name, metric="accuracy") for name in datasets]
        )

    def test_reduction_averages_only_replicates_and_excludes_ties(self):
        errors = np.array([np.zeros((1, 3, 2)), np.full((1, 3, 2), 0.3)])
        baseline = np.array([np.full((1, 3, 2), 0.4), np.full((1, 3, 2), 0.1)])
        result = summarize_errors(errors, baseline, ["a", "b"], ["m1", "m2"], [1])
        metrics = result["result"]["metrics"]
        self.assertEqual(metrics["num_settings"], 4)
        self.assertEqual(metrics["setting_improvement_fraction"], 0.5)
        self.assertEqual(
            [row["expected_gain"] > 0 for row in result["per_setting"]],
            [True, True, False, False],
        )
        tied = summarize_errors(baseline, baseline, ["a", "b"], ["m1", "m2"], [1])
        self.assertEqual(tied["result"]["metrics"]["settings_improved"], 0)

    def test_incomplete_or_nonfinite_errors_are_rejected(self):
        errors = np.zeros((1, 2, 3, 4))
        for invalid in (errors[:, :1], np.full_like(errors, np.nan)):
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                summarize_errors(invalid, errors, ["a"], list("abcd"), [1, 2])
        with self.assertRaisesRegex(ValueError, "Invalid.*panel"):
            summarize_errors(errors, errors, ["a"], list("abcd"), [1])

    def test_loader_aligns_real_helm_formats_and_rejects_missing_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.fixture(root)
            panels, targets = load_panels(root, manifest)
            self.assertEqual([p.pool_size for p in panels], [8, 8])
            self.assertEqual(len(targets), 12)
            path = next(root.glob("med_qa*/per_instance_stats.json"))
            rows = json.loads(path.read_text())
            rows.pop()
            path.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, "Missing accuracy"):
                load_panels(root, manifest)

    def test_loader_rejects_unaligned_item_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.fixture(root)
            path = next(root.glob("med_qa*/scenario_state.json"))
            data = json.loads(path.read_text())
            data["request_states"][0]["instance"]["input"]["text"] = (
                "different question"
            )
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "content differs"):
                load_panels(root, manifest)

    def test_live_evaluation_fits_dkps_and_holds_out_families(self):
        from dkps.dkps import DataKernelPerspectiveSpace

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.fixture(root)
            seen = []

            def inspect_predict(
                embeddings, reference_scores, target, target_query_scores, **kw
            ):
                self.assertEqual(len(target_query_scores), 2)
                self.assertTrue(
                    all(
                        m.split("/")[0] != target.split("/")[0]
                        for m in reference_scores
                    )
                )
                self.assertEqual(len(reference_scores), 8)
                seen.append(target)
                return predict(
                    embeddings, reference_scores, target, target_query_scores, **kw
                )

            original = DataKernelPerspectiveSpace.fit_transform
            fits = []

            def fit(instance, data, **kw):
                fits.append(tuple(data))
                self.assertEqual(len(data), 9)
                return original(instance, data, **kw)

            with (
                patch(
                    "jhu_ta1.magnet.pair_coverage.predict", side_effect=inspect_predict
                ),
                patch.object(DataKernelPerspectiveSpace, "fit_transform", fit),
            ):
                result = evaluate(
                    root, manifest, query_budgets=[2], num_replicates=3, dimensions=2
                )
            self.assertEqual(len(seen), 72)
            self.assertEqual(len(fits), 72)
            self.assertEqual(result["result"]["metrics"]["dkps_fits"], 72)
            self.assertEqual(result["result"]["metrics"]["embedding_batches"], 6)
            self.assertEqual(len(result["per_setting"]), 24)

    def test_unqueried_target_scores_cannot_change_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.fixture(root)
            manifest["datasets"] = manifest["datasets"][:1]
            manifest["target_models"] = ["family0/model0"]
            first, second = io.StringIO(), io.StringIO()
            evaluate(
                root,
                manifest,
                query_budgets=[1],
                num_replicates=2,
                dimensions=2,
                trial_stream=first,
            )
            trials = [json.loads(line) for line in first.getvalue().splitlines()]
            queried = {json.loads(q)[1] for row in trials for q in row["query_ids"]}
            path = root / "med_qa,model=family0_model0/per_instance_stats.json"
            data = json.loads(path.read_text())
            for row in data:
                if row["instance_id"] not in queried:
                    row["stats"][0]["mean"] = 0.0
            path.write_text(json.dumps(data))
            evaluate(
                root,
                manifest,
                query_budgets=[1],
                num_replicates=2,
                dimensions=2,
                trial_stream=second,
            )
            after = [json.loads(line) for line in second.getvalue().splitlines()]
            for a, b in zip(trials, after):
                self.assertEqual(a["prediction"], b["prediction"])
                self.assertEqual(a["raw_prediction"], b["raw_prediction"])
                self.assertEqual(a["sample_score"], b["sample_score"])
                self.assertNotEqual(a["truth"], b["truth"])

    def test_insufficient_references_fail_before_fitting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.fixture(root)
            with patch("jhu_ta1.magnet.pair_coverage.predict") as estimator:
                with self.assertRaisesRegex(ValueError, "eligible references"):
                    evaluate(
                        root,
                        manifest,
                        query_budgets=[1],
                        num_replicates=2,
                        dimensions=9,
                    )
                estimator.assert_not_called()

    def test_multiple_budgets_execute_and_tag_every_trial(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.fixture(root)
            manifest["target_models"] = ["family0/model0"]
            stream = io.StringIO()
            result = evaluate(
                root,
                manifest,
                query_budgets=[1, 2],
                num_replicates=2,
                dimensions=2,
                trial_stream=stream,
            )
            m = result["result"]["metrics"]
            self.assertEqual(m["num_settings"], 4)
            self.assertEqual(m["dkps_fits"], 8)
            self.assertEqual(m["confidence_comparisons"], 4)
            rows = [json.loads(line) for line in stream.getvalue().splitlines()]
            self.assertEqual(len(rows), 8)
            self.assertEqual({r["n_eval"] for r in rows}, {1, 2})
            self.assertTrue(all(len(r["query_ids"]) == r["n_eval"] for r in rows))

    def test_card_requires_complete_fixed_alpha_setting_coverage(self):
        root = Path(__file__).resolve().parents[1]
        card = yaml.safe_load(
            (
                root / "jhu_ta1/cards/jhu_run_predict_pair_coverage_kwdagger.yaml"
            ).read_text()
        )
        pipeline = yaml.safe_load(
            (root / "jhu_ta1/magnet/pair_coverage_pipeline.yaml").read_text()
        )
        self.assertEqual(
            pipeline["nodes"]["pair_coverage"]["algo_params"]["alpha"], 0.8
        )
        self.assertEqual(
            card["kwdagger"]["matrix"]["pair_coverage.query_budgets"], "1,2,4,8"
        )
        self.assertNotIn("pair_coverage.n_eval", card["kwdagger"]["matrix"])
        values = dict(
            alpha=0.8,
            n_components_cmds=8,
            budget_spec="1,2,4,8",
            num_models=25,
            num_datasets=4,
            num_replicates=10,
            num_pairs=100,
            num_budgets=4,
            num_settings=400,
            dkps_fits=4000,
            embedding_batches=160,
            settings_improved=384,
            setting_improvement_fraction=0.96,
            setting_improvement_fraction_with_confidence=0.90,
        )
        self.run_claim(values)
        for update, message in [
            (dict(alpha=0.85), "Expected sample weight"),
            (dict(dkps_fits=0), "Expected one DKPS fit"),
            (dict(num_settings=100), "Every dataset/model/budget combination"),
            (dict(budget_spec="1,2,4,4"), "Budgets must be distinct"),
            (dict(settings_improved=401), "Improvement count must be between"),
            (
                dict(settings_improved=380, setting_improvement_fraction=0.95),
                "the claim requires more than",
            ),
        ]:
            with (
                self.subTest(update=update),
                self.assertRaisesRegex(AssertionError, message),
            ):
                self.run_claim(values | update)


if __name__ == "__main__":
    unittest.main()
