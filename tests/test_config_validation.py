import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import li_4yp.experiments as experiments_pkg
from li_4yp.experiments.config import ExperimentConfig
import scripts.train as train_script


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config" / "experiments"


class FakeExperiment:
    last_instance = None

    def __init__(self, config):
        self.config = config
        self.run_kwargs = None
        FakeExperiment.last_instance = self

    def run(self, **kwargs):
        self.run_kwargs = kwargs
        return {}


class ExperimentConfigValidationTests(unittest.TestCase):
    def test_valid_pytorch_config_loads(self):
        config = ExperimentConfig.from_yaml(CONFIG_DIR / "attentive_statnet.yaml")
        self.assertEqual(config.model_type, "pytorch")
        self.assertEqual(config.dataset_class, "DigitalSwatchDataset")
        self.assertEqual(config.train_split, 0.8)

    def test_valid_sklearn_config_loads_without_dataset_class(self):
        config = ExperimentConfig.from_yaml(CONFIG_DIR / "example_sklearn.yaml")
        self.assertEqual(config.model_type, "sklearn")
        self.assertIsNone(config.dataset_class)
        self.assertEqual(config.train_split, 0.8)

    def test_missing_shared_field_fails(self):
        with self.assertRaisesRegex(ValueError, "Shared required fields: train_split"):
            ExperimentConfig.from_yaml(CONFIG_DIR / "failure_missing_train_split.yaml")

    def test_missing_pytorch_required_field_fails(self):
        with self.assertRaisesRegex(
            ValueError, "PyTorch required fields: dataset_class"
        ):
            ExperimentConfig.from_yaml(
                CONFIG_DIR / "failure_missing_dataset_class.yaml"
            )

    def test_missing_sklearn_required_field_fails(self):
        with self.assertRaisesRegex(
            ValueError, "sklearn required fields: feature_cols"
        ):
            ExperimentConfig.from_yaml(CONFIG_DIR / "failure_missing_feature_cols.yaml")

    def test_transform_conflict_fails(self):
        with self.assertRaisesRegex(
            ValueError, "Cannot use both transform preset and custom augmentations"
        ):
            ExperimentConfig.from_yaml(CONFIG_DIR / "failure_transform_conflict.yaml")


class TrainEntrypointContractTests(unittest.TestCase):
    def test_sklearn_main_does_not_require_dataset_class(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            data_dir = temp_root / "data"
            save_dir = temp_root / "experiments"
            data_dir.mkdir()
            save_dir.mkdir()
            (data_dir / "masterlist.csv").write_text(
                "Base,Primary,Secondary,LAB_L_mean,LAB_A_mean,LAB_B_mean\n"
                "1,2,3,10.0,1.0,2.0\n",
                encoding="utf-8",
            )
            config_path = temp_root / "valid_sklearn.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        'name: "sklearn_cli_validation"',
                        'model_type: "sklearn"',
                        'model_name: "multi_output_random_forest"',
                        "data_dir: " + f'"{data_dir}"',
                        'csv_file: "masterlist.csv"',
                        "save_dir: " + f'"{save_dir}"',
                        "train_split: 0.8",
                        "feature_cols:",
                        '  - "LAB_L_mean"',
                        '  - "LAB_A_mean"',
                        '  - "LAB_B_mean"',
                        "label_cols:",
                        '  - "Base"',
                        '  - "Primary"',
                        '  - "Secondary"',
                    ]
                ),
                encoding="utf-8",
            )

            FakeExperiment.last_instance = None
            with patch.object(train_script, "register_sklearn_models"), patch.object(
                train_script.ModelRegistry, "is_registered", return_value=True
            ), patch.dict(experiments_pkg.__dict__, {"Experiment": FakeExperiment}):
                with patch.object(
                    sys,
                    "argv",
                    ["train.py", "--config_path", str(config_path)],
                ):
                    train_script.main()

            self.assertIsNotNone(FakeExperiment.last_instance)
            self.assertEqual(
                FakeExperiment.last_instance.run_kwargs["dataset_class"], None
            )


if __name__ == "__main__":
    unittest.main()
