import contextlib
import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


class FakeTensor:
    __module__ = "torch.fake"

    def __init__(self, data):
        self.data = np.array(data)
        self.device = None

    @property
    def shape(self):
        return self.data.shape

    def to(self, device):
        self.device = device
        return self

    def size(self, dim):
        return self.data.shape[dim]

    def argmax(self, dim=-1):
        return FakeTensor(np.argmax(self.data, axis=dim))

    def cpu(self):
        return self

    def numpy(self):
        return self.data

    def __len__(self):
        return len(self.data)


class FakeLossValue:
    def __init__(self, value):
        self.value = value
        self.backward_called = False

    def item(self):
        return self.value

    def backward(self):
        self.backward_called = True


class FakeParameter:
    def __init__(self, size, requires_grad=True):
        self._size = size
        self.requires_grad = requires_grad

    def numel(self):
        return self._size


class FakeOptimizer:
    def __init__(self):
        self.zero_grad_calls = 0
        self.step_calls = 0

    def zero_grad(self):
        self.zero_grad_calls += 1

    def step(self):
        self.step_calls += 1


class FakeLossFn:
    def __init__(self, value):
        self.value = value
        self.calls = []
        self.loss_values = []

    def __call__(self, outputs, labels):
        self.calls.append((outputs, labels))
        loss_value = FakeLossValue(self.value)
        self.loss_values.append(loss_value)
        return loss_value


class FakeHybridModel:
    def __init__(self, hidden_dim=8, output_dim=3):
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.forward_calls = []
        self.to_device = None
        self.mode = None

    def to(self, device):
        self.to_device = device
        return self

    def parameters(self):
        return [FakeParameter(12), FakeParameter(6, requires_grad=False)]

    def train(self):
        self.mode = "train"

    def eval(self):
        self.mode = "eval"

    def __call__(self, images, features=None):
        self.forward_calls.append((images, features))
        batch_size = images.shape[0]
        return FakeTensor(np.full((batch_size, self.output_dim), 1.0))


class FakeDictOutputModel(FakeHybridModel):
    def __call__(self, images, features=None):
        self.forward_calls.append((images, features))
        return {
            "base": FakeTensor([[0.1, 0.9], [0.8, 0.2]]),
            "primary": FakeTensor([[0.7, 0.3], [0.2, 0.8]]),
            "secondary": FakeTensor([[0.4, 0.6], [0.9, 0.1]]),
        }


class FakeEvaluator:
    def __init__(self, summary_payload=None):
        self.summary_payload = summary_payload or {
            "base_acc": 1.0,
            "primary_acc": 0.5,
        }
        self.reset_calls = 0
        self.updates = []

    def reset(self):
        self.reset_calls += 1

    def update(self, preds, labels):
        self.updates.append((preds, labels))

    def summary(self):
        return dict(self.summary_payload)


class FakeLoader:
    def __init__(self, batches, dataset_size):
        self.batches = batches
        self.dataset = [None] * dataset_size

    def __iter__(self):
        return iter(self.batches)


class FakeSubset:
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)


class FakeDataLoader:
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers


class FakeDataset:
    instances = []

    def __init__(self, data_dir, csv_file, transform=None, **kwargs):
        self.data_dir = data_dir
        self.csv_file = csv_file
        self.transform = transform
        self.kwargs = kwargs
        self.df = _FakeDataFrame([0, 1, 0, 1])
        FakeDataset.instances.append(self)

    def __len__(self):
        return 4

    def get_info(self):
        return {"info_marker": "fake_dataset"}


class _FakeIndex(list):
    def __getitem__(self, mask):
        return _FakeIndex([value for value, keep in zip(self, mask) if keep])

    def tolist(self):
        return list(self)


class _FakeSeries(list):
    def __eq__(self, other):
        return [value == other for value in self]

    def __ne__(self, other):
        return [value != other for value in self]


class _FakeDataFrame:
    def __init__(self, folds):
        self.columns = ["fold"]
        self.index = _FakeIndex(list(range(len(folds))))
        self._folds = _FakeSeries(folds)

    def __getitem__(self, key):
        if key != "fold":
            raise KeyError(key)
        return self._folds


class DummyRegisteredLoss:
    def __call__(self, *args, **kwargs):
        return FakeLossValue(1.0)


class FakePandasDataFrame:
    def __init__(self, rows):
        self.rows = rows

    def to_csv(self, path, index=False):
        Path(path).write_text("fake_dataframe\n", encoding="utf-8")


class SilentExperimentLogger:
    def __init__(self, log_dir, experiment_name, console_output=True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.console_output = console_output
        self.logged_epochs = []
        self.logged_metrics = []

    def info(self, message):
        return None

    def warning(self, message):
        return None

    def error(self, message):
        return None

    def log_config(self, config):
        self.config = config

    def log_model_info(self, model_info):
        self.model_info = model_info

    def log_data_info(self, data_info):
        self.data_info = data_info

    def log_epoch(self, **kwargs):
        self.logged_epochs.append(kwargs)

    def log_metrics(self, phase, metrics, step=None):
        self.logged_metrics.append((phase, metrics, step))

    def save_predictions(self, predictions, targets, phase="test"):
        self.saved_predictions = (predictions, targets, phase)

    def finalize(self, final_metrics=None):
        self.final_metrics = final_metrics


def _install_fake_torch_modules():
    fake_torch = types.ModuleType("torch")
    fake_torch._codex_fake = True
    fake_nn = types.ModuleType("torch.nn")
    fake_utils = types.ModuleType("torch.utils")
    fake_data = types.ModuleType("torch.utils.data")

    class FakeModule:
        def __call__(self, *args, **kwargs):
            if hasattr(self, "forward"):
                return self.forward(*args, **kwargs)
            raise NotImplementedError

    class FakeGenerator:
        def manual_seed(self, seed):
            self.seed = seed
            return self

    fake_nn.Module = FakeModule
    fake_nn.CrossEntropyLoss = lambda **kwargs: DummyRegisteredLoss()
    fake_nn.MSELoss = lambda **kwargs: DummyRegisteredLoss()
    fake_nn.L1Loss = lambda **kwargs: DummyRegisteredLoss()

    fake_torch.nn = fake_nn
    fake_torch.manual_seed = lambda seed: None
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: False, manual_seed_all=lambda seed: None
    )
    fake_torch.backends = types.SimpleNamespace(
        cudnn=types.SimpleNamespace(deterministic=False, benchmark=False),
        mps=types.SimpleNamespace(is_available=lambda: False),
    )
    fake_torch.device = lambda value: value
    fake_torch.Tensor = FakeTensor
    fake_torch.Generator = FakeGenerator
    fake_torch.no_grad = contextlib.nullcontext
    fake_torch.stack = lambda tensors, dim=0: FakeTensor(
        np.stack([tensor.data for tensor in tensors], axis=dim)
    )
    fake_torch.cat = lambda tensors, dim=0: FakeTensor(
        np.concatenate([tensor.data for tensor in tensors], axis=dim)
    )
    fake_torch.FloatTensor = lambda data: FakeTensor(data)
    fake_torch.zeros = lambda size: FakeTensor(np.zeros(size))

    fake_data.DataLoader = FakeDataLoader
    fake_data.Subset = FakeSubset
    fake_data.random_split = None
    fake_utils.data = fake_data

    sys.modules["torch"] = fake_torch
    sys.modules["torch.nn"] = fake_nn
    sys.modules["torch.utils"] = fake_utils
    sys.modules["torch.utils.data"] = fake_data


_install_fake_torch_modules()
fake_pandas = types.ModuleType("pandas")
fake_pandas.DataFrame = FakePandasDataFrame
sys.modules["pandas"] = fake_pandas

from li_4yp.experiments.config import ExperimentConfig
from li_4yp.experiments.loss_registry import LossRegistry
from li_4yp.experiments.model_registry import ModelRegistry, register_custom_model

experiment_module = importlib.import_module("li_4yp.experiments.experiment")
Experiment = experiment_module.Experiment


class ExperimentBranchCoverageTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)
        self.base_config = {
            "name": "branch_coverage",
            "model_type": "pytorch",
            "model_name": "branch_test_model",
            "dataset_class": "FakeDataset",
            "data_dir": str(self.workspace / "data"),
            "csv_file": "labels.csv",
            "train_split": 0.75,
            "save_dir": str(self.workspace / "experiments"),
            "batch_size": 2,
            "num_epochs": 1,
            "learning_rate": 0.001,
            "optimizer": "Adam",
            "loss_name": "CrossEntropyLoss",
            "image_size": [224, 224],
            "normalize": False,
            "device": "cpu",
            "num_workers": 0,
            "use_transform_preset": False,
            "augmentation": {},
        }
        FakeDataset.instances = []

    def tearDown(self):
        for model_name in ("branch_test_model", "dict_output_test_model"):
            if ModelRegistry.is_registered(model_name):
                ModelRegistry.unregister(model_name)
        if LossRegistry.is_registered("ToyLoss"):
            LossRegistry.unregister("ToyLoss")
        self.temp_dir.cleanup()

    def _make_config(self, **overrides):
        config_dict = dict(self.base_config)
        config_dict.update(overrides)
        return ExperimentConfig.from_dict(config_dict)

    def _make_experiment(self, config):
        utils_module = importlib.import_module("li_4yp.utils")
        with (
            patch.dict(
                utils_module.__dict__,
                {"seed_all": lambda seed: None, "get_device": lambda: "cpu"},
            ),
            patch.object(experiment_module, "ExperimentLogger", SilentExperimentLogger),
        ):
            return Experiment(config)

    def test_setup_model_uses_custom_registry_model(self):
        register_custom_model(
            name="branch_test_model",
            model_class=FakeHybridModel,
            default_params={"hidden_dim": 16},
            description="Toy model for branch coverage tests",
        )
        experiment = self._make_experiment(
            self._make_config(model_params={"output_dim": 4})
        )

        experiment.setup_model()

        self.assertIsInstance(experiment.model, FakeHybridModel)
        self.assertEqual(experiment.model.hidden_dim, 16)
        self.assertEqual(experiment.model.output_dim, 4)
        self.assertEqual(experiment.model.to_device, "cpu")

    def test_setup_loss_fn_uses_registered_loss(self):
        captured_kwargs = {}

        def build_toy_loss(**kwargs):
            captured_kwargs.update(kwargs)
            return FakeLossFn(0.25)

        LossRegistry.register(
            "ToyLoss", build_toy_loss, "Toy loss for branch coverage tests"
        )
        experiment = self._make_experiment(
            self._make_config(loss_name="ToyLoss", loss_params={"margin": 0.4})
        )

        experiment.setup_loss_fn()

        self.assertIsInstance(experiment.loss_fn, FakeLossFn)
        self.assertEqual(captured_kwargs, {"margin": 0.4})

    def test_train_pytorch_epoch_handles_hybrid_batches(self):
        experiment = self._make_experiment(self._make_config())
        experiment.model = FakeHybridModel()
        experiment.optimizer = FakeOptimizer()
        experiment.loss_fn = FakeLossFn(2.5)
        experiment.train_loader = FakeLoader(
            batches=[
                (
                    FakeTensor([[1.0], [2.0]]),
                    FakeTensor([[0.1], [0.2]]),
                    FakeTensor([[0, 1, 2], [1, 2, 0]]),
                ),
                (
                    FakeTensor([[3.0]]),
                    FakeTensor([[0.3]]),
                    FakeTensor([[2, 0, 1]]),
                ),
            ],
            dataset_size=3,
        )

        average_loss = experiment.train_pytorch_epoch()

        self.assertEqual(average_loss, 2.5)
        self.assertEqual(experiment.model.mode, "train")
        self.assertEqual(len(experiment.model.forward_calls), 2)
        self.assertIsNotNone(experiment.model.forward_calls[0][1])
        self.assertEqual(experiment.optimizer.zero_grad_calls, 2)
        self.assertEqual(experiment.optimizer.step_calls, 2)
        self.assertTrue(
            all(
                loss_value.backward_called
                for loss_value in experiment.loss_fn.loss_values
            )
        )

    def test_evaluate_pytorch_handles_dict_outputs(self):
        experiment = self._make_experiment(self._make_config())
        experiment.model = FakeDictOutputModel()
        experiment.loss_fn = FakeLossFn(0.75)
        experiment.evaluator = FakeEvaluator(
            {"base_acc": 0.5, "primary_acc": 0.75, "secondary_acc": 0.25}
        )
        val_loader = FakeLoader(
            batches=[
                (
                    FakeTensor([[1.0], [2.0]]),
                    FakeTensor([[1, 0, 1], [0, 1, 0]]),
                )
            ],
            dataset_size=2,
        )

        average_loss, metrics = experiment.evaluate_pytorch(val_loader)

        self.assertEqual(average_loss, 0.75)
        self.assertEqual(metrics["base_acc"], 0.5)
        self.assertEqual(len(experiment.evaluator.updates), 1)
        preds, labels = experiment.evaluator.updates[0]
        np.testing.assert_array_equal(preds.data, np.array([[1, 0, 1], [0, 1, 0]]))
        np.testing.assert_array_equal(labels.data, np.array([[1, 0, 1], [0, 1, 0]]))

    def test_setup_data_uses_fixed_fold_split_indices(self):
        experiment = self._make_experiment(self._make_config(val_fold_idx=1))

        with (
            patch.object(experiment, "_build_train_transform", return_value="train"),
            patch.object(experiment, "_build_val_transform", return_value="val"),
            patch.object(
                experiment_module.Experiment,
                "_require_torch",
                return_value=(
                    sys.modules["torch"],
                    sys.modules["torch.nn"],
                    FakeDataLoader,
                    FakeSubset,
                    None,
                ),
            ),
        ):
            experiment.setup_data(dataset_class=FakeDataset)

        self.assertEqual(experiment.train_loader.dataset.indices, [0, 2])
        self.assertEqual(experiment.val_loader.dataset.indices, [1, 3])
        self.assertEqual(FakeDataset.instances[1].transform, "train")
        self.assertEqual(FakeDataset.instances[2].transform, "val")

    def test_build_train_transform_uses_preset_lookup(self):
        experiment = self._make_experiment(
            self._make_config(use_transform_preset=True, transform_preset="imagenet")
        )
        utils_module = importlib.import_module("li_4yp.utils")
        build_calls = []
        preset_calls = []

        with (
            patch.object(experiment_module.Experiment, "_require_torch"),
            patch.dict(
                utils_module.__dict__,
                {
                    "build_transform": lambda **kwargs: build_calls.append(kwargs)
                    or "custom",
                    "get_transform_from_preset": lambda name: preset_calls.append(name)
                    or "preset",
                },
            ),
        ):
            transform = experiment._build_train_transform()

        self.assertEqual(transform, "preset")
        self.assertEqual(preset_calls, ["imagenet"])
        self.assertEqual(build_calls, [])

    def test_build_train_transform_uses_custom_augmentation_branch(self):
        experiment = self._make_experiment(
            self._make_config(
                augmentation={"random_horizontal_flip": True},
                use_transform_preset=False,
            )
        )
        utils_module = importlib.import_module("li_4yp.utils")
        build_calls = []

        with (
            patch.object(experiment_module.Experiment, "_require_torch"),
            patch.dict(
                utils_module.__dict__,
                {
                    "build_transform": lambda **kwargs: build_calls.append(kwargs)
                    or "custom",
                    "get_transform_from_preset": lambda name: "preset",
                },
            ),
        ):
            transform = experiment._build_train_transform()

        self.assertEqual(transform, "custom")
        self.assertEqual(len(build_calls), 1)
        self.assertEqual(
            build_calls[0]["augmentation"], {"random_horizontal_flip": True}
        )
        self.assertTrue(build_calls[0]["is_training"])


if __name__ == "__main__":
    unittest.main()
