import importlib
import importlib.machinery
import os
import sys
import types
import unittest
from unittest import mock
from types import SimpleNamespace


def load_sft_module():
    fake_trl = types.ModuleType("trl")
    fake_trl.__spec__ = importlib.machinery.ModuleSpec("trl", loader=None)
    fake_trl.SFTConfig = object
    fake_trl.SFTTrainer = object

    fake_peft = types.ModuleType("peft")
    fake_peft.__spec__ = importlib.machinery.ModuleSpec("peft", loader=None)
    fake_peft.LoraConfig = object

    with mock.patch.dict(sys.modules, {"trl": fake_trl, "peft": fake_peft}):
        if "training.trl.sft" in sys.modules:
            return importlib.reload(sys.modules["training.trl.sft"])
        return importlib.import_module("training.trl.sft")


sft = load_sft_module()


class TrlSftDeviceBindingTest(unittest.TestCase):
    @mock.patch.object(sft.torch.cuda, "set_device")
    @mock.patch.object(sft.torch.cuda, "device_count", return_value=4)
    @mock.patch.object(sft.torch.cuda, "is_available", return_value=True)
    def test_bind_local_rank_device_when_running_under_torchrun(
        self, _is_available, _device_count, set_device
    ):
        with mock.patch.dict(os.environ, {"LOCAL_RANK": "2"}, clear=False):
            device_index = sft.maybe_configure_torch_device_for_distributed()

        self.assertEqual(device_index, 2)
        set_device.assert_called_once_with(2)

    @mock.patch.object(sft.torch.cuda, "set_device")
    @mock.patch.object(sft.torch.cuda, "is_available", return_value=False)
    def test_skip_binding_when_cuda_unavailable(self, _is_available, set_device):
        with mock.patch.dict(os.environ, {"LOCAL_RANK": "1"}, clear=False):
            device_index = sft.maybe_configure_torch_device_for_distributed()

        self.assertIsNone(device_index)
        set_device.assert_not_called()

    @mock.patch.object(sft.torch.cuda, "set_device")
    @mock.patch.object(sft.torch.cuda, "is_available", return_value=True)
    def test_skip_binding_when_not_under_torchrun(self, _is_available, set_device):
        with mock.patch.dict(os.environ, {}, clear=True):
            device_index = sft.maybe_configure_torch_device_for_distributed()

        self.assertIsNone(device_index)
        set_device.assert_not_called()


class TrlSftDataSplitTest(unittest.TestCase):
    def test_split_train_eval_records_reserves_validation_examples(self):
        records = [{"id": i} for i in range(10)]

        train_records, eval_records = sft.split_train_eval_records(
            records,
            train_ratio=0.8,
            seed=123,
        )

        self.assertEqual(len(train_records), 8)
        self.assertEqual(len(eval_records), 2)
        self.assertEqual(
            sorted(item["id"] for item in train_records + eval_records),
            list(range(10)),
        )

    def test_split_train_eval_records_keeps_at_least_one_eval_record(self):
        records = [{"id": 1}, {"id": 2}]

        train_records, eval_records = sft.split_train_eval_records(
            records,
            train_ratio=0.95,
            seed=42,
        )

        self.assertEqual(len(train_records), 1)
        self.assertEqual(len(eval_records), 1)


class TrlSftConfigTest(unittest.TestCase):
    def test_build_sft_config_kwargs_enables_assistant_only_loss_and_eval(self):
        args = SimpleNamespace(
            output_dir="/tmp/out",
            num_epochs=3,
            batch_size=2,
            grad_accum=4,
            lr=1e-5,
            max_length=8192,
            seed=42,
            bf16=True,
            gradient_checkpointing=True,
            max_steps=-1,
            use_lora=True,
        )

        kwargs = sft.build_sft_config_kwargs(args, has_eval=True)

        self.assertTrue(kwargs["assistant_only_loss"])
        self.assertEqual(kwargs["eval_strategy"], "steps")
        self.assertEqual(kwargs["eval_steps"], 50)
        self.assertTrue(kwargs["load_best_model_at_end"])
        self.assertEqual(kwargs["metric_for_best_model"], "eval_loss")
        self.assertFalse(kwargs["greater_is_better"])

    def test_build_sft_config_kwargs_skips_eval_fields_without_eval_dataset(self):
        args = SimpleNamespace(
            output_dir="/tmp/out",
            num_epochs=3,
            batch_size=2,
            grad_accum=4,
            lr=1e-5,
            max_length=8192,
            seed=42,
            bf16=True,
            gradient_checkpointing=True,
            max_steps=-1,
            use_lora=False,
        )

        kwargs = sft.build_sft_config_kwargs(args, has_eval=False)

        self.assertTrue(kwargs["assistant_only_loss"])
        self.assertNotIn("eval_strategy", kwargs)
        self.assertNotIn("load_best_model_at_end", kwargs)


if __name__ == "__main__":
    unittest.main()
