import importlib
import importlib.machinery
import os
import sys
import tempfile
import types
import unittest
from unittest import mock
from types import SimpleNamespace
from pathlib import Path


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


class TrlSftResumeCheckpointTest(unittest.TestCase):
    def test_resolve_resume_checkpoint_accepts_empty_values(self):
        args = SimpleNamespace(output_dir="/tmp/out", resume_from_checkpoint=None)
        self.assertIsNone(sft.resolve_resume_checkpoint(args))

        args.resume_from_checkpoint = ""
        self.assertIsNone(sft.resolve_resume_checkpoint(args))

    def test_resolve_resume_checkpoint_uses_explicit_path(self):
        args = SimpleNamespace(output_dir="/tmp/out", resume_from_checkpoint="/tmp/out/checkpoint-50")
        self.assertEqual(sft.resolve_resume_checkpoint(args), "/tmp/out/checkpoint-50")

    def test_resolve_resume_checkpoint_auto_finds_latest_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "checkpoint-50").mkdir()
            (root / "checkpoint-100").mkdir()
            (root / "checkpoint-bad").mkdir()
            args = SimpleNamespace(output_dir=str(root), resume_from_checkpoint="auto")

            self.assertEqual(sft.resolve_resume_checkpoint(args), str(root / "checkpoint-100"))

    def test_resolve_resume_checkpoint_auto_returns_none_without_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(output_dir=tmpdir, resume_from_checkpoint="latest")

            self.assertIsNone(sft.resolve_resume_checkpoint(args))

    def test_resolve_resume_checkpoint_rejects_unknown_keyword(self):
        args = SimpleNamespace(output_dir="/tmp/out", resume_from_checkpoint="checkpoint")

        with self.assertRaisesRegex(ValueError, "resume_from_checkpoint"):
            sft.resolve_resume_checkpoint(args)


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
            chat_template_path=None,
            deepspeed=None,
            fsdp=None,
            fsdp_config=None,
        )

        kwargs = sft.build_sft_config_kwargs(args, has_eval=True)

        self.assertTrue(kwargs["assistant_only_loss"])
        self.assertEqual(kwargs["max_length"], 8192)
        self.assertNotIn("max_seq_length", kwargs)
        self.assertEqual(kwargs["per_device_eval_batch_size"], 2)
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
            chat_template_path=None,
            deepspeed=None,
            fsdp=None,
            fsdp_config=None,
        )

        kwargs = sft.build_sft_config_kwargs(args, has_eval=False)

        self.assertTrue(kwargs["assistant_only_loss"])
        self.assertEqual(kwargs["max_length"], 8192)
        self.assertNotIn("max_seq_length", kwargs)
        self.assertNotIn("per_device_eval_batch_size", kwargs)
        self.assertNotIn("eval_strategy", kwargs)
        self.assertNotIn("load_best_model_at_end", kwargs)

    def test_build_sft_config_kwargs_forwards_distributed_backend_options(self):
        args = SimpleNamespace(
            output_dir="/tmp/out",
            num_epochs=3,
            batch_size=1,
            grad_accum=8,
            lr=1e-5,
            max_length=8192,
            seed=42,
            bf16=True,
            gradient_checkpointing=True,
            max_steps=-1,
            use_lora=False,
            chat_template_path=None,
            deepspeed="configs/deepspeed_zero3_bf16.json",
            fsdp=None,
            fsdp_config=None,
        )

        kwargs = sft.build_sft_config_kwargs(args, has_eval=False)

        self.assertEqual(kwargs["deepspeed"], "configs/deepspeed_zero3_bf16.json")

    def test_build_sft_config_kwargs_skips_prepare_for_tokenized_dataset(self):
        args = SimpleNamespace(
            output_dir="/tmp/out",
            num_epochs=3,
            batch_size=1,
            grad_accum=8,
            lr=1e-5,
            max_length=8192,
            seed=42,
            bf16=True,
            gradient_checkpointing=True,
            max_steps=-1,
            use_lora=False,
            chat_template_path=None,
            deepspeed=None,
            fsdp=None,
            fsdp_config=None,
            tokenized_dataset_dir="/tmp/tokenized",
            resume_from_checkpoint=None,
        )

        kwargs = sft.build_sft_config_kwargs(args, has_eval=False)

        self.assertEqual(kwargs["dataset_kwargs"], {"skip_prepare_dataset": True})

    def test_validate_distributed_backend_args_rejects_deepspeed_with_fsdp(self):
        args = SimpleNamespace(
            deepspeed="configs/deepspeed_zero3_bf16.json",
            fsdp="full_shard auto_wrap",
            fsdp_config=None,
        )

        with self.assertRaisesRegex(ValueError, "不能同时启用 DeepSpeed 和 FSDP"):
            sft.validate_distributed_backend_args(args)


class TrlSftParamStatsTest(unittest.TestCase):
    class FakeParam:
        def __init__(self, numel, element_size=2, requires_grad=True, ds_numel=None):
            self._numel = numel
            self._element_size = element_size
            self.requires_grad = requires_grad
            if ds_numel is not None:
                self.ds_numel = ds_numel

        def numel(self):
            return self._numel

        def element_size(self):
            return self._element_size

    class FakeModel:
        def __init__(self, params):
            self._params = params

        def parameters(self):
            return iter(self._params)

    def test_collect_param_stats_uses_deepspeed_logical_numel(self):
        model = self.FakeModel(
            [
                self.FakeParam(numel=0, ds_numel=100, requires_grad=True),
                self.FakeParam(numel=0, ds_numel=50, requires_grad=False),
            ]
        )

        stats = sft.collect_param_stats(model)

        self.assertEqual(stats["total"], 150)
        self.assertEqual(stats["trainable"], 100)
        self.assertAlmostEqual(stats["trainable_ratio"], 100 / 150)
        self.assertEqual(stats["trainable_bytes"], 200)
        self.assertEqual(stats["frozen_bytes"], 100)

    def test_collect_param_stats_handles_empty_zero3_partition_without_dividing_by_zero(self):
        model = self.FakeModel([])

        stats = sft.collect_param_stats(model)

        self.assertEqual(stats["total"], 0)
        self.assertEqual(stats["trainable"], 0)
        self.assertEqual(stats["trainable_ratio"], 0.0)


class TrlSftAssistantMaskSupportTest(unittest.TestCase):
    def test_resolve_training_chat_template_path_uses_qwen3_template_when_available(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            template_path = Path(tmpdir) / "qwen3_training.jinja"
            template_path.write_text("{{ 'qwen3' }}", encoding="utf-8")

            with mock.patch.object(sft, "TRL_CHAT_TEMPLATES_DIR", Path(tmpdir)):
                resolved = sft.resolve_training_chat_template_path(
                    model_path="/root/workspace/models/Qwen3-8B",
                    explicit_path=None,
                )

        self.assertEqual(resolved, str(template_path))

    def test_validate_assistant_mask_support_rejects_empty_masks(self):
        tokenizer = mock.Mock()
        tokenizer.apply_chat_template.return_value = {
            "input_ids": [1, 2, 3],
            "assistant_masks": [0, 0, 0],
        }

        records = [
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ]
            }
        ]

        with self.assertRaisesRegex(RuntimeError, "assistant mask"):
            sft.validate_assistant_mask_support(records, tokenizer)

    def test_validate_assistant_mask_support_accepts_non_empty_masks(self):
        tokenizer = mock.Mock()
        tokenizer.apply_chat_template.return_value = {
            "input_ids": [1, 2, 3],
            "assistant_masks": [0, 1, 1],
        }

        records = [
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ]
            }
        ]

        sft.validate_assistant_mask_support(records, tokenizer)

    def test_validate_transformers_version_for_assistant_masks_requires_supported_version(self):
        with mock.patch.object(sft.transformers, "__version__", "4.51.1"):
            with self.assertRaisesRegex(RuntimeError, "transformers>=4.56.2"):
                sft.validate_transformers_version_for_assistant_masks()


if __name__ == "__main__":
    unittest.main()
