import importlib
import importlib.machinery
import os
import sys
import types
import unittest
from unittest import mock


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


if __name__ == "__main__":
    unittest.main()
