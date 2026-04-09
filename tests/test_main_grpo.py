import sys
import types
import unittest
from unittest import mock

from omegaconf import OmegaConf

from training.verl import main_grpo


class MainGrpoWorkerSelectionTest(unittest.TestCase):
    def test_build_worker_components_uses_async_worker_for_fsdp(self):
        config = OmegaConf.create(
            {
                "actor_rollout_ref": {"actor": {"strategy": "fsdp"}},
                "critic": {"strategy": "fsdp"},
            }
        )

        fake_async_worker = type("AsyncActorRolloutRefWorker", (), {})
        fake_critic_worker = type("CriticWorker", (), {})
        fake_worker_group = type("RayWorkerGroup", (), {})

        fake_fsdp_module = types.SimpleNamespace(
            AsyncActorRolloutRefWorker=fake_async_worker,
            CriticWorker=fake_critic_worker,
        )
        fake_ray_module = types.SimpleNamespace(RayWorkerGroup=fake_worker_group)

        with mock.patch.dict(
            sys.modules,
            {
                "verl.workers.fsdp_workers": fake_fsdp_module,
                "verl.single_controller.ray": fake_ray_module,
            },
        ), mock.patch.object(main_grpo.ray, "remote", side_effect=lambda cls: cls):
            role_worker_mapping, ray_worker_group_cls = main_grpo._build_worker_components(config)

        self.assertIs(fake_async_worker, role_worker_mapping[main_grpo.Role.ActorRollout])
        self.assertIs(fake_async_worker, role_worker_mapping[main_grpo.Role.RefPolicy])
        self.assertIs(fake_critic_worker, role_worker_mapping[main_grpo.Role.Critic])
        self.assertIs(fake_worker_group, ray_worker_group_cls)


if __name__ == "__main__":
    unittest.main()
