"""
DB-Opt GRPO 训练入口

参考 Agent-R1 的 main_agent.py，将编译器工具替换为 DB 调优工具。

Usage:
    python3 -m training.verl.main_grpo \
        --config configs/grpo_trainer.yaml \
        ...hydra overrides...
"""

import ray
import hydra
import torch

from verl import DataProto

from training.verl.agent_ray_trainer import RayAgentTrainer, ResourcePoolManager, Role
from training.reward_score import (
    compute_score_format,
    compute_score_answer,
    compute_score_format_answer,
)


# ============================================================
# Reward Manager
# ============================================================

class DBRewardManager:
    """DB 调优 Reward 管理器"""

    def __init__(self, tokenizer, num_examine=0, cost_model=None):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.cost_model = cost_model

    def __call__(self, data: DataProto):
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(
            data.batch['responses'], dtype=torch.float32
        )
        answer_lst = []
        format_lst = []

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][
                :prompt_length
            ].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][
                prompt_length:
            ].sum()
            valid_response_ids = response_ids[:valid_response_length].long()

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(
                sequences, skip_special_tokens=False
            )
            pad_token_id = self.tokenizer.pad_token_id
            sequences_str = sequences_str.split(
                self.tokenizer.decode([pad_token_id])
            )[0]

            ground_truth = data_item.non_tensor_batch['reward_model'][
                'ground_truth'
            ]
            data_source = data_item.non_tensor_batch['data_source']

            # 计算 reward
            score = compute_score_format_answer(
                solution_str=sequences_str,
                ground_truth=ground_truth,
                cost_model=self.cost_model,
            )
            answer_score = compute_score_answer(
                solution_str=sequences_str,
                ground_truth=ground_truth,
                cost_model=self.cost_model,
            )
            format_score = compute_score_format(solution_str=sequences_str)

            answer_lst.append(answer_score)
            format_lst.append(format_score)

            reward_tensor[i, valid_response_length - 1] = score

            # 调试打印
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt+response]", sequences_str[:500])
                print("[ground_truth]", ground_truth)
                print("[score]", score)

        return reward_tensor, answer_lst, format_lst


# ============================================================
# 训练入口
# ============================================================

@hydra.main(config_path='../../configs', config_name='grpo_trainer', version_base=None)
def main(config):
    run_grpo(config)


def run_grpo(config) -> None:
    if not ray.is_initialized():
        ray.init(runtime_env={
            'env_vars': {
                'TOKENIZERS_PARALLELISM': 'true',
                'NCCL_DEBUG': 'WARN',
                'VLLM_LOGGING_LEVEL': 'WARN',
                'VLLM_USE_V1': '1',
            }
        })

    ray.get(main_task.remote(config))


def _build_worker_components(config):
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(AsyncActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
            Role.RefPolicy: ray.remote(AsyncActorRolloutRefWorker),
        }
        return role_worker_mapping, RayWorkerGroup

    raise NotImplementedError(
        f"Strategy {config.actor_rollout_ref.actor.strategy} not supported"
    )


@ray.remote(num_cpus=1)
def main_task(config):
    from verl.utils.fs import copy_to_local
    from pprint import pprint
    from omegaconf import OmegaConf

    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    # 下载模型 checkpoint
    local_path = copy_to_local(config.actor_rollout_ref.model.path)

    # 初始化 tokenizer
    from verl.utils import hf_tokenizer, hf_processor
    tokenizer = hf_tokenizer(local_path)
    processor = hf_processor(local_path, use_fast=True)

    role_worker_mapping, ray_worker_group_cls = _build_worker_components(config)

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # Reward Model（如果有）
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec=resource_pool_spec, mapping=mapping
    )

    # 构建 DB 工具环境（使用已有的 DBToolEnv）
    from environment.tools import DBToolEnv

    # 加载 Cost Model（如果配置了路径）
    cost_model = None
    cost_model_path = getattr(config, 'cost_model_path', None)
    if cost_model_path:
        try:
            from cost_model.model import CostModel
            cost_model = CostModel.load(cost_model_path)
            print(f"已加载 Cost Model: {cost_model_path}")
        except Exception as e:
            print(f"[WARNING] Cost Model 加载失败: {e}，answer_score 将为 0")

    scenario_dir = getattr(config, 'scenario_dir', None)
    knob_space_path = getattr(config, 'knob_space_path', 'configs/knob_space.yaml')

    env = DBToolEnv(
        mode="train",
        scenario_dir=scenario_dir,
        cost_model=cost_model,
        max_turns=config.tool.max_turns,
        knob_space_path=knob_space_path,
    )

    # 创建 Trainer
    trainer = RayAgentTrainer(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=DBRewardManager(
            tokenizer=tokenizer, num_examine=0, cost_model=cost_model
        ),
        val_reward_fn=DBRewardManager(
            tokenizer=tokenizer, num_examine=1, cost_model=cost_model
        ),
        env=env,
    )
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
