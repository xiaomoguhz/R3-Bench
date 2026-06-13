

import json
import os

import ray
from omegaconf import OmegaConf

from ..single_controller.ray import RayWorkerGroup
from ..utils.tokenizer import get_processor, get_tokenizer
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import BatchFunctionRewardManager, SequentialFunctionRewardManager
from .config import PPOConfig
from .data_loader import create_dataloader
from .ray_trainer import RayPPOTrainer, ResourcePoolManager, Role


# please make sure main_task is not scheduled on head
@ray.remote(num_cpus=1)
class Runner:
    """A runner for RL training."""

    def run(self, config: PPOConfig):
        # print config
        print(json.dumps(config.to_dict(), indent=2))

        # instantiate tokenizer
        tokenizer = get_tokenizer(
            config.worker.actor.model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )
        processor = get_processor(
            config.worker.actor.model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )

        # define worker classes
        ray_worker_group_cls = RayWorkerGroup
        role_worker_mapping = {
            Role.ActorRolloutRef: ray.remote(FSDPWorker),
            Role.Critic: ray.remote(FSDPWorker),
        }
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRolloutRef: global_pool_id,
            Role.Critic: global_pool_id,
        }
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        if config.worker.reward.reward_type == "sequential":
            RewardManager = SequentialFunctionRewardManager
        elif config.worker.reward.reward_type == "batch":
            RewardManager = BatchFunctionRewardManager
        else:
            raise NotImplementedError(f"Unknown reward type {config.worker.reward.reward_type}.")

        RemoteRewardManager = ray.remote(RewardManager).options(num_cpus=config.worker.reward.num_cpus)
        reward_fn = RemoteRewardManager.remote(config.worker.reward, tokenizer)
        val_reward_fn = RemoteRewardManager.remote(config.worker.reward, tokenizer)

        train_dataloader, val_dataloader = create_dataloader(config.data, tokenizer, processor)

        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        trainer.init_workers()
        trainer.fit()


def main():
    cli_args = OmegaConf.from_cli()
    default_config = OmegaConf.structured(PPOConfig())

    if hasattr(cli_args, "config"):
        config_path = cli_args.pop("config", None)
        file_config = OmegaConf.load(config_path)
        default_config = OmegaConf.merge(default_config, file_config)

    ppo_config = OmegaConf.merge(default_config, cli_args)
    ppo_config: PPOConfig = OmegaConf.to_object(ppo_config)
    ppo_config.deep_post_init()

    if not ray.is_initialized():
        # Base environment variables
        runtime_env_vars = {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
                "PYTHONUNBUFFERED": "1",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            }
        
        # Forward wandb-related env vars to Ray workers
        wandb_api_key = os.getenv("WANDB_API_KEY")
        wandb_mode = os.getenv("WANDB_MODE")
        if wandb_api_key:
            runtime_env_vars["WANDB_API_KEY"] = wandb_api_key
        if wandb_mode:
            runtime_env_vars["WANDB_MODE"] = wandb_mode
        
        # Forward other env vars that workers may need
        for key in ["HF_ENDPOINT", "VLLM_DISABLE_SYMMETRIC_MEMORY", "VLLM_USE_V1"]:
            value = os.getenv(key)
            if value:
                runtime_env_vars[key] = value
        
        # Force-set VLLM_DISABLE_SYMMETRIC_MEMORY and VLLM_USE_V1 if not already present.
        # Critical for avoiding GPU allocation conflicts when tensor_parallel_size > 1.
        if "VLLM_DISABLE_SYMMETRIC_MEMORY" not in runtime_env_vars:
            runtime_env_vars["VLLM_DISABLE_SYMMETRIC_MEMORY"] = "1"
        if "VLLM_USE_V1" not in runtime_env_vars:
            runtime_env_vars["VLLM_USE_V1"] = "1"
        
        runtime_env = {"env_vars": runtime_env_vars}

        ray_init_kwargs = {
            "runtime_env": runtime_env,
            "ignore_reinit_error": True,
            "log_to_driver": True,
        }
        ray_address = os.getenv("RAY_ADDRESS")
        if ray_address:
            ray_init_kwargs["address"] = ray_address
        else:
            include_dashboard = os.getenv("RAY_INCLUDE_DASHBOARD", "0").lower() in {"1", "true", "yes"}
            ray_init_kwargs["include_dashboard"] = include_dashboard

            if os.getenv("RAY_NUM_CPUS"):
                ray_init_kwargs["num_cpus"] = int(os.environ["RAY_NUM_CPUS"])
            if os.getenv("RAY_NUM_GPUS"):
                ray_init_kwargs["num_gpus"] = int(os.environ["RAY_NUM_GPUS"])
            if os.getenv("RAY_OBJECT_STORE_MEMORY"):
                ray_init_kwargs["object_store_memory"] = int(os.environ["RAY_OBJECT_STORE_MEMORY"])
            if os.getenv("RAY_TMPDIR"):
                ray_init_kwargs["_temp_dir"] = os.environ["RAY_TMPDIR"]
            if os.getenv("RAY_WORKER_REGISTER_TIMEOUT_SECONDS"):
                ray_init_kwargs["_system_config"] = {
                    "worker_register_timeout_seconds": int(os.environ["RAY_WORKER_REGISTER_TIMEOUT_SECONDS"])
                }

        print(f"[INFO] Initializing Ray with kwargs: { {k: v for k, v in ray_init_kwargs.items() if k != 'runtime_env'} }", flush=True)
        ray.init(**ray_init_kwargs)
        print(f"[INFO] Ray initialized. cluster_resources={ray.cluster_resources()}", flush=True)

    runner = Runner.remote()
    ray.get(runner.run.remote(ppo_config))

    if ppo_config.trainer.ray_timeline is not None:
        # use `export RAY_PROFILING=1` to record the ray timeline
        ray.timeline(filename=ppo_config.trainer.ray_timeline)


if __name__ == "__main__":
    main()
