# Services

R3-Refiner uses HTTP services for image editing and reward scoring.

Typical setup:

1. Edit node: runs BAGEL or Qwen-Image-Edit.
2. Reward node: runs self-reward, CLIP, SAM3, or mixed SAM3-plus-base reward services.
3. Training node: sources the service endpoints and starts VERL training.

The roles can share one machine for local tests. For larger runs, use separate GPU machines for edit and reward services.

## Start The Edit Service

Run on the edit-service machine:

```bash
cd /path/to/R3-Refiner

export EDIT_MODEL_PATH=/path/to/BAGEL-7B-MoT
export EDIT_MODEL_TYPE=bagel
bash distributed_services/scripts/deploy_services.sh edit_server \
  "$EDIT_MODEL_PATH" "$EDIT_MODEL_TYPE"
```

For Qwen-Image-Edit:

```bash
export EDIT_MODEL_PATH=/path/to/Qwen-Image-Edit
export EDIT_MODEL_TYPE=qwen_image_edit
bash distributed_services/scripts/deploy_services.sh edit_server \
  "$EDIT_MODEL_PATH" "$EDIT_MODEL_TYPE"
```

The shortcuts below start one instance per GPU by default:

```bash
export EDIT_MODEL_PATH=/path/to/BAGEL-7B-MoT
bash distributed_services/scripts/start_bagel_simple.sh

export EDIT_MODEL_PATH=/path/to/Qwen-Image-Edit
bash distributed_services/scripts/start_qwen_simple.sh
```

Edit services provide `/edit` and `/health`.

## Start The Reward Service

For self-reward:

```bash
cd /path/to/R3-Refiner

export REWARD_TYPE=self_reward
export SELF_REWARD_MODEL_PATH=/path/to/self_reward_checkpoint
export SELF_REWARD_MODEL_TYPE=qwen2_5vl  # qwen2_5vl or qwen3vl

bash distributed_services/scripts/deploy_services.sh reward_server
```

`REWARD_TYPE` selects the reward family. `SELF_REWARD_MODEL_TYPE` selects the self-reward server implementation.

To disable stage-2 editing during training:

```bash
export ENABLE_STAGE2=false
bash distributed_services/scripts/train_quick.sh
```

To run the full two-stage objective:

```bash
export ENABLE_STAGE2=true
bash distributed_services/scripts/train_quick.sh
```

Stage-2 is enabled by default.

CLIP reward:

```bash
export REWARD_TYPE=clip
bash distributed_services/scripts/deploy_services.sh reward_server ViT-B/32
```

SAM3 reward:

```bash
export REWARD_TYPE=sam3
export SAM3_BPE_PATH=/path/to/sam3/bpe_simple_vocab_16e6.txt.gz
export SAM3_CKPT_PATH=/path/to/sam3/sam3.pt
bash distributed_services/scripts/deploy_services.sh reward_server
```

`SAM3_METADATA_JSONL` is optional. Data records can also store SAM3 metadata in `ground_truth`.

Self-reward and CLIP services provide `/compute_reward` and `/health`. SAM3 services provide `/compute_sam3_reward` and `/health`.

## Source Endpoints On The Training Node

Service startup writes endpoint files under `distributed_services/config/`, for example `edit_server_endpoints.txt`, `self_reward_server_endpoints.txt`, and `sam3_reward_server_endpoints.txt`.

If the service nodes and training node share the repository directory:

```bash
cd /path/to/R3-Refiner
bash distributed_services/scripts/deploy_services.sh get_config
source distributed_services/config/service_endpoints.env
```

If they do not share a filesystem, copy the endpoint text files into the training node's `distributed_services/config/` directory before running `get_config`.

Manual endpoint variables:

```bash
export EDIT_SERVER_ENDPOINTS="http://edit-node:5001,http://edit-node:5002"
export REWARD_TYPE=self_reward
export REWARD_SERVER_ENDPOINTS="http://reward-node:6001,http://reward-node:6002"
export SELF_REWARD_SERVER_ENDPOINTS="$REWARD_SERVER_ENDPOINTS"
```

`get_config` reads existing endpoint files and writes `service_endpoints.env`.

## Start Training

After the edit and reward services are reachable:

```bash
cd /path/to/R3-Refiner
source distributed_services/config/service_endpoints.env

export MODEL_PATH=/path/to/policy_or_base_vlm
export TRAIN_DATA_PATH=examples/data/demo_train.json
export VAL_DATA_PATH=examples/data/demo_train.json
export IMAGE_DIR=examples/data/images

bash distributed_services/scripts/train_quick.sh
```

`train_quick.sh` defaults to:

```bash
FORMAT_PROMPT=examples/format_prompt/refiner_edit.jinja
REWARD_FUNCTION=examples/reward_function/self_reward_staged_reward_api.py:compute_score
```

The training scripts use `examples/data/demo_train.json` by default and set `ROLLOUT_BATCH_SIZE=64` for that file unless you override it.

For LLaVA-OneVision:

```bash
export ROLLOUT_BATCH_SIZE=64  # useful when using the demo JSON
bash distributed_services/scripts/train_llava_onevision.sh
```

## Scaling

Use `GPUS_PER_NODE` or `distributed_services/config/config.yaml` to set the number of service GPUs. Use `INSTANCES_PER_GPU` to set service instances per GPU.

To add service nodes, start the same service command on each node and copy their endpoint files to the training node before running `get_config`.

Mixed reward mode averages SAM3 with one base reward. Use either `self_reward` or `clip` as the base reward for a run; do not mix both in the same `REWARD_TYPE_PER_GPU` mapping.

This mapping decides which reward service starts on each local GPU:

```bash
export REWARD_TYPE=mixed
export REWARD_TYPE_PER_GPU="0:self_reward,1:self_reward,2:sam3,3:sam3"
bash distributed_services/scripts/deploy_services.sh reward_server
```

For multi-node Ray training:

```bash
# Head training node
ray stop --force
ray start --head

# Worker training node
ray stop --force
ray start --address='HEAD_NODE_IP:6379'
bash distributed_services/scripts/ray.sh

# Head node
export NNODES=2
export N_GPUS_PER_NODE=4
source distributed_services/config/service_endpoints.env
bash distributed_services/scripts/train_quick.sh
```

`ray.sh` is only a keep-alive helper after `ray start --address=...`; it does not start Ray by itself.

## Checks And Shutdown

```bash
curl http://edit-node:5001/health
curl http://reward-node:6001/health

bash distributed_services/scripts/deploy_services.sh stop_edit
bash distributed_services/scripts/deploy_services.sh stop_reward
bash distributed_services/scripts/deploy_services.sh stop
```

Service logs are written under `distributed_services/logs/`.
