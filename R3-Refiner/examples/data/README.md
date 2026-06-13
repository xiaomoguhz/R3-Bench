# Data Format

Datasets are not included in this repository. Put local JSON/JSONL files and images here locally, or point the launch scripts to external paths with:

```bash
export TRAIN_DATA_PATH=/path/to/train.json
export VAL_DATA_PATH=/path/to/val.json
export IMAGE_DIR=/path/to/images
```

## Demo Training Set

`demo_train.json` is a compact training demo with 97 manually reviewed examples. It is kept roughly balanced across eight dimensions (`object`, `color`, `shape`, `texture`, `spatial`, `numeracy`, `non`, and `complex`).

The examples are sampled from the paper's training data construction sources:

- T2I-R1 generate-and-rank examples.
- BLIP-3O counterfactual rewriting examples.
- PICO-Banana visual inversion examples.

The JSON uses image paths relative to `examples/data/images`:

```bash
export TRAIN_DATA_PATH=examples/data/demo_train.json
export VAL_DATA_PATH=examples/data/demo_train.json
export IMAGE_DIR=examples/data/images
export ROLLOUT_BATCH_SIZE=64
```

The demo split has fewer than 128 records, so use a rollout batch size such as 64 unless you replace it with a larger training set.

The image files are kept out of git. The demo image archive is released at [nickname-xingxing/R3-Refiner_demoTrain](https://huggingface.co/datasets/nickname-xingxing/R3-Refiner_demoTrain). Download and unpack it so the directory layout is:

```text
examples/data/images/demo_train/
```

Example download flow:

```bash
export HF_DATASET_REPO=nickname-xingxing/R3-Refiner_demoTrain
huggingface-cli download "$HF_DATASET_REPO" demo_train_images.zip --repo-type dataset --local-dir examples/data
unzip -o examples/data/demo_train_images.zip -d examples/data/images
```

The default config uses `data.answer_key=ground_truth`, so each item should contain at least:

```json
{
  "prompt": "Describe the target image or verification question.",
  "images": ["demo_train/example_000001.png"],
  "ground_truth": "{\"answer\": false, \"category\": \"object\", \"prompt\": \"...\"}"
}
```

`ground_truth` is a JSON-encoded string. Released examples should use paths relative to `IMAGE_DIR`.

## Fields

- `prompt`: the original generation prompt or verification target. It is formatted by `examples/format_prompt/*.jinja` before being sent to the policy.
- `images`: a list of input image paths. The current training setup uses one image per item.
- `ground_truth.answer`: boolean label used by the first-stage MLLM judgment reward.
- `ground_truth.category`: optional task category, such as `object`, `color`, `shape`, `texture`, `spatial`, `numeracy`, `non`, or `complex`.
- `ground_truth.prompt`: prompt text used by standalone reward calls and stage-2 reward scoring.
- `ground_truth.reward_type`: optional override for the second-stage reward, for example `self_reward`, `clip`, `sam3`, or `mixed`.

For stage-1 MLLM judgment, `answer` is required. For the default two-stage training path, keep `prompt`, `category`, and the metadata required by the selected stage-2 reward.

## Direct MLLM Reward

```json
{
  "prompt": "a red apple and a green orange",
  "images": ["demo_train/example_000001.png"],
  "ground_truth": "{\"answer\": false, \"category\": \"color\", \"prompt\": \"a red apple and a green orange\"}"
}
```

## Question Decomposition Reward

The self-reward server also accepts decomposed yes/no questions. Store them in `ground_truth.generated_qa`:

```json
{
  "prompt": "a red apple and a green orange",
  "images": ["demo_train/example_000001.png"],
  "ground_truth": "{\"answer\": false, \"category\": \"color\", \"prompt\": \"a red apple and a green orange\", \"reward_type\": \"self_reward\", \"generated_qa\": {\"yn_question_list\": [\"Is there a red apple in the image?\", \"Is there a green orange in the image?\"]}}"
}
```

If `generated_qa` is provided, the reward service scores the edited image by the fraction of yes/no questions answered true.

## CLIP Reward

CLIP only needs the image and prompt. To use CLIP globally with `train_quick.sh`, set `REWARD_KWARGS` with `default_reward_type=clip`. Per-sample routing can use `ground_truth.reward_type`:

```json
{
  "prompt": "a black candle and a white holder",
  "images": ["demo_train/example_000002.png"],
  "ground_truth": "{\"answer\": false, \"category\": \"color\", \"prompt\": \"a black candle and a white holder\", \"reward_type\": \"clip\"}"
}
```

## SAM3 Reward

SAM3 needs object metadata in `ground_truth`. For `object`/`non` use `nouns`:

```json
{
  "prompt": "a photo of a red car",
  "images": ["demo_train/example_000003.png"],
  "ground_truth": "{\"answer\": true, \"category\": \"object\", \"prompt\": \"a photo of a red car\", \"reward_type\": \"sam3\", \"nouns\": [\"red car\"]}"
}
```

For `color`, `shape`, and `texture`, use `attr_nouns` for attribute-aware targets:

```json
{
  "prompt": "a purple airplane and a pink toaster",
  "images": ["demo_train/example_000004.png"],
  "ground_truth": "{\"answer\": false, \"category\": \"color\", \"prompt\": \"a purple airplane and a pink toaster\", \"reward_type\": \"sam3\", \"nouns\": [\"airplane\", \"toaster\"], \"attr_nouns\": [\"purple airplane\", \"pink toaster\"]}"
}
```

For `spatial`, add `spatial_info`:

```json
{
  "prompt": "the cup is on the left of the plate",
  "images": ["demo_train/example_000005.png"],
  "ground_truth": "{\"answer\": false, \"category\": \"spatial\", \"prompt\": \"the cup is on the left of the plate\", \"reward_type\": \"sam3\", \"nouns\": [\"cup\", \"plate\"], \"spatial_info\": {\"obj1\": \"cup\", \"obj2\": \"plate\", \"locality\": \"on the left of\"}}"
}
```

For `numeracy`, add `numeracy_info`:

```json
{
  "prompt": "a photo of three donuts",
  "images": ["demo_train/example_000006.png"],
  "ground_truth": "{\"answer\": false, \"category\": \"numeracy\", \"prompt\": \"a photo of three donuts\", \"reward_type\": \"sam3\", \"nouns\": [\"donut\"], \"numeracy_info\": [{\"obj_name\": \"donut\", \"num\": 3}]}"
}
```

For `complex`, use `nouns` and optionally `spatial_info`.
