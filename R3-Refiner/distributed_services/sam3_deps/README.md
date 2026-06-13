# SAM3 Reward Support

This directory contains the SAM3-based reward helpers used by the distributed reward service. SAM3 can be used as a standalone reward backend (`REWARD_TYPE=sam3`) or as one branch of a mixed reward setup.

## Inputs

The server receives an image, a prompt, a category, and the original `ground_truth` JSON string. SAM3 metadata is read from `ground_truth` first. `SAM3_METADATA_JSONL` is optional and is only used when metadata is not embedded in the training item.

Supported metadata fields:

- `nouns`: object names for `object`, `non`, and `complex`.
- `attr_nouns`: attribute-qualified targets for `color`, `shape`, and `texture`.
- `spatial_info`: `{"obj1": "...", "obj2": "...", "locality": "..."}` for spatial relations.
- `numeracy_info`: `[{"obj_name": "...", "num": 3}]` for counting.

Supported `spatial_info.locality` values are:

```text
on the left of
on the right of
on the top of
on the bottom of
next to
near
on side of
```

The scorer also normalizes simple aliases such as `left of`, `right of`, `above`, and `below`.

## Service Setup

SAM3 service startup requires the SAM3 tokenizer and checkpoint paths:

```bash
export REWARD_TYPE=sam3
export SAM3_BPE_PATH=/path/to/sam3/bpe_simple_vocab_16e6.txt.gz
export SAM3_CKPT_PATH=/path/to/sam3/sam3.pt
bash distributed_services/scripts/deploy_services.sh reward_server
```

Optional metadata file:

```bash
export SAM3_METADATA_JSONL=/path/to/metadata.jsonl
```

The service exposes:

```text
GET  /health
POST /compute_sam3_reward
```

## Scoring

Spatial:

1. Detect `obj1` and `obj2`.
2. Use the highest-confidence box for each object.
3. Combine object confidence and normalized position score.

Numeracy:

1. Detect each expected object type.
2. Score both object presence and exact count match.

Object/color/shape/texture/non:

1. Detect the expected targets.
2. Score detection coverage weighted by confidence.

Complex:

- If `spatial_info` exists, use the spatial score.
- Otherwise use object-presence scoring over `nouns`.

## Metadata Merge Utility

If your data keeps SAM3 metadata in a separate JSONL file, merge it into `ground_truth` before training:

```bash
python distributed_services/sam3_deps/merge_metadata.py \
  --train_data /path/to/train.json \
  --metadata_jsonl /path/to/metadata.jsonl \
  --output /path/to/train_with_metadata.json
```

New public demo data already stores the required SAM3 fields in `ground_truth`, so this merge step is not needed for the demo set.
