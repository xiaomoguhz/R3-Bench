

from typing import Optional

import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..utils.dataset import RLHFDataset, collate_fn
from .config import DataConfig


def create_dataloader(config: DataConfig, tokenizer: PreTrainedTokenizer, processor: Optional[ProcessorMixin]) -> None:
    train_filter = config.train_filter_overlong_prompts if getattr(config, "train_filter_overlong_prompts", None) is not None else config.filter_overlong_prompts
    train_dataset = RLHFDataset(
        data_path=config.train_files,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=config.prompt_key,
        answer_key=config.answer_key,
        image_key=config.image_key,
        video_key=config.video_key,
        image_dir=config.image_dir,
        video_fps=config.video_fps,
        max_prompt_length=config.max_prompt_length,
        truncation="right",
        format_prompt=config.format_prompt,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
        filter_overlong_prompts=train_filter,
        filter_overlong_prompts_workers=config.filter_overlong_prompts_workers,
    )
    print(f"[INFO] Training dataset size: {len(train_dataset)} (filter_overlong_prompts={train_filter})")
    # use sampler for better ckpt resume
    if config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(config.seed)
        sampler = RandomSampler(data_source=train_dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=train_dataset)

    if config.mini_rollout_batch_size is not None:
        train_batch_size = config.mini_rollout_batch_size
    else:
        train_batch_size = config.rollout_batch_size

    train_dataloader = StatefulDataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        sampler=sampler,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=True,
    )

    # If val_files is empty, create an empty dummy dataset
    if not config.val_files or config.val_files.strip() == "":
        from torch.utils.data import TensorDataset
        # Create an empty dummy dataset
        dummy_tensor = torch.zeros(0, dtype=torch.long)
        val_dataset = TensorDataset(dummy_tensor)
        val_batch_size = 1
    else:
        val_filter = config.val_filter_overlong_prompts if config.val_filter_overlong_prompts is not None else config.filter_overlong_prompts
        val_dataset = RLHFDataset(
            data_path=config.val_files,
            tokenizer=tokenizer,
            processor=processor,
            prompt_key=config.prompt_key,
            answer_key=config.answer_key,
            image_key=config.image_key,
            image_dir=config.image_dir,
            max_prompt_length=config.max_prompt_length,
            truncation="right",
            format_prompt=config.format_prompt,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
            filter_overlong_prompts=val_filter,
        )
        print(f"[INFO] Validation dataset size: {len(val_dataset)} (filter_overlong_prompts={val_filter})")
        # Cap validation set size if val_max_samples is set
        if config.val_max_samples is not None and len(val_dataset) > config.val_max_samples:
            print(f"[INFO] Capping validation dataset: {len(val_dataset)} -> {config.val_max_samples}")
            # Use a fixed random seed for reproducibility
            import random
            random.seed(config.seed)
            indices = list(range(len(val_dataset)))
            random.shuffle(indices)
            selected_indices = indices[:config.val_max_samples]
            # Build the subset
            from torch.utils.data import Subset
            val_dataset = Subset(val_dataset, selected_indices)
            print(f"[INFO] Validation dataset subsampled to {len(val_dataset)} samples")
        
        if config.val_batch_size == -1:
            val_batch_size = len(val_dataset)
        else:
            val_batch_size = config.val_batch_size

    val_dataloader = StatefulDataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False,
    )

    if len(train_dataloader) < 1:
        raise ValueError(
            "Training dataloader is empty. "
            f"dataset_size={len(train_dataset)}, train_batch_size={train_batch_size}, drop_last=True. "
            "Lower ROLLOUT_BATCH_SIZE or provide at least one full training batch."
        )
    # The validation dataloader may legitimately be empty
    if len(val_dataloader) == 0:
        print("Warning: Validation dataloader is empty. Validation will be skipped.")
    print(f"Size of train dataloader: {len(train_dataloader)}")
    print(f"Size of val dataloader: {len(val_dataloader)}")
    return train_dataloader, val_dataloader
