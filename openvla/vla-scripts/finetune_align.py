"""
finetune.py

Simple script for parameter-efficient fine-tuning of OpenVLA models loaded through the HuggingFace AutoClasses, using
HuggingFace PEFT library for low-rank adaptation (LoRA).

Notes & Benchmarks:
    - Requires PEFT (`pip install peft==0.11.1`)
    - LoRA fine-tuning (see parameters below -- no quantization, LoRA rank = 32, target_modules = all-linear):
        + One 48 GB GPU can fit a Batch Size of 12
        + One 80 GB GPU can fit a Batch Size of 24

Run with:
    - [Single Node Multi-GPU (= $K) ]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py
    - [Override Config Values]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py \
                                    --data_root_dir <PATH/TO/RLDS/DATASETS/DIRECTORY> \
                                    --dataset_name <DATASET_NAME> \
                                    --run_root_dir <PATH/TO/LOGS/DIR> \
                                    ...
"""

import os
import math
import time
import json
import types
import importlib.util
from datetime import datetime
from collections import deque
from dataclasses import dataclass, asdict, is_dataclass
from pathlib import Path
from typing import List, Optional, Union

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import tqdm
import timm
from PIL import Image
import draccus
from accelerate import PartialState
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoImageProcessor,
    BitsAndBytesConfig,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import TensorType
from huggingface_hub import HfFileSystem, hf_hub_download

from prismatic.models.registry import GLOBAL_REGISTRY, MODEL_REGISTRY
from prismatic.models.materialize import get_vision_backbone_and_transform
from prismatic.models.backbones.llm.prompting import (
    PurePromptBuilder,
    VicunaV15ChatPromptBuilder,
)
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import (
    PrismaticImageProcessor,
    PrismaticProcessor,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"                            # Path to OpenVLA model (on HuggingFace Hub)

    # Directory Paths
    data_root_dir: Path = Path("datasets/open-x-embodiment")        # Path to Open-X dataset directory
    dataset_name: str = "droid_wipe"                                # Name of fine-tuning dataset (e.g., `droid_wipe`)
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    # adapter_tmp_dir: Path = Path("adapter-tmp")                     # Temporary directory for LoRA weights before fusing

    # Fine-tuning Parameters
    batch_size: int = 16                                            # Fine-tuning batch size
    max_steps: int = 200_000                                        # Max number of fine-tuning steps
    eval_steps: int = 50                                            # Interval for checkpoint saving
    save_steps: str = "0"
    learning_rate: float = 5e-4                                     # Fine-tuning learning rate
    grad_accumulation_steps: int = 1                                # Gradient accumulation steps
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 100_000                              # Dataloader shuffle buffer size (can reduce if OOM)
    # save_latest_checkpoint_only: bool = True                        # Whether to save only one checkpoint per run and
                                                                    #   continually overwrite the latest checkpoint
                                                                    #   (If False, saves all checkpoints)
    # LoRA Arguments
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = False                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    #   => CAUTION: Reduces memory but hurts performance
    # Tracking Parameters
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases
    unnorm_key: Optional[str] = None                                # Unnorm key from your fine-tuning dataset

    mode: str = 'alig' # 'orig'                                     # Use Visual representation Alignment or not 

    align_coeff: float = 0.2                                        # Aligning loss weight for total loss 
    align_layers: str = "16, "                                      # VLA layers to align with frozen teacher

    exp_note: str = ''                                              # Experiment note
    projector_dim: int = 2048                                       # Inner dim for projector from vla's dim to teacher dim 
    teacher_model_id: str = "c-radio_v3-l" # 'dinov2-vit-l', 'dinov2-vit-g', "c-radio_v3-h", "theaiinstitute/theia-base-patch16-224-cdiv", ""theaiinstitute/theia-base-patch16-224-cddsv""
    freeze_alignment_projector: bool = True


def denormalize_openvla(x: torch.Tensor) -> torch.Tensor:
    mean_openvla = [0.484375, 0.455078125, 0.40625]
    std_openvla  = [0.228515625, 0.2236328125, 0.224609375]

    mean = torch.tensor(mean_openvla, device=x.device).view(1, -1, 1, 1)
    std = torch.tensor(std_openvla, device=x.device).view(1, -1, 1, 1)
    return x * std + mean

class AlignmentProjector(nn.Module):
    def __init__(self, hidden_size, projector_dim, z_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, projector_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(projector_dim, projector_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(projector_dim, z_dim),
        )
    def forward(self, x):
        return self.net(x)
    
class MultiProjector(nn.Module):
    def __init__(self, proj_list: nn.ModuleList):
        super().__init__()
        self.projs = proj_list  

    def forward(self, feats):
        return [m(x) for m, x in zip(self.projs, feats)]


def load_teacher_vision_backbone(model_id_or_path):

    if 'dinov2' in model_id_or_path:        
        alignment_encoder = torch.hub.load('facebookresearch/dinov2', model_id_or_path)
        del alignment_encoder.head
        patch_resolution = 16

        alignment_encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
            alignment_encoder.pos_embed.data, [patch_resolution, patch_resolution],
        )
        alignment_encoder.head = nn.Identity()
    elif 'radio' in model_id_or_path:
        alignment_encoder = torch.hub.load('NVlabs/RADIO', 'radio_model', version=model_id_or_path, progress=True, skip_validation=True)

    elif 'theia' in model_id_or_path:
        alignment_encoder = AutoModel.from_pretrained(model_id_or_path, trust_remote_code=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    alignment_encoder.to(device, dtype=dtype).eval()

    for p in alignment_encoder.parameters():
        p.requires_grad_(False)
    print('Teacher Vision Backbone ready!')

    return alignment_encoder


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_step_list = [int(x) for x in cfg.save_steps.split(",") if x.strip() != ""]
    cfg.align_layers = tuple(int(x) for x in cfg.align_layers.split(',') if x.strip())

    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")
    assert cfg.use_lora

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    # Configure Unique Experiment ID & Log Directory
    exp_id = f"steps_{cfg.max_steps}"
    if not cfg.image_aug:
        exp_id += "-no_aug"

    # Start =>> Build Directories

    name = f"{cfg.exp_note}"
    run_dir = cfg.run_root_dir / name
    os.makedirs(run_dir, exist_ok=True)

    # Quantization Config =>> only if LoRA fine-tuning
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    # [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()


    ### MODIFICATION: LOAD TEACHER 
    if cfg.mode == 'alig':
        teacher_vision_backbone = load_teacher_vision_backbone(cfg.teacher_model_id)

        in_features = vla.language_model.model.embed_tokens.embedding_dim
        if 'dinov' in cfg.teacher_model_id:
            out_features = teacher_vision_backbone.patch_embed.proj.out_channels #teacher_vision_backbone.dino_featurizer.patch_embed.proj.out_channels
        elif 'radio' in cfg.teacher_model_id:
            out_features = teacher_vision_backbone.model.patch_generator.embed_dim
        elif 'theia' in cfg.teacher_model_id:
            out_features = teacher_vision_backbone._modules['backbone'].model.embeddings.patch_embeddings.projection.out_channels

        projectors_list = nn.ModuleList([
            AlignmentProjector(in_features, cfg.projector_dim, out_features) for _ in range(len(cfg.align_layers))
        ]).train()
        alignment_projector = MultiProjector(projectors_list).to(device_id).train()


        if cfg.freeze_alignment_projector:
            for p in alignment_projector.parameters():
                p.requires_grad = False

            alignment_projector.eval() 

        print('Vision Teacher Loaded!!!')


    # Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training
    vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

    # Create Optimizer =>> note that we default to a simple constant learning rate!
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    )
    unnorm_stats = vla.module.base_model.norm_stats[cfg.unnorm_key] if cfg.unnorm_key else None
    vla_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
        train=True,
        unnorm_stats=unnorm_stats
    )

    # [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
    if distributed_state.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)
        print(f"Using {cfg.unnorm_key} for dataset statistics")

    # Create Collator and DataLoader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )


    # eval
    vla_dataset_eval = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
        train=False,
        unnorm_stats=unnorm_stats
    )
    dataloader_eval = DataLoader(
        vla_dataset_eval,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )

    # Initialize Logging =>> TensorBoard
    if distributed_state.is_main_process:
        writer = SummaryWriter(log_dir=run_dir)
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in asdict(cfg).items()])),
        )

 
    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)

    # Train!
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            with torch.autocast("cuda", dtype=torch.bfloat16):

                pixel_val = batch["pixel_values"].to(torch.bfloat16).to(device_id)
                output: CausalLMOutputWithPast = vla(
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    pixel_values=pixel_val,
                    labels=batch["labels"],
                    output_hidden_states=True,
                )
                loss = output.loss

            ### MODIFICATION: align loss

                if cfg.mode == 'alig':

                    ### MODIFICATION: get vla features
                    n_vis = output.projector_features.shape[1]
                    pos = 1
                    pos_end = pos + n_vis

                    vla_features = [output.hidden_states[i][:, pos:pos_end] for i in cfg.align_layers]
                    vla_features = alignment_projector(vla_features)

                    del output.hidden_states

                    ### MODIFICATION: get teacher features 
                    with torch.no_grad():
                        pixel_values_init, pixel_values_fused = torch.split(pixel_val, [3, 3], dim=1)
                        if 'dinov2' in cfg.teacher_model_id:
                            teacher_features = teacher_vision_backbone.forward_features(pixel_values_init)['x_norm_patchtokens']
                        elif 'radio' in cfg.teacher_model_id:
                            pixel_values_init = F.interpolate(pixel_values_init, size=(256, 256), mode="bilinear")
                            pixel_values_init = denormalize_openvla(pixel_values_init) 
                            if "e-radio" in cfg.teacher_model_id:
                                teacher_vision_backbone.model.set_optimal_window_size(x.shape[2:])

                            summary, teacher_features = teacher_vision_backbone(pixel_values_init)
                            del summary
                        elif 'theia' in cfg.teacher_model_id:
                            pixel_values_init = F.interpolate(pixel_values_init, size=(256, 256), mode="bilinear")
                            teacher_features = teacher_vision_backbone.forward_feature(pixel_values_init.to(torch.float), interpolate_pos_encoding=True, do_resize=False, do_rescale=False, do_normalize=False).to(torch.bfloat16)


                    if cfg.mode == 'alig':

                        align_loss = 0.0
                        for idx in range(len(cfg.align_layers)):

                            emb_t = F.normalize(teacher_features, dim=-1)
                            emb_s = F.normalize(vla_features[idx], dim=-1)

                            cossim = (emb_t * emb_s).sum(dim=-1)
                            align_loss += (-cossim).mean()

                            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps
                            if distributed_state.is_main_process and gradient_step_idx % 10 == 0:
                                layer_n = cfg.align_layers[idx]
                                writer.add_scalar(f"cossim_layer_{layer_n}", cossim.mean().item(), gradient_step_idx)

                    loss += cfg.align_coeff * align_loss

            # Normalize loss to account for gradient accumulation
            normalized_loss = loss / cfg.grad_accumulation_steps

            # Backward pass
            normalized_loss.backward()

            # Compute Accuracy and L1 Loss for Logging
            action_logits = output.logits[:, vla.module.vision_backbone.featurizer.patch_embed.num_patches : -1]
            action_preds = action_logits.argmax(dim=2)
            action_gt = batch["labels"][:, 1:].to(action_preds.device)
            mask = action_gt > action_tokenizer.action_token_begin_idx

            # Compute Accuracy
            correct_preds = (action_preds == action_gt) & mask
            action_accuracy = correct_preds.sum().float() / mask.sum().float()

            # Compute L1 Loss on Predicted (Continuous) Actions
            continuous_actions_pred = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
            )
            continuous_actions_gt = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
            )
            action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

            # Store recent train metrics
            recent_losses.append(loss.item())
            recent_action_accuracies.append(action_accuracy.item())
            recent_l1_losses.append(action_l1_loss.item())

            # Compute gradient step index
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps

            # Compute smoothened train metrics
            #   =>> Equal to current step metrics when not using gradient accumulation
            #   =>> Otherwise, equal to the average of metrics observed over micro-batches used for gradient accumulation
            smoothened_loss = sum(recent_losses) / len(recent_losses)
            smoothened_action_accuracy = sum(recent_action_accuracies) / len(recent_action_accuracies)
            smoothened_l1_loss = sum(recent_l1_losses) / len(recent_l1_losses)

            # Push Metrics to TensorBoard (every 10 gradient steps)
            if distributed_state.is_main_process and gradient_step_idx % 10 == 0:
                writer.add_scalar("train_loss", smoothened_loss, gradient_step_idx)
                writer.add_scalar("action_accuracy", smoothened_action_accuracy, gradient_step_idx)
                writer.add_scalar("l1_loss", smoothened_l1_loss, gradient_step_idx)

                if cfg.mode == 'alig':
                    writer.add_scalar("align_loss", align_loss.mean().item(), gradient_step_idx)


            # Optimizer Step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                progress.update()


            if gradient_step_idx % cfg.eval_steps == 0 or gradient_step_idx == cfg.max_steps:
                eval_losses, eval_action_accuracies, eval_l1_losses = [], [], []
                for eval_idx, eval_batch in enumerate(dataloader_eval):
                    with torch.no_grad():
                        output_eval: CausalLMOutputWithPast = vla(
                            input_ids=eval_batch["input_ids"].to(device_id),
                            attention_mask=eval_batch["attention_mask"].to(device_id),
                            pixel_values=eval_batch["pixel_values"].to(torch.bfloat16).to(device_id),
                            labels=eval_batch["labels"],
                        )
                        loss = output_eval.loss

                        # Compute Accuracy and L1 Loss for Logging
                        action_logits = output_eval.logits[:,
                                        vla.module.vision_backbone.featurizer.patch_embed.num_patches: -1]
                        action_preds = action_logits.argmax(dim=2)
                        action_gt = eval_batch["labels"][:, 1:].to(action_preds.device)
                        mask = action_gt > action_tokenizer.action_token_begin_idx

                        # Compute Accuracy
                        correct_preds = (action_preds == action_gt) & mask
                        action_accuracy = correct_preds.sum().float() / mask.sum().float()

                        # Compute L1 Loss on Predicted (Continuous) Actions
                        continuous_actions_pred = torch.tensor(
                            action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
                        )
                        continuous_actions_gt = torch.tensor(
                            action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
                        )
                        action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

                    eval_losses.append(loss.item())
                    eval_action_accuracies.append(action_accuracy.item())
                    eval_l1_losses.append(action_l1_loss.item())

                eval_loss = sum(eval_losses) / len(eval_losses)
                eval_action_accuracy = sum(eval_action_accuracies) / len(eval_action_accuracies)
                eval_l1_loss = sum(eval_l1_losses) / len(eval_l1_losses)

                # Push Eval Metrics
                if distributed_state.is_main_process:
                    writer.add_scalar("eval_loss", eval_loss, gradient_step_idx)
                    writer.add_scalar("eval_action_accuracy", eval_action_accuracy, gradient_step_idx)
                    writer.add_scalar("eval_l1_loss", eval_l1_loss, gradient_step_idx)

                dist.barrier()


            # Save Model Checkpoint =>> by default, only keeps the latest checkpoint, continually overwriting it!
            if gradient_step_idx in save_step_list:
                if distributed_state.is_main_process:
                    print(f"Saving Model Checkpoint for Step {gradient_step_idx}")

                    # If LoRA, we first save adapter weights, then merge into full model; otherwise, default save!
                    lora_save_dir = run_dir / f"lora_{gradient_step_idx:0>6d}"

                    # Save Processor & Weights
                    processor.save_pretrained(run_dir)
                    vla.module.save_pretrained(lora_save_dir)

                    # ---- NEW: сохраняем alignment projector (только первый из списка) ----
                    ap_dir = lora_save_dir / "alignment_projector"
                    ap_dir.mkdir(parents=True, exist_ok=True)

                    torch.save(projectors_list[0].state_dict(), ap_dir / "pytorch_model.bin")
                    print(f"[alignment_projector] saved → {ap_dir / 'pytorch_model.bin'}")

                    save_dataset_statistics(vla_dataset.dataset_statistics, lora_save_dir)
                    print(f"Using {cfg.unnorm_key} for dataset statistics")

                # Wait for processor and adapter weights to be saved by main process
                dist.barrier()

            # Stop training when max_steps is reached
            if gradient_step_idx == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                writer.close()

                break


if __name__ == "__main__":
    finetune()
