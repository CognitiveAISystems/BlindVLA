from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import draccus
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForVision2Seq, AutoProcessor

@dataclass
class AttentionMapsConfig:
    """Configuration for OpenVLA attention map visualization."""

    vla_path: str = "openvla/openvla-7b"                     # Base OpenVLA model HF id or path
    lora_root: Optional[Path] = None                         # Root dir with LoRA adapter (or None)
    lora_rank: int = 8                                       # Rank for new LoRA if adapter not found
    unnorm_key: Optional[str] = "train"                      # Key for norm_stats injection from dataset_statistics.json

    attn_implementation: str = "eager"                       # "eager"
    dtype: str = "bfloat16"                                  # "bfloat16", "float16", "float32"
    device: str = "cuda"                                     # "cuda" or "cpu"

    image_path: Path = Path("PATH_TO_INPUT_IMAGE")
    question: str = "Do you see a can?"
    general_prompt: str = "Describe the scene briefly."

    layers: str = "15,16,17,18"                              # string with comma separated layers, for example: "15,16,17,18" or "16,"
    output_dir: Path = Path("attention_maps")

    dpi: int = 140


def str_to_dtype(name: str) -> torch.dtype:
    """Map a string dtype name to a torch.dtype."""
    name = name.lower()
    if name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if name in {"float16", "fp16", "half"}:
        return torch.float16
    if name in {"float32", "fp32", "full"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def get_device(name: str) -> torch.device:
    """Return torch.device from a user-friendly string."""
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def find_lora_path(root: str | Path) -> Optional[Path]:
    """Search for a LoRA adapter folder under the given root.

    A valid adapter directory contains `adapter_config.json` and
    either `adapter_model.bin` or `adapter_model.safetensors`.
    """
    root = Path(root)
    if not root.exists():
        return None

    if (root / "adapter_config.json").exists() and (
        (root / "adapter_model.bin").exists()
        or (root / "adapter_model.safetensors").exists()
    ):
        return root

    for p in root.rglob("adapter_config.json"):
        parent = p.parent
        if (parent / "adapter_model.bin").exists() or (parent / "adapter_model.safetensors").exists():
            return parent

    return None


def attach_openvla_lora(
    base_model: torch.nn.Module,
    device: torch.device,
    lora_rank: Optional[int] = None,
    lora_path: Optional[str | Path] = None,
    unnorm_key: Optional[str] = None,
    make_trainable: bool = False,
) -> torch.nn.Module:
    """Attach LoRA to an OpenVLA model, from path or by creating a new adapter."""
    if lora_path is None:
        if lora_rank is None:
            raise ValueError("lora_rank must be provided when lora_path is None.")

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=min(lora_rank, 16),
            lora_dropout=0.0,
            target_modules=[
                "proj", "qkv", "fc1", "fc2",         # vision
                "q", "kv", "fc3",                    # projector
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                "lm_head",                           # LLM
            ],
            init_lora_weights="gaussian",
        )
        peft_model = get_peft_model(base_model, lora_config)
        for p in peft_model.parameters():
            p.requires_grad_(make_trainable)
        return peft_model.to(device).eval()

    lora_path = Path(lora_path)
    peft_model = PeftModel.from_pretrained(
        base_model,
        lora_path,
        is_trainable=make_trainable,
    ).to(device).eval()
    print(f"[LoRA] Loaded from: {lora_path}")

    try:
        ds_path = lora_path / "dataset_statistics.json"
        if (
            ds_path.exists()
            and hasattr(peft_model, "base_model")
            and hasattr(peft_model.base_model, "norm_stats")
        ):
            with ds_path.open("r") as f:
                ds = json.load(f)
            if unnorm_key and (unnorm_key not in peft_model.base_model.norm_stats) and (unnorm_key in ds):
                peft_model.base_model.norm_stats[unnorm_key] = ds[unnorm_key]
                print(f"[LoRA] Injected norm_stats['{unnorm_key}'] from dataset_statistics.json")
    except Exception as e:
        print(f"[LoRA] norm_stats injection skipped: {e}")

    return peft_model


def guess_grid_hw(n_tokens: int) -> Tuple[int, int, int]:
    """Heuristically infer (offset, H, W) from the number of visual tokens."""
    r = int(math.sqrt(n_tokens))
    if r * r == n_tokens:
        return 0, r, r

    r1 = int(math.sqrt(n_tokens - 1))
    if (n_tokens - 1) > 0 and r1 * r1 == (n_tokens - 1):
        return 1, r1, r1

    r2 = int(math.floor(math.sqrt(n_tokens)))
    return 0, r2, r2


def load_model_and_processor(
    cfg: AttentionMapsConfig,
) -> tuple[AutoProcessor, torch.nn.Module, torch.device, torch.dtype]:
    """Load OpenVLA processor and model, attach LoRA if available."""
    device = get_device(cfg.device)
    dtype = str_to_dtype(cfg.dtype)

    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        trust_remote_code=True,
        attn_implementation=cfg.attn_implementation,
        torch_dtype=dtype,
    ).eval().to(device)

    resolved_lora_path: Optional[Path]
    if cfg.lora_root is not None:
        resolved_lora_path = find_lora_path(cfg.lora_root)
    else:
        resolved_lora_path = None

    if resolved_lora_path is not None:
        model = attach_openvla_lora(
            base_model=model,
            device=device,
            lora_path=resolved_lora_path,
            unnorm_key=cfg.unnorm_key,
            make_trainable=False,
        )
    else:
        if cfg.lora_root is not None:
            print(
                f"[LoRA] Adapter not found under '{cfg.lora_root}', "
                f"creating new LoRA (rank={cfg.lora_rank})"
            )
        model = attach_openvla_lora(
            base_model=model,
            device=device,
            lora_rank=cfg.lora_rank,
            make_trainable=True,
        )

    try:
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()
    except Exception:
        pass

    return processor, model, device, dtype


def prepare_inputs(
    processor: AutoProcessor,
    image: Image.Image,
    question: str,
    general_prompt: str,
    device: torch.device,
    model_dtype: torch.dtype,
) -> tuple[dict, dict]:
    """Tokenize image + prompts for query and general descriptions."""
    prompt_query = f"In: {question}\nOut:"
    prompt_general = f"In: {general_prompt}\nOut:"

    inputs_query = processor(prompt_query, image, return_tensors="pt").to(device)
    inputs_general = processor(prompt_general, image, return_tensors="pt").to(device)

    for d in (inputs_query, inputs_general):
        for k, v in list(d.items()):
            if torch.is_floating_point(v):
                d[k] = v.to(dtype=model_dtype)

    return inputs_query, inputs_general


def parse_layers(layers_str: str) -> tuple[int, ...]:
    """Parse comma-separated layer indices into a tuple of ints.

    Example:
        "15,16,17,18" -> (15, 16, 17, 18)
        "16,"         -> (16,)
    """
    return tuple(int(x) for x in layers_str.split(",") if x.strip())


def run_attention_maps(cfg: AttentionMapsConfig) -> None:
    """ Compute and save attention maps for selected layers, inspired by https://github.com/WanyueZhang-ai/spatial-understanding """

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    processor, model, device, model_dtype = load_model_and_processor(cfg)

    layer_indices = parse_layers(cfg.layers)
    if not layer_indices:
        raise ValueError(f"No valid layers parsed from string: {cfg.layers!r}")

    img = Image.open(cfg.image_path).convert("RGB")
    original_img = mpimg.imread(cfg.image_path)

    inputs_query, inputs_general = prepare_inputs(
        processor=processor,
        image=img,
        question=cfg.question,
        general_prompt=cfg.general_prompt,
        device=device,
        model_dtype=model_dtype,
    )

    with torch.no_grad():
        out_query = model(
            **inputs_query,
            output_attentions=True,
            output_projector_features=True,
        )
        out_general = model(
            **inputs_general,
            output_attentions=True,
            output_projector_features=True,
        )

    n_vis = out_query.projector_features.shape[1]
    pos, pos_end = 1, 1 + n_vis
    offset, H, W = guess_grid_hw(n_vis)
    eps = 1e-6

    print(f"[Attention] n_vis={n_vis}, grid={H}x{W}, offset={offset}, layers={layer_indices}")

    for layer_idx in layer_indices:
        if layer_idx < 0 or layer_idx >= len(out_query.attentions):
            print(f"[Warning] Layer {layer_idx} is out of range, skipping.")
            continue

        att_q = out_query.attentions[layer_idx][0, :, -1, pos:pos_end].to(torch.float32).mean(dim=0)
        att_g = out_general.attentions[layer_idx][0, :, -1, pos:pos_end].to(torch.float32).mean(dim=0)

        att = (att_q / (att_g + eps)).detach().cpu().numpy()

        att_to_plot = att[1:] if (offset == 1 and att.shape[0] > 1) else att

        grid_n = H * W
        if att_to_plot.shape[0] >= grid_n:
            att_to_plot = att_to_plot[:grid_n]
        else:
            att_to_plot = np.pad(att_to_plot, (0, grid_n - att_to_plot.shape[0]))

        att_img = att_to_plot.reshape(H, W)

        plt.figure(figsize=(5, 5))
        plt.imshow(att_img, cmap="viridis", interpolation="nearest")
        plt.title(f"Layer {layer_idx + 1} — {cfg.question}")
        plt.axis("off")

        layer_path = cfg.output_dir / f"openvla_layer_{layer_idx + 1:02d}.png"
        plt.savefig(layer_path, bbox_inches="tight", dpi=cfg.dpi)
        plt.close()
        print(f"[Attention] Saved: {layer_path}")

    orig_path = cfg.output_dir / "original_image.png"
    plt.figure(figsize=(5, 5))
    plt.imshow(original_img)
    plt.title("Original Image")
    plt.axis("off")
    plt.savefig(orig_path, bbox_inches="tight", dpi=cfg.dpi)
    plt.close()
    print(f"[Attention] Saved: {orig_path}")


@draccus.wrap()
def main(cfg: AttentionMapsConfig) -> None:
    run_attention_maps(cfg)


if __name__ == "__main__":
    main()