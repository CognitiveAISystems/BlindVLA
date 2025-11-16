from __future__ import annotations

import json
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import draccus
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.manifold import TSNE
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor

@dataclass
class TsneConfig:
    """Configuration for OpenVLA t-SNE visualization on COCO."""

    dataset_dir: Path = Path("./coco2017")           # Root COCO dir (with train2017/ and annotations/)
    instances_filename: str = "instances_train2017.json"

    selected_classes: str = "cup,bottle,knife"       # Comma-separated COCO class names
    max_per_class: int = 10_000                     # Max images per class when building pool
    max_samples: int = 10_000                       # Max images to use for t-SNE

    model_id: str = "openvla/openvla-7b"
    attn_implementation: str = "eager"              # "eager" / "sdpa" / etc.
    dtype: str = "bfloat16"                          # "bfloat16", "float16", "float32"
    device: str = "cuda"                            # "cuda" or "cpu"

    mode: str = "text_object_token"                 # "last_text_token" | "text_object_token" | "vis_mean" | "vis_cls" | "vis_pool_attn"
    layers: str = "1,10,20"                         # Comma-separated layer indices, e.g. "1,10,20"

    batch_size: int = 64
    random_seed: int = 42

    tsne_perplexity: float = 30.0
    tsne_max_iter: int = 1000

    layout: str = "auto"                            # "auto" | "horizontal" | "vertical"
    figure_dpi: int = 120
    save_path: Optional[Path] = None                # Optional: where to save the figure (PNG). If None -> only show()

    global_prompt: str = "In: describe the object from the picture.\nOut:"
    per_class_prompt: str = "In: Do you see {class_name} on picture?\nOut:"

def seed_everything(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def parse_layers(layers_str: str) -> Tuple[int, ...]:
    """Parse comma-separated layer indices into a tuple of ints."""
    return tuple(int(x) for x in layers_str.split(",") if x.strip())


def parse_class_list(classes_str: str) -> List[str]:
    """Parse comma-separated class names into a list of strings."""
    return [c.strip() for c in classes_str.split(",") if c.strip()]


def _to_np(x: torch.Tensor) -> np.ndarray:
    """Detach tensor and move it to CPU numpy float32."""
    return x.detach().to(torch.float32).cpu().numpy()

def build_coco_index(instances_json_path: Path):
    """Build COCO indices: image→file, image→cats, cat-id→name, name→cat-id."""
    with instances_json_path.open("r") as f:
        data = json.load(f)

    catid_to_name = {c["id"]: c["name"] for c in data["categories"]}
    name_to_catid = {v: k for k, v in catid_to_name.items()}
    imgid_to_file = {im["id"]: im["file_name"] for im in data["images"]}

    imgid_to_cats: Dict[int, set] = defaultdict(set)
    for ann in data["annotations"]:
        imgid_to_cats[ann["image_id"]].add(ann["category_id"])

    return imgid_to_file, imgid_to_cats, catid_to_name, name_to_catid


def select_coco_images_for_classes(
    images_dir: Path,
    imgid_to_file: Dict[int, str],
    imgid_to_cats: Dict[int, Iterable[int]],
    catid_to_name: Dict[int, str],
    name_to_catid: Dict[str, int],
    selected_classes: Sequence[str],
    max_per_class: int,
) -> List[Tuple[Path, str]]:
    """Select image paths and class labels for given COCO classes."""
    selected_cat_ids = [name_to_catid[c] for c in selected_classes]
    chosen: List[Tuple[Path, str]] = []
    per_class_count: Dict[str, int] = defaultdict(int)

    for img_id, fname in tqdm(imgid_to_file.items(), desc="Selecting images"):
        cats = imgid_to_cats.get(img_id, set())
        for cid in cats:
            if cid not in selected_cat_ids:
                continue
            cname = catid_to_name[cid]
            if per_class_count[cname] >= max_per_class:
                continue

            img_path = images_dir / fname
            if not img_path.exists():
                continue

            chosen.append((img_path, cname))
            per_class_count[cname] += 1

    print(f"[COCO] Chosen {len(chosen)} images: {dict(per_class_count)}")
    return chosen

def guess_grid_has_cls(n_vis: int) -> Tuple[bool, int]:
    """Guess whether visual tokens include a CLS token and infer grid side."""
    r = int(math.sqrt(n_vis))
    if r * r == n_vis:
        return False, r
    if n_vis > 1:
        r1 = int(math.sqrt(n_vis - 1))
        if r1 * r1 == (n_vis - 1):
            return True, r1
    return False, int(math.floor(math.sqrt(n_vis)))


def find_visual_span(out) -> Tuple[int, int, int]:
    """Return [pos, pos_end, n_vis] for visual tokens in projector features."""
    n_vis = out.projector_features.shape[1]
    pos = 1
    return pos, pos + n_vis, n_vis


def tokens_str_batch(input_ids: torch.Tensor, tokenizer) -> List[List[str]]:
    """Convert token ids to string tokens for each batch element."""
    return [tokenizer.convert_ids_to_tokens(row.tolist()) for row in input_ids]


def locate_word_token_idx_list(tokens_list: List[List[str]], word: str = "object") -> List[Optional[int]]:
    """Locate first token index containing `word` (substring, case-insensitive) in each sequence."""
    lw = word.lower()
    idxs: List[Optional[int]] = []
    for toks in tokens_list:
        low = [t.lower() for t in toks]
        hit = next((i for i, t in enumerate(low) if lw in t), None)
        idxs.append(hit)
    return idxs


def load_openvla_model(cfg: TsneConfig):
    """Load OpenVLA model and processor with desired dtype and attention impl."""
    device = get_device(cfg.device)
    dtype = str_to_dtype(cfg.dtype)

    processor = AutoProcessor.from_pretrained(cfg.model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        cfg.model_id,
        torch_dtype=dtype,
        attn_implementation=cfg.attn_implementation,
        trust_remote_code=True,
    ).to(device)

    return processor, model, device

def _batched(iterable: Sequence, batch_size: int):
    """Yield slices of size `batch_size` from a sequence."""
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]


def run_tsne_for_layers_batched(
    cfg: TsneConfig,
    chosen: Sequence[Tuple[Path, str]],
    layers: Sequence[int],
    selected_classes_order: Sequence[str],
) -> None:
    """Run t-SNE for given layers and plot per-layer embeddings."""

    # Subsample images for t-SNE
    rng = np.random.default_rng(cfg.random_seed)
    n_total = len(chosen)
    n_use = min(cfg.max_samples, n_total)
    idxs = rng.choice(n_total, size=n_use, replace=False)
    subset = [chosen[i] for i in idxs]

    print(f"[t-SNE] Using {len(subset)} images (from {n_total} total).")

    # Prepare containers
    layer_to_vecs = {L: [] for L in layers}
    layer_to_labs = {L: [] for L in layers}

    need_attn = (cfg.mode == "vis_pool_attn")
    need_proj = (cfg.mode in ["vis_mean", "vis_pool_attn", "vis_cls"])

    processor, model, device = load_openvla_model(cfg)
    torch.backends.cuda.matmul.allow_tf32 = True
    model.eval()

    for batch in tqdm(list(_batched(subset, cfg.batch_size)), desc="Embedding"):
        imgs: List[Image.Image] = []
        batch_labels: List[str] = []
        prompts: List[str] = []

        for img_path, cname in batch:
            try:
                imgs.append(Image.open(img_path).convert("RGB"))
                batch_labels.append(cname)
                prompts.append(cfg.per_class_prompt.format(class_name=cname))
            except Exception:
                pass

        if not imgs:
            continue

        with torch.inference_mode():
            inputs = processor(prompts, imgs, return_tensors="pt", padding=True)

            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)
                    # Cast float tensors to model dtype if needed
                    if v.dtype == torch.float32 and isinstance(model.dtype, torch.dtype):
                        inputs[k] = inputs[k].to(model.dtype)

            out = model(
                **inputs,
                output_hidden_states=True,
                output_attentions=need_attn,
                output_projector_features=need_proj,
                use_cache=False,
                return_dict=True,
            )

            hidden_all = out.hidden_states           # list of [B, S, H]
            B, S, H = hidden_all[-1].shape
            attn_all = out.attentions if need_attn else None

            attn_mask = inputs["attention_mask"]
            last_idxs = (attn_mask.sum(dim=1) - 1).tolist()

            toks_list = tokens_str_batch(inputs["input_ids"], processor.tokenizer)
            obj_idxs = locate_word_token_idx_list(toks_list, word="object")

            if need_proj:
                pos, pos_end, n_vis = find_visual_span(out)
                has_cls, _ = guess_grid_has_cls(n_vis)

            for L in layers:
                hL = hidden_all[L]

                if cfg.mode == "last_text_token":
                    # Take final text token embedding
                    vecs = [_to_np(hL[b, last_idxs[b], :]) for b in range(B)]

                elif cfg.mode == "text_object_token":
                    # Take token matching class name; fallback to last token
                    vecs = []
                    for b in range(B):
                        class_word = batch_labels[b].lower()
                        toks = toks_list[b]
                        low = [t.lower() for t in toks]
                        hit = next((i for i, t in enumerate(low) if class_word in t), None)
                        if hit is None:
                            hit = last_idxs[b]
                        vecs.append(_to_np(hL[b, hit, :]))

                elif cfg.mode == "vis_mean":
                    # Mean over visual tokens
                    vecs = []
                    for b in range(B):
                        vis = hL[b, pos:pos_end, :]
                        if has_cls and n_vis > 1:
                            vis = vis[1:, :]
                        vecs.append(_to_np(vis.mean(dim=0)))

                elif cfg.mode == "vis_cls":
                    # First visual token (CLS-like)
                    vecs = [_to_np(hL[b, pos, :]) for b in range(B)]

                elif cfg.mode == "vis_pool_attn":
                    # Attention-pooled visual features w.r.t. "object" word
                    vecs = []
                    A_L = attn_all[L]    # [B, Hh, S_q, S_k]
                    for b in range(B):
                        idx_q = obj_idxs[b] if obj_idxs[b] is not None else last_idxs[b]
                        A = A_L[b, :, idx_q, pos:pos_end]          # [num_heads, n_vis]
                        alpha = A.mean(dim=0).to(torch.float32)   # [n_vis]
                        alpha = alpha / (alpha.sum() + 1e-8)

                        vis = hL[b, pos:pos_end, :]
                        if has_cls and n_vis > 1:
                            alpha = alpha[1:]
                            alpha = alpha / (alpha.sum() + 1e-8)
                            vis = vis[1:, :]

                        v = (alpha[:, None] * vis.to(torch.float32)).sum(dim=0)
                        vecs.append(_to_np(v))
                else:
                    raise ValueError(f"Unknown mode: {cfg.mode}")

                layer_to_vecs[L].extend(vecs)
                layer_to_labs[L].extend(batch_labels)

    n_layers = len(layers)
    if cfg.layout == "horizontal":
        rows, cols = 1, n_layers
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4), squeeze=False, dpi=cfg.figure_dpi)
        axes = axes[0]
    elif cfg.layout == "vertical":
        rows, cols = n_layers, 1
        fig, axes = plt.subplots(rows, cols, figsize=(6, 4 * rows), squeeze=False, dpi=cfg.figure_dpi)
        axes = axes[:, 0]
    else:  # auto grid
        cols = int(math.ceil(math.sqrt(n_layers)))
        rows = int(math.ceil(n_layers / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), dpi=cfg.figure_dpi)
        axes = np.array(axes).reshape(-1)

    for ax, L in zip(axes, layers):
        X = np.asarray(layer_to_vecs[L], dtype=np.float32)
        labs = np.asarray(layer_to_labs[L])

        if len(X) == 0:
            ax.set_title(f"{cfg.mode}, layer {L} (no data)")
            ax.axis("off")
            continue

        perplexity = min(
            cfg.tsne_perplexity,
            max(5.0, (len(X) - 1) / 3.0),
        )

        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            max_iter=cfg.tsne_max_iter,
            metric="cosine",
            random_state=cfg.random_seed,
            verbose=0,
        )
        X2 = tsne.fit_transform(X)

        classes = sorted(
            set(labs.tolist()),
            key=lambda x: selected_classes_order.index(x) if x in selected_classes_order else 999,
        )

        cmap = plt.get_cmap("tab10")
        colors = {c: cmap(i % 10) for i, c in enumerate(classes)}

        for c in classes:
            m = (labs == c)
            ax.scatter(X2[m, 0], X2[m, 1], s=10, alpha=0.85, c=[colors[c]], label=c)

        ax.set_title(f"{cfg.mode}, layer {L}")
        ax.set_xticks([])
        ax.set_yticks([])

    # Turn off unused axes
    for ax in axes[len(layers):]:
        ax.axis("off")

    handles, labels_legend = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels_legend,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=min(len(labels_legend), 6),
            frameon=False,
            fontsize=12,
        )

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    if cfg.save_path is not None:
        cfg.save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(cfg.save_path, bbox_inches="tight", dpi=cfg.figure_dpi)
        print(f"[t-SNE] Figure saved to: {cfg.save_path}")

    plt.show()


@draccus.wrap()
def main(cfg: TsneConfig) -> None:
    seed_everything(cfg.random_seed)

    # Paths
    images_dir = cfg.dataset_dir / "train2017"
    anns_dir = cfg.dataset_dir / "annotations"
    instances_json = anns_dir / cfg.instances_filename

    selected_classes = parse_class_list(cfg.selected_classes)
    layers = parse_layers(cfg.layers)

    print(f"[Config] Selected classes: {selected_classes}")
    print(f"[Config] Layers: {layers}")

    imgid_to_file, imgid_to_cats, catid_to_name, name_to_catid = build_coco_index(instances_json)
    chosen = select_coco_images_for_classes(
        images_dir=images_dir,
        imgid_to_file=imgid_to_file,
        imgid_to_cats=imgid_to_cats,
        catid_to_name=catid_to_name,
        name_to_catid=name_to_catid,
        selected_classes=selected_classes,
        max_per_class=cfg.max_per_class,
    )

    if not chosen:
        raise RuntimeError("No images were selected for the given classes / paths.")

    run_tsne_for_layers_batched(
        cfg=cfg,
        chosen=chosen,
        layers=layers,
        selected_classes_order=selected_classes,
    )


if __name__ == "__main__":
    main()