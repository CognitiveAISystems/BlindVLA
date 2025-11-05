import json
import os
from dataclasses import dataclass
from pathlib import Path
import shutil

import draccus
import torch
from peft import PeftModel
from transformers import AutoModelForVision2Seq, AutoProcessor

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class MergeConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"                            # Path to OpenVLA model (on HuggingFace Hub)
    run_path: str = ""
    lora_name: str = "lora_000000"
    # fmt: on

@draccus.wrap()
def merge(cfg: MergeConfig) -> None:
    run_path = Path(cfg.run_path)
    lora_path = run_path / cfg.lora_name
    merge_path = run_path / cfg.lora_name.replace("lora", "merged")
    os.makedirs(merge_path, exist_ok=True)

    processor = AutoProcessor.from_pretrained(str(run_path), trust_remote_code=True)
    base_vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    )

    merged_vla = PeftModel.from_pretrained(base_vla, lora_path)
    merged_vla = merged_vla.merge_and_unload()

    # dataset_statistics.json
    shutil.copy(run_path / "dataset_statistics.json", merge_path / "dataset_statistics.json")

    # Save processor and model weights to new directory
    processor.save_pretrained(merge_path)
    merged_vla.save_pretrained(merge_path)

    # process norm_states
    config = json.load((merge_path / "config.json").open())
    new_norm_stat = json.load((merge_path / "dataset_statistics.json").open())
    config["norm_stats"].update(new_norm_stat)
    json.dump(config, (merge_path / "config.json").open("w"), indent=2)



if __name__ == "__main__":
    merge()
