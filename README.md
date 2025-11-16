<!-- <div align="center"> -->

# Don’t Blind Your VLA: Aligning Visual Representations for OOD Generalization


<!-- <div align="left"> -->

<img width="5567" height="4133" alt="method_1" src="figs/method_1.png" />

To address the degradation of visual-language (VL) representations during VLA supervised fine-tuning (SFT), we introduce **Visual Representation Alignment**. During SFT, we pull a VLA’s visual tokens toward a frozen teacher’s patch features using cosine similarity through a lightweight frozen projector. This keeps perception anchored while the model learns to act — improving OOD generalization with almost no added cost.


<div align="center">

[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2510.25616) [![Project-Page](https://img.shields.io/badge/Project--Page-%2300B4AB?style=for-the-badge&logo=logolol&logoColor=white&labelColor=000000)](https://blind-vla-paper.github.io) 
[![Twitter](https://img.shields.io/badge/post-%23000000.svg?style=for-the-badge&logo=x&logoColor=white)](https://x.com/judokach/status/1988616551516225859?s=20) 
[![HF Papers](https://img.shields.io/badge/Model--Dataset-%23FFD14D?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/collections/tttonyalpha/dont-blind-your-vla)

</div>


<!-- <div align="center"> -->

## Contents:
 * [**⚙️ Installation**](#installation) 
 * [**✨ Visual Representation Alignment**](#visual-representation-alignment)
 * [**🔍 VL Representations Analysis**](#vl-anal)
 * [**📊 VL-Think**](#vl-think)
 * [**📈 Evaluation**](#evaluation)
 * [**❤️ Citation**](#citation)
<!-- </div> -->


## TODO List

- [x] Code for OpenVLA Visual Representation Alignmnet.
- [x] Dataset and warmupt checkpoint.
- [x] VL-Think Task Suite.
- [x] Vizualization code.
- [ ] Code for aligning pi0.5 on real robot.


<h2 id="installation">⚙️ Installation</h2>

Use the enviroment setup commands below to get started:

```bash
# Create and activate conda environment
conda create -n blindvla python=3.10 -y
conda activate blindvla

# Install PyTorch. Below is a sample command to do this, but you should check the following link
# to find installation instructions that are specific to your compute platform:
# https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio

# Clone and install the BlindVLA repo
git clone https://github.com/CognitiveAISystems/BlindVLA.git
cd BlindVLA
pip install -e ./openvla

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip3 install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation
pip install diffusers==0.33.0

pip install -e ./ManiSkill
pip install -e ./SimplerEnv
pip install -U "typeguard>=3"

```

We warm up the pretrained OpenVLA using 140 episodes collected with Octo-Small and a motion planner for 2k steps.
You can download the training dataset (1.4k episodes) [here](https://huggingface.co/datasets/tttonyalpha/openvla_1k-dataset)￼ and the warm-up checkpoint [here](https://huggingface.co/tttonyalpha/openvla-7b-warmup-checkpoint_lora_002000).

<h2 id="visual-representation-alignment">✨ Visual Representation Alignment</h2>


<img width="10468" height="2292" alt="scheme1_1" src="figs/scheme1_1.png" />

When a VLA model is adapted to downstream control tasks, its VL representations often drifts away from the rich, semantically grounded features. To mitigate this, we introduce **Visual Representation Alignment** — a lightweight regularization that anchors the model’s mid-layer visual embeddings to a frozen teacher’s features via a cosine-similarity loss.

Below is a minimal example of how easily you can integrate Visual Representation Alignment into your VLA’s training pipeline. Just plug in these few lines right after your forward pass — no architecture changes are needed.

```python
# ....
# out = vla.forward(..., output_hidden_states=True)
# pixel_values = preprocessor(image, ...)
# ....
#

n_vis = out.projector_features.shape[1]
pos, pos_end = 1, 

# 1. Extract VLA's visual features from specific layer and project to visual teacher dimention
vla_features = out.hidden_states[align_layer][:, pos:pos_end]     
vla_features = alignment_projector(vla_feats)                     

# 2. Get teacher patch features
with torch.no_grad():
    teacher_features = teacher_vision_backbone(pixel_values)  

# 3. Compute cosine alignment loss
emb_t = F.normalize(teacher_features, dim=-1)         
emb_s = F.normalize(vla_features, dim=-1)             

cossim = (emb_t * emb_s).sum(dim=-1)                  
align_loss = (-cossim).mean()

loss += cfg.align_coeff * align_loss
```

You can run LoRA fine-tuning with Visual Representation Alignment using this script:

```bash

openvla_path="tttonyalpha/openvla-7b-warmup-checkpoint_merged_002000_lora_002000"

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "$openvla_path" \
  --data_root_dir "datasets" \
  --dataset_name "sft" \
  --run_root_dir "runs" \
  --lora_rank 32 \
  --batch_size 8 \
  --max_steps 60000 \
  --eval_steps 200 \
  --save_steps "0,5000,10000,20000,30000,40000,50000,60000" \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True

```

<h2 id="vl-anal">🔍 VL Representations Analysis</h2>

<!-- <img width="4456" height="1564" alt="att_sink_rep_collapse" src="figs/att_sink_rep_collapse.png" /> -->

We investigate how VL representations evolve in VLA models after action fine-tuning. Specifically, we ask whether semantic grounding and knowledge transfer from pretrained VLMs are preserved. To assess degradation we probed VL representations of OpenVLA after naive fine-tuning on robotic data and observed three major problems: (1) **Attention sink** - the attention maps become diffuse, noisy, and weakly correlated
with the target object referenced in instruction, (2) **Representation collapse** - action fine-tuning disrupts the structured organization of VL representations, and (3) **Domain forgetting** - VLA models lose knowledge about domains that are absent in robotics fine-tuning datasets.


To visualize OpenVLA attention maps, run the script [vizualize_attention.py](openvla/vizualization/vizualize_attention.py):

```bash
python vizualize_attention.py \
  --image_path /path/to/image.png \
  --output_dir runs/attention_maps \
  --lora_root /path/to/lora_dir \
  --question "Do you see a can?" \
  --layers "15,16,17,18" \
  --device cuda \
  --dtype bfloat16
```

To visualize t-SNE of OpenVLA features, run the script [vizualize_tsne.py](openvla/vizualization/vizualize_tsne.py):

```bash
python vizualize_tsne.py \
  --dataset_dir /path/to/coco2017_dataset \
  --selected_classes "cup,bottle,knife" \
  --layers "1,10,20" \
  --mode text_object_token \
  --max_samples 5000 \
  --tsne_perplexity 30 \
  --tsne_max_iter 1000 \
  --save_path runs/tsne_openvla_coco.png
```

<h2 id="vl-think">📊 VL-Think</h2>


<img width="4456" height="1564" alt="vl_think" src="figs/VL_THINK.png" />

We introduce the **VL-Think Task Suite**, a diagnostic suite assessing the transfer of VL understanding and knowledge from VLMs to VLAs independently of low-level control. The suite focuses on whether models retain the ability to interpret visual symbols, compositional cues, and categorical relations rather than pure manipulation skills. Control complexity is intentionally minimized so that any degradation reflects a loss of VL understanding.

### Task description:

* a) `PutOnShapeInSceneMultiColor-v1`: **13 shapes** (trapezoid, triangle, right triangle, rectangle, square, parallelogram, pentagon, hexagon, circle, heart, star, arrow, cross )
* b) `PutOnColorInSceneMulti-v1`: **8 colors** (black, red, green, blue, orange, purple, yellow, brown)
* c) `PutOnLaundryIconInSceneMulti-v1`: **17 laundry icons** (any solvent, bleach allowed, cold wash, do not bleach, do not dryclean, do not iron, do not wash, dryclean, hand wash, hot wash, iron, machine wash delicate, machine wash permanent press, machine wash, non chlorine bleach, warm wash, wet cleaning)
* d) `PutOnNumberInSceneParity-v1`: **8 numbers**
* e) `PutOnPublicInfoSignInSceneMulti-v1`: **14 public info signs** (disabled access, escalator, fire escape, hairdresser, information, no dogs, no entry, no parking, no smoking, recycle, stairs, taxi, telephone, toilets)
* f) `PutOnSignTrafficInSceneMulti-v1`: **24 traffic signs** (ahead only, falling rocks, loose chippings, max speed, minimum speed, no U-turn, no entry, no left turn, no overtaking, no right turn, no stopping, no through road, no waiting, road narrows right, road works, roundabout, sharp route deviation, steep downwards, steep upwards, stop give way, turn left ahead, uneven road, wild animals, yield)
* g) `PutOnWeatherIconInSceneMulti-v1`: **9 weather icons** (clear night, cloudy, rainy, snowing, storm, sunny, sunrise, windy, windy and cloudy)
* h) `PutOnArrowSignInSceneMulti-v1`: **4 directions**

<h2 id="evaluation">📈 Evaluation</h2>

Evaluation is performed using batched environments for efficient parallel processing. The script [openvla_eval_batched.py](SimplerEnv/simpler_env/openvla_eval_batched.py)￼ runs evaluation with `num_envs` parallel environments in a single batch. 

Each environment implements several methods designed for evaluating VLM models: `where_target()` - determines the position of the target board — one of "left", "center", or "right", returns a list of strings corresponding to the target position in each environment instance; `get_target_name()`: returns the semantic name of the target board for each environment (e.g., “square”, “escalator sign”, “orange”); `get_language_instruction()` - returns the language instruction associated with each environment, used as the input text prompt for evaluation.

You can run OpenVLA evaluation using this script: 

```bash

openvla_path="tttonyalpha/openvla-7b-warmup-checkpoint_merged_002000_lora_002000"
lora_load_path="<YOUR_PROJECT_DIR>/<PATH_TO_LORA>"  # or set empry 

for seed in 0 1 2 4 5 6 7 8; do
    for env_id in \

        ### OOD Generalization envs: 
        "PutOnPlateInScene25VisionImage-v1" "PutOnPlateInScene25VisionTexture03-v1" "PutOnPlateInScene25VisionTexture05-v1" \
        "PutOnPlateInScene25VisionWhole03-v1"  "PutOnPlateInScene25VisionWhole05-v1" \
        "PutOnPlateInScene25Carrot-v1" "PutOnPlateInScene25Plate-v1" "PutOnPlateInScene25Instruct-v1" \
        "PutOnPlateInScene25MultiCarrot-v1" "PutOnPlateInScene25MultiPlate-v1" \
        "PutOnPlateInScene25Position-v1" "PutOnPlateInScene25EEPose-v1" "PutOnPlateInScene25PositionChangeTo-v1"

        ### VL-Think envs
        "PutOnShapeInSceneMultiColor-v1" "PutOnColorInSceneMulti-v1"
        "PutOnSignTrafficInSceneMulti-v1" "PutOnLaundryIconInSceneMulti-v1"
        "PutOnWeatherIconInSceneMulti-v1" "PutOnArrowSignInSceneMulti-v1"
        "PutOnPublicInfoSignInSceneMulti-v1" "PutOnNumberInSceneParity-v1" ;
         do
      
      CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
      python ./SimplerEnv/simpler_env/openvla_eval_batched.py \
        --vla_path="$openvla_path" --vla_unnorm_key="sft" \
        --vla_load_path="${lora_load_path}" \
        --env_id="${env_id}" \
        --seed=${seed} \
        --buffer_inferbatch=64 \
        --num_envs=128 --obj_set="test"
    done
done

```

<h2 id="citation">❤️ Citation</h2>

If you find our code useful, please cite [our paper](https://arxiv.org/abs/2510.25616): 

```BibTeX

@misc{kachaev2025dontblindvlaaligning,
      title={Don't Blind Your VLA: Aligning Visual Representations for OOD Generalization}, 
      author={Nikita Kachaev and Mikhail Kolosov and Daniil Zelezetsky and Alexey K. Kovalev and Aleksandr I. Panov},
      year={2025},
      eprint={2510.25616},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.25616}, 
}

```

## 🙏 Acknowledgement
BlindVLA is built with reference on: [RL4VLA](https://github.com/gen-robot/RL4VLA), [Simpler](https://github.com/simpler-env/SimplerEnv), [REPA](https://github.com/sihyun-yu/REPA), [OpenVLA](https://github.com/openvla/openvla). Many thanks for their awesome work!
