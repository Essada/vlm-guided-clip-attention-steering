## Overview
CLIP performs well on broad image-text alignment tasks, but can struggle when predictions depend on localized visual evidence, such as fine-grained attributes or question-specific image regions. This repository contains code and reproducible experiments for a VLM-guided inference-time attention steering pipeline for CLIP's Vision Transformer. The method uses vision-language model guidance (Qwen2.5-VL-7B) to generate task-relevant visual attributes, grounds those attributes into image regions with Grounding DINO, converts the regions into CLIP patch-level targets, and reweights selected attention heads (discovered through a seperate profiling phase) at inference time.

1. Generate class- or question-relevant visual attributes using a VLM with in-context learning examples.
2. Ground attributes into image regions using Grounding DINO.
3. Convert grounded boxes into CLIP ViT-B/16 patch targets.
4. Profile attention heads on a held-out subset.
5. Reweight selected attention heads during inference using [PASTA-style](https://arxiv.org/abs/2311.02262) steering adapted to the visual domain.


## Approach

**Offline Pre-processing pipeline for classification:**

<img width="1047" height="450" alt="image" src="https://github.com/user-attachments/assets/f46cb598-8087-4597-b2b9-f80803ec0ae8" />


**Offline Pre-processing pipeline for VQA:**

<img width="1034" height="398" alt="image" src="https://github.com/user-attachments/assets/901c6ff0-da3f-4d99-ba6e-46846a117111" />


**Attention steering code** 

See `ConformationCLIP_ViT.py` for core attention steering implemenation.

## Technical Report 
See `report/Enhancing CLIP for Fine-Grained Classification and VQA with VLM-Guided Attention Steering` for the full writeup.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Set dataset paths:

```bash
export CUB_DATA_ROOT=/path/to/CUB_200_2011
export VQA_DATA_ROOT=/path/to/vqa
```

Start Ollama and pull the VLMs:

```bash
ollama pull gemma4:e4b
ollama pull qwen2.5vl:7b
ollama serve
```

Grounding DINO weights are downloaded automatically on first run.

## Run CUB

```bash
python precompute_birds_cache_perclass.py
python test_birds_perclass.py
```

## Run VQA

```bash
python precompute_tapc_cache.py
python test_tapc_pasta.py
```

