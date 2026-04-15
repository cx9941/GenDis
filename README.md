# GenDis: Generative-Discriminative Dual-View Co-Training for Generalized Category Discovery

| **[Overview](#overview)**
| **[Installation](#installation)**
| **[Dataset](#dataset)**
| **[Folder Structure](#folder-structure)**
| **[How to Run Model](#how-to-run-model)**

## Overview

Official code for article "[GenDis: Generative-Discriminative Dual-View Co-Training for Generalized Category Discovery (ACL 2026)](https://aclanthology.org/2026.acl-main.xxx)".

Generalized Category Discovery (GCD) aims to identify both known and novel categories from partially labeled data, reflecting more realistic open-world learning scenarios. However, most existing methods rely solely on one-hot discriminative supervision, leading to overfitting on seen classes and poor generalization to unseen ones. Recent advances introduce large language models (LLMs) to incorporate external semantics, yet they often suffer from semantic–label misalignment and weak semantic integration during training.
We propose **GenDis**, a **Gen**erative–**Dis**criminative Dual-View Co-Training framework that unifies discriminative classification and semantic label generation within an LLM. Discriminative pseudo-labels guide the formation of a separable generative latent space, enabling semantically meaningful supervision for novel classes. To ensure consistency between the two views, we employ Canonical Correlation Analysis (CCA)-based alignment and a curriculum-guided, dispersion-aware pseudo-labeling strategy for iterative refinement.
Extensive experiments on five GCD benchmarks demonstrate that **GenDis** substantially outperforms prior methods, validating the effectiveness of dual-view co-training with semantically enriched supervision.
Code and datasets are available at: https://github.com/cx9941/GenDis.

## Installation

Create a python 3.12 environment and install dependencies:

```
conda create -n py312 python=3.12
conda activate py312
```

Install library

```
pip install -r requirements.txt
```

Note that pytorch >= 2.3.1

## Dataset
The datasets used in all experiments are derived from previously published scientific papers. We provide the links from which we obtain the datasets.

| Dataset | Link |
|---------|------|
| BANKING | [https://aclanthology.org/D19-1131](https://aclanthology.org/D19-1131) |
| CLINC | [https://aclanthology.org/D19-1131](https://aclanthology.org/D19-1131) |
| StackOverflow | [https://doi.org/10.3115/v1/W15-1509](https://doi.org/10.3115/v1/W15-1509) |
| MICD | [https://aclanthology.org/2020.nlpcovid19-acl.15/](https://aclanthology.org/2020.nlpcovid19-acl.15/) |
| HWU | [https://aclanthology.org/2020.nlp4convai-1.5/](https://aclanthology.org/2020.nlp4convai-1.5/) |

Each dataset is split into three known label ratios (0.25, 0.5, 0.75) for GCD evaluation. Preprocessing steps are provided in `data/step0-process.ipynb`, `step1-data_split.ipynb`, and `step2-data_statics.ipynb`.

## Folder Structure

```tex
├── code
│   ├── init_parameters.py         # Parameter initialization
│   ├── __init__.py
│   ├── main.py                    # Entry point for training and testing
│   ├── models                     # Model architectures
│   ├── test.py                    # Testing utilities
│   ├── trainer_callback.py        # Training callbacks
│   ├── trainer.py                 # Core training logic
│   └── utils                      # Utility functions
├── configs
│   ├── args.json                  # Default parameters
│   ├── json                       # JSON configs
│   └── yaml                       # Accelerate configs
├── data
│   ├── banking                    # BANKING Dataset
│   ├── clinc                      # CLINC Dataset
│   ├── hwu                        # HWU Dataset
│   ├── mcid                       # MICD Dataset
│   ├── stackoverflow              # StackOverflow Dataset
│   ├── data_statics.json          # Dataset statistics
│   ├── data_statics.xlsx          # Dataset statistics (excel)
│   ├── step0-process.ipynb        # Raw data preprocessing
│   ├── step1-data_split.ipynb     # Train/val/test split
│   └── step2-data_statics.ipynb   # Statistics generation
├── pretrained_models
│   ├── Meta-Llama-3.1-8B-Instruct # Llama 3.1 pretrained weights
│   └── Qwen2.5-7B-Instruct        # Qwen2.5 pretrained weights
├── README.md                      # This document
├── requirements.txt               # Dependencies
└── scripts
    └── run.sh                     # Pipeline training and inference
```

## How to Run Model

Train and test GenDis on all datasets with different known label ratios:

```
sh scripts/run.sh
```

For specific dataset or custom configuration, modify `configs/args.json` or pass command-line arguments:

```
python code/main.py --dataset banking --known_ratio 0.5 --batch_size 32 --epochs 50
```

## Citation

If you find our work is useful for your research, please consider citing:

```
@inproceedings{chen2026gendis,
title={GenDis: Generative-Discriminative Dual-View Co-Training for Generalized Category Discovery},
author={Xi Chen and Chuan Qin and Ziqi Wang and Shasha Hu and Chao Wang and Hengshu Zhu and Hui Xiong},
booktitle={The 64th Annual Meeting of the Association for Computational Linguistics (ACL)},
year={2026},
address={San Diego, California},
url={https://github.com/cx9941/GenDis}
}
```

## License

This project is licensed under the MIT License.