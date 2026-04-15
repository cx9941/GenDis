# GenDis: Generative-Discriminative Dual-View Co-Training for Generalized Category Discovery

| **[Overview](#overview)**
| **[Installation](#installation)**
| **[Dataset](#dataset)**
| **[Folder Structure](#folder-Structure)**
| **[How to Run Model](#how-to-run-model)**

## Overview


## Installation

Create a python 3.12 environment and install dependencies:

```
  conda create -n py312 python=3.12
  source activate py312
```

Install library

```
  pip install -r requirements.txt
```

Note that pytorch >= 2.3.1

## Dataset
The datasets used in all experiments are derived from previously published scientific papers. We provide the links from which we obtain the datasets.

[BANKING](https://aclanthology.org/D19-1131), [CLINC](https://aclanthology.org/D19-1131), [StackOverflow](https://doi.org/10.3115/v1/W15-1509), [MICD](https://aclanthology.org/2020.nlpcovid19-acl.15/), [HWU](https://aclanthology.org/2020.nlp4convai-1.5/), [medical](https://github.com/sebischair/Medical-Abstracts-TC-Corpus)

## Folder Structure

```tex
├── code
│   ├── init_parameters.py
│   ├── __init__.py
│   ├── main.py
│   ├── models
│   ├── test.py
│   ├── trainer_callback.py
│   ├── trainer.py
│   └── utils
├── configs
│   ├── args.json
│   ├── json
│   └── yaml
├── data
│   ├── banking
│   ├── clinc
│   ├── hwu
│   ├── mcid
│   ├── stackoverflow
│   ├── data_statics.json
│   ├── data_statics.xlsx
│   ├── step0-process.ipynb
│   ├── step1-data_split.ipynb
│   └── step2-data_statics.ipynb
├── pretrained_models
│   ├── Meta-Llama-3.1-8B-Instruct
│   └── Qwen2.5-7B-Instruct
├── README.md
├── requirements.txt
└── scripts
    └── run.sh
```

## How to Run Model

Train and test:

```
sh scripts/run.sh # for all dataset with 0.25, 0.5, 0.75 known ratio of labels
```