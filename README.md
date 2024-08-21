# EXP3RT

Repo for paper [Review-driven Personalized Preference Reasoning with Large Language Models for Recommendation](https://arxiv.org/abs/2408.06276)

## Quick Start

First, download the dataset from (google drive link), and unzip the file in root folder.

Next, follow the steps below ⬇️

### Set Up

```sh
conda env create -f qlora.yaml
conda env create -n merge
conda env create -n vllm
```

### Train & Inference

train with command:

```sh
conda activate qlora
(you might need to install additional libraries to train with no error)
sh shell/train_imdb.sh
sh shell/train_amazon-book.sh
```

merge the model for inference with command:

```sh
conda activate merge
pip install transformsers
(you might need to install additional libraries to merge with no error)
sh shell/merge.sh
```

inference with command:

```sh
conda activate vllm
pip install vllm
(you might need to install additional libraries to inference with no error)
sh shell/test_imdb.sh
sh shell/test_amazon-book.sh
```
