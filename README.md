# EXP3RT

## Quick Start

First, download the dataset from https://drive.google.com/drive/folders/1F7aN-r2ixAaOBkxBm_-VJd_iCQyPo4Ih?usp=drive_link, https://drive.google.com/drive/folders/1VQ5W_GQcJCherIN1vzvOI6e5qLIgXt71?usp=drive_link, https://drive.google.com/drive/folders/14Vd4choyshdvt1vq4yc4K5GKN8XKt0pc?usp=drive_link, and unzip the file in root folder.

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
