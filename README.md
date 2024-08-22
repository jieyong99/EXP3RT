# EXP3RT

## Quick Start

Before training the model, we have split some of the train datasets due to the size of the dataset, so you need to merge them into the original train dataset using merge_data.py (set the paths).

Follow the steps below ⬇️

### Set Up

```sh
conda env create -f qlora.yaml
conda env create -n merge
conda env create -n vllm
```
### Dataset

make dataset for training:

```sh
sh data_gen/generate.sh
```

### Train & Inference

train with command:

```sh
conda activate qlora
(you might need to install additional libraries to train with no error)

## for step 1. preference extraction
sh shell/train_preference.sh

## for step 2. profile construction
sh shell/train_user.sh
sh shell/train_item.sh

## for step 3. reasoning-enhanced rating prediction
sh shell/train_imdb.sh
sh shell/train_amazon-book.sh
```

merge the model for inference with command:

```sh
conda activate merge
pip install transformers
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

check the results with command:

```sh
python test_result_inspect.py
```
