# EXP3RT

## Quick Start

Before training the model, we have split some of the train datasets due to the size of the dataset, so you need to merge them into the original train dataset using merge_data.py (set the paths).

Follow the steps below ⬇️

### Dataset

make dataset for training:

```sh
sh data_gen/generate.sh
```

### Train & Inference

train with command:

```sh
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
sh shell/merge.sh
```

inference with command:

```sh
sh shell/test_imdb.sh
sh shell/test_amazon-book.sh
```

check the results with command:

```sh
python test_result_inspect.py
```
