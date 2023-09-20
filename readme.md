## Introduction

The purpose of this project is to enable language models to adapt to code completion tasks in different domains without the need for fine-tuning. The primary method used is the kNN-LM framework. Unlike the standard kNN-LM, we only store samples where the language model makes mistakes when predicting. This decouples the database from the language model's capabilities, allowing us to use Bayesian inference to merge the predictions of the database and the language model.

![image-20230920140126952](https://github.com/zetang94/ASE2023_kNM-LM/blob/main/structure.png)

## Quickstart

### 1. Install requirements

faiss-gpu==1.7.2

transformers==4.27.2

fuzzywuzzy==0.18.0

torch==1.13.0+cu117

### 2. Dataset and Processing

#### 2.1 Build Intra-Project Dataset

Since developer completions typically occur within projects, we utilize specific commit versions of project snapshots as our retrieval database and employ newly added methods after that commit as our test set. For the database partitioning, we employ the Miner tool, which can be downloaded from [here](https://zenodo.org/record/6040745). For detailed information about this tool, please refer to their original [paper](https://arxiv.org/abs/2206.03333) [1].


As an example of building the [Froyo_Email](https://github.com/Dustinmj/Froyo_Email) database, follow these steps:

- **Step 1:** Use Miner to partition the dataset.

```shell
git clone git@github.com:Dustinmj/Froyo_Email.git

# create a new file 'projects.txt' and then save the absolute path to the project 'Froyo_Email', then run

./run_miner.sh projects.txt outputs/ stats.json
```

- Step 2: To partition the database, tokenize, and label code token categories using the [process.py](https://github.com/zetang94/ASE2023_kNM-LM/blob/main/dataset/intra_project_completion/process.py).

```shell
# change the dir_path to the location of 'Froyo_Email' in your computer, then
python process.py 
```

The data used in the paper can be found [here](https://github.com/zetang94/ASE2023_kNM-LM/tree/main/dataset/intra_project_completion/dataset). Here's an explanation of the provided files:

- **train.txt**: This file is used to build the database. Each line contains a complete method body that has already been tokenized.
- **test.txt**: This file is used to test token-level completion performance. Similar to the train.txt file, each line contains a complete method body that has already been tokenized.
- ***_type.txt**: These files contain information about token types corresponding to the tokens.

#### 2.2 Build Intra-Scenario Dataset

We utilized the APIBench[2] dataset to validate code completion performance in the same application scenario. For token-level code completion, we used the training set from APIBench to construct the database and the test set to evaluate completion performance. You can find the token-level completion data [here](https://zenodo.org/record/8254171).

For Android scenario line-level completion, we employed JavaParser to identify code lines containing Android API calls, allowing us to extract complete signatures for Android APIs. The processed code has been uploaded [here](https://zenodo.org/record/8362214), and the processed data can be found  [here](https://github.com/zetang94/ASE2023_kNM-LM/blob/main/dataset/intra_scenario_completion/java/line_completion/Android/test.json). The json format is like: 

```json
{
    "input": "code context to be completed",
    "gt": "the ground truth of next code line",
    "api_signature": "the signature of the Android API used in the line"
}
```

### 3. Evaluate

#### 3.1 Intra-Project token-level Completion (RQ1):

Tip: Since the dataset for the project is relatively small in scale, we do not use faiss to train vector database.

```shell
# These params are used for all models. So for any model you want to use, run this script first.

model_type=gpt2
pretrained_dir=microsoft/CodeGPT-small-java-adaptedGPT2
# or use unixcoder 
# model_type=unixCoder
# pretrained_dir=microsoft/unixcoder-base

data_dir=./dataset/intra_project_completion
project_dir=${data_dir}/dataset/large_projects/AmazeFileManager
lit_file=${data_dir}/literals.json
output_dir=./save/intra_project/AmazeFileManager/${model_type}
```

##### - Base model

```shell
CUDA_VISIBLE_DEVICES=0 python ./code/run_lm.py \
	--data_dir=${project_dir} \
	--lit_file=${lit_file} \
    --langs=java \
    --output_dir=${output_dir}/base \
    --pretrain_dir=${pretrained_dir} \
    --log_file=log.log \
    --model_type=${model_type} \
    --block_size=1024 \
    --per_gpu_eval_batch_size=8 \
    --logging_steps=100 \
    --seed=42 \
    --do_eval_token
    
# expect output 
# word acc: [54.82]
```

##### - kNN-LM [3]

```shell
dstore_dir=${output_dir}/knn_lm/db    # path to store database
dstore_size=100000000    # the max number of tokens in the db
k=8    # set for k-nearest neighbors search

CUDA_VISIBLE_DEVICES=0 python ./code/run_lm.py \
	--data_dir=${project_dir} \
	--lit_file=${lit_file} \
    --langs=java \
    --output_dir=${output_dir}/knn_lm \
    --pretrain_dir=${pretrained_dir} \
    --log_file=log.log \
    --model_type=${model_type} \
    --block_size=1024 \
    --per_gpu_eval_batch_size=8 \
    --logging_steps=100 \
    --seed=42 \
    --do_eval_token \
    --with_knn \
    --build_index \
    --dstore_dir=${dstore_dir} \
    --dstore_size=${dstore_size} \
    --k=${k}

# expect output
# word acc: [58.65]
```

##### - kNM-LM (Ours)

```shell
dstore_dir=${output_dir}/knm_lm/db    # path to store database
dstore_size=100000000    # the max number of tokens in the db
k=8    # set for k-nearest neighbors search

CUDA_VISIBLE_DEVICES=0 python ./code/run_lm.py \
	--data_dir=${project_dir} \
	--lit_file=${lit_file} \
    --langs=java \
    --output_dir=${output_dir}/knm_lm \
    --pretrain_dir=${pretrained_dir} \
    --log_file=log.log \
    --model_type=${model_type} \
    --block_size=1024 \
    --per_gpu_eval_batch_size=8 \
    --logging_steps=100 \
    --seed=42 \
    --do_eval_token \
    --with_knn \
    --build_index \
    --dstore_dir=${dstore_dir} \
    --dstore_size=${dstore_size} \
    --k=${k} \
    --only_errors \
    --use_bayes

# expect output
# word acc: [68.38]
```

##### - BM25

```shell
CUDA_VISIBLE_DEVICES=0 python ./reacc/run_token_completion.py \
 	--data_dir=${project_dir} \
    --lit_file=${lit_file} \
    --langs=java  \
    --output_dir=${output_dir}/bm25 \
    --pretrain_dir=${pretrained_dir} \
    --model_type=${model_type} \
    --dstore_file=${project_dir}/train.txt \
    --data_process  \
    --build_index \
    --do_search \
    --do_generate \
    --use_bm25 \
    --bm_name amazefilemanager
    
# expect result
# word acc: 0.56879
```

#### 3.2 Intra-Scenario token-level Completion (RQ2):

Tip: We use faiss to train vector database for faster searching.

```shell
# These params are used for all models. So for any model you want to use, run this script first.

model_type=gpt2
pretrained_dir=microsoft/CodeGPT-small-py-adaptedGPT2
# or use unixcoder 
# model_type=unixCoder
# pretrained_dir=microsoft/unixcoder-base

data_dir=./dataset/intra_scenario_completion/python
scenario_dir=${data_dir}/token_completion/DL
lit_file=${data_dir}/literals.json
output_dir=./save/intra_scenario/python/token/DL/${model_type}
```

##### - Base model

```shell
CUDA_VISIBLE_DEVICES=0 python ./code/run_lm.py \
	--data_dir=${scenario_dir} \
	--lit_file=${lit_file} \
    --langs=python  \
    --output_dir=${output_dir}/base \
    --pretrain_dir=${pretrained_dir} \
    --log_file=log.log \
    --model_type=${model_type} \
    --block_size=1024 \
    --per_gpu_eval_batch_size=8 \
    --logging_steps=100 \
    --seed=42 \
    --do_eval_token
```

##### - kNN-LM

```shell
dstore_dir=${output_dir}/knn_lm/db    # path to store database

CUDA_VISIBLE_DEVICES=0 python ./code/run_lm.py \
	--data_dir=${scenario_dir} \
	--lit_file=${lit_file} \
    --langs=python  \
    --output_dir=${output_dir}/knnlm \
    --pretrain_dir=${pretrained_dir} \
    --log_file=log.log \
    --model_type=${model_type} \
    --block_size=1024 \
    --per_gpu_eval_batch_size=8 \
    --logging_steps=100 \
    --seed=42 \
    --do_eval_token \
    --with_knn \
    --build_index \
    --need_knn_train
```

##### - kNM-LM (Ours)

```shell
dstore_dir=${output_dir}/knm_lm/db    # path to store database

CUDA_VISIBLE_DEVICES=0 python ./code/run_lm.py \
	--data_dir=${scenario_dir} \
	--lit_file=${lit_file} \
    --langs=python  \
    --output_dir=${output_dir}/knmlm \
    --pretrain_dir=${pretrained_dir} \
    --log_file=log.log \
    --model_type=${model_type} \
    --block_size=1024 \
    --per_gpu_eval_batch_size=8 \
    --logging_steps=100 \
    --seed=42 \
    --do_eval_token \
    --with_knn \
    --build_index \
    --need_knn_train \
    --only_errors \
    --use_bayes
```

##### - ReACC 

```shell
# step 1. bm25 search
CUDA_VISIBLE_DEVICES=0 python ./reacc/run_token_completion.py \
	--data_dir=${scenario_dir} \ 
	--lit_file=${lit_file} \
    --output_dir=${output_dir}/reacc \
    --pretrain_dir=${pretrained_dir} \
    --model_type=${model_type} \
    --dstore_file=${scenario_dir}/train.txt \
    --data_process \
    --do_search \
    --do_generate \
    --use_bm25 \
    --bm_name py_dl

# step 2. dense search
CUDA_VISIBLE_DEVICES=0 python ./reacc/run_token_completion.py \
	--data_dir=${scenario_dir} \ 
	--lit_file=${lit_file} \
    --output_dir=${output_dir}/reacc \
    --pretrain_dir=${pretrained_dir} \
    --model_type=${model_type} \
    --dstore_file=${scenario_dir}/train.txt \
    --do_search \
    --do_generate \
    --use_dense \
    --build_index
    
# step 3. hybrid
CUDA_VISIBLE_DEVICES=0 python ./reacc/run_token_completion.py \
	--data_dir=${scenario_dir} \ 
	--lit_file=${lit_file} \
    --output_dir=${output_dir}/reacc \
    --pretrain_dir=${pretrained_dir} \
    --model_type=${model_type} \
    --dstore_file=${scenario_dir}/train.txt \
    --do_search \
    --do_generate \
    --use_hybrid
```

#### 3.3 Complete code line with Android APIs

##### - kNM-LM (Ours)

```shel
model_type=gpt2
pretrained_dir=microsoft/CodeGPT-small-java-adaptedGPT2

data_dir=./dataset/intra_scenario_completion/java
lit_file=${data_dir}/literals.json
output_dir=./save/intra_scenario/java/line/Android/${model_type}

dstore_dir=${output_dir}/knm_lm/db

# 1. build the database (same as token completion for Android.)
scenario_dir=${data_dir}/token_completion/Android

CUDA_VISIBLE_DEVICES=0 python ./code/run_lm.py \
    --data_dir=${scenario_dir} \
    --lit_file=${lit_file} \
    --langs=java \
    --output_dir=${output_dir}/knm_lm \
    --pretrain_dir=${pretrained_dir} \
    --log_file=log.log \
    --model_type=${model_type} \
    --block_size=1024 \
    --per_gpu_eval_batch_size=4 \
    --logging_steps=100 \
    --seed=42 \
    --build_index \
    --with_knn \
    --dstore_dir=${dstore_dir} \
    --only_errors \
    --use_bayes

# step 2. inference next line.
scenario_dir=${data_dir}/line_completion/Android

CUDA_VISIBLE_DEVICES=0 python ./code/run_lm.py \
    --data_dir=${scenario_dir} \
    --lit_file=${lit_file} \
    --langs=java \
    --output_dir=${output_dir}/knm_lm \
    --pretrain_dir=${pretrained_dir} \
    --log_file=log.log \
    --model_type=${model_type} \
    --block_size=1024 \
    --per_gpu_eval_batch_size=4 \
    --logging_steps=100 \
    --seed=42 \
    --do_eval_line \
    --with_knn \
    --dstore_dir=${dstore_dir} \
    --only_errors \
    --use_bayes
```

### Reference

[1] Egor Bogomolov, Sergey Zhuravlev, Egor Spirin, Timofey Bryksin:Assessing Project-Level Fine-Tuning of ML4SE Models. CoRR abs/2206.03333 (2022)

[2] Yun Peng, Shuqing Li, Wenwei Gu, Yichen Li, Wenxuan Wang, Cuiyun Gao, Michael R. Lyu:Revisiting, Benchmarking and Exploring API Recommendation: How Far Are We? IEEE Trans. Software Eng. 49(4): 1876-1897 (2023)

[3] Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, Mike Lewis:Generalization through Memorization: Nearest Neighbor Language Models. ICLR 2020

### Acknowledge 

This repository is inspired by the code from this repository: https://github.com/neulab/knn-transformers. We greatly appreciate the authors for providing their code.
