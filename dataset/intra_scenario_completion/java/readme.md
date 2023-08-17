## 项目介绍

## Quickstart

### 1. Install requirements

faiss-gpu==1.7.2

transformers==4.27.2

fuzzywuzzy==0.18.0

torch==1.13.0+cu117

### 2. Dataset and Processing

2.1 构建项目内的代码补全数据集

- commit划分数据库（训练集）和测试集 我们使用了[1]中的代码来划分项目代码，工具可以从这边[下载](https://zenodo.org/record/6040745 ).

  [1] Bogomolov E, Zhuravlev S, Spirin E, et al. Assessing Project-Level Fine-Tuning of ML4SE Models[J]. arXiv preprint arXiv:2206.03333, 2022.

- 标注代码类别

- 处理好的代码在这边可以找到

2.2 构建同一应用场景的代码补全数据集

- Token数据集构建
- Android API行级别补全数据集构建
- 处理好的代码可以在这边[找到](https://zenodo.org/record/8254171)

3. 验证基础的性能

4. 利用训练集构建解耦数据库

   4.1 构建项目内的代码数据集

   4.2 构建同一场景的代码数据集

5. 验证Token补全准确率

   5.1 验证项目内的代码补全准确率

   5.2 验证同一场景的代码补全准确率

6. 验证补全Android API行级别补全

7. 验证和Finetuning对比

8. 超参数验证



3. Intra-project code completion



4. Intra-scenario code completion



5. Acknowledge
This repository is inspired by the code from this repository: https://github.com/neulab/knn-transformers. We greatly appreciate the authors for providing their code.