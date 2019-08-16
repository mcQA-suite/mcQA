# mcQA : Multiple Choice Questions Answering 

Answering multiple choice questions with Language Models.

[![CircleCI](https://circleci.com/gh/mcQA-suite/mcQA.svg?style=svg)](https://circleci.com/gh/mcQA-suite/mcQA)
[![PyPI Version](https://img.shields.io/pypi/v/mcqa.svg)](https://pypi.org/project/mcqa/)
![GitHub](https://img.shields.io/github/license/mcqa-suite/mcqa.svg)
[![codecov](https://codecov.io/gh/mcqa-suite/mcQA/branch/master/graph/badge.svg)](https://codecov.io/gh/mcqa-suite/mcQA)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

## Installation

### With pip

```shell
pip install mcqa
```

### From source

```shell
git clone https://github.com/mcqa-suite/mcqa.git
cd mcQA
pip install -e .
```

## Getting started

### Data preparation

To train a `mcQA` model, you need to create a csv file with n+2 columns, n being the number of choices for each question. The first column should be the context sentence, the n following columns should be the choices for that question and the last column is the selected answer. 

Below is an example of a 3 choice question (taken from the [CoS-E dataset](https://arxiv.org/pdf/1906.02361.pdf)) :

| Context sentence  | Choice 1                | Choice 2            | Choice 3    | Label|
| ----------------- | --------------------|--------------------|--------------------|-------------|
| People do what during their time off from work?| take trips | brow shorter | become hysterical | take trips |

If you have a trained `mcQA` model and want to infer on a dataset, it should have the same format as the train data, but the `label` column. 

See example data preparation below:

```python
from mcqa.data import MCQAData

mcqa_data = MCQAData(bert_model="bert-base-uncased", 
                     lower_case=True, 
                     max_seq_length=256) 
                     
train_dataset = mcqa_data.read(data_file='swagaf/data/train.csv', is_training=True)
test_dataset = mcqa_data.read(data_file='swagaf/data/test.csv', is_training=False)
```

### Model training 

```python
from mcqa.models import Model

mdl = Model(bert_model="bert-base-uncased",
            device="cuda") 
            
mdl.fit(train_dataset, 
        train_batch_size=32, 
        num_train_epochs=20)
```

### Prediction

```python
preds = mdl.predict(test_dataset, 
                    eval_batch_size=32)
```

### Evaluation

```python
from sklearn.metrics import accuracy_score
from mcqa.data import get_labels

print(accuracy_score(preds, get_labels(train_dataset)))
```
## References

| Type                 | Title                                                                                                                                        | Author                                                                                 | Year |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ---- |
|:newspaper: Paper| [Explain Yourself! Leveraging Language Models for Commonsense Reasoning](https://arxiv.org/pdf/1906.02361.pdf)|Nazneen Fatema Rajani, Bryan McCann, Caiming Xiong and Richard Socher| ACL 2019|
|:newspaper: Paper|[SWAG: A Large-Scale Adversarial Dataset for Grounded Commonsense Inference](https://arxiv.org/abs/1808.05326)|Rowan Zellers, Yonatan Bisk, Roy Schwartz and Yejin Choi|2018|
