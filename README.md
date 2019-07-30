# mcQA : Multiple Choice Questions Answering 

[![Build Status](https://travis-ci.com/mcQA-suite/mcQA.svg?branch=master)](https://travis-ci.com/mcQA-suite/mcQA)
[![PyPI Version](https://img.shields.io/pypi/v/mcqa.svg)](https://pypi.org/project/mcqa/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
![GitHub](https://img.shields.io/github/license/mcqa-suite/mcqa.svg)
[![codecov](https://codecov.io/gh/mcqa-suite/mcQA/branch/master/graph/badge.svg)](https://codecov.io/gh/mcqa-suite/mcQA)


Answering multiple choice questions with Language Models.


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

