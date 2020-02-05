<a href="https://mcqa.readthedocs.io"><img src="https://avatars0.githubusercontent.com/u/52794440" width="125" height="125" align="right" /></a>

# mcQA : Multiple Choice Questions Answering 

[![CircleCI](https://circleci.com/gh/mcQA-suite/mcQA.svg?style=svg)](https://circleci.com/gh/mcQA-suite/mcQA)
[![codecov](https://codecov.io/gh/mcqa-suite/mcQA/branch/master/graph/badge.svg)](https://codecov.io/gh/mcqa-suite/mcQA)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/26f497010c934b688c70bda4304c7100)](https://app.codacy.com/app/tayciryahmed/mcQA?utm_source=github.com&utm_medium=referral&utm_content=mcQA-suite/mcQA&utm_campaign=Badge_Grade_Dashboard)
[![PyPI Version](https://img.shields.io/pypi/v/mcqa.svg)](https://pypi.org/project/mcqa/)

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

To train a `mcQA` model, you need to create a csv file with n+2 columns, n being the number of choices for each question. The first column should be the context sentence, the n following columns should be the choices for that question and the last column is the selected answer. 

Below is an example of a 3 choice question (taken from the [CoS-E dataset](https://arxiv.org/pdf/1906.02361.pdf)) :

| Context sentence  | Choice 1                | Choice 2            | Choice 3    | Label|
| ----------------- | --------------------|--------------------|--------------------|-------------|
| People do what during their time off from work?| take trips | brow shorter | become hysterical | take trips |

If you have a trained `mcQA` model and want to infer on a dataset, it should have the same format as the train data, but the `label` column. 

See example data preparation below:

```python
from mcqa.data import MCQAData

mcqa_data = MCQAData(bert_model="bert-base-uncased", lower_case=True, max_seq_length=256) 
                     
train_dataset = mcqa_data.read(data_file='swagaf/data/train.csv', is_training=True)
test_dataset = mcqa_data.read(data_file='swagaf/data/test.csv', is_training=False)
```

### Model training 

```python
from mcqa.models import Model

mdl = Model(bert_model="bert-base-uncased", device="cuda") 
            
mdl.fit(train_dataset, train_batch_size=32, num_train_epochs=20)
```

### Prediction

```python
preds = mdl.predict(test_dataset, eval_batch_size=32)
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
|:newspaper: Paper| [Are We Modeling the Task or the Annotator? An Investigation of Annotator Bias in Natural Language Understanding Datasets](https://arxiv.org/pdf/1908.07898.pdf)|Mor Geva, Yoav Goldberg, Jonathan Berant| 2019|
|:newspaper: Paper| [Explain Yourself! Leveraging Language Models for Commonsense Reasoning](https://arxiv.org/pdf/1906.02361.pdf)|Nazneen Fatema Rajani, Bryan McCann, Caiming Xiong and Richard Socher| 2019|
|:newspaper: Paper|[SWAG: A Large-Scale Adversarial Dataset for Grounded Commonsense Inference](https://arxiv.org/abs/1808.05326)|Rowan Zellers, Yonatan Bisk, Roy Schwartz and Yejin Choi|2018|
|:newspaper: Paper|[Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering](https://arxiv.org/abs/1809.02789)|Todor Mihaylov, Peter Clark, Tushar Khot, Ashish Sabharwal|2018|
|:newspaper: Paper|[CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge](https://arxiv.org/abs/1811.00937)|Alon Talmor, Jonathan Herzig, Nicholas Lourie, Jonathan Berant|2018|
|:newspaper: Paper|[RACE: Large-scale ReAding Comprehension Dataset From Examinations](https://arxiv.org/abs/1704.04683)|Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang and Eduard Hovy|2017|
| :computer: Framework | [Scikit-learn: Machine Learning in Python](http://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html)                                           | Pedregosa et al.                                                                       | 2011 |
| :computer: Framework | [PyTorch](https://arxiv.org/abs/1906.04980)                                                                                                  | Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan                               | 2016 |
| :computer: Framework | [Transformers: State-of-the-art Natural Language Processing for TensorFlow 2.0 and PyTorch.](https://github.com/huggingface/transformers) | Hugging Face                                                                           | 2018 |
| :video_camera: Video | [Stanford CS224N: NLP with Deep Learning Lecture 10 – Question Answering](https://youtube.com/watch?v=yIdF-17HwSk)                           | Christopher Manning                                                                    | 2019 |

## LICENSE
[Apache-2.0](LICENSE)

## Contributing
Read our [Contributing Guidelines](.github/CONTRIBUTING.md).

## Citation

```
@misc{Taycir2019,
  author = {mcQA-suite},
  title = {mcQA},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/mcQA-suite/mcQA/}}
}
```
