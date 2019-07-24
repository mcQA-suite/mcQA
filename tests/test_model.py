import pytest
from mcqa.models import Model
from pytorch_transformers import BertForMultipleChoice


def test_fit(mcqa_dataset):
    mdl = Model(bert_model="bert-base-uncased",
                device="cpu")

    mdl.fit(mcqa_dataset,
            train_batch_size=1,
            num_train_epochs=1)

    assert isinstance(mdl.model, BertForMultipleChoice)


@pytest.fixture(scope='session')
def trained_model(mcqa_dataset):

    mdl = Model(bert_model="bert-base-uncased",
                device="cpu")

    mdl.fit(mcqa_dataset,
            train_batch_size=1,
            num_train_epochs=1)

    yield mdl


def test_predict(mcqa_dataset, trained_model):
    outputs = trained_model.predict(mcqa_dataset)
    assert len(outputs) == len(mcqa_dataset)
