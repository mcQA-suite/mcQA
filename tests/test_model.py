import pytest
from mcqa.models import Model
import numpy as np
from pytorch_transformers import BertForMultipleChoice


def test_fit(mcqa_dataset):
    mdl = Model(bert_model="bert-base-uncased",
                device="cpu")

    mdl.fit(mcqa_dataset,
            train_batch_size=1,
            num_train_epochs=1)

    assert isinstance(mdl.model, BertForMultipleChoice)


@pytest.fixture()
def trained_model(mcqa_dataset):

    mdl = Model(bert_model="bert-base-uncased",
                device="cpu")

    mdl.fit(mcqa_dataset,
            train_batch_size=1,
            num_train_epochs=1)

    yield mdl


def test_predict(mcqa_dataset, trained_model):
    outputs = trained_model.predict(mcqa_dataset,
                                    eval_batch_size=1)
    assert len(outputs) == len(mcqa_dataset)


def test_predict_proba(mcqa_dataset, trained_model):
    outputs_proba = trained_model.predict_proba(mcqa_dataset,
                                                eval_batch_size=1)

    assert len(outputs_proba) == len(mcqa_dataset)
    assert (np.abs(outputs_proba.sum(axis=1) - 1) < 1e-5).all()


def test_training_reproducibility(trained_model, mcqa_dataset):
    mdl1 = trained_model
    mdl2 = Model(bert_model="bert-base-uncased",
                 device="cpu")
    mdl2.fit(mcqa_dataset,
             train_batch_size=1,
             num_train_epochs=1)

    for p1, p2 in zip(mdl1.model.parameters(), mdl2.model.parameters()):
        assert p1.data.allclose(p2.data)
