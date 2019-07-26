import pytest
from mcqa.models import Model
import numpy as np
from pytorch_transformers import BertForMultipleChoice
from mcqa.data import get_labels
from sklearn.metrics import accuracy_score


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


def test_fit_reproducibility(trained_model, mcqa_dataset):
    mdl1 = trained_model
    mdl2 = Model(bert_model="bert-base-uncased",
                 device="cpu")
    mdl2.fit(mcqa_dataset,
             train_batch_size=1,
             num_train_epochs=1)

    for p1, p2 in zip(mdl1.model.parameters(), mdl2.model.parameters()):
        assert p1.data.allclose(p2.data)


def test_save_load(trained_model, mcqa_dataset, tmpdir):
    model_path = str(tmpdir)

    trained_model.save_model(model_path)

    mdl_clone = Model(bert_model="bert-base-uncased",
                      device="cpu")

    mdl_clone.model = BertForMultipleChoice.from_pretrained(model_path,
                                                            num_choices=4)

    for p1, p2 in zip(mdl_clone.model.parameters(),
                      trained_model.model.parameters()):

        assert p1.data.allclose(p2.data)

    mdl_clone.fit(mcqa_dataset,
                  train_batch_size=1,
                  num_train_epochs=1)

    _ = mdl_clone.predict_proba(mcqa_dataset,
                                eval_batch_size=1)


def test_unfitted_error(mcqa_dataset):
    mdl = Model(bert_model="bert-base-uncased",
                device="cpu")
    with pytest.raises(Exception):
        mdl.predict_proba(mcqa_dataset,
                          eval_batch_size=1)


def test_integration_sklearn_metrics(trained_model, mcqa_dataset):
    outputs = trained_model.predict(mcqa_dataset,
                                    eval_batch_size=1)
    labels = get_labels(mcqa_dataset)

    accuracy = accuracy_score(outputs, labels)

    assert isinstance(accuracy, float)
    assert 0 <= accuracy <= 1
