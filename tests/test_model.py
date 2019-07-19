from mcqa.models import Model
from pytorch_transformers import BertForMultipleChoice


def test_fit(mcqa_dataset):
    mdl = Model(bert_model="bert-base-uncased",
                device="cpu")

    mdl.fit(mcqa_dataset)

    assert isinstance(mdl.model, BertForMultipleChoice)
