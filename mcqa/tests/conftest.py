import csv

import pytest
from transformers import AutoTokenizer
from mcqa.data import McqaDataset, Split


@pytest.fixture(scope="session")
def dummy_data():
    data = [
        {
            "index": "0",
            "video-id": "lsmdc1052_Harry_Potter_and_the_order_of_phoenix-94857",
            "fold-ind": "18313",
            "startphrase": "Members of the procession walk down \
                      the street holding small horn brass \
                      instruments. A drum line",
            "sent1": "Students lower their eyes nervously.",
            "sent2": "Students lower their eyes nervously.",
            "gold-source": "Students lower their eyes nervously.",
            "ending0": "passes by walking down the street playing \
                        their instruments.",
            "ending1": "has heard approaching them.",
            "ending2": "arrives and they're outside dancing and asleep.",
            "ending3": "turns the lead singer watches the performance.",
            "label": "0"
        }
    ]
    yield data


@pytest.fixture(scope="function")
def dummy_data_path(dummy_data, tmpdir):
    data = dummy_data

    data_path = str(tmpdir.join("train.csv"))
    file = csv.writer(open(data_path, "w"))
    file.writerow(list(data[0].keys()))
    for exp in data:
        file.writerow(exp.values())

    data_path = str(tmpdir.join("val.csv"))
    file = csv.writer(open(data_path, "w"))
    file.writerow(list(data[0].keys()))
    for exp in data:
        file.writerow(exp.values())

    return str(tmpdir)


@pytest.fixture(scope="session")
def tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        'bert-base-uncased',
        cache_dir='/tmp',
    )
    yield tokenizer


@pytest.fixture(scope="function")
def mcqa_dataset(dummy_data_path, tokenizer):
    dataset = McqaDataset(
        data_dir=dummy_data_path,
        tokenizer=tokenizer,
        task='swag',
        max_seq_length=10,
        overwrite_cache=False,
        mode=Split.train,
    )
    yield dataset
