import pytest
import csv

from mcqa.data import MCQAData


@pytest.fixture(scope="session")
def dummy_data():
    data = [
        {
            "startphrase": "Members of the procession walk down \
                      the street holding small horn brass \
                      instruments. A drum line",
            "ending0": "passes by walking down the street playing their instruments.",
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
    data_path = str(tmpdir.join("data.csv"))

    f = csv.writer(open(data_path, "w"))
    f.writerow(list(data[0].keys()))
    for exp in data:
        f.writerow(exp.values())

    return data_path


@pytest.fixture(scope="session")
def mcqa_data():
    bert_model = "bert-base-uncased"
    lower_case = True
    max_seq_length = 10

    yield MCQAData(bert_model, lower_case, max_seq_length)


@pytest.fixture(scope="function")
def mcqa_dataset(mcqa_data, dummy_data_path):
    dataset = mcqa_data.read(dummy_data_path,
                             is_training=True)
    yield dataset