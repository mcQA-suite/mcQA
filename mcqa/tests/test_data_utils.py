from mcqa.data import _truncate_seq_pair
from mcqa.data import select_field
from mcqa.data import InputFeatures
from mcqa.data import read_mcqa_examples
from mcqa.data import get_labels
import numpy as np


def test_truncate_seq_pair():
    a_tokens = ['a', 'b', 'c', 'a']
    b_tokens = ['b', 'c', 'd']

    _truncate_seq_pair(a_tokens, b_tokens, 5)

    print(b_tokens, a_tokens)

    assert len(a_tokens) == 3 and len(b_tokens) == 2
    assert len(a_tokens) + len(b_tokens) == 5


def test_select_field():
    tokens = ['This', 'is', 'a', 'test', '.']
    input_ids = [1, 2, 3, 4, 5]
    input_mask = [1, 1, 1, 1, 1]
    segment_ids = [0, 0, 0, 0, 0]
    choices_features = [(tokens, input_ids, input_mask, segment_ids)]
    features = [
        InputFeatures(example_id=1,
                      choices_features=choices_features,
                      label=None)
    ]
    field = 'input_ids'
    res = select_field(features, field)

    assert len(res) == 1 and len(res[0]) == 1 and isinstance(res[0][0], list)
    assert len(res[0][0]) == 5


def test_read_mcqa_examples(dummy_data_path):
    examples = read_mcqa_examples(dummy_data_path,
                                  is_training=False)
    assert len(examples) == 1
    assert examples[0].label == None

    examples = read_mcqa_examples(dummy_data_path,
                                  is_training=True)
    assert len(examples) == 1
    assert examples[0].label == 0


def test_get_labels(mcqa_dataset):
    labels = get_labels(mcqa_dataset)

    assert labels == np.array([0])
