from mcqa.data import MCQAData, read_mcqa_examples
from torch.utils.data import TensorDataset
import torch


def test_convert_examples_to_features(mcqa_data, dummy_data_path):
    examples = read_mcqa_examples(dummy_data_path,
                                  is_training=True)
    features = mcqa_data.convert_examples_to_features(examples)

    assert len(features) == len(examples) == 1
    assert features[0].label == 0
    assert len(features[0].choices_features) == 4
    assert list(features[0].choices_features[0].keys()) \
        == ['input_ids', 'input_mask', 'segment_ids']


def test_read(mcqa_data, dummy_data_path):
    dataset = mcqa_data.read(dummy_data_path, is_training=True)

    assert isinstance(dataset, TensorDataset)
    assert len(dataset) == 1  # Nb of examples
    # 4 : all_input_ids, all_input_mask, all_segment_ids, all_label
    assert len(dataset[0]) == 4
    assert len(dataset[0][0]) == 4  # Nb of encodings for all_input_ids
    assert len(dataset[0][1]) == 4  # Nb of encodings for all_input_mask
    assert len(dataset[0][2]) == 4  # Nb of encodings for all_segment_ids
    assert dataset[0][3] == torch.tensor(0)  # all_label
    # all_input_ids of encoding 0
    assert len(dataset[0][0][0]) == mcqa_data.max_seq_length
    # all_segment_ids of encoding 1
    assert len(dataset[0][2][1]) == mcqa_data.max_seq_length
