import torch
from torch.utils.data.dataset import TensorDataset
from mcqa.data import McqaDataset


def test_McqaDataset(mcqa_dataset):
    dataset = mcqa_dataset.get_dataset()
    assert isinstance(dataset, TensorDataset)
    assert len(dataset) == 1  # Nb of examples
    # 4 : all_input_ids, all_input_mask, all_segment_ids, all_label
    assert len(dataset[0]) == 4
    assert len(dataset[0][0]) == 4  # Nb of encodings for all_input_ids
    assert len(dataset[0][1]) == 4  # Nb of encodings for all_input_mask
    assert len(dataset[0][2]) == 4  # Nb of encodings for all_segment_ids
    assert dataset[0][3] == torch.tensor(0)  # all_label
    # all_input_ids of encoding 0
    assert len(dataset[0][0][0]) == 10
    # all_segment_ids of encoding 1
    assert len(dataset[0][2][1]) == 10
