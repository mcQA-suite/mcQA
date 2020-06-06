import numpy as np

from mcqa.data import (convert_examples_to_features,
                       SwagProcessor)


def test_convert_examples_to_features(tokenizer, dummy_data_path):
    processor = SwagProcessor()
    label_list = processor.get_labels()
    examples = processor.get_train_examples(dummy_data_path)
    features = convert_examples_to_features(
        examples=examples,
        label_list=label_list,
        tokenizer=tokenizer,
        max_length=10)

    assert len(features) == len(examples) == 1
    assert features[0].label == 0
