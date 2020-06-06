import logging
import os
from typing import List
from typing import Optional
import numpy as np
import torch
from filelock import FileLock
from torch.utils.data.dataset import TensorDataset
from transformers import PreTrainedTokenizer
from .processors import ArcProcessor
from .processors import RaceProcessor
from .processors import SwagProcessor
from .processors import SynonymProcessor
from .utils import convert_examples_to_features
from .utils import InputFeatures
from .utils import Split, select_field

PROCESSORS = {"race": RaceProcessor,
              "swag": SwagProcessor,
              "arc": ArcProcessor,
              "syn": SynonymProcessor}


LOGGER = logging.getLogger(__name__)


class McqaDataset(TensorDataset):
    """Read and prepare the input data. Returns a Dataset."""

    features: List[InputFeatures]

    def __init__(
        self,
        data_dir: str,
        tokenizer: PreTrainedTokenizer,
        task: str,
        max_seq_length: Optional[int] = None,
        overwrite_cache=False,
        mode: Split = Split.train,
    ):
        processor = PROCESSORS[task]()

        cached_features_file = os.path.join(
            data_dir,
            "cached_{}_{}_{}_{}".format(
                mode.value, tokenizer.__class__.__name__, str(max_seq_length), task,),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                LOGGER.info(
                    "Loading features from cached file %s", cached_features_file)
                self.features = torch.load(cached_features_file)
            else:
                LOGGER.info(
                    "Creating features from dataset file at %s", data_dir)
                label_list = processor.get_labels()
                if mode == Split.dev:
                    examples = processor.get_dev_examples(data_dir)
                elif mode == Split.test:
                    examples = processor.get_test_examples(data_dir)
                else:
                    examples = processor.get_train_examples(data_dir)
                LOGGER.info("Training examples: %s", len(examples))
                # TODO clean up all this to leverage built-in features of tokenizers
                self.features = convert_examples_to_features(
                    examples,
                    label_list,
                    max_seq_length,
                    tokenizer)
                LOGGER.info("Saving features into cached file %s",
                            cached_features_file)
                torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    def get_dataset(self):
        all_input_ids = torch.tensor(select_field(self.features, 'input_ids'),
                                     dtype=torch.long)
        all_input_mask = torch.tensor(select_field(self.features, 'attention_mask'),
                                      dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(self.features, 'token_type_ids'),
                                       dtype=torch.long)

        all_label = torch.tensor([f.label for f in self.features],
                                 dtype=torch.long)

        self.data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                  all_label)

        return self.data

    def get_labels(self):
        """Get labels from a dataset

        Returns:
            [np.array] -- A numpy array of the labels
        """
        labels = [self.data[i][3] for i in range(len(self.data))]
        return np.array(labels)
