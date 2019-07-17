import csv
import torch
from torch.utils.data import TensorDataset

from pytorch_transformers.tokenization_bert import BertTokenizer

from.utils import MCQAExample, InputFeatures, _truncate_seq_pair, select_field


class MCQAData():
    def __init__(self, bert_model, lower_case, max_seq_length):
        self.tokenizer = BertTokenizer.from_pretrained(
            bert_model, lower_case=lower_case)
        self.max_seq_length = max_seq_length

    def read_mcqa_examples(self, input_file, is_training):

        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            lines = []
            for line in reader:
                lines.append(line)

        if is_training and lines[0][-1] != 'label':
            raise ValueError(
                "For training, the input file must contain a label column."
            )

        examples = [
            MCQAExample(
                mcqa_id=line[2],
                context_sentence=line[4],
                start_ending=line[5],  # the common beginning of each
                # choice is stored in "sent2".
                ending_0=line[7],
                ending_1=line[8],
                ending_2=line[9],
                ending_3=line[10],
                label=int(line[11]) if is_training else None
            ) for line in lines[1:] 
        ]

        return examples

    def convert_examples_to_features(self, examples, max_seq_length):
        """Loads a data file into a list of `InputBatch`s."""

        # MCQA is a multiple choice task. To perform this task using Bert,
        # we will use the formatting proposed in "Improving Language
        # Understanding by Generative Pre-Training" and suggested by
        # @jacobdevlin-google in this issue
        # https://github.com/google-research/bert/issues/38.
        #
        # Each choice will correspond to a sample on which we run the
        # inference. For a given MCQA example, we will create the 4
        # following inputs:
        # - [CLS] context [SEP] choice_1 [SEP]
        # - [CLS] context [SEP] choice_2 [SEP]
        # - [CLS] context [SEP] choice_3 [SEP]
        # - [CLS] context [SEP] choice_4 [SEP]
        # The model will output a single value for each input. To get the
        # final decision of the model, we will run a softmax over these 4
        # outputs.
        features = []
        for _, example in enumerate(examples):
            context_tokens = self.tokenizer.tokenize(example.context_sentence)
            start_ending_tokens = self.tokenizer.tokenize(example.start_ending)

            choices_features = []
            for _, ending in enumerate(example.endings):
                # We create a copy of the context tokens in order to be
                # able to shrink it according to ending_tokens
                context_tokens_choice = context_tokens[:]
                ending_tokens = start_ending_tokens + \
                    self.tokenizer.tokenize(ending)
                # Modifies `context_tokens_choice` and `ending_tokens` in
                # place so that the total length is less than the
                # specified length.  Account for [CLS], [SEP], [SEP] with
                # "- 3"
                _truncate_seq_pair(
                    context_tokens_choice, ending_tokens, max_seq_length - 3)

                tokens = ["[CLS]"] + context_tokens_choice + \
                    ["[SEP]"] + ending_tokens + ["[SEP]"]
                segment_ids = [
                    0] * (len(context_tokens_choice) + 2) + [1] * (len(ending_tokens) + 1)

                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)

                # Zero-pad up to the sequence length.
                padding = [0] * (max_seq_length - len(input_ids))
                input_ids += padding
                input_mask += padding
                segment_ids += padding

                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length

                choices_features.append(
                    (tokens, input_ids, input_mask, segment_ids))

            label = example.label

            features.append(
                InputFeatures(
                    example_id=example.mcqa_id,
                    choices_features=choices_features,
                    label=label
                )
            )

        return features

    def read(self, data_file, is_training):
        """Read and preprocess data

        Arguments:
            data_file {str} -- Path to the data : Should be in same
                                                     format as SWAG dataset.

        Returns:
            dataset {TensorDataset}
        """

        examples = self.read_mcqa_examples(data_file, is_training=is_training)
        features = self.convert_examples_to_features(
            examples, self.max_seq_length)

        all_input_ids = torch.tensor(select_field(features, 'input_ids'),
                                     dtype=torch.long)
        all_input_mask = torch.tensor(select_field(features, 'input_mask'),
                                      dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(features, 'segment_ids'),
                                       dtype=torch.long)

        all_label = torch.tensor([f.label for f in features],
                                 dtype=torch.long)

        data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                             all_label)

        return data
