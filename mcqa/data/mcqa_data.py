import torch
from torch.utils.data import TensorDataset
from pytorch_transformers.tokenization_bert import BertTokenizer
from.utils import InputFeatures, _truncate_seq_pair, select_field, \
    read_mcqa_examples


class MCQAData():
    """Read and prepare the input data. Returns a TensorDataSet."""

    def __init__(self, bert_model, lower_case, max_seq_length):
        self.tokenizer = BertTokenizer.from_pretrained(
            bert_model, lower_case=lower_case)
        self.max_seq_length = max_seq_length

    def convert_examples_to_features(self, examples):
        """Convert a list of `MCQAExample`s to a list of `InputFeatures`s

        MCQA is a multiple choice task. To perform this task using Bert,
        we will use the formatting proposed in "Improving Language
        Understanding by Generative Pre-Training" and suggested by
        @jacobdevlin-google in this issue
        https://github.com/google-research/bert/issues/38 :
        Each choice will correspond to a sample on which we run the
        inference. For a given MCQA example, we will create the 4
        following inputs:
            - [CLS] context [SEP] choice_1 [SEP]
            - [CLS] context [SEP] choice_2 [SEP]
            - [CLS] context [SEP] choice_3 [SEP]
            - [CLS] context [SEP] choice_4 [SEP]
        The model will output a single value for each input. To get the
        final decision of the model, we will run a softmax over these 4
        outputs.

        Arguments:
            examples [MCQAExample] -- list of `MCQAExample`s

        Returns:
            [InputFeatures] -- list of `InputFeatures`
        """

        features = []
        for _, example in enumerate(examples):
            context_tokens = self.tokenizer.tokenize(example.context_sentence)
            choices_features = []
            for _, ending in enumerate(example.endings):
                # We create a copy of the context tokens in order to be
                # able to shrink it according to ending_tokens
                context_tokens_choice = context_tokens[:]
                ending_tokens = self.tokenizer.tokenize(ending)
                # Modifies `context_tokens_choice` and `ending_tokens` in
                # place so that the total length is less than the
                # specified length.  Account for [CLS], [SEP], [SEP] with
                # "- 3"
                _truncate_seq_pair(
                    context_tokens_choice, ending_tokens, self.max_seq_length - 3)

                tokens = ["[CLS]"] + context_tokens_choice + \
                    ["[SEP]"] + ending_tokens + ["[SEP]"]

                segment_ids = [
                    0] * (len(context_tokens_choice) + 2) + [1] * (len(ending_tokens) + 1)

                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)

                # Zero-pad up to the sequence length.
                padding = [0] * (self.max_seq_length - len(input_ids))
                input_ids += padding
                input_mask += padding
                segment_ids += padding

                assert len(input_ids) == self.max_seq_length
                assert len(input_mask) == self.max_seq_length
                assert len(segment_ids) == self.max_seq_length

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

        examples = read_mcqa_examples(data_file, is_training=is_training)
        features = self.convert_examples_to_features(examples)

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
