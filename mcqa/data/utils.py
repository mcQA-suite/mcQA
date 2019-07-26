import csv
import numpy as np


class MCQAExample():
    """A single training/test example for the MCQA dataset."""

    def __init__(self,
                 mcqa_id,
                 context_sentence,
                 start_ending,
                 ending_0,
                 ending_1,
                 ending_2,
                 ending_3,
                 label=None):
        self.mcqa_id = mcqa_id
        self.context_sentence = context_sentence
        self.start_ending = start_ending
        self.endings = [
            ending_0,
            ending_1,
            ending_2,
            ending_3,
        ]
        self.label = label

    def __str__(self):
        return self.__repr__()


class InputFeatures():
    """Input features for each example."""

    def __init__(self,
                 example_id,
                 choices_features,
                 label
                 ):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for _, input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length.

    This is a simple heuristic which will always truncate the longer sequence
    one token at a time. This makes more sense than truncating an equal percent
    of tokens from each, since if one sequence is very short then each token
    that's truncated likely contains more information than a longer sequence.

    Arguments:
        tokens_a {[str]} -- First sequence
        tokens_b {[str]} -- Second sequence
        max_length {int} -- Maximum length of the sequence pair
    """

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def select_field(features, field):
    """Select a field from the features

    Arguments:
        features {InputFeatures} -- List of features : Instances of InputFeatures
                                    with attribute choice_features being a list of dicts.
        field {str} -- Field to consider.

    Returns:
        [list] -- List of corresponding features for field.
    """

    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]


def read_mcqa_examples(input_file, is_training):
    """Read an input file and return the corresponding MCQAExample instances.

    Arguments:
        input_file {str} -- File containing the data.
        is_training {bool} -- Whether this is a training data
                                (labels not None).

    Returns:
        [MCQAExample] -- Returns a list of instances of `MCQAExample`.
    """

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
        ) for line in lines[1:]  # we skip the line with the column names
    ]

    return examples


def get_labels(dataset):
    """Get labels from a dataset

    Arguments:
        dataset {MCQADataset} -- A dataset with valid labels

    Returns:
        [np.array] -- A numpy array of the labels
    """
    labels = [dataset[i][3] for i in range(len(dataset))]
    return np.array(labels)
