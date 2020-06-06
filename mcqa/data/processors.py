""" Multiple choice processors to work with multiple choice tasks of reading comprehension."""
import os
import logging
import csv
import glob
import json
from typing import List
import tqdm

from .utils import InputExample

MULTIPLE_CHOICE_TASKS_NUM_LABELS = {"race", 4, "swag",
                                    4, "arc", 4, "syn", 5}

LOGGER = logging.getLogger(__name__)


class DataProcessor:
    """Base class for data processors for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class RaceProcessor(DataProcessor):
    """Processor for the RACE data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        LOGGER.info("LOOKING AT %s train", data_dir)
        high = os.path.join(data_dir, "train/high")
        middle = os.path.join(data_dir, "train/middle")
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        LOGGER.info("LOOKING AT %s dev", LOGGER)
        high = os.path.join(data_dir, "dev/high")
        middle = os.path.join(data_dir, "dev/middle")
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        LOGGER.info("LOOKING AT %s test", data_dir)
        high = os.path.join(data_dir, "test/high")
        middle = os.path.join(data_dir, "test/middle")
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_txt(self, input_dir):
        lines = []
        files = glob.glob(input_dir + "/*txt")
        for file in tqdm.tqdm(files, desc="read files"):
            with open(file, "r", encoding="utf-8") as f:
                data_raw = json.load(f)
                data_raw["race_id"] = file
                lines.append(data_raw)
        return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (_, data_raw) in enumerate(lines):
            race_id = "%s-%s" % (set_type, data_raw["race_id"])
            article = data_raw["article"]
            for i in range(len(data_raw["answers"])):
                truth = str(ord(data_raw["answers"][i]) - ord("A"))
                question = data_raw["questions"][i]
                options = data_raw["options"][i]

                examples.append(
                    InputExample(
                        example_id=race_id,
                        question=question,
                        # this is not efficient but convenient
                        contexts=[article, article, article, article],
                        endings=[options[0], options[1],
                                 options[2], options[3]],
                        label=truth,
                    )
                )
        return examples


class SynonymProcessor(DataProcessor):
    """Processor for the Synonym data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        LOGGER.info("LOOKING AT %s train", data_dir)
        return self._create_examples(self._read_csv(
            os.path.join(data_dir, "mctrain.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        LOGGER.info("LOOKING AT %s dev", data_dir)
        return self._create_examples(self._read_csv(
            os.path.join(data_dir, "mchp.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        LOGGER.info("LOOKING AT %s dev", data_dir)

        return self._create_examples(self._read_csv(os.path.join(
            data_dir, "mctest.csv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3", "4"]

    def _read_csv(self, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            return list(csv.reader(f))

    def _create_examples(self, lines: List[List[str]], type: str):
        """Creates examples for the training and dev sets."""

        examples = [
            InputExample(
                example_id=line[0],
                question="",  # in the swag dataset, the
                # common beginning of each
                # choice is stored in "sent2".
                contexts=[line[1], line[1], line[1], line[1], line[1]],
                endings=[line[2], line[3], line[4], line[5], line[6]],
                label=line[7],
            )
            for line in lines  # we skip the line with the column names
        ]

        return examples


class SwagProcessor(DataProcessor):
    """Processor for the SWAG data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        LOGGER.info("LOOKING AT %s train", data_dir)
        return self._create_examples(self._read_csv(
            os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        LOGGER.info("LOOKING AT %s dev", data_dir)
        return self._create_examples(self._read_csv(
            os.path.join(data_dir, "val.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        LOGGER.info("LOOKING AT %s dev", data_dir)
        raise ValueError(
            "For swag testing, the input file does not contain a label column."
            "It can not be tested in current code setting!"
        )
        return self._create_examples(self._read_csv(
            os.path.join(data_dir, "test.csv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_csv(self, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            return list(csv.reader(f))

    def _create_examples(self, lines: List[List[str]], type: str):
        """Creates examples for the training and dev sets."""
        if type == "train" and lines[0][-1] != "label":
            raise ValueError(
                "For training, the input file must contain a label column.")

        examples = [
            InputExample(
                example_id=line[2],
                question=line[5],  # in the swag dataset, the
                # common beginning of each
                # choice is stored in "sent2".
                contexts=[line[4], line[4], line[4], line[4]],
                endings=[line[7], line[8], line[9], line[10]],
                label=line[11],
            )
            for line in lines[1:]  # we skip the line with the column names
        ]

        return examples


class ArcProcessor(DataProcessor):
    """Processor for the ARC data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        LOGGER.info("LOOKING AT %s train", data_dir)
        return self._create_examples(self._read_json(os.path.join(
            data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        LOGGER.info("LOOKING AT %s dev", data_dir)
        return self._create_examples(self._read_json(
            os.path.join(data_dir, "dev.jsonl")), "dev")

    def get_test_examples(self, data_dir):
        LOGGER.info("LOOKING AT %s test", data_dir)
        return self._create_examples(self._read_json(
            os.path.join(data_dir, "test.jsonl")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_json(self, input_file):
        with open(input_file, "r", encoding="utf-8") as fin:
            lines = fin.readlines()
            return lines

    def _create_examples(self, lines, type):
        """Creates examples for the training and dev sets."""

        # There are two types of labels. They should be normalized
        def normalize(truth):
            if truth in "ABCD":
                return ord(truth) - ord("A")
            if truth in "1234":
                return int(truth) - 1

            LOGGER.info("truth ERROR! %s", str(truth))
            return None

        examples = []
        three_choice = 0
        four_choice = 0
        five_choice = 0
        other_choices = 0
        # we deleted example which has more than or less than four choices
        for line in tqdm.tqdm(lines, desc="read arc data"):
            data_raw = json.loads(line.strip("\n"))
            if len(data_raw["question"]["choices"]) == 3:
                three_choice += 1
                continue
            if len(data_raw["question"]["choices"]) == 5:
                five_choice += 1
                continue
            if len(data_raw["question"]["choices"]) != 4:
                other_choices += 1
                continue
            four_choice += 1
            truth = str(normalize(data_raw["answerKey"]))
            assert truth != "None"
            question_choices = data_raw["question"]
            question = question_choices["stem"]
            id = data_raw["id"]
            options = question_choices["choices"]
            if len(options) == 4:
                examples.append(
                    InputExample(
                        example_id=id,
                        question=question,
                        contexts=[
                            options[0]["para"].replace("_", ""),
                            options[1]["para"].replace("_", ""),
                            options[2]["para"].replace("_", ""),
                            options[3]["para"].replace("_", ""),
                        ],
                        endings=[options[0]["text"], options[1]["text"],
                                 options[2]["text"], options[3]["text"]],
                        label=truth,
                    )
                )

        if type == "train":
            assert len(examples) > 1
            assert examples[0].label is not None
        LOGGER.info("len examples: %s}", str(len(examples)))
        LOGGER.info("Three choices: %s", str(three_choice))
        LOGGER.info("Five choices: %s", str(five_choice))
        LOGGER.info("Other choices: %s", str(other_choices))
        LOGGER.info("four choices: %s", str(four_choice))

        return examples
