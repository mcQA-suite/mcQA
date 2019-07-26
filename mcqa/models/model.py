import os
from tqdm import tqdm, trange
import logging
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from apex.parallel import DistributedDataParallel as DDP
from apex.optimizers import FP16_Optimizer
from apex.optimizers import FusedAdam

from pytorch_transformers import BertForMultipleChoice
from pytorch_transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_transformers import WEIGHTS_NAME, CONFIG_NAME
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule


class Model():
    """Multiple choice question answering model."""

    def __init__(self, device, bert_model, num_choices=4,
                 local_rank=-1, fp16=False, seed=0):

        self.bert_model = bert_model
        self.num_choices = num_choices

        if not torch.cuda.is_available() and (device == "cuda" or local_rank != -1):
            raise ValueError(
                "No cuda device detected, can't set `device` to cuda.")

        if local_rank == -1 or device == "cpu":
            device = torch.device("cuda" if torch.cuda.is_available() and
                                  not device == "cpu" else "cpu")
            n_gpu = torch.cuda.device_count()
        else:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            n_gpu = 1
            # Initializes the distributed backend which will take care
            #  of sychronizing nodes/GPUs
            torch.distributed.init_process_group(backend='nccl')

        logging.info("device: {} n_gpu: {}, distributed training: {}, \
                     16-bits training: {}".format(device, n_gpu,
                                                  bool(local_rank != -1), fp16))

        self.local_rank = local_rank
        self.device = device
        self.fp16 = fp16
        self.n_gpu = n_gpu
        self.model = None

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if n_gpu > 0:
            torch.cuda.manual_seed_all(seed)

    def _prepare_model(self, freeze):
        """Prepare a model to be trained

        Arguments:
            freeze {bool} -- Whether to freeze the BERT layers.

        Returns:
            [BertForMultipleChoice] -- BertForMultipleChoice model to train
        """
        model = BertForMultipleChoice.from_pretrained(self.bert_model,
                                                      cache_dir=os.path.join(
                                                          str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(self.local_rank)),
                                                      num_choices=self.num_choices)

        if self.fp16:
            model.half()

        model.to(self.device)

        if freeze:
            for param in model.bert.parameters():
                param.requires_grad = False

        if self.local_rank != -1:
            model = DDP(model)
        elif self.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        return model

    def _prepare_optimizer(self, learning_rate, loss_scale, warmup_proportion,
                           num_train_optimization_steps):
        """Initialize the optimizer

        Arguments:
            learning_rate {float} -- The initial learning rate for Adam
            loss_scale {float} -- Loss scaling to improve fp16 numeric 
                                  stability. Only used when fp16 set to True.
                                  0 (default value): dynamic loss scaling.
                                  Positive power of 2: static loss scaling value.
            warmup_proportion {float} -- Proportion of training to perform
                                         linear learning rate warmup for
                                         E.g., 0.1 = 10%% of training
            num_train_optimization_steps {int} -- Number of optimization steps

        Returns:
            Optimizer -- The optimizer to use while training
        """
        param_optimizer = list(self.model.named_parameters())

        # hack to remove pooler, which is not used
        # thus it produce None grad that break apex

        param_optimizer = [n for n in param_optimizer]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        if self.fp16:

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=learning_rate,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            if loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(
                    optimizer, static_loss_scale=loss_scale)

            warmup_linear = WarmupLinearSchedule(warmup_steps=warmup_proportion,
                                                 t_total=num_train_optimization_steps)

        else:
            optimizer = AdamW(optimizer_grouped_parameters,
                              lr=learning_rate,)
            warmup_linear = None

        return optimizer, warmup_linear

    def fit(self, train_dataset, train_batch_size, num_train_epochs,
            learning_rate=5e-5, loss_scale=0, gradient_accumulation_steps=1,
            warmup_proportion=0.1, freeze=True):
        """Train the multiple choice QA model

        Arguments:
            train_dataset {MCQADataset} -- The training dataset
            train_batch_size {int} -- Total batch size for training.
            num_train_epochs {[type]} -- Total number of training epochs
                                         to perform

        Keyword Arguments:
            learning_rate {float} -- The initial learning rate for Adam
                                     (default: {5e-5})
            loss_scale {int} -- Loss scaling to improve fp16 numeric stability
                                (default: {0})
            gradient_accumulation_steps {int} -- Number of updates steps to 
                                                 accumulate before performing 
                                                 a backward/update pass (default: {1})
            warmup_proportion {float} -- Proportion of training to perform linear 
                                         learning rate warmup for.  (default: {0.1})
            freeze {bool} -- Whether to freeze BERT layers (default: {True})

        Raises:
            ValueError: Invalid gradient_accumulation_steps

        Returns:
            [BertForMultipleChoice] -- Trained Multiple Choice QA model
        """

        if gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, \
                              should be >= 1".format(gradient_accumulation_steps))

        train_batch_size = train_batch_size // gradient_accumulation_steps

        self.model = self._prepare_model(freeze)

        if self.local_rank == -1:
            train_sampler = RandomSampler(train_dataset)
        else:
            train_sampler = DistributedSampler(train_dataset)

        train_dataloader = DataLoader(train_dataset,
                                      sampler=train_sampler,
                                      batch_size=train_batch_size)

        num_train_optimization_steps = len(
            train_dataloader) // gradient_accumulation_steps * num_train_epochs
        if self.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

        optimizer, warmup_linear = self._prepare_optimizer(learning_rate,
                                                           loss_scale,
                                                           warmup_proportion,
                                                           num_train_optimization_steps)

        logging.info("  Num examples = %d", len(train_dataset))
        logging.info("  Batch size = %d", train_batch_size)
        logging.info("  Num steps = %d", num_train_optimization_steps)

        global_step = 0

        self.model.train()

        for _ in trange(int(num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss = self.model(input_ids, segment_ids,
                                  input_mask, label_ids)[0]
                if self.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if self.fp16 and loss_scale != 1.0:
                    # rescale loss for fp16 training
                    # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                    loss = loss * loss_scale
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                if self.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                if (step + 1) % gradient_accumulation_steps == 0:
                    if self.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = learning_rate * \
                            warmup_linear.get_lr(
                                global_step, warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

        return self.model

    def save_model(self, path):
        """Save model to local

        Arguments:
            path {str} -- path to directory where to save the model
        """
        model_to_save = self.model.module if hasattr(self.model,
                                                     'module') else self.model

        # If we save using the predefined names, we can load using `from_pretrained`
        if not os.path.exists(path) or os.path.isfile(path):
            os.makedirs(path)

        output_model_file = os.path.join(path, WEIGHTS_NAME)
        output_config_file = os.path.join(path, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)

    def predict_proba(self, eval_dataset, eval_batch_size):
        """Predict probabilities of classes 

        Arguments:
            eval_dataset {MCQADataset} -- The eval dataset
            eval_batch_size {int} -- The evaluation batch size

        Raises:
            ValueError: Invalid local rank parameter

        Returns:
            [np.array] -- Numpy array with the probabilities
        """
        if not (self.local_rank == -1 or torch.distributed.get_rank() == 0):
            raise ValueError("Invalid local rank parameter.")

        logging.info("  Num examples = %d", len(eval_dataset))
        logging.info("  Batch size = %d", eval_batch_size)

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset,
                                     sampler=eval_sampler,
                                     batch_size=eval_batch_size)

        self.model.eval()

        outputs_proba = []

        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader,
                                                                  desc="Evaluating"):

            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)
            label_ids = label_ids.to(self.device)

            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask)[0]

            logits = torch.nn.functional.softmax(logits, dim=1)
            logits = logits.detach().cpu().numpy()
            outputs_proba.extend(logits.tolist())

        return np.array(outputs_proba)

    def predict(self, eval_dataset, eval_batch_size):
        """Genrate prediction of the eval dataset

        Arguments:
            eval_dataset {MCQADataset} -- The eval dataset
            eval_batch_size {int} -- The evaluation batch size

        Returns:
            [np.array] -- The predictions
        """
        outputs_proba = self.predict_proba(eval_dataset,
                                           eval_batch_size)

        return np.argmax(outputs_proba, axis=1)
