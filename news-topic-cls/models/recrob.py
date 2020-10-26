import pandas as pd
from math import ceil, inf
from tqdm import tqdm
import torch
from torch.nn import Sigmoid, BCELoss, Linear, Dropout
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler, RandomSampler
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.core.step_result import TrainResult, EvalResult
from pytorch_lightning.metrics.sklearns import Accuracy
from math import sqrt

from models.extension import RNNExtension, ClassifierExtension
from data.data import split_articles
from utils.optim import weighted_bce
from utils.utils import get_labels, IncreasedLoggingLevel, TransformerOptions


class RecRob(LightningModule):
    def __init__(self, hparams):
        super(RecRob, self).__init__()
        self.hparams = hparams

        roberta = TransformerOptions(
            "roberta-base", lm_path=self.hparams.lm_path)

        self.roberta = roberta.model
        self.dropout = Dropout(roberta.config.hidden_dropout_prob)
        self.rnn = RNNExtension(hidden_size=hparams.rnn_hidden_size)

        if int(self.hparams.amount_labels) == 1:
            self.num_labels = int(self.hparams.amount_labels) + 1
        else:
            self.num_labels = int(self.hparams.amount_labels)

        self.classifier = ClassifierExtension(input_size=hparams.rnn_hidden_size,
                                              hidden_size=hparams.cls_hidden_size,
                                              num_labels=int(self.num_labels))

        self.tokenizer = roberta.tokenizer
        self.loss_function = weighted_bce

        self.eval_conf_matrix = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        self.test_conf_matrix = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    def forward(self, input_ids, attention_mask):
        split_length = self.hparams.split_size
        position_ids = torch.arange(split_length).view(
            1, split_length).repeat(input_ids.shape[0], 1)
        outputs = self.roberta(input_ids=input_ids,
                               attention_mask=attention_mask,
                               position_ids=position_ids.to(self.device))
        cls_embeddings = outputs[0][:, 0, :]
        rnn_input_dim = (cls_embeddings.shape[0], 1, cls_embeddings.shape[1])
        document_embedding = self.rnn(
            self.dropout(cls_embeddings.view(rnn_input_dim)))
        logits = self.classifier(self.dropout(document_embedding))
        return logits

    def prepare_data(self):

        split_seed = self.hparams.data_path.strip("/").split("/")[-1]
        if split_seed != "test":
            train_data_path = "{}/df_train_{}_augmented_{}labels.csv".format(self.hparams.data_path,
                                                                             split_seed,
                                                                             self.hparams.amount_labels)

            eval_data_path = "{}/df_eval_{}.csv".format(self.hparams.data_path,
                                                        split_seed)
        else:
            train_data_path = "{}/df_test.csv".format(self.hparams.data_path)
            eval_data_path = "{}/df_test.csv".format(self.hparams.data_path)

        train_eval_path = "/".join(
            self.hparams.data_path.strip("/").split("/")[:-1])
        test_data_path = "{}/gold-standard-testset.csv".format(
            train_eval_path)

        training_set = pd.read_csv(train_data_path)
        evaluation_set = pd.read_csv(eval_data_path)
        try:
            test_set = pd.read_csv(test_data_path)
        except:
            test_set = pd.read_csv("/" + test_data_path)

        labels = get_labels(self.hparams.amount_labels)

        split_length = self.hparams.split_size
        shift = self.hparams.shift
        self.tokenizer.padding_side = "right"
        self.tokenizer.pad_token = "<pad>"

        with IncreasedLoggingLevel("transformers.tokenization_utils_base"):
            max_length_training = max([len(self.tokenizer(text)["input_ids"])
                                       for text in training_set.text.values])
            max_length_evaluation = max([len(self.tokenizer(text)["input_ids"])
                                         for text in evaluation_set.text.values])
            max_length_test = max([len(self.tokenizer(text)["input_ids"])
                                   for text in test_set.text.values])

        max_padding_training = ceil(max_length_training/split_length) * \
            split_length + split_length
        max_padding_evaluation = ceil(max_length_evaluation/split_length) * \
            split_length + split_length
        max_padding_test = ceil(max_length_test/split_length) * \
            split_length + split_length

        print("Computing Input")
        with IncreasedLoggingLevel("transformers.tokenization_utils_base"):
            training_inputs = [self.tokenizer(text,
                                              max_length=max_padding_training,
                                              padding="max_length",
                                              truncation=True)
                               for text in tqdm(training_set.text.values,
                                                total=len(training_set))]

        training_input_ids, training_attention_mask = split_articles(training_inputs,
                                                                     max_length=max_padding_training,
                                                                     split_length=split_length,
                                                                     shift=shift)

        training_labels = training_set.loc[:, labels].values

        with IncreasedLoggingLevel("transformers.tokenization_utils_base"):
            evaluation_inputs = [self.tokenizer(text,
                                                max_length=max_padding_evaluation,
                                                padding="max_length",
                                                truncation=True)
                                 for text in tqdm(evaluation_set.text.values,
                                                  total=len(evaluation_set))]

        evaluation_input_ids, evaluation_attention_mask = split_articles(evaluation_inputs,
                                                                         max_length=max_padding_evaluation,
                                                                         split_length=split_length,
                                                                         shift=shift)

        evaluation_labels = evaluation_set.loc[:, labels].values

        with IncreasedLoggingLevel("transformers.tokenization_utils_base"):
            test_inputs = [self.tokenizer(text,
                                          max_length=max_padding_test,
                                          padding="max_length",
                                          truncation=True)
                           for text in tqdm(test_set.text.values,
                                            total=len(test_set))]

        test_input_ids, test_attention_mask = split_articles(test_inputs,
                                                             max_length=max_padding_test,
                                                             split_length=split_length,
                                                             shift=shift)

        test_labels = test_set.loc[:, labels].values

        training_input_ids = torch.tensor(training_input_ids)
        training_attention_mask = torch.tensor(training_attention_mask)
        training_labels = torch.tensor(training_labels)

        evaluation_input_ids = torch.tensor(evaluation_input_ids)
        evaluation_attention_mask = torch.tensor(evaluation_attention_mask)
        evaluation_labels = torch.tensor(evaluation_labels)

        test_input_ids = torch.tensor(test_input_ids)
        test_attention_mask = torch.tensor(test_attention_mask)
        test_labels = torch.tensor(test_labels)

        self.label_weights = 1 / \
            evaluation_labels.shape[0] * evaluation_labels.sum(dim=0)

        self.train_data = TensorDataset(training_input_ids,
                                        training_attention_mask,
                                        training_labels)
        self.valid_data = TensorDataset(evaluation_input_ids,
                                        evaluation_attention_mask,
                                        evaluation_labels)
        self.test_data = TensorDataset(test_input_ids,
                                       test_attention_mask,
                                       test_labels)

    def collate(self, batch):
        input_ids_batch, mask_batch, labels_batch = batch[0]
        amount_splits = input_ids_batch.shape[0]
        sequence_length = input_ids_batch.shape[1]

        input_ids_batch = input_ids_batch.view(amount_splits, sequence_length)
        mask_batch = mask_batch.view(amount_splits, sequence_length)
        labels_batch = labels_batch.view(-1)

        for index, batch_part in enumerate(input_ids_batch):
            if batch_part[0] == self.tokenizer.pad_token_id or index == amount_splits-1:
                number_parts = index
                break

        type_ = torch.LongTensor
        batch = (input_ids_batch[:number_parts].type(type_),
                 mask_batch[:number_parts].type(type_),
                 labels_batch.type(torch.float).view(1, self.num_labels))

        return batch

    def train_dataloader(self):
        train_sampler = RandomSampler(self.train_data)
        train_dataloader = DataLoader(self.train_data,
                                      sampler=train_sampler,
                                      collate_fn=self.collate,
                                      batch_size=self.hparams.batch_size,
                                      num_workers=self.hparams.num_workers)
        return train_dataloader

    def val_dataloader(self):
        valid_sampler = SequentialSampler(self.valid_data)
        valid_dataloader = DataLoader(self.valid_data,
                                      sampler=valid_sampler,
                                      collate_fn=self.collate,
                                      batch_size=self.hparams.batch_size,
                                      num_workers=self.hparams.num_workers)
        return valid_dataloader

    def test_dataloader(self):
        test_sampler = SequentialSampler(self.test_data)
        test_dataloader = DataLoader(self.test_data,
                                     sampler=test_sampler,
                                     collate_fn=self.collate,
                                     batch_size=self.hparams.batch_size,
                                     num_workers=self.hparams.num_workers)
        return test_dataloader

    def configure_optimizers(self):
        if not self.hparams.scheduler:
            layers_with_parameters_lm = []
            for module in self.roberta.modules():
                for child in module.children():
                    if (list(child.children()) == [] and
                        list(child.parameters()) != [] and
                            child.__class__.__name__ != "LayerNorm"):
                        layers_with_parameters_lm.append(child)

            params_lm = [
                {"params": layer.parameters(),
                 "lr": self.hparams.learning_rate_lm * self.hparams.lr_decay**index}
                for index, layer in enumerate(reversed(layers_with_parameters_lm))
            ]

            layers_with_parameters_ext = []
            for module in self.rnn.modules():
                for child in module.children():
                    if (list(child.children()) == [] and
                        list(child.parameters()) != [] and
                            child.__class__.__name__ != "LayerNorm"):
                        layers_with_parameters_ext.append(child)

            for module in self.classifier.modules():
                for child in module.children():
                    if (list(child.children()) == [] and
                        list(child.parameters()) != [] and
                            child.__class__.__name__ != "LayerNorm"):
                        layers_with_parameters_ext.append(child)

            params_ext = [
                {"params": layer.parameters(),
                 "lr": self.hparams.learning_rate_ext * self.hparams.lr_decay**index}
                for index, layer in enumerate(reversed(layers_with_parameters_ext))
            ]

            return Adam(params_lm + params_ext,
                        lr=self.hparams.learning_rate_ext,
                        weight_decay=self.hparams.weight_decay)
        else:
            param_optimizer = list(self.rnn.parameters())
            param_optimizer += list(self.classifier.parameters())

            optimizer_grouped_parameters = [
                {"params": [param for param in param_optimizer]}]

            optimizer = Adam(optimizer_grouped_parameters,
                             lr=self.hparams.learning_rate_ext,
                             weight_decay=self.hparams.weight_decay)

            scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                          factor=0.5,
                                          patience=1,
                                          verbose=True)

            scheduler_dict = {
                "scheduler": scheduler,
                "name": "LR-Scheduler",
                "interval": "epoch",
                "reduce_on_plateau": True,
                "monitor": "val_checkpoint_on"
            }

            return [optimizer], [scheduler_dict]

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        outputs = self.forward(input_ids, attention_mask)

        loss = self.loss_function(predictions=outputs,
                                  targets=labels,
                                  weight=self.hparams.loss_weight)

        result = TrainResult(loss)
        metrics = {
            "train_loss": loss
        }

        result.log_dict(metrics,
                        prog_bar=True,
                        logger=self.logger.experiment)
        return result

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        outputs = self.forward(input_ids, attention_mask)
        loss = self.loss_function(outputs, labels)

        result = EvalResult(checkpoint_on=loss)

        predictions = torch.where(outputs > 0.5,
                                  torch.ones(outputs.shape,
                                             device=self.device),
                                  torch.zeros(outputs.shape,
                                              device=self.device))

        if int(self.hparams.amount_labels) == 1:
            batch_size = labels.shape[0]
            labels = labels[:, 0].view(batch_size, 1)
            predictions = predictions[:, 0].view(batch_size, 1)

        accuracy = Accuracy()
        acc = accuracy(predictions, labels)
        tp = float(torch.logical_and(predictions == 1, labels == 1).sum())
        fp = float(torch.logical_and(predictions == 1, labels == 0).sum())
        tn = float(torch.logical_and(predictions == 0, labels == 0).sum())
        fn = float(torch.logical_and(predictions == 0, labels == 1).sum())

        self.eval_conf_matrix["tp"] += tp
        self.eval_conf_matrix["fp"] += fp
        self.eval_conf_matrix["tn"] += tn
        self.eval_conf_matrix["fn"] += fn

        result.log("eval_loss", loss)
        result.log("eval_acc", acc)
        return result

    def on_validation_epoch_end(self):

        tp = self.eval_conf_matrix["tp"]
        fp = self.eval_conf_matrix["fp"]
        fn = self.eval_conf_matrix["fn"]
        tn = self.eval_conf_matrix["tn"]

        try:
            recall = tp/(tp + fn)
        except:
            recall = 0

        try:
            precision = tp/(tp + fp)
        except:
            precision = 0

        if recall == 0 and precision == 0:
            f1 = 0
        else:
            f1 = 2*recall*precision/(recall + precision)

        try:
            mcc = (tp*tn-fp*fn)/sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        except:
            mcc = 0

        metrics = {"eval_tp": torch.tensor(tp),
                   "eval_fp": torch.tensor(fp),
                   "eval_fn": torch.tensor(fn),
                   "eval_tn": torch.tensor(tn),
                   "eval_precision": torch.tensor(precision),
                   "eval_recall": torch.tensor(recall),
                   "eval_f1": torch.tensor(f1),
                   "eval_mcc": torch.tensor(mcc)}

        self.logger.log_metrics(metrics=metrics, step=self.global_step)
        self.eval_conf_matrix = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        outputs = self.forward(input_ids, attention_mask)
        loss = self.loss_function(outputs, labels)

        predictions = torch.where(outputs > 0.5,
                                  torch.ones(outputs.shape,
                                             device=self.device),
                                  torch.zeros(outputs.shape,
                                              device=self.device))

        if int(self.hparams.amount_labels) == 1:
            batch_size = labels.shape[0]
            labels = labels[:, 0].view(-1)
            predictions = predictions[:, 0].view(-1)

        accuracy = Accuracy()
        acc = accuracy(predictions, labels)

        tp = float(torch.logical_and(predictions == 1, labels == 1).sum())
        fp = float(torch.logical_and(predictions == 1, labels == 0).sum())
        tn = float(torch.logical_and(predictions == 0, labels == 0).sum())
        fn = float(torch.logical_and(predictions == 0, labels == 1).sum())

        self.test_conf_matrix["tp"] += tp
        self.test_conf_matrix["fp"] += fp
        self.test_conf_matrix["tn"] += tn
        self.test_conf_matrix["fn"] += fn

        result = EvalResult()
        result.log("test_loss", loss)
        result.log("acc", acc)
        return result

    def on_test_epoch_end(self):

        tp = self.test_conf_matrix["tp"]
        fp = self.test_conf_matrix["fp"]
        fn = self.test_conf_matrix["fn"]
        tn = self.test_conf_matrix["tn"]

        try:
            recall = tp/(tp + fn)
        except:
            recall = 0

        try:
            precision = tp/(tp + fp)
        except:
            precision = 0

        if recall == 0 and precision == 0:
            f1 = 0
        else:
            f1 = 2*recall*precision/(recall + precision)

        try:
            mcc = (tp*tn-fp*fn)/sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        except:
            mcc = 0

        metrics = {"tp": torch.tensor(tp),
                   "fp": torch.tensor(fp),
                   "fn": torch.tensor(fn),
                   "tn": torch.tensor(tn),
                   "precision": torch.tensor(precision),
                   "recall": torch.tensor(recall),
                   "f1": torch.tensor(f1),
                   "mcc": torch.tensor(mcc)}

        self.logger.log_metrics(metrics)

        print("\n--------------------------------------------------------------------------------")
        print("METRICS ON TESTSET")
        for metric, value in metrics.items():
            print("{}: {}".format(metric, value))
