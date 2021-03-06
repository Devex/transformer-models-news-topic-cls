import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.nn import Sigmoid, BCELoss, Linear, Dropout
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler, RandomSampler
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.core.step_result import TrainResult, EvalResult
from pytorch_lightning.metrics.sklearns import Accuracy
from tqdm import tqdm
from math import sqrt
import json

from models.extension import ClassifierExtension
from data.augment_dataframe import augment_dataframe
from utils.utils import get_labels, TransformerOptions
from utils.optim import adam_discriminative_lr, weighted_bce


class RobSum(LightningModule):
    def __init__(self,
                 hparams):
        super(RobSum, self).__init__()

        self.hparams = hparams

        roberta = TransformerOptions("roberta-base",
                                     lm_path=self.hparams.lm_path)

        self.roberta = roberta.model
        self.dropout = Dropout(roberta.config.hidden_dropout_prob)
        if int(self.hparams.amount_labels) == 1:
            num_labels = int(self.hparams.amount_labels) + 1
        else:
            num_labels = int(self.hparams.amount_labels)
        self.classifier = ClassifierExtension(input_size=roberta.config.hidden_size,
                                              hidden_size=hparams.cls_hidden_size,
                                              num_labels=num_labels)

        self.tokenizer = roberta.tokenizer
        self.loss_function = weighted_bce

        self.test_conf_matrix = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        self.eval_conf_matrix = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    def forward(self, input_ids, attention_mask):
        position_ids = torch.arange(self.tokenizer.model_max_length).view(
            1, self.tokenizer.model_max_length).repeat(input_ids.shape[0], 1)

        outputs = self.roberta(input_ids=input_ids,
                               attention_mask=attention_mask,
                               position_ids=position_ids.to(self.device))

        cls_embedding = outputs[0][:, 0, :]
        return self.classifier(self.dropout(cls_embedding))

    def prepare_data(self):
        split_seed = self.hparams.data_path.strip("/").split("/")[-1]
        if split_seed != "test":
            if self.hparams.label is not None:
                train_data_path = "{}/df_train_{}{}_augmented_{}labels.csv".format(self.hparams.data_path,
                                                                                   self.hparams.label.lower(),
                                                                                   split_seed,
                                                                                   self.hparams.amount_labels)

            else:
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

        try:
            training_set = pd.read_csv(train_data_path)
        except:
            if self.hparams.augment:
                print("Augmenting Data...")
                train_data_path_unaug = "{}/df_train_{}.csv".format(self.hparams.data_path,
                                                                    split_seed)
                unaug_training_set = pd.read_csv(train_data_path_unaug)
                augmentations_path = "/".join(
                    self.hparams.data_path.strip("/").split("/")[:-2])

                augmentations_df = pd.read_csv(
                    "/{}/article_confirmed_summary_augmentations.csv".format(augmentations_path))
                training_set = augment_dataframe(df=unaug_training_set,
                                                 augmentations_df=augmentations_df,
                                                 categories=[self.hparams.label.upper()])
                training_set.to_csv(train_data_path, index=False)
            else:
                print("Loading Data without Augmentation...")
                train_data_path = "{}/df_train_{}.csv".format(self.hparams.data_path,
                                                              split_seed)
                training_set = pd.read_csv(train_data_path)
                training_set.loc[:, "none"] = 0
                for index, row in training_set.iterrows():
                    if row.loc["none"] == 0:
                        training_set.loc[index, "none"] = 1

        evaluation_set = pd.read_csv(eval_data_path)
        try:
            test_set = pd.read_csv(test_data_path)
        except:
            test_set = pd.read_csv("/" + test_data_path)

        if self.hparams.label is not None:
            not_label = "Not{}".format(self.hparams.label.upper())
            training_set.rename(columns={"none": not_label}, inplace=True)
            evaluation_set.loc[:, not_label] = 0
            for index, row in evaluation_set.iterrows():
                if row.UD == 0:
                    evaluation_set.loc[index, not_label] = 1
            test_set.loc[:, not_label] = 0
            for index, row in test_set.iterrows():
                if row.UD == 0:
                    test_set.loc[index, not_label] = 1
            labels = [self.hparams.label.upper(), not_label]
        else:
            labels = get_labels(self.hparams.amount_labels)

        max_length = 512

        data_column = "summary_{}".format(self.hparams.summary_type)

        print("Computing Input")
        training_inputs = [self.tokenizer(text,
                                          max_length=max_length,
                                          padding="max_length",
                                          truncation=True)
                           for text in tqdm(training_set.loc[:, data_column].values,
                                            total=len(training_set))]

        training_input_ids = [training_input["input_ids"]
                              for training_input in training_inputs]
        training_attention_mask = [training_input["attention_mask"]
                                   for training_input in training_inputs]
        training_labels = training_set.loc[:, labels].values

        evaluation_inputs = [self.tokenizer(text,
                                            max_length=max_length,
                                            padding="max_length",
                                            truncation=True)
                             for text in tqdm(evaluation_set.loc[:, data_column].values,
                                              total=len(evaluation_set))]

        evaluation_input_ids = [evaluation_input["input_ids"]
                                for evaluation_input in evaluation_inputs]
        evaluation_attention_mask = [
            evaluation_input["attention_mask"] for evaluation_input in evaluation_inputs]
        evaluation_labels = evaluation_set.loc[:, labels].values

        test_inputs = [self.tokenizer(text,
                                      max_length=max_length,
                                      padding="max_length",
                                      truncation=True)
                       for text in tqdm(test_set.loc[:, data_column].values,
                                        total=len(test_set))]

        test_input_ids = [test_input["input_ids"]
                          for test_input in test_inputs]
        test_attention_mask = [
            test_input["attention_mask"] for test_input in test_inputs]
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

    def train_dataloader(self):
        train_sampler = RandomSampler(self.train_data)
        train_dataloader = DataLoader(self.train_data,
                                      sampler=train_sampler,
                                      batch_size=self.hparams.batch_size,
                                      num_workers=self.hparams.num_workers)
        return train_dataloader

    def val_dataloader(self):
        valid_sampler = SequentialSampler(self.valid_data)
        valid_dataloader = DataLoader(self.valid_data,
                                      sampler=valid_sampler,
                                      batch_size=self.hparams.batch_size,
                                      num_workers=self.hparams.num_workers)
        return valid_dataloader

    def test_dataloader(self):
        test_sampler = SequentialSampler(self.test_data)
        test_dataloader = DataLoader(self.test_data,
                                     sampler=test_sampler,
                                     batch_size=self.hparams.batch_size,
                                     num_workers=self.hparams.num_workers)
        return test_dataloader

    def configure_optimizers(self):
        optimizer = adam_discriminative_lr(model=self,
                                           learning_rate=self.hparams.learning_rate,
                                           weight_decay=self.hparams.weight_decay,
                                           lr_decay=self.hparams.lr_decay)
        return optimizer

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

        eval_result = EvalResult(checkpoint_on=loss)

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

        eval_result.log("eval_loss", loss)
        eval_result.log("eval_acc", acc)
        return eval_result

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

        print("\n--------------------------------------------------------------------------------")
        print("METRICS ON EVALSET")
        for metric, value in metrics.items():
            print("{}: {}".format(metric, value))
        print("--------------------------------------------------------------------------------")

        self.logger.log_metrics(metrics=metrics, step=self.global_step)
        self.eval_conf_matrix = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        outputs = self.forward(input_ids, attention_mask)
        loss = self.loss_function(outputs, labels, weight=1.0)

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
        result.write("predictions",
                     predictions,
                     "{}{}/{}/predictions.pt".format(self.logger.save_dir, self.logger.name, self.logger.version))
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
