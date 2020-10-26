import pandas as pd
import logging
import sys
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skmultilearn.problem_transform import BinaryRelevance
from math import sqrt
import numpy as np
from argparse import ArgumentParser
import torch

from utils.utils import get_labels


def scoring(targets, predictions, verbose=True):
    acc = accuracy_score(targets, predictions)
    if verbose:
        print("Accuracy: {}".format(acc))

    targets = torch.tensor(targets)
    predictions = torch.tensor(predictions)
    tp = float(torch.logical_and(predictions == 1, targets == 1).sum())
    fp = float(torch.logical_and(predictions == 1, targets == 0).sum())
    tn = float(torch.logical_and(predictions == 0, targets == 0).sum())
    fn = float(torch.logical_and(predictions == 0, targets == 1).sum())

    recall = tp/(tp + fn)
    precision = tp/(tp + fp)
    mcc = (tp*tn-fp*fn)/sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    f1 = 2*recall*precision/(recall + precision)

    if verbose:
        print("F1-Score: {}".format(f1))
        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("MCC: {}".format(mcc))
    metrics = {
        "acc": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "mcc": mcc
    }
    return metrics


def main(hparams):
    try:
        logger.handlers.clear()
    except:
        pass

    logger_name = "SVM - Cross Validation"
    logger = logging.getLogger(logger_name)
    logger.setLevel("INFO")

    file_handler = logging.StreamHandler(sys.stdout)
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    metrics = {
        "eval": {
            "acc": [],
            "f1": [],
            "precision": [],
            "recall": [],
            "mcc": []
        },
        "test": {
            "acc": [],
            "f1": [],
            "precision": [],
            "recall": [],
            "mcc": []
        }
    }

    logger.info("Hyperparameter:")

    print(json.dumps(vars(hparams), indent=4))

    df_test = pd.read_csv(
        "{}/gold-standard-testset.csv".format(hparams.data_path))

    if hparams.amount_labels == 1:
        label_cols = ["GH"]
    else:
        label_cols = get_labels(hparams.amount_labels)

    for seed in range(5):
        logger.info(
            "Starting Classification for training split {}".format(seed))

        logger.info(
            "Loading Text and Labels - use augmentations: {}".format(hparams.augment))

        data_path_split = "{}/{}/".format(hparams.data_path, seed)
        if hparams.augment:
            df_train = pd.read_csv("{}df_train_{}_augmented_{}labels.csv".format(data_path_split,
                                                                                 seed,
                                                                                 hparams.amount_labels))
        else:
            assert hparams.amount_labels == 21
            df_train = pd.read_csv(
                "{}df_train_{}.csv".format(data_path_split, seed))

        df_eval = pd.read_csv("{}df_eval_{}.csv".format(data_path_split, seed))

        text_train = df_train.text.values
        text_eval = df_eval.text.values
        text_test = df_test.text.values
        labels_train = df_train.loc[:, label_cols].values
        labels_eval = df_eval.loc[:, label_cols].values
        labels_test = df_test.loc[:, label_cols].values

        logger.info("Computing Features")
        vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3))
        features_train = vectorizer.fit_transform(text_train)
        features_eval = vectorizer.transform(text_eval)
        features_test = vectorizer.transform(text_test)

        logger.info("Starting Training")
        strategy = BinaryRelevance
        model = SVC(verbose=True, class_weight="balanced")
        classifier = strategy(model)
        classifier.fit(features_train, labels_train)
        print("\n")
        logger.info("Starting Prediction for Evaluation and Testset")
        predictions_eval = classifier.predict(features_eval).todense()
        predictions_test = classifier.predict(features_test).todense()

        logger.info("Metrics on Evaluationset")
        metrics_eval = scoring(labels_eval, predictions_eval)

        logger.info("Metrics on Testset")
        metrics_test = scoring(labels_test, predictions_test)

        for metric, value in metrics_eval.items():
            metrics["eval"][metric].append(value)
        for metric, value in metrics_test.items():
            metrics["test"][metric].append(value)

        print("--------------------------------------------------------------------------------")

    logger.info("Cross Validation complete")
    logger.info("Averaged Metrics on Evaluationset")

    metrics_eval = metrics["eval"]
    for metric, value in metrics_eval.items():
        logger.info("{}: {}".format(metric, np.array(value).mean()))

    logger.info("Averaged Metrics on Testset")

    metrics_test = metrics["test"]
    for metric, value in metrics_test.items():
        logger.info("{}: {}".format(metric, np.array(value).mean()))

    logger.handlers.clear()


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--data_path", default="../../data/train-eval-split")
    parser.add_argument("--amount_labels", type=int, default=21)
    parser.add_argument("--augment", action="store_true")

    main(parser.parse_args())
