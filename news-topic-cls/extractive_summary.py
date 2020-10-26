from models.base import TransformerOptions
from data.data import load_data, clean_data
from summarization.extraction import summarize, textrank_summary

import pandas as pd
import traceback
import json
from os import path
import sys
import logging
from tqdm import tqdm
from argparse import ArgumentParser
import spacy
from pytextrank import TextRank


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")
    tr = TextRank()
    nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)

    parser = ArgumentParser()
    parser.add_argument(
        "--data_path", default="../../data/train-eval-split/gold-standard-testset.csv")
    parser.add_argument("--extend", action="store_true")
    parser.add_argument(
        "--lm_path", default="../../data/models/language_models/roberta-lm")

    params = parser.parse_args()

    print("Parameter: \n{}".format(json.dumps(vars(params), indent=3)))

    transformer = TransformerOptions(
        "roberta-base", lm_path=params.lm_path)

    logging.getLogger(
        "transformers.tokenization_utils_base").setLevel(logging.ERROR)
    for param in transformer.model.parameters():
        param.requires_grad = False

    name, file_ending = params.data_path.split(".csv")

    if params.lm_path is not None:
        folders = params.lm_path.split("/")
        summary_path = "{}_{}_summary.csv".format(name, folders[-1])

    else:
        summary_path = name + "_summary.csv"

    if params.extend:
        try:
            summary_path = path.abspath(summary_path)
            df = pd.read_csv(summary_path)
        except:
            print("Could not load CSV from {}".format(summary_path))
            sys.exit(1)
        df = clean_data(df)
    else:
        data_path = path.abspath(params.data_path)
        df = load_data(data_path)
        df["summary_cluster"] = [""]*len(df)
        df["summary_textrank"] = [""]*len(df)

    print("Starting Summarization of Articles")
    print("Summary File Location: {}".format(summary_path))
    for index, row in tqdm(df.iterrows(), total=len(df)):
        if row.summary_cluster == "" or row.summary_textrank == "":
            article = row.text

            if len(transformer.tokenizer(article)["input_ids"]) <= 510:
                summary_cluster = article
                summary_textrank = article
            else:
                summary_cluster = summarize(article=article,
                                            cluster_alg="hdbscan",
                                            transformer=transformer)
                summary_textrank = textrank_summary(article=article,
                                                    processor=nlp)
            row.summary_cluster = summary_cluster
            row.summary_textrank = summary_textrank
            df.iloc[index, :] = row
            df.to_csv(summary_path, index=False)
