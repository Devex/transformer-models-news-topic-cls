from models.base import TransformerOptions
from data.data import load_data

from multiprocessing import cpu_count
from tqdm import tqdm
from sklearn.cluster import KMeans
import hdbscan
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch
import pandas as pd
import re
import nltk
nltk.download("punkt")


def compute_embeddings(tokenized_sentences: list, language_model):
    language_model.eval()
    for step, tokens in enumerate(tokenized_sentences):
        outputs = language_model(tokens)
        # set_trace()
        embedding = outputs[0][:, 0, :]
        if step == 0:
            embeddings = embedding.numpy()
        else:
            embeddings = np.concatenate((embeddings, embedding), axis=0)
    return embeddings


def get_sentences_for_summary(summary_df, cluster_center):
    summary_sentences = [0]*len(cluster_center)

    for index, center in enumerate(cluster_center):
        is_cluster = summary_df.cluster == index
        cluster_df = summary_df[is_cluster]
        min_distance = np.Inf
        for _, row in cluster_df.iterrows():
            distance = np.linalg.norm(center-row.embedding)

            if distance < min_distance:
                min_distance = distance
                summary_sentences[index] = (row.position, row.sentence)
    return sorted(summary_sentences, key=lambda x: x[0])


def get_row_of_point(summary_df, point):
    for _, row in summary_df.iterrows():
        if np.all(row.embedding == point):
            return row


def get_sentences_for_summary_by_exemplar(summary_df, exemplars_):
    exemplars = exemplars_.copy()
    number_clusters = len(exemplars)
    summary_sentences = []

    while len(summary_sentences) < 10:
        summary_sentences_before_loop = summary_sentences.copy()
        for index, exemplar in enumerate(exemplars):
            if len(exemplar) == 0:
                continue
            point = exemplar[0]

            row = get_row_of_point(summary_df, point)
            # set_trace()
            summary_sentences.append((row.position, row.sentence))

            exemplar = exemplar[1:]
            exemplars[index] = exemplar

            if len(summary_sentences) == 10:
                break
        if summary_sentences == summary_sentences_before_loop:
            break

    index = 0
    while len(summary_sentences) < 10:
        summary_sentences_before_loop = summary_sentences.copy()
        for cluster in range(number_clusters):
            cluster_points = summary_df[summary_df.cluster == cluster]
            try:
                row = cluster_points.iloc[index, :]
                summary_sentences.append((row.position, row.sentence))
            except IndexError:
                continue
            if len(summary_sentences) == 10:
                break
        if summary_sentences == summary_sentences_before_loop:
            break
        else:
            index += 1

    # set_trace()
    if len(summary_sentences) < 10:
        outlier_df = summary_df[summary_df.cluster == -1]
        for _, row in outlier_df.iterrows():
            summary_sentences.append((row.position, row.sentence))
            if len(summary_sentences) == 10:
                return sorted(summary_sentences, key=lambda x: x[0])
    else:
        return sorted(summary_sentences, key=lambda x: x[0])


def find_quotes(sentence):
    condition = re.compile(".*?”")
    matches = condition.finditer(sentence)
    quotes = [match.group().strip() for match in matches]
    if quotes == []:
        return [sentence]
    last_match = quotes[-1]
    lower_bound = sum([len(quote) for quote in quotes]) - len(last_match)
    last_match_end = sentence.find(last_match, lower_bound) + len(last_match)
    after_quotes = sentence[last_match_end:].strip()
    if after_quotes != "":
        quotes.append(after_quotes)
    return quotes


def sentenize(article):
    sentences = nltk.tokenize.sent_tokenize(article)
    for index, sentence in enumerate(sentences):
        quotes = find_quotes(sentence)
        sentences[index] = quotes

    def flatten(list_): return [item for sublist in list_ for item in sublist]
    return flatten(sentences)


def sentenize_without_quotes(article):
    article_cleaned = article.replace("“", "")
    article_cleaned = article_cleaned.replace("”", "")

    return nltk.tokenize.sent_tokenize(article_cleaned)


def postprocess_sentences(sentences):
    for index, sentence in enumerate(sentences):
        if sentence.count(",") > 15:
            tokens = nltk.word_tokenize(sentence)
            word_counter = len(tokens)
            if word_counter >= 200:
                sentences[index] = "List of upcoming events."

    return sentences


def cluster_embeddings_kmeans(sentences, sentence_embeddings, n_cluster):
    clusterer = KMeans(n_clusters=n_cluster)
    clusterer.fit(sentence_embeddings)

    summary_df = pd.DataFrame(data={
        "position": range(len(sentences)),
        "sentence": sentences,
        "embedding": sentence_embeddings.tolist(),
        "cluster": clusterer.labels_
    })
    return summary_df, clusterer.cluster_centers_


def cluster_embeddings_hdbscan(sentences, sentence_embeddings):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=None)
    clusterer.fit(sentence_embeddings)

    summary_df = pd.DataFrame(data={
        "position": range(len(sentences)),
        "sentence": sentences,
        "embedding": sentence_embeddings.tolist(),
        "cluster": clusterer.labels_
    })

    return summary_df, clusterer.exemplars_


def summarize(article: str, transformer, cluster_alg: str = "hdbscan", n_cluster=None):
    sentences = sentenize_without_quotes(article)
    if len(sentences) <= 10:
        return article

    tokenized_sentences = [transformer.tokenizer(text=sentence,
                                                 truncation=True,
                                                 return_tensors="pt")["input_ids"]
                           for sentence in sentences]

    with torch.no_grad():
        sentence_embeddings = compute_embeddings(
            tokenized_sentences, transformer.model)

    if cluster_alg == "kmeans":
        assert n_cluster is not None
        summary_df, cluster_centers = cluster_embeddings_kmeans(sentences,
                                                                sentence_embeddings,
                                                                n_cluster=10)
        summary_sentences = get_sentences_for_summary(
            summary_df, cluster_centers)

    elif cluster_alg == "hdbscan":
        summary_df, cluster_exemplars = cluster_embeddings_hdbscan(sentences,
                                                                   sentence_embeddings)
        summary_sentences = get_sentences_for_summary_by_exemplar(
            summary_df, cluster_exemplars)

    return " ".join([sentence for _, sentence in summary_sentences])


def textrank_summary(article, processor):
    summ = processor(article)
    spacy_spans = summ._.textrank.summary(
        limit_phrases=None, limit_sentences=10)
    summary_sentences = [spacy_span.text for spacy_span in spacy_spans]
    return " ".join(summary_sentences)
