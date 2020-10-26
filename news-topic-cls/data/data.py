import numpy as np
from math import ceil, floor
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from tqdm import tqdm


def clean_data(dataframe):
    id_list = [62094]
    cleaned_df = dataframe[~dataframe.id.isin(id_list)]
    cleaned_df.replace("<.*>", "", regex=True, inplace=True)
    return cleaned_df


def load_data(data_path, return_abbrev=False):
    if "test" in data_path and "gold-standard" not in data_path:
        df = pd.read_csv(data_path).drop(labels="Unnamed: 0", axis=1)
    else:
        df = pd.read_csv(data_path)

    df = clean_data(df)
    df, label_encoder = encode_labels(df)

    abbrev_mapping = get_abbreveation_mapping(label_encoder)
    df = binarize_labels(df, abbrev_mapping)

    if return_abbrev:
        return df, abbrev_mapping
    else:
        return df


def encode_labels(df):
    categories = df.categories.str.replace(pat=":", repl="")
    labels = set()
    for category in categories:
        category_split = category.split("/")[1:]
        labels.update(set(category_split))
    label_encoder = LabelEncoder()
    label_encoder.fit(list(labels))

    def get_label_ids(row, encoder):
        labels = row.categories.replace(":", "").split("/")[1:]
        label_ids = encoder.transform(labels)
        row.category_ids = label_ids
        return row

    df["category_ids"] = ""
    df = df.apply(lambda x: get_label_ids(x, encoder=label_encoder), axis=1)

    return df.drop(labels=["type", "categories"], axis=1), label_encoder


def binarize_labels(df, abbrev_mapping):
    assert "category_ids" in df.columns
    ids = [tuple(id_) for id_ in df.category_ids.values]

    binarizer = MultiLabelBinarizer()
    ids_binarized = binarizer.fit_transform(ids)

    for index, key in enumerate(abbrev_mapping.keys()):
        category_binarized = ids_binarized[:, index]
        df[key] = category_binarized

    return df.drop(labels=["category_ids"], axis=1)


def get_abbreveation_mapping(label_encoder):
    categories = [label_encoder.inverse_transform(
        [number])[0] for number in range(21)]
    abbrev_mapping = {category[1:3]: category[5:] for category in categories}
    return abbrev_mapping


def get_max_length(df, tokenizer):
    max_length = max([len(tokenizer.tokenize(sentence))
                      for sentence in df.text.values])
    return max_length


def split_articles(inputs, max_length, split_length=200, shift=50):
    splitted_input_ids, splitted_attention_mask = [], []
    for input_ in inputs:
        splitted_input_ids.append([input_["input_ids"][i:i+split_length] for
                                   i in range(0, max_length, split_length-shift)][:-1])

        splitted_attention_mask.append([input_["attention_mask"][i:i+split_length] for
                                        i in range(0, max_length, split_length-shift)][:-1])

    return splitted_input_ids, splitted_attention_mask
