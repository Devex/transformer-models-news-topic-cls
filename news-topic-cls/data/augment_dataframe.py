import math
from tqdm import tqdm
import numpy as np
import pandas as pd
from os import path


def generate_augmentations_for_row(row, categories, label_counter, augmentations_df):
    labels = row.loc[categories].values
    active_labels = label_counter[labels.astype(bool)]

    min_active = min(active_labels)
    max_overall = max(label_counter)

    augmented_df = pd.DataFrame(columns=["id", "text"] + categories)

    if max_overall in active_labels:
        return augmented_df

    if len(categories) == 2:
        number_augmentations = min(math.ceil(max_overall/min_active), 12)
    elif len(categories) == 6:
        number_augmentations = min(round(max_overall/min_active), 12)
    elif len(categories) == 21:
        number_augmentations = min(math.floor(max_overall/min_active), 12)

    augmented_df = sample_from_augmentations(
        id_=row.id,
        number_augmentations=number_augmentations,
        categories=categories,
        augmentations_df=augmentations_df)

    return augmented_df


def sample_from_augmentations(id_, number_augmentations, categories, augmentations_df):
    augmentations = augmentations_df[augmentations_df.reference_id == id_]

    sampled_indicees = np.random.choice(
        range(12), size=number_augmentations, replace=False)

    sampled_augmentations = augmentations.iloc[sampled_indicees, :]

    augmented_ids = [int(str(row.reference_id) + str(row.augmentation_id))
                     for _, row in sampled_augmentations.loc[:, ["reference_id",
                                                                 "augmentation_id"]].iterrows()]

    sampled_augmentations.drop(
        columns=["reference_id", "augmentation_id"], inplace=True)
    sampled_augmentations.loc[:, "id"] = augmented_ids

    sampled_augmentations.rename(columns={
        "augmented_article": "text",
        "augmented_summary_cluster": "summary_cluster",
        "augmented_summary_textrank": "summary_textrank"
    }, inplace=True)

    return sampled_augmentations


def label_none(row, categories):
    if sum(row.loc[categories].values) == 0:
        row.none = 1
    return row


def augment_dataframe(df, augmentations_df, categories):
    if "none" in df.columns:
        df.drop(columns=["none"], inplace=True)
    if "none" in augmentations_df.columns:
        augmentations_df.drop(columns=["none"], inplace=True)
    if len(categories) < 21:
        df["none"] = [0] * len(df)
        augmentations_df["none"] = [0] * len(augmentations_df)

        df = df.apply(func=lambda x: label_none(
            x, categories=categories), axis=1)
        augmentations_df = augmentations_df.apply(
            func=lambda x: label_none(x, categories=categories), axis=1)
        categories += ["none"]

    label_counter_original = df.loc[:, categories].sum().values
    counter_dict = {category: counter for category,
                    counter in zip(categories, label_counter_original)}

    sorted_labels = sorted(
        categories, key=lambda key: counter_dict[key], reverse=False)
    df_aug = df
    for category in tqdm(sorted_labels, total=len(sorted_labels), position=0, leave=True):
        if len(sorted_labels) == 2 and category == "none":
            continue
        active_df = df[df.loc[:, category] == 1]
        for index, row in active_df.iterrows():
            if int(str(row.id) + "0") not in df_aug.id.values:
                label_counter = df_aug.loc[:, categories].sum().values
                augmented_df = generate_augmentations_for_row(
                    row, categories, label_counter, augmentations_df)
                df_aug = pd.concat([df_aug, augmented_df])

    df_aug.loc[:, categories] = df_aug.loc[:, categories].astype(int)
    return df_aug
