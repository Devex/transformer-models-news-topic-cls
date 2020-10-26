from data.augment_dataframe import augment_dataframe

import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    augmentations_df = pd.read_csv(
        "../../data/article_confirmed_summary_augmentations.csv")
    data_path = "../../data/train-eval-split"

    for seed in tqdm(range(5)):
        csv_path = "{}/{}/df_train_{}.csv".format(
            data_path, str(seed), str(seed))
        train_df = pd.read_csv(csv_path)

        try:
            categories = list(train_df.drop(
                columns=["id", "text", "summary_cluster", "summary_textrank", "none"]).columns)
        except KeyError:
            categories = list(train_df.drop(
                columns=["id", "text", "summary_cluster", "summary_textrank"]).columns)

        number_labels = len(categories)
        augmented_df = augment_dataframe(df=train_df,
                                         augmentations_df=augmentations_df,
                                         categories=categories)
        augmented_df_path = "{}/{}/df_train_{}_augmented_{}labels.csv".format(
            data_path, str(seed), str(seed), number_labels)
        augmented_df.to_csv(augmented_df_path, index=False)

        categories = ["GH", "FU", "TP", "ID", "SD"]
        number_labels = len(categories)
        augmented_df = augment_dataframe(df=train_df,
                                         augmentations_df=augmentations_df,
                                         categories=categories)

        augmented_df = augmented_df.loc[:, [
            "id", "text", "summary_cluster", "summary_textrank"] + categories]
        augmented_df.drop(columns=["none"], inplace=True)
        augmented_df_path = "{}/{}/df_train_{}_augmented_{}labels.csv".format(
            data_path, str(seed), str(seed), number_labels)
        augmented_df.to_csv(augmented_df_path, index=False)

        categories = ["GH"]
        number_labels = len(categories)
        augmented_df = augment_dataframe(df=train_df,
                                         augmentations_df=augmentations_df,
                                         categories=categories)
        augmented_df = augmented_df.loc[:, [
            "id", "text", "summary_cluster", "summary_textrank"] + categories]
        augmented_df.drop(columns=["none"], inplace=True)
        augmented_df_path = "{}/{}/df_train_{}_augmented_{}labels.csv".format(
            data_path, str(seed), str(seed), number_labels)
        augmented_df.to_csv(augmented_df_path, index=False)
