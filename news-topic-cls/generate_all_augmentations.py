import nlpaug.augmenter.word as naw
from sklearn.model_selection import train_test_split
import pandas as pd
import json
from tqdm import tqdm
from argparse import ArgumentParser

from data.data import load_data
from data.augmentation import generate_augmentations

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--data_path", default="../../data/test_data_summary.csv")
    parser.add_argument("--extend", action="store_true")
    params = parser.parse_args()
    print("Parameter: \n{}".format(json.dumps(vars(params), indent=3)))

    summary_df = pd.read_csv(params.data_path)

    categories = list(summary_df.drop(
        columns=["id", "text", "summary_cluster", "summary_textrank"]).columns)
    name, file_ending = params.data_path.split(".csv")
    augmentations_path = name + "_augmentations.csv"

    if params.extend:
        augmentations_df = pd.read_csv(augmentations_path)
    else:
        augmentations_df = pd.DataFrame(
            columns=["augmentation_id",
                     "reference_id",
                     "augmented_article",
                     "augmented_summary_cluster",
                     "augmented_summary_textrank"] + categories)

    print("Loaded Data successfully.")
    aug_syn_ppdb = naw.SynonymAug(aug_src="ppdb",
                                  model_path="../../notebooks/aug_models/ppdb-2.0-s-all",
                                  aug_max=None)

    aug_syn_wordnet = naw.SynonymAug(aug_src="wordnet",
                                     aug_max=None)

    languages = ["zh-cn", "de", "fi", "hi", "ar", "tr"]
    print("Loaded Augmentors successfully.")
    for index, row in tqdm(summary_df.iterrows(), total=len(summary_df), position=0, leave=True):
        if row.id not in augmentations_df.reference_id.values:
            augmented_df = generate_augmentations(row,
                                                  categories,
                                                  db_augmentors=[
                                                      aug_syn_wordnet, aug_syn_ppdb],
                                                  languages=["zh-cn", "de", "fi", "hi", "ar", "tr"])

            augmentations_df = pd.concat([augmentations_df, augmented_df])

            augmentations_df.to_csv(augmentations_path, index=False)
