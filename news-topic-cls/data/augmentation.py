from googletrans import Translator
import math
import nltk
import numpy as np
import pandas as pd
import itertools
import traceback
import sys
from httpx._config import Timeout


def split_text(text, sentences_per_part=10):
    sentences = nltk.tokenize.sent_tokenize(text)
    parts = []
    for index, sentence in enumerate(sentences):
        if index % sentences_per_part == 0:
            new_part = " ".join(sentences[index:index+sentences_per_part])
            assert len(new_part) < 15000
            parts.append(new_part)
    return parts


def _backtranslation(text_parts, translator, language):
    if len(" ".join(text_parts)) > 20000:
        first_parts = text_parts[:math.ceil(len(text_parts)/2)]
        second_parts = text_parts[math.ceil(len(text_parts)/2):]

        first_parts_backtrans = _backtranslation(
            first_parts, translator, language)
        second_parts_backtrans = _backtranslation(
            second_parts, translator, language)
        return first_parts_backtrans + second_parts_backtrans

    parts_trans = translator.translate(
        text=text_parts, dest=language, src="en")
    parts_trans = [
        part_trans.text for part_trans in parts_trans]
    parts_backtrans = translator.translate(
        text=parts_trans, dest="en", src=language)
    parts_backtrans = [
        part_backtrans.text for part_backtrans in parts_backtrans]
    return parts_backtrans


def backtranslation(text, translator, language):
    start_counter = {
        "de": 50,
        "fi": 50,
        "tr": 50,
        "ar": 20,
        "zh-cn": 20,
        "hi": 10
    }

    sentence_counter = start_counter[language]
    text_parts = split_text(text, sentences_per_part=sentence_counter)

    parts_backtrans = _backtranslation(text_parts=text_parts,
                                       translator=translator,
                                       language=language)

    is_lang_correct = evaluate_backtranslation(parts_backtrans, translator)
    while sum(is_lang_correct) != len(is_lang_correct):
        incorrect_parts = [text_part for (text_part, lang_correct) in
                           zip(text_parts, is_lang_correct) if not lang_correct]
        incorrect_indices = np.where(~np.array(is_lang_correct))[
            0].astype(list)
        sentence_counter = math.floor(sentence_counter/2)
        for index, incorrect_part in enumerate(incorrect_parts):
            new_text_parts = split_text(incorrect_part,
                                        sentences_per_part=sentence_counter)
            new_backtrans = _backtranslation(text_parts=new_text_parts,
                                             translator=translator,
                                             language=language)
            parts_backtrans[incorrect_indices[index]] = new_backtrans

        parts_backtrans = flatten_backtrans(parts_backtrans)
        is_lang_correct = evaluate_backtranslation(parts_backtrans, translator)

    return " ".join(parts_backtrans)


def flatten_backtrans(backtrans):
    flattened_backtrans = []
    for el in backtrans:
        if isinstance(el, str):
            flattened_backtrans.append(el)
        elif isinstance(el, list):
            for el2 in el:
                flattened_backtrans.append(el2)
        else:
            raise TypeError("Unrecognized Type: {}".format(type(el)))

    return flattened_backtrans


def evaluate_backtranslation(backtrans, translator):
    if isinstance(backtrans, list):
        is_trans_correct = []
        for trans in backtrans:
            characters_for_evaluation = 100
            # set_trace()
            while translator.detect(trans[:characters_for_evaluation]).confidence < 0.95:
                characters_for_evaluation *= 2
            is_trans_correct.append(translator.detect(
                trans[:characters_for_evaluation]).lang == "en")
        return is_trans_correct
    elif isinstance(backtrans, str):
        characters_for_evaluation = 50
        while translator.detect(backtrans[:characters_for_evaluation]).confidence < 0.95:
            characters_for_evaluation *= 2
        return translator.detect(backtrans[:characters_for_evaluation]).lang == "en"
    else:
        raise TypeError("Unrecognized Type: {}".format(backtrans))


def augment_text(text, augmentors=[], augmentations_per_augmenter=None, languages=[]):
    augmented_texts = []
    for augmentor in augmentors:
        augmented_text = augmentor.augment(text, n=augmentations_per_augmenter)
        if augmentations_per_augmenter == 1:
            augmented_texts.append(augmented_text)
        else:
            augmented_texts += augmented_text

    translator = Translator(timeout=Timeout(10.0))
    for language in languages:
        try:
            backtranslated_text = backtranslation(
                text=text, translator=translator, language=language)
            augmented_texts.append(backtranslated_text)
        except Exception as ex:
            print("Exception: {}".format(ex))
            augmented_texts.append(
                "MISSING {} ".format(language))

    return augmented_texts


def generate_augmentations(row, categories, db_augmentors, languages):
    labels = row.loc[categories].values

    number_augmentations = 12

    augmented_article = augment_text(text=row.text,
                                     augmentors=db_augmentors,
                                     augmentations_per_augmenter=3,
                                     languages=languages)
    augmented_summary_cluster = augment_text(text=row.summary_cluster,
                                             augmentors=db_augmentors,
                                             augmentations_per_augmenter=3,
                                             languages=languages)
    augmented_summary_textrank = augment_text(text=row.summary_textrank,
                                              augmentors=db_augmentors,
                                              augmentations_per_augmenter=3,
                                              languages=languages)

    augmented_labels = np.array([list(labels), ] * number_augmentations)

    augmented_df = pd.DataFrame(
        columns=["augmentation_id",
                 "reference_id",
                 "augmented_article",
                 "augmented_summary_cluster",
                 "augmented_summary_textrank"] + categories)
    augmented_df["augmentation_id"] = range(number_augmentations)
    augmented_df["reference_id"] = [row.id] * number_augmentations
    augmented_df["augmented_article"] = augmented_article
    augmented_df["augmented_summary_cluster"] = augmented_summary_cluster
    augmented_df["augmented_summary_textrank"] = augmented_summary_textrank
    augmented_df[categories] = augmented_labels

    return augmented_df
