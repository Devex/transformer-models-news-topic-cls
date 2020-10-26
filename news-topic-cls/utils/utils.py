import logging
from sklearn.metrics import f1_score, recall_score, precision_score
import torch
from transformers import \
    DistilBertTokenizer, DistilBertConfig, DistilBertModel, \
    RobertaTokenizer, RobertaConfig, RobertaModel, \
    BertTokenizer, BertConfig, BertModel, \
    AlbertTokenizer, AlbertConfig, AlbertModel, \
    GPT2Tokenizer, GPT2Config, GPT2Model, \
    XLNetTokenizer, XLNetConfig, XLNetModel, \
    LongformerTokenizer, LongformerConfig, LongformerModel


class TransformerOptions:
    options = {
        "albert-xxlarge-v2": (AlbertTokenizer, AlbertConfig, AlbertModel),
        "albert-base-v2": (AlbertTokenizer, AlbertConfig, AlbertModel),
        "albert-base-v1": (AlbertTokenizer, AlbertConfig, AlbertModel),
        "bert-base-uncased": (BertTokenizer, BertConfig, BertModel),
        "bert-large-uncased": (BertTokenizer, BertConfig, BertModel),
        "xlnet-base-cased": (XLNetTokenizer, XLNetConfig, XLNetModel),
        "gpt2": (GPT2Tokenizer, GPT2Config, GPT2Model),
        "distilbert-base-uncased": (DistilBertTokenizer, DistilBertConfig, DistilBertModel),
        "roberta-base": (RobertaTokenizer, RobertaConfig, RobertaModel),
        "allenai/longformer-base-4096": (LongformerTokenizer, LongformerConfig, LongformerModel)
    }

    def __init__(self, config_name, output_hidden_states=True, output_past=False,
                 lm_path=None, max_length=None, num_labels=21):

        self.tokenizer = self.options[config_name][0].from_pretrained(
            config_name)

        self.config = self.options[config_name][1].from_pretrained(config_name,
                                                                   output_hidden_states=output_hidden_states,
                                                                   output_past=output_past,
                                                                   num_labels=num_labels)
        if lm_path is None:
            self.model = self.options[config_name][2](self.config)
        else:
            self.model = self.options[config_name][2].from_pretrained(lm_path,
                                                                      config=self.config)


def get_number_splits(batch):
    return batch[0].shape[1]


def get_labels(number):
    label_options = {
        "1": ["GH", "NotGH"],
        "5": ["FU", "GH", "TP", "SD", "ID"],
        "6": ["FU", "GH", "TP", "SD", "ID", "none"],
        "21": ['AG', 'BF', 'CE', 'ED', 'EN', 'FU', 'GH', 'HA', 'HR', 'ID', 'II', 'IN', 'MC', 'NR', 'PM', 'PS', 'RE', 'SD', 'TP', 'UD', 'WS']
    }

    return label_options[str(number)]


class IncreasedLoggingLevel():
    def __init__(self, logger_name, target_level=logging.ERROR):
        self.logger = logging.getLogger(logger_name)
        self.target_level = logging.ERROR
        self.original_level = self.logger.getEffectiveLevel()

    def __enter__(self):
        self.logger.setLevel(self.target_level)

    def __exit__(self, type, value, traceback):
        self.logger.setLevel(self.original_level)
