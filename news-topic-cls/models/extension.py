from torch.nn import Module, Linear, ReLU, Tanh, Sigmoid, Softmax, LSTM, TransformerEncoderLayer, TransformerEncoder, Conv2d, MaxPool2d
from math import floor


class RNNExtension(Module):
    def __init__(self, input_size=768, hidden_size=768):
        super(RNNExtension, self).__init__()
        self.name = "rnn"
        self.lstm = LSTM(input_size=input_size,
                         hidden_size=hidden_size,
                         num_layers=1,
                         batch_first=True,
                         bias=True)

        self.output_size = hidden_size

    def forward(self, cls_embeddings):
        out, _ = self.lstm(cls_embeddings)
        return out[-1]


class ClassifierExtension(Module):
    def __init__(self, input_size, hidden_size=30, num_labels=21):
        super(ClassifierExtension, self).__init__()
        self.name = "cls"
        self.linear1 = Linear(input_size, hidden_size)
        self.hidden_activation = Tanh()
        self.linear2 = Linear(hidden_size, num_labels)

        if num_labels == 1:
            self.final_activation = Softmax()
        else:
            self.final_activation = Sigmoid()

    def forward(self, document_embedding):
        output = self.hidden_activation(self.linear1(document_embedding))
        logits = self.final_activation(self.linear2(output))
        return logits
