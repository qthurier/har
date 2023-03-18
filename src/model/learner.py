import tensorflow as tf
from keras.layers import (GRU, Conv1D, Dense, GlobalMaxPooling1D,
                          LayerNormalization, Softmax)
from keras.models import Sequential


class Classifier(tf.keras.Model):
    def __init__(self, feat_dim, kernel_size, n_classes=6):
        super().__init__()
        self.normalise = LayerNormalization()
        self.cnn = Sequential(
            [
                Conv1D(feat_dim, kernel_size, padding="same", activation="relu"),
                GlobalMaxPooling1D(),
            ]
        )
        self.rnn = GRU(feat_dim, activation="relu")
        self.logits = Dense(n_classes)
        self.scores = Softmax()

    def call(self, inputs):
        norm_inputs = self.normalise(inputs)
        cnn_repr = self.cnn(norm_inputs)
        rnn_repr = self.rnn(norm_inputs)
        representation = cnn_repr + rnn_repr
        logits = self.logits(representation)
        return self.scores(logits)
