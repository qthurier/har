import tensorflow as tf
from keras.layers import (GRU, Conv1D, Dense, GlobalMaxPooling1D,
                          LayerNormalization, Softmax, Layer, Concatenate, Bidirectional)
from keras.models import Sequential
from typing import Any

# norm_inputs = Concatenate()([block1, block2, block3]) smooth decrease
# 0.0978
# 0.0717


class ExtractFeatRange(Layer):

    def __init__(self, start: int, end: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.start = start
        self.end = end

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return inputs[:,:,self.start:self.end]

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({"start": self.start, "end": self.end})
        return config


class TimeWiseNormalisation(Layer):

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        mu = tf.math.reduce_mean(inputs, axis=-2, keepdims=True)
        sigma = tf.math.reduce_std(inputs, axis=-2, keepdims=True)
        return (inputs - mu / sigma)


class TimeAndFeatWiseNormalisation(Layer):

    def __init__(self, n_feat: int, feat_block_size: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.n_feat = n_feat
        self.feat_block_size = feat_block_size
        self.blocks = [Sequential([ExtractFeatRange(i, i+self.feat_block_size), LayerNormalization()], name=f"block{1}") for i in range(0, self.n_feat, self.feat_block_size)]

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        timewise_norm_inputs = TimeWiseNormalisation()(inputs)
        featwise_norm_inputs_blocks = [block(inputs) for block in self.blocks]
        norm_inputs = Concatenate()([timewise_norm_inputs] + featwise_norm_inputs_blocks)
        return norm_inputs

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({"n_feat": self.n_feat, "feat_block_size": self.feat_block_size})
        return config


class Classifier(tf.keras.Model):
    def __init__(self, n_feat, feat_block_size, encoder_dim, dense_dim, small_kernel_size, large_kernel_size, n_classes, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.normalise = TimeAndFeatWiseNormalisation(n_feat, feat_block_size)
        self.local_cnns = [Sequential(
            [
                Conv1D(encoder_dim, small_kernel_size, padding="same", activation="relu"),
                LayerNormalization()
            ], name=f"local_cnn{i}"
        ) for i in range(2)]
        self.global_cnns = [Sequential(
            [
                Conv1D(encoder_dim, large_kernel_size, padding="same", activation="relu"),
                LayerNormalization()
            ], name=f"global_cnn{i}"
        ) for i in range(2)]
        self.rnns = [Sequential([Bidirectional(GRU(encoder_dim, return_sequences=True, activation="relu"), merge_mode='sum'), LayerNormalization()], name=f"rnn{i}") for i in range(2)]
        self.denses = [Dense(dense_dim, activation='relu')]*3
        self.pool = GlobalMaxPooling1D()
        self.logits = Dense(n_classes)
        self.scores = Softmax()

    def call(self, inputs) -> tf.Tensor:
        norm_inputs = self.normalise(inputs)
        local_cnn_representation = Concatenate()([self.local_cnns[0](norm_inputs), self.local_cnns[1](inputs)])
        global_cnn_representation = Concatenate()([self.global_cnns[0](norm_inputs), self.global_cnns[1](inputs)])
        rnn_representation = Concatenate()([self.rnns[0](norm_inputs), self.rnns[1](inputs)])
        mixed_representation = Concatenate()([self.denses[0](local_cnn_representation), self.denses[1](global_cnn_representation), self.denses[2](rnn_representation)])
        flat_representation = self.pool(mixed_representation)
        logits = self.logits(flat_representation)
        return self.scores(logits)
