#!/usr/bin/env python
# _*_coding:utf-8_*_

"""
@Project: ctrip_tkt_deep_model
@File   : modules.py
@Author : sichenghe(sichenghe@trip.com)
@Time   : 2022/7/28 
"""


import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer, Dropout, Input, BatchNormalization
from tensorflow.keras.regularizers import l2


class Linear(Layer):
    def __init__(self, sparse_feature_length, w_reg=1e-6):
        """
        Linear Part
        :param feature_length: A scalar. The length of features.
        :param w_reg: A scalar. The regularization coefficient of parameter w.
        """
        super(Linear, self).__init__()
        self.sparse_feature_length = sparse_feature_length
        self.w_reg = w_reg
        self.dense_layer = Dense(1, activation=None, kernel_regularizer=l2(self.w_reg))  # 连续特征直接过Dense

    def build(self, input_shape):
        # w 的维度是self.sparse_feature_length x 1
        # 之后要使用embedding_lookup
        if self.sparse_feature_length != 0:
            self.w = self.add_weight(name="w",
                                     shape=(self.sparse_feature_length, 1),
                                     regularizer=l2(self.w_reg),
                                     trainable=True)

    def call(self, inputs: "[dense_input, sparse_input]", **kwargs):
        dense_input, sparse_input = inputs
        if dense_input is not None:
            dense_output = self.dense_layer(dense_input)  # batch_size, 1
        else:
            dense_output = None
        # (batch_size, len, 1) 沿axis=1 sum, 直接变成 (batch_size, 1)
        if sparse_input is not None:
            sparse_output = tf.reduce_sum(tf.nn.embedding_lookup(self.w, sparse_input), axis=1)  # (batch_size, 1)
        else:
            sparse_output = None

        if dense_output is not None and sparse_output is not None:
            output = dense_output + sparse_output
        elif dense_output is not None:
            output = dense_output
        else:
            output = sparse_output
        return output  # (batch_size, 1)

class DNN(Layer):
    def __init__(self, hidden_units, activation='relu', dropout=0., w_reg=1e-6):

        super(DNN, self).__init__()
        self.dnn_network = [Dense(units=unit, activation=activation, kernel_regularizer=l2(w_reg)) for unit in hidden_units]
        self.dropout = Dropout(dropout)

    def call(self, inputs, **kwargs):
        x = inputs
        for dnn in self.dnn_network:
            x = dnn(x)
            x = self.dropout(x)
        return x

def sparseFeature(feat, feat_num, embed_dim=4):
    """
    create dictionary for sparse feature
    :param feat: feature name
    :param feat_num: the total number of sparse features that do not repeat
    :param embed_dim: embedding dimension
    :return:
    """
    return {'feat_name': feat, 'feat_num': feat_num, 'embed_dim': embed_dim, 'type': 'sparse'}

def denseFeature(feat):
    """
    create dictionary for dense feature
    :param feat: dense feature name
    :return:
    """
    return {'feat_name': feat, 'type': 'dense'}


class Attention_Layer(Layer):
    def __init__(self, att_hidden_units, activation='relu', w_reg=1e-6):
        """
        """
        super(Attention_Layer, self).__init__()
        self.att_dense = [Dense(unit, activation=activation, kernel_regularizer=l2(w_reg)) for unit in att_hidden_units]
        self.att_final_dense = Dense(1)

    def call(self, inputs):
        # query: candidate item  (None, d * 2), d is the dimension of embedding
        # key: hist items  (None, seq_len, d * 2)
        # value: hist items  (None, seq_len, d * 2)
        # mask: (None, seq_len)
        q, k, v, mask = inputs
        q = tf.tile(q, multiples=[1, k.shape[1]])  # (None, seq_len * d * 2)
        q = tf.reshape(q, shape=[-1, k.shape[1], k.shape[2]])  # (None, seq_len, d * 2)

        # q, k, out product should concat
        info = tf.concat([q, k, q - k, q * k], axis=-1)

        # dense
        for dense in self.att_dense:
            info = dense(info)

        outputs = self.att_final_dense(info)  # (None, seq_len, 1)
        outputs = tf.squeeze(outputs, axis=-1)  # (None, seq_len)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)  # (None, seq_len)
        outputs = tf.where(tf.equal(mask, 0), paddings, outputs)  # (None, seq_len)

        # softmax
        outputs = tf.nn.softmax(logits=outputs)  # (None, seq_len)
        outputs = tf.expand_dims(outputs, axis=1)  # None, 1, seq_len)

        outputs = tf.matmul(outputs, v)  # (None, 1, d * 2)
        outputs = tf.squeeze(outputs, axis=1)

        return outputs


class Dice(Layer):
    def __init__(self):
        super(Dice, self).__init__()
        # Dice 可以用一个没有参数的BN层+sigmiod实现
        self.bn = BatchNormalization(center=False, scale=False)
        self.alpha = self.add_weight(shape=(), dtype=tf.float32, name='alpha')

    def call(self, x):
        x_normed = self.bn(x)
        x_p = tf.sigmoid(x_normed)

        return self.alpha * (1.0 - x_p) * x + x_p * x


if __name__ == '__main__':
    dense_feature_length = 4
    sparse_feature_length = 100
    linear_layer = Linear(dense_feature_length, sparse_feature_length)
    dense_input = Input(shape=(4,), dtype=tf.float32)
    sparse_input = Input(shape=(100,), dtype=tf.int32)
    output = linear_layer([dense_input, sparse_input])
