#!/usr/bin/env python
# _*_coding:utf-8_*_

"""
@Project: ctrip_tkt_deep_model
@File   : model_with_seq.py
@Author : sichenghe(sichenghe@trip.com)
@Time   : 8/5/22
加入行为序列attention模块的 deep & wide
"""
#!/usr/bin/env python
# _*_coding:utf-8_*_

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Embedding, Dropout, Input
from tensorflow.keras.regularizers import l2

from WDL.modules import Linear, DNN, Attention_Layer

class WideDeepDIN(Model):
    def __init__(self, dense_feature_list, sparse_feature_list,
                 wide_feature_list=[], deep_feature_list=[], hidden_units=[64, 32], activation='relu',
                 behavior_feature_list=[], att_hidden_units=[40, 20], att_activation='relu', seq_len=-1,
                 dnn_dropout=0.1, embed_reg=1e-6, w_reg=1e-6):
        """
        @param dense_feature_list: dense features, [dict()]
        @param sparse_feature_list: sparse features, [dict()]
        @param wide_feature_list: wide side features, [dict()]
        @param deep_feature_list: deep side features, [dict()]
        @param hidden_units: Neural network hidden units.
        @param activation: Activation function of dnn.
        @param dnn_dropout: Dropout of dnn.
        @param embed_reg: The regularizer of embedding.
        @param w_reg: The regularizer of Linear.
        """
        super(WideDeepDIN, self).__init__()
        self.dense_feature_list = dense_feature_list  # 连续特征
        self.sparse_feature_list = sparse_feature_list  # 离散特征(不要包含序列特征如poi id，attention层会自动加入)
        self.behavior_feature_list = behavior_feature_list # 序列特征(离散)
        self.seq_len = seq_len # 序列长度(定长)
        self.behavior_num = len(self.behavior_feature_list) #
        self.attention_layer = Attention_Layer(att_hidden_units, att_activation) # attention layer

        all_feature_list = dense_feature_list + sparse_feature_list  # 所有特征 [连续， 离散]
        self.feature_idx = {feat['feat_name']: i for i, feat in enumerate(all_feature_list)} # 特征顺序记录

        if len(wide_feature_list) == 0:  # wide侧默认使用全部特征
            wide_feature_list = all_feature_list
        if len(deep_feature_list) == 0:  # deep侧默认使用全部特征
            deep_feature_list = all_feature_list
        self.wide_feature_list = wide_feature_list  # wide侧特征
        self.deep_feature_list = deep_feature_list  # deep侧特征

        self.len_dense_feature = len(dense_feature_list)  # 连续特征个数
        self.len_sparse_feature = len(sparse_feature_list)  # 离散特征个数
        self.wide_dense_feature_list = [feat for feat in self.wide_feature_list
                                        if feat['type'] == 'dense']  # wide侧的连续特征
        self.wide_sparse_feature_list = [feat for feat in self.wide_feature_list
                                        if feat['type'] == 'sparse']  # wide侧的离散特征
        self.deep_dense_feature_list = [feat for feat in self.deep_feature_list
                                        if feat['type'] == 'dense']  # wide侧的连续特征
        self.deep_sparse_feature_list = [feat for feat in self.deep_feature_list
                                        if feat['type'] == 'sparse']  # wide侧的离散特征

        # deep侧sparse feature的embed layer
        self.embed_layers = {
            'embed_' + feat['feat_name']: Embedding(input_dim=feat['feat_num'] + 1, # 0 留给未知/无
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_uniform',
                                         embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.deep_sparse_feature_list)
        }
        # 序列特征的embed layer
        self.seq_embed_layers = {
            'embed_' + feat['feat_name']: Embedding(input_dim=feat['feat_num'] + 1, # 0 留给未知/无
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_uniform',
                                         embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.behavior_feature_list)
        }

        # wide 部分变量记录累计长度，初始化wide部分的dense layer
        self.index_mapping = []
        self.feature_length = 0
        for feat in self.wide_sparse_feature_list:
            self.index_mapping.append(self.feature_length)
            self.feature_length += feat['feat_num']

        # wide, deep, output layer
        self.dnn_network = DNN(hidden_units, activation, dnn_dropout)
        self.linear = Linear(self.feature_length, w_reg=w_reg)
        self.final_dense = Dense(1, activation=None)

    def call(self, inputs, **kwargs):
        inputs, seq_input, item_input = inputs
        wide_dense_input, wide_sparse_input, deep_dense_input, deep_sparse_input = self._get_input(inputs)
        # deep侧sparse embedding
        # batch, len * embed_dim
        deep_sparse_input_embed = None
        if deep_sparse_input is not None:
            deep_sparse_input_embed = tf.concat([self.embed_layers['embed_' + feat['feat_name']](deep_sparse_input[:, i])
                                                 for i, feat in enumerate(self.deep_sparse_feature_list)], axis=-1)

        # sequence mask
        mask = tf.cast(tf.not_equal(seq_input[:, :, 0], 0), dtype=tf.float32)  # (None, maxlen)
        # 序列 和 target item embedding
        seq_input_embed = tf.concat([self.seq_embed_layers['embed_' + feat['feat_name']](seq_input[:, :, i])
                                     for i, feat in enumerate(self.behavior_feature_list)], axis=-1) # batch, seq_len, dim * num

        item_input_embed = tf.concat([self.seq_embed_layers['embed_' + feat['feat_name']](item_input[:, i])
                                     for i, feat in enumerate(self.behavior_feature_list)], axis=-1) # batch, dim * num

        # att
        # query: item_embed, batch, d*?
        # key, value: seq_embed, batch, maxlen, d*?
        print(item_input_embed.shape)
        print(seq_input_embed.shape)
        att_output = self.attention_layer([item_input_embed, seq_input_embed, seq_input_embed, mask])  # (None, d * 2)

        # Wide
        if wide_dense_input is not None and wide_sparse_input is not None:
            wide_inputs = [wide_dense_input, wide_sparse_input + tf.convert_to_tensor(self.index_mapping)]
        elif wide_dense_input is not None:
            wide_inputs = [wide_dense_input, None]
        else:
            wide_inputs = [None, wide_sparse_input + tf.convert_to_tensor(self.index_mapping)]

        wide_out = self.linear(wide_inputs)  # (batch, 1)
        # Deep
        if deep_dense_input is not None and deep_sparse_input_embed is not None:
            deep_inputs = tf.concat([deep_dense_input, deep_sparse_input_embed], axis=-1)
        elif deep_dense_input is not None:
            deep_inputs = deep_dense_input
        else:
            deep_inputs = deep_sparse_input_embed

        # 连接上attention输出
        deep_inputs = tf.concat([deep_inputs, att_output], axis=-1)
        deep_out = self.dnn_network(deep_inputs)
        deep_out = self.final_dense(deep_out)
        # out
        outputs = tf.nn.sigmoid(0.5 * wide_out + 0.5 * deep_out)
        return outputs

    def _get_input(self, inputs):
        # inputs: 一个list
        # 非序列输入list，按照初始化model时给的特征顺序排列,
        # return: wide_dense_input, wide_sparse_input, deep_dense_input, deep_sparse_input
        # 非序列特征处理
        inputs = tf.concat(inputs, axis=-1)  # batch, 特征个数
        assert inputs.shape[1] == self.len_dense_feature + self.len_sparse_feature, '特征个数不匹配 {} != {}'.format(inputs.shape[1], self.len_dense_feature + self.len_sparse_feature)
        # wide_dense
        if len(self.wide_dense_feature_list) == 0:
            wide_dense_input = None
        else:
            idx = [self.feature_idx[feat['feat_name']] for feat in self.wide_dense_feature_list]
            wide_dense_input = tf.gather(inputs, idx, axis=1)
        # wide_sparse
        if len(self.wide_sparse_feature_list) == 0:
            wide_sparse_input = None
        else:
            idx = [self.feature_idx[feat['feat_name']] for feat in self.wide_sparse_feature_list]
            wide_sparse_input = tf.cast(tf.gather(inputs, idx, axis=1), tf.int32)  # sparse特征id要cast int
        # deep_dense
        if len(self.deep_dense_feature_list) == 0:
            deep_dense_input = None
        else:
            idx = [self.feature_idx[feat['feat_name']] for feat in self.deep_dense_feature_list]
            deep_dense_input = tf.gather(inputs, idx, axis=1)
        # deep_sparse
        if len(self.deep_sparse_feature_list) == 0:
            deep_sparse_input = None
        else:
            idx = [self.feature_idx[feat['feat_name']] for feat in self.deep_sparse_feature_list]
            deep_sparse_input = tf.cast(tf.gather(inputs, idx, axis=1), tf.int32) # sparse特征id要cast int

        return wide_dense_input, wide_sparse_input, deep_dense_input, deep_sparse_input




    def summary(self, **kwargs):
        print('wide side featrues: ', ','.join(feat['feat_name']for feat in self.wide_feature_list))
        print('deep side featrues: ', ','.join(feat['feat_name']for feat in self.deep_feature_list))
        non_seq_inputs = [Input(shape=(1,), dtype=tf.float32) for _ in
                          range(self.len_dense_feature + self.len_sparse_feature)]

        seq_inputs = Input(shape=(self.behavior_num, 1), dtype=tf.float32)
        item_inputs = Input(shape=(1,), dtype=tf.float32)
        inputs = [non_seq_inputs, seq_inputs, item_inputs]
        Model(inputs=inputs, outputs=self.call(inputs)).summary()

# class WideDeep(Model):
#     def __init__(self, feature_columns, hidden_units, activation='relu',
#                  dnn_dropout=0., embed_reg=1e-6, w_reg=1e-6):
#         """
#         Wide&Deep
#         :param feature_columns: A list. sparse column feature information.
#         :param hidden_units: A list. Neural network hidden units.
#         :param activation: A string. Activation function of dnn.
#         :param dnn_dropout: A scalar. Dropout of dnn.
#         :param embed_reg: A scalar. The regularizer of embedding.
#         :param w_reg: A scalar. The regularizer of Linear.
#         """
#         super(WideDeep, self).__init__()
#         self.sparse_feature_columns = feature_columns
#         self.embed_layers = {
#             'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
#                                          input_length=1,
#                                          output_dim=feat['embed_dim'],
#                                          embeddings_initializer='random_uniform',
#                                          embeddings_regularizer=l2(embed_reg))
#             for i, feat in enumerate(self.sparse_feature_columns)
#         }
#         self.index_mapping = []
#         self.feature_length = 0
#         for feat in self.sparse_feature_columns:
#             self.index_mapping.append(self.feature_length)
#             self.feature_length += feat['feat_num']
#         self.dnn_network = DNN(hidden_units, activation, dnn_dropout)
#         self.linear = Linear(self.feature_length, w_reg=w_reg)
#         self.final_dense = Dense(1, activation=None)
#
#     def call(self, inputs, **kwargs):
#         # inputs: batch_size x field
#         # 按最后一个维度(列)concat
#         sparse_embed = tf.concat([self.embed_layers['embed_{}'.format(i)](inputs[:, i])
#                                   for i in range(inputs.shape[1])], axis=-1)
#         x = sparse_embed  # (batch_size, field * embed_dim)
#         # Wide
#         wide_inputs = inputs + tf.convert_to_tensor(self.index_mapping)
#         wide_out = self.linear(wide_inputs) # (batch, 1)
#         # Deep
#         deep_out = self.dnn_network(x)
#         deep_out = self.final_dense(deep_out)
#         # out
#         outputs = tf.nn.sigmoid(0.5 * wide_out + 0.5 * deep_out)
#         return outputs
#
#     def summary(self, **kwargs):
#         sparse_inputs = Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
#         Model(inputs=sparse_inputs, outputs=self.call(sparse_inputs)).summary()