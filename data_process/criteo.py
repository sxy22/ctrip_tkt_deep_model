#!/usr/bin/env python
# _*_coding:utf-8_*_

"""
@Project: ctrip_tkt_deep_model
@File   : criteo.py
@Author : sichenghe(sichenghe@trip.com)
@Time   : 2022/7/28 
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.model_selection import train_test_split

from data_process.utils import sparseFeature


def create_criteo_dataset(file, embed_dim=8, read_part=True, sample_num=100000, test_size=0.2):
    """
    a example about creating criteo dataset
    :param file: dataset's path
    :param embed_dim: the embedding dimension of sparse features
    :param read_part: whether to read part of it
    :param sample_num: the number of instances if read_part is True
    :param test_size: ratio of test dataset
    :return: feature columns, train, test
    """
    # names = ['label', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11',
    #          'I12', 'I13', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
    #          'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22',
    #          'C23', 'C24', 'C25', 'C26']

    if read_part:
        # data_df = pd.read_csv(file, sep='\t', iterator=True, header=None,
        #                   names=names)
        data_df = pd.read_csv(file, sep=',', header=0, iterator=True)
        data_df = data_df.get_chunk(sample_num)

    else:
        # data_df = pd.read_csv(file, sep='\t', header=None, names=names)
        data_df = pd.read_csv(file, sep=',', header=0)

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
    features = sparse_features + dense_features

    data_df[sparse_features] = data_df[sparse_features].fillna('-1')
    data_df[dense_features] = data_df[dense_features].fillna(0)

    # Bin continuous data into intervals.
    est = KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='uniform')
    data_df[dense_features] = est.fit_transform(data_df[dense_features])

    for feat in sparse_features:
        le = LabelEncoder()
        data_df[feat] = le.fit_transform(data_df[feat])

    # ==============Feature Engineering===================

    # ====================================================
    feature_columns = [sparseFeature(feat, int(data_df[feat].max()) + 1, embed_dim=embed_dim)
                        for feat in features]
    train, test = train_test_split(data_df, test_size=test_size)

    train_X = train[features].values.astype('int32')
    train_y = train['label'].values.astype('int32')
    test_X = test[features].values.astype('int32')
    test_y = test['label'].values.astype('int32')

    return feature_columns, (train_X, train_y), (test_X, test_y)

# ??????wide???deep????????????
def create_criteo_dataset_wide_deep(file, wide_features=[], deep_features=[],
                                    embed_dim=8, read_part=True, sample_num=100000, test_size=0.2):
    """
    @param file: dataset's path
    @param wide_features: features for wide part, default value [] will use all features
    @param deep_features: features for deep part, default value [] will use all features
    @param embed_dim: the embedding dimension of sparse features
    @param read_part: whether to read part of it
    @param sample_num: the number of instances if read_part is True
    @param test_size: ratio of test dataset
    @return: feature columns, train, test
    """

    # names = ['label', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11',
    #          'I12', 'I13', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
    #          'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22',
    #          'C23', 'C24', 'C25', 'C26']

    if read_part:
        # data_df = pd.read_csv(file, sep='\t', iterator=True, header=None,
        #                   names=names)
        data_df = pd.read_csv(file, sep=',', header=0, iterator=True)
        data_df = data_df.get_chunk(sample_num)
    else:
        # data_df = pd.read_csv(file, sep='\t', header=None, names=names)
        data_df = pd.read_csv(file, sep=',', header=0)

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
    features = sparse_features + dense_features

    data_df[sparse_features] = data_df[sparse_features].fillna('-1')
    data_df[dense_features] = data_df[dense_features].fillna(0)

    # Bin continuous data into intervals.
    est = KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='uniform')
    data_df[dense_features] = est.fit_transform(data_df[dense_features])

    for feat in sparse_features:
        le = LabelEncoder()
        data_df[feat] = le.fit_transform(data_df[feat])

    # ==============Feature Engineering===================

    if len(wide_features) == 0:
        wide_features = features
    if len(deep_features) == 0:
        deep_features = features
    wide_feature_columns = [sparseFeature(feat, int(data_df[feat].max()) + 1, embed_dim=embed_dim)
                        for feat in wide_features]
    deep_feature_columns = [sparseFeature(feat, int(data_df[feat].max()) + 1, embed_dim=embed_dim)
                        for feat in deep_features]

    train, test = train_test_split(data_df, test_size=test_size)

    train_X_wide = train[wide_features].values.astype('int32')
    train_X_deep = train[deep_features].values.astype('int32')
    test_X_wide = test[wide_features].values.astype('int32')
    test_X_deep = test[deep_features].values.astype('int32')

    train_y = train['label'].values.astype('int32')
    test_y = test['label'].values.astype('int32')

    return (wide_feature_columns, deep_feature_columns), (train_X_wide, train_X_deep, train_y), (test_X_wide, test_X_deep, test_y)
if __name__ == '__main__':
    file = '../dataset/criteo/criteo_sampled_data.csv'
    create_criteo_dataset(file, embed_dim=8, read_part=True, sample_num=50000, test_size=0.2)