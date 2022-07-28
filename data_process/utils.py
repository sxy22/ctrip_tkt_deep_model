#!/usr/bin/env python
# _*_coding:utf-8_*_

"""
@Project: ctrip_tkt_deep_model
@File   : utils.py
@Author : sichenghe(sichenghe@trip.com)
@Time   : 2022/7/28 
"""

def sparseFeature(feat, feat_num, embed_dim=4):
    """
    create dictionary for sparse feature
    :param feat: feature name
    :param feat_num: the total number of sparse features that do not repeat
    :param embed_dim: embedding dimension
    :return:
    """
    return {'feat_name': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}


def denseFeature(feat):
    """
    create dictionary for dense feature
    :param feat: dense feature name
    :return:
    """
    return {'feat_name': feat}