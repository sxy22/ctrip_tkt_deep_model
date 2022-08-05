# !/usr/bin/env python 
# -*- coding:utf-8 -*-  
# author:xudongyu
'''
说明：get：auc/gauc/topn/ndcg
'''
import os
# os.environ['PYSPARK_PYTHON'] = '/usr/bin/python3.6.5'
# os.environ["SPARK_CONF_DIR"] = "/opt/app/spark/conf"
# os.sys.path.insert(0, '/opt/app/spark/python')
# os.sys.path.insert(1, '/opt/app/spark/python/lib/py4j-0.10.4-src.zip')

import pandas as pd
import numpy as np
# from pyspark.sql import *
import math
from sklearn import metrics


def get_gauc_v2(data,qid_col,label_col,y_pred_col):
    df = data.copy()
#     df[label_col] = df[label_col].astype(int)
    # traceid下不同label的个数
    label_nunique = df.groupby(qid_col)[label_col].nunique().to_dict()
    df['label_nunique'] = df[qid_col].map(label_nunique)
    # 选择均有正负例的traceid
#    df = df[df['label_nunique'] > 1]
    # traceid下产品曝光数
    expos_cnt = df.groupby(qid_col)[label_col].size().to_dict()
    # 每个traceid下的auc
    auc_per_qid = df.groupby(qid_col).apply(get_auc, label_col, y_pred_col).to_dict()
    df_auc = pd.DataFrame()
    df_auc[qid_col] = df[qid_col].unique()
    df_auc['auc_per_qid'] = df_auc[qid_col].map(auc_per_qid)
    df_auc['expos_cnt'] = df_auc[qid_col].map(expos_cnt)
    df_auc["auc_weight"] = df_auc['auc_per_qid'] * df_auc['expos_cnt']
    df_auc_sum = df_auc.sum()
    gauc = df_auc_sum["auc_weight"] / df_auc_sum["expos_cnt"]
    return gauc


def get_topn_v2(data, qid_col, label_col, rank_col, n):
    def tophit_at_k(r, k):
        r = np.asfarray(r)[:k] >= 1
        return float(r.__contains__(1.0))

    def get_tophit_k(target, pred_score, k):
        zpd = list(zip(target, pred_score))
        zpd.sort(key=lambda x: x[1], reverse=True)
        pred_rank, _ = list(zip(*zpd))
        return tophit_at_k(list(pred_rank), k)

    traceids = set(data[qid_col].tolist())
    data['score'] = 1 / data['fine_rank']
    sums = 0
    for traceid in traceids:
        batch_data = data[data[qid_col] == traceid]
        sums += get_tophit_k(batch_data[label_col].tolist(), batch_data['score'].tolist(), 3)
    return (sums / len(traceids))

# -- 转换dataframe --
# use rdd partitions to solve the problem out of memory
def _map_to_pandas(rdds):
    """ Needs to be here due to pickling issues """
    return [pd.DataFrame(list(rdds))]


def toPandas(df, n_partitions=None):
    """
    Returns the contents of `df` as a local `pandas.DataFrame` in a speedy fashion. The DataFrame is
    repartitioned if `n_partitions` is passed.
    :param df:              pyspark.sql.DataFrame
    :param n_partitions:    int or None
    :return:                pandas.DataFrame
    """
    if n_partitions is not None: df = df.repartition(n_partitions)
    df_pand = df.rdd.mapPartitions(_map_to_pandas).collect()
    df_pand = pd.concat(df_pand)
    df_pand.columns = df.columns
    return df_pand


# -- 加载数据 --
def load_data(tablename, startdate=None, enddate=None):
    if not startdate:
        data_spark = spark.sql("select * from %s " % (tablename))
    elif not enddate:
        data_spark = spark.sql("select * from %s where d='%s'" % (tablename, startdate))
    else:
        data_spark = spark.sql("select * from %s where d>='%s' and d<='%s'" % (tablename, startdate, enddate))
    return data_spark


# def get_auc(data,y_label_col,y_pred_col):
# #     data[y_label_col] = data[y_label_col].astype(int)
#     auc = metrics.roc_auc_score(data[y_label_col], data[y_pred_col])
#     return auc


def get_auc(data, y_label_col, y_pred_col):    
    data[y_label_col] = data[y_label_col].astype(int)
    if len(np.unique(data[y_label_col])) != 2:
        return 0
    auc = metrics.roc_auc_score(data[y_label_col], data[y_pred_col])
    return auc


def get_topn(data,label_col,rank_col,n):
#     data[label_col] = data[label_col].astype(int)
#     data[rank_col] = data[rank_col].astype(int)
    topn_cnt = len(data[(data[rank_col] <= n) & (data[label_col] == 1)])
    sum = len(data[(data[label_col] == 1)])
    if sum !=0:
        topn_rate = topn_cnt/sum
    else:
        topn_rate = None
    return topn_rate


def get_gauc(data,qid_col,label_col,y_pred_col):
    df = data.copy()
#     df[label_col] = df[label_col].astype(int)
    # traceid下不同label的个数
    label_nunique = df.groupby(qid_col)[label_col].nunique().to_dict()
    df['label_nunique'] = df[qid_col].map(label_nunique)

    # 选择均有正负例的traceid
    df = df[df['label_nunique'] > 1]

    # traceid下产品曝光数
    expos_cnt = df.groupby(qid_col)[label_col].size().to_dict()
    # 每个traceid下的auc
    auc_per_qid = df.groupby(qid_col).apply(get_auc, label_col, y_pred_col).to_dict()

    df_auc = pd.DataFrame()
    df_auc[qid_col] = df[qid_col].unique()
    df_auc['auc_per_qid'] = df_auc[qid_col].map(auc_per_qid)
    df_auc['expos_cnt'] = df_auc[qid_col].map(expos_cnt)
    df_auc["auc_weight"] = df_auc['auc_per_qid'] * df_auc['expos_cnt']
    df_auc_sum = df_auc.sum()
    gauc = df_auc_sum["auc_weight"] / df_auc_sum["expos_cnt"]
    return gauc


def get_ndcg(data, qid_col, target, rank_col, n):
    '''

    Args:
        data:
        qid_col: qid
        target: 真实target
        rank_col: 模型打分的排名
        n: ndcg@n

    Returns:

    '''
#     data[target] = data[target].astype(int)
#     data[rank_col] = data[rank_col].astype(int)
    data["rank_by_target"] = data[target].groupby(data[qid_col]).rank(ascending=0, method='first')

    if n == "all":
        df1 = data.copy()
        df2 = data.copy()
    else:
        df1 = data[data[rank_col] <= n].copy()                  # dcg按照模型分数取topn
        df2 = data[data["rank_by_target"] <= n].copy()          # idcg按照target取topn

    df1["dcg_mol"] = df1[target].apply(lambda x: np.power(2, x) - 1)
    df1["dcg_den"] = df1[rank_col].apply(lambda x: math.log(x + 1, 2))
    df1["dcg"] = df1["dcg_mol"] / df1["dcg_den"]
    df1_trace = df1[["dcg"]].groupby(df1[qid_col]).sum().reset_index()

    df2["idcg_mol"] = df2[target].apply(lambda x: np.power(2, x) - 1)
    df2["idcg_den"] = df2["rank_by_target"].apply(lambda x: math.log(x + 1, 2))
    df2["idcg"] = df2["idcg_mol"] / df2["idcg_den"]
    df2_trace = df2[["idcg"]].groupby(df2[qid_col]).sum().reset_index()

    df_trace = pd.merge(df1_trace,df2_trace,on=qid_col)
    df_trace["ndcg"] = df_trace["dcg"] / df_trace["idcg"]
    ndcg = df_trace["ndcg"].mean()
    return ndcg
    

def run_pipline(data,qid_col,y_pred_col,rank_col):
    '''
    
    Args:
        data: DataFrame格式，包含qid_col,y_pred_col,rank_col
        qid_col: qid列名
        y_pred_col: 模型分数列名
        rank_col: 模型分数在qid下的排名

    Returns: auc/gauc/topn/ndcg
    '''

    res = dict()
#     res.setdefault("date", [])
    res.setdefault("item", [])
    res.setdefault("value", [])

#     testdate = data["d"].unique()[0]
    # -- AUC --
    for label_col in ["click", "order_create", "order_ground_truth"]:
        auc = get_auc(data, label_col, y_pred_col)
#         res["date"].append(testdate)
        res["item"].append("{}_auc".format(label_col))
        res["value"].append(auc)


    # -- gauc --
    for label_col in ["click", "order_create", "order_ground_truth"]:
        gauc = get_gauc(data,qid_col,label_col,y_pred_col)
#         res["date"].append(testdate)
        res["item"].append("{}_gauc".format(label_col))
        res["value"].append(gauc)


    # -- topn --
    for label_col in ["click", "order_create", "order_ground_truth"]:
        for n in [1, 3, 5, 10,15,20,30]:
            topn_rate = get_topn(data, label_col, rank_col, n)
#             res["date"].append(testdate)
            res["item"].append("{}_top{}".format(label_col, n))
            res["value"].append(topn_rate)
   

    # -- ndcg --
    target = "click"
    for n in [1, 3, 5, 10,15,20,30, "all"]:
        ndcg = get_ndcg(data,qid_col,target,rank_col,n)
#         res["date"].append(testdate)
        res["item"].append("{}_ndcg{}".format(target, n))
        res["value"].append(ndcg)

    return pd.DataFrame(res)
    








