"""
Helper functions
"""
import os
import pdb

import numpy as np
import pandas as pd

import statistics
import math



def get_pos_level_dist(weights, level_counts, mode='non-uniform'):
    """
    Returns the sampling distribution for positive
    feedback channels L using either a `non-uniform` or `uniform` approach

    Args:
        weights (:obj:`np.array`): (w, ) `w` rating values representing distinct
            positive feedback channels
        level_counts (:obj:`np.array`): (s, ) count `s` of ratings for each
            positive feedback channel
        mode (str): either `uniform` meaning all positive levels are
            equally relevant or `non-uniform` which imposes
            a (rating*count)-weighted distribution of positive levels

    Returns:
        dist (dict): positive channel sampling distribution
    """
    if mode == 'non-uniform':
        nominators = weights * level_counts
        denominator = sum(nominators)
        dist = nominators / denominator
    else:
        n_levels = len(weights)
        dist = np.ones(n_levels) / n_levels

    dist = dict(zip(list(weights), dist))

    return dist


def get_neg_level_dist(weights, level_counts, mode='non-uniform'):
    """
    Compute negative feedback channel distribution

    Args:
        weights (:obj:`np.array`): (w, ) `w` rating values representing distinct
            negative feedback channels
        level_counts (:obj:`np.array`): (s, ) count `s` of ratings for each
            negative feedback channel
        mode: either `uniform` meaning all negative levels are
            equally relevant or `non-uniform` which imposes
            a (rating*count)-weighted distribution of negative levels

    Returns:
        dist (dict): negative channel sampling distribution
    """
    if mode == 'non-uniform':
        # print("ok")
        nominators = [weight * count for weight, count in zip(weights, level_counts)]
        denominator = sum(nominators)
        if denominator != 0:
            dist = list(nom / denominator for nom in nominators)
        else:
            dist = [0] * len(nominators)
    else:
        # print("not ok")
        n_levels = len(weights)
        dist = [1 / n_levels] * n_levels

    # if np.abs(np.sum(dist)-1) > 0.00001:
    #     _logger.warning("Dist sum unequal 1.")

    dist = dict(zip(list(weights), dist))

    return dist


def load_rating_data(path):
    """
    loads the movielens 1M dataset, ignoring temporal information

    Args:
        path (str): path pointing to folder with interaction data `ratings.dat`

    Returns:
        ratings (:obj:`pd.DataFrame`): overall interaction instances (rows)
            with three columns `[user, item, rating]`
        m (int): no. of unique users in the dataset
        n (int): no. of unique items in the dataset
    """
    df = pd.read_csv(path, delimiter = "\t", header = None, names=['user','item','rating'])

    m = df['user'].unique().shape[0]
    n = df['item'].unique().shape[0]

    # Contiguation of user and item IDs
    user_rehasher = dict(zip(df['user'].unique(), np.arange(m)))
    item_rehasher = dict(zip(df['item'].unique(), np.arange(n)))
    df['user'] = df['user'].map(user_rehasher).astype(int)
    df['item'] = df['item'].map(item_rehasher)

    return df, m, n, user_rehasher, item_rehasher


def get_channels(inter_df):
    """
    Return existing feedback channels ordered by descending preference level

    Args:
        inter_df (:obj:`pd.DataFrame`): overall interaction instances (rows)
            with three columns `[user, item, rating]`
    Returns:
        channels ([int]): rating values representing distinct feedback channels
    """
    channels = list(inter_df['rating'].unique())
    channels.sort()

    return channels[::-1]



def calculate_Recall(active_watching_log, topk_program):
    unique_played_amount = len(set(active_watching_log))
    hit = 0
    for program in topk_program:
        if program in active_watching_log:
            hit += 1
            
    if unique_played_amount == 0:
        return 0
    else:
        return hit / unique_played_amount

def calculate_Precision(active_watching_log, topk_program):
    recommend_amount = len(topk_program)
    hit = 0
    for program in topk_program:
        if program in active_watching_log:
            hit += 1
    return hit / recommend_amount
    
def calculate_NDCG(active_watching_log, topk_program):
    dcg = 0
    idcg = 0
    ideal_length = min(len(active_watching_log), len(topk_program))
    #dcg
    for i in range(len(topk_program)):
        if topk_program[i] in active_watching_log:
            dcg += (1/math.log2(i+2))
    #idcg
    for i in range(ideal_length):
        idcg += (1/math.log2(i+2))
    
    if idcg == 0:
        return 0
    else:
        return float(dcg/idcg)
