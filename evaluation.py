"""
Evaluation module
"""
import argparse
import faiss
import numpy as np
import json
import pandas as pd
from utils import calculate_Recall, calculate_NDCG, calculate_Precision
import numpy as np




def top_k_evaluate(k_amount, test_data_address, user_emb, item_emb, dimension):
    """
    Computes mean average precision, mean average recall and
    mean reciprocal rank based on the One-plus-random testing methodology
    outlined in
    "Performance of recommender algorithms on top-n recommendation tasks."
    by Cremonesi, Paolo, Yehuda Koren, and Roberto Turrin (2010)

    Args:
        k (int): no. of most relevant items
        test_inter (:obj:`pd.DataFrame`): `M` testing instances (rows)
            with three columns `[user, item, rating]`
        user_reps (dict): representations for all `m` unique users
        item_reps (:obj:`np.array`): (n, d) `d` latent features for all `n` items
        n_random (int): no. of unobserved items to sample randomly
        verbose (bool): verbosity

    Returns:
        prec (float): mean average precision @ k
        rec (float): mean average recall @ k
        mrr (float): mean reciprocal rank @ k
    """

    test_df=pd.read_csv(test_data_address, delimiter='\t', names=['user', 'item', 'rating'])
    
    user_emb_vec=np.array(list(user_emb.values()))
    item_emb_vec=np.array(list(item_emb.values()))
    index=faiss.IndexFlatIP(dimension)
    index.add(item_emb_vec)
    D, I=index.search(user_emb_vec, max(k_amount))

    precision=0
    rec=0
    ndcg=0

    user_recomm_dict=dict(zip(list(user_emb.keys()), I))
    count = 0
    total_precision=[0, 0,0,0,0]
    total_rec=[0, 0, 0, 0, 0]
    total_ndcg=[0, 0, 0 ,0, 0]

    for user in list(test_df['user'].unique()):
        if user in list(user_recomm_dict.keys()):
            count+=1
            item_key=np.array(list(item_emb.keys()))
            recomm_list=list(item_key[user_recomm_dict[user]])

            watched_list=list(test_df[test_df['user']==user]['item'])

            for k in range(len(k_amount)):
                recomm_k=recomm_list[:k_amount[k]]
                total_precision[k]+=calculate_Precision(watched_list, recomm_k)
                total_rec[k]+=calculate_Recall(watched_list, recomm_k)
                total_ndcg[k]+=calculate_NDCG(watched_list, recomm_k)
    total_precision=np.array(total_precision)
    total_rec=np.array(total_rec)
    total_ndcg=np.array(total_ndcg)
            
    return total_precision/count, total_rec/count, total_ndcg/count

   
