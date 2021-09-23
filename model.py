"""
Model initialization and training methods
"""

from collections import OrderedDict
import logging

from evaluation import top_k_evaluate
from sampling import (get_pos_channel, get_neg_channel,
                       get_pos_user_item, get_neg_item)
from utils import *
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import threading
import time


class MCBPR(nn.Module):
    def __init__(self, d, beta, rd_seed, channels, n_user, n_item, n_random=1000):
		# super(MCBPR, self).__init__()
        super(MCBPR, self).__init__()
        self.d = d
        self.beta = beta
        self.rd_seed = rd_seed
        self.channels = channels
        self.n_user = n_user
        self.n_item = n_item
        self.n_random = n_random
        self.embed_item = nn.Embedding(n_item, d)
        self.embed_user = nn.Embedding(n_user, d)
        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)


    def forward(self, u, i, j): 
            user = self.embed_user(torch.LongTensor(u).cuda())
            item_i = self.embed_item(torch.LongTensor(i).cuda())
            item_j = self.embed_item(torch.LongTensor(j).cuda())
            prediction_i = (user * item_i).sum(dim=-1)
            prediction_j = (user * item_j).sum(dim=-1)

            return prediction_i, prediction_j

    def fit(self, lr, reg_params, n_epochs, train_loader, optimizer_params):
        self.train()
        self.cuda()

        print(self)
        if optimizer_params == 'sgd':
            print('sgd')
            optimizer = optim.SGD(self.parameters(), lr=lr, weight_decay=reg_params['u'])
        elif optimizer_params == 'adam':
            print('adam')
            optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=reg_params['u'])
        

        for epoch in tqdm(range(n_epochs)):
            for u, i, j in tqdm(train_loader):
                self.zero_grad()
                prediction_i, prediction_j = self.forward(u, i, j)
                loss = - (prediction_i - prediction_j).sigmoid().log().sum()
                loss.backward()
		        
                optimizer.step()


                                                   
    def save_user_item_embedding(self, user_hasher, item_hasher, file_path):
        file1 = open(file_path,"w")

        for i, user in enumerate(list(user_hasher.keys())):
            user_embed = torch.flatten(self.embed_user(torch.LongTensor([i]).cuda()))
            user_embed = user_embed.detach().cpu().numpy()
            file1.write(user+"\t")
            file1.write(" ".join(map(str,  user_embed))+"\n")

        for j, item in enumerate(list(item_hasher.keys())):
            item_embed = torch.flatten(self.embed_item(torch.LongTensor([j]).cuda()))
            item_embed = item_embed.detach().cpu().numpy()
            file1.write(item+"\t")
            file1.write(" ".join(map(str,  item_embed))+"\n")

        file1.close()


    def predict(self, users, k):
        """
        Returns the `k` most-relevant items for every user in `users`

        Args:
            users ([int]): list of user ID numbers
            k (int): no. of most relevant items

        Returns:
            top_k_items (:obj:`np.array`): (len(users), k) array that holds
                the ID numbers for the `k` most relevant items
        """
        top_k_items = np.zeros((len(users), k))

        for idx, user in enumerate(users):
            user_embed = self.user_reps[user]
            pred_ratings = np.dot(self.item_reps, user_embed)
            user_items = np.argsort(pred_ratings)[::-1][:k]
            top_k_items[idx] = user_items

        return top_k_items

    def evaluate(self, test_data_address, k, user_hasher, item_hasher):
        """
        Offline evaluation of the model performance using precision,
        recall, and mean reciprocal rank computed for top-`k` positions
        and averaged across all users

        Args:
            test_ratings (:obj:`pd.DataFrame`): `M` testing instances (rows)
                with three columns `[user, item, rating]`
            `k` (int): no. of most relevant items

        Returns:
            result (tuple): mean average precision (MAP), mean average recall (MAR),
                and mean reciprocal rank (MRR) - all at `k` positions
        """

        user_emb={}
        item_emb={}


        for i, user in enumerate(list(user_hasher.keys())):
            user_embed = torch.flatten(self.embed_user(torch.LongTensor([i]).cuda()))
            user_embed = user_embed.detach().cpu().numpy()
            user_emb.update({ user: user_embed })

        for j, item in enumerate(list(item_hasher.keys())):
            item_embed = torch.flatten(self.embed_item(torch.LongTensor([j]).cuda()))
            item_embed = item_embed.detach().cpu().numpy()
            item_emb.update({ item: item_embed })

        result = top_k_evaluate(k, test_data_address, user_emb, item_emb, self.d )

        return result


# def get_pos_neg_splits(train_inter_df):
#     """
#     Calculates the rating mean for each user and splits the train
#     ratings into positive (greater or equal as every user's
#     mean rating) and negative ratings (smaller as mean ratings)

#     Args:
#         train_inter_df (:obj:`pd.DataFrame`): `M` training instances (rows)
#             with three columns `[user, item, rating]`

#     Returns:
#         train_inter_pos (:obj:`pd.DataFrame`): training instances (rows)
#             where `rating_{user}` >= `mean_rating_{user}
#         train_inter_neg (:obj:`pd.DataFrame`): training instances (rows)
#             where `rating_{user}` < `mean_rating_{user}
#     """
#     user_mean_ratings = \
#         train_inter_df[['user', 'rating']].groupby('user').mean().reset_index()
#     user_mean_ratings.rename(columns={'rating': 'mean_rating'},
#                              inplace=True)

#     train_inter_df = train_inter_df.merge(user_mean_ratings, on='user')
#     train_inter_pos = train_inter_df[
#         train_inter_df['rating'] >= train_inter_df['mean_rating']]
#     train_inter_neg = train_inter_df[
#         train_inter_df['rating'] < train_inter_df['mean_rating']]

#     return train_inter_pos, train_inter_neg


# def get_overall_level_distributions(train_inter_pos, train_inter_neg, beta):
#     """
#     Computes the frequency distributions for discrete ratings

#     Args:
#         train_inter_pos (:obj:`pd.DataFrame`): training instances (rows)
#             where `rating_{user}` >= `mean_rating_{user}
#         train_inter_neg (:obj:`pd.DataFrame`): training instances (rows)
#             where `rating_{user}` < `mean_rating_{user}
#         beta (float): share of unobserved feedback within the overall
#             negative feedback

#     Returns:
#         pos_level_dist (dict): positive level sampling distribution
#         neg_level_dist (dict): negative level sampling distribution
#     """

#     pos_counts = train_inter_pos['rating'].value_counts().sort_index(
#             ascending=False)
#     neg_counts = train_inter_neg['rating'].value_counts().sort_index(
#             ascending=False)

#     pos_level_dist = get_pos_level_dist(pos_counts.index.values,
#                                         pos_counts.values)
#     neg_level_dist = get_neg_level_dist(neg_counts.index.values,
#                                         neg_counts.values, beta)

#     return pos_level_dist, neg_level_dist


# def get_pos_channel_item_dict(train_inter_pos):
#     """
#     Creates buckets for each possible rating in `train_inter_pos`
#     and subsumes all observed (user, item) interactions with
#     the respective rating within

#     Args:
#         train_inter_pos (:obj:`pd.DataFrame`): training instances (rows)
#             where `rating_{user}` >= `mean_rating_{user}

#     Returns:
#         train_inter_pos_dict (dict): collection of all (user, item) interaction
#             tuples for each positive feedback channel
#     """

#     pos_counts = train_inter_pos['rating'].value_counts().sort_index(
#         ascending=False)
#     train_inter_pos_dict = OrderedDict()

#     for key in pos_counts.index.values:
#         u_i_tuples = [tuple(x) for x in
#                       train_inter_pos[train_inter_pos['rating'] == key][['user', 'item']].values]
#         train_inter_pos_dict[key] = u_i_tuples

#     return train_inter_pos_dict


# def get_user_reps(m, d, train_inter, test_inter, channels, beta):
#     """
#     Creates user representations that encompass user latent features
#     and additional user-specific information
#     User latent features are drawn from a standard normal distribution

#     Args:
#         m (int): no. of unique users in the dataset
#         d (int): no. of latent features for user and item representations
#         train_inter (:obj:`pd.DataFrame`): `M` training instances (rows)
#             with three columns `[user, item, rating]`
#         test_ratings (:obj:`pd.DataFrame`): `M` testing instances (rows)
#                 with three columns `[user, item, rating]`
#         channels ([int]): rating values representing distinct feedback channels
#         beta (float): share of unobserved feedback within the overall
#             negative feedback

#     Returns:
#         user_reps (dict): representations for all `m` unique users
#     """
#     user_reps = {}
#     train_inter = train_inter.sort_values('user')

#     for user_id in range(m):
#         user_reps[user_id] = {}
#         user_reps[user_id]['embed'] = np.random.normal(size=(d,))
#         user_item_ratings = train_inter[train_inter['user'] == user_id][['item', 'rating']]
#         user_reps[user_id]['mean_rating'] = user_item_ratings['rating'].mean()
#         user_reps[user_id]['items'] = list(user_item_ratings['item'])
#         user_reps[user_id]['all_items'] = list(set(user_reps[user_id]['items']).union(
#                                                set(list(test_inter[test_inter['user'] == user_id]['item']))))
#         user_reps[user_id]['pos_channel_items'] = OrderedDict()
#         user_reps[user_id]['neg_channel_items'] = OrderedDict()
#         for channel in channels:
#             if channel >= user_reps[user_id]['mean_rating']:
#                 user_reps[user_id]['pos_channel_items'][channel] = \
#                     list(user_item_ratings[user_item_ratings['rating'] == channel]['item'])
#             else:
#                 user_reps[user_id]['neg_channel_items'][channel] = \
#                     list(user_item_ratings[user_item_ratings['rating'] == channel]['item'])

#         pos_channels = np.array(list(user_reps[user_id]['pos_channel_items'].keys()))
#         neg_channels = np.array(list(user_reps[user_id]['neg_channel_items'].keys()))
#         pos_channel_counts = [len(user_reps[user_id]['pos_channel_items'][key]) for key in pos_channels]
#         neg_channel_counts = [len(user_reps[user_id]['neg_channel_items'][key]) for key in neg_channels]

#         user_reps[user_id]['pos_channel_dist'] = \
#             get_pos_level_dist(pos_channels, pos_channel_counts, 'non-uniform')

#         if sum(neg_channel_counts) != 0:
#             user_reps[user_id]['neg_channel_dist'] = \
#                 get_neg_level_dist(neg_channels, neg_channel_counts, 'non-uniform')

#             # correct for beta
#             for key in user_reps[user_id]['neg_channel_dist'].keys():
#                 user_reps[user_id]['neg_channel_dist'][key] = \
#                     user_reps[user_id]['neg_channel_dist'][key] * (1 - beta)
#             user_reps[user_id]['neg_channel_dist'][-1] = beta

#         else:
#             # if there is no negative feedback, only unobserved remains
#             user_reps[user_id]['neg_channel_dist'] = {-1: 1.0}

#     return user_reps

# def get_user_reps_torch(m, train_inter, test_inter, channels, beta):
#     """
#     Creates user representations that encompass user latent features
#     and additional user-specific information
#     User latent features are drawn from a standard normal distribution

#     Args:
#         m (int): no. of unique users in the dataset
#         d (int): no. of latent features for user and item representations
#         train_inter (:obj:`pd.DataFrame`): `M` training instances (rows)
#             with three columns `[user, item, rating]`
#         test_ratings (:obj:`pd.DataFrame`): `M` testing instances (rows)
#                 with three columns `[user, item, rating]`
#         channels ([int]): rating values representing distinct feedback channels
#         beta (float): share of unobserved feedback within the overall
#             negative feedback

#     Returns:
#         user_reps (dict): representations for all `m` unique users
#     """
#     user_reps = {}
#     train_inter = train_inter.sort_values('user')

#     for user_id in range(m):
#         user_reps[user_id] = {}
#         # user_reps[user_id]['embed'] = 
#         user_item_ratings = train_inter[train_inter['user'] == user_id][['item', 'rating']]
#         user_reps[user_id]['mean_rating'] = user_item_ratings['rating'].mean()
#         user_reps[user_id]['items'] = list(user_item_ratings['item'])
#         user_reps[user_id]['all_items'] = list(set(user_reps[user_id]['items']).union(
#                                                set(list(test_inter[test_inter['user'] == user_id]['item']))))
#         user_reps[user_id]['pos_channel_items'] = OrderedDict()
#         user_reps[user_id]['neg_channel_items'] = OrderedDict()
#         for channel in channels:
#             if channel >= user_reps[user_id]['mean_rating']:
#                 user_reps[user_id]['pos_channel_items'][channel] = \
#                     list(user_item_ratings[user_item_ratings['rating'] == channel]['item'])
#             else:
#                 user_reps[user_id]['neg_channel_items'][channel] = \
#                     list(user_item_ratings[user_item_ratings['rating'] == channel]['item'])

#         pos_channels = np.array(list(user_reps[user_id]['pos_channel_items'].keys()))
#         neg_channels = np.array(list(user_reps[user_id]['neg_channel_items'].keys()))
#         pos_channel_counts = [len(user_reps[user_id]['pos_channel_items'][key]) for key in pos_channels]
#         neg_channel_counts = [len(user_reps[user_id]['neg_channel_items'][key]) for key in neg_channels]

#         user_reps[user_id]['pos_channel_dist'] = \
#             get_pos_level_dist(pos_channels, pos_channel_counts, 'non-uniform')

#         if sum(neg_channel_counts) != 0:
#             user_reps[user_id]['neg_channel_dist'] = \
#                 get_neg_level_dist(neg_channels, neg_channel_counts, 'non-uniform')

#             # correct for beta
#             for key in user_reps[user_id]['neg_channel_dist'].keys():
#                 user_reps[user_id]['neg_channel_dist'][key] = \
#                     user_reps[user_id]['neg_channel_dist'][key] * (1 - beta)
#             user_reps[user_id]['neg_channel_dist'][-1] = beta

#         else:
#             # if there is no negative feedback, only unobserved remains
#             user_reps[user_id]['neg_channel_dist'] = {-1: 1.0}

#     return user_reps


# def get_item_reps(n, d):
#     """
#     Initializes item latent features from a standard normal distribution

#     Args:
#         n (int): no. of unique items in the dataset
#         d (int): no. of latent features for user and item representations

#     Returns:
#         item_reps (:obj:`np.array`): (n, d) `d` latent features for all `n` items
#     """
#     item_reps = np.random.normal(size=(n, d))

#     return item_reps

# def get_item_reps_torch(item_embed):
#     """
#     Initializes item latent features from a standard normal distribution

#     Args:
#         n (int): no. of unique items in the dataset
#         d (int): no. of latent features for user and item representations

#     Returns:
#         item_reps (:obj:`np.array`): (n, d) `d` latent features for all `n` items
#     """
#     # item_reps = np.random.normal(size=(n, d))

#     return np.array(item_embed.weight.data)

