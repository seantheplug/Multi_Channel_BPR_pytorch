"""
Model initialization and training methods
"""

from collections import OrderedDict
import logging

from evaluation import top_k_evaluate
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


