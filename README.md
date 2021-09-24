# Pytorch MCBPR

This is a Pytorch implementation of Multi Channel BPR (MCBPR)
[Bayesian Personalized Ranking with Multi-channel User Feedback - Loni et al. (2016)](https://dl.acm.org/doi/pdf/10.1145/2959100.2959163)

## Description

This project is a Pytorch implementation of the earilist recommendation system model that taken into account the information of multi-behavior user-item interactions. MCBPR alters the typical sampling procedure of BPR (Rendle et al. 2009) to prioritize different user behaviors; specifically, MCBPR samples training pairs according to different positive level, that is the more positive the interaction is, the more likely it will be sampled. For more details, please see the paper.

with a type of positive behavior and another type of weaker behavior for model training.

## Getting Started

### Dependencies
```
Python                  3.8.10
torch                   1.9.0
pandas                  1.2.0
numpy                   1.19.4
faiss-cpu               1.7.0
scipy                   1.5.4
tqdm                    4.55.0
```

### Executing program

* To check data format see the data inside example_data 

* A running example
```
python3 run.py -results ./embeddings/ -eval_results ./eval_results/ -data ./example_data/amz_beauty.train -test_data ./example_data/amz_beauty.test -d 100 -beta 0.9 -k 1 2 3 10 20 -epochs 10 -batch_size 512 -lr 0.01 -reg 0.001 0.001 0.001 -sampling 'non-uniform' -seed 1
```

* `-results` (str): address for storing the trained user and item embeddings
* `-eval_results` (str): address for storing the evaluation results
* `-data` (str): training data address
* `-test_data` (str): testing data address
* `-d` (int): the number of features for users and items embeddings
* `-beta` [(float)]: share of unobserved feedback within the overall negative feedback
* `-k` [(int)]: number of top-k evaluation ranking
* `-epochs` (int): number of training epochs
* `-batch_size` (int): number of data within each batch
* `-lr` (float): learning rate
* `-reg` [(float)]: regularization parameters for user, positive and negative item
* `-sampling` [(str)]: list of negative item sampling modes, `uniform` and/or `non-uniform`
* `-seed` (int): random number generator seed

## Authors


* SeanThePlug (https://github.com/seantheplug)
* Marcel Kurovski (https://github.com/mkurovski)

## Acknowledgments

This project is the pytorch implementation of MCBPR and it is built by modifying [Python implementation of MCBPR](https://github.com/mkurovski/multi_channel_bpr#bayesian-personalized-ranking-with-multi-channel-user-feedback---loni-et-al-2016)
