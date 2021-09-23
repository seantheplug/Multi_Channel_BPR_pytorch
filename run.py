
from datetime import datetime
import os
import pickle
import sys
import torch.utils.data as data



from cli import parse_args
from model import MCBPR
from utils import load_rating_data, get_channels
from data_util import MCBPRData
import evaluation
from tqdm import tqdm






def main(args):
    """Main entry point allowing external calls

    Args:
      args ([str]): command line parameter list
    """
    args = parse_args(args)
    print(args)


    ratings, m, n, user_hasher, item_hasher = load_rating_data(args.data_path)
    channels = get_channels(ratings)

    reg_params = {'u': args.reg_param_list[0],
                  'i': args.reg_param_list[0],
                  'j': args.reg_param_list[0]}


    train_inter = ratings


    print('start training')
    for neg_sampling_mode in args.neg_sampling_modes:

        train_dataset = MCBPRData(rd_seed=args.rd_seed,channels=channels, n_user=m, n_item=n)
        train_dataset.set_data(train_inter, args.beta_list[0])
        train_dataset.ng_sample(neg_sampling_mode)
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)


        for lr in tqdm(args.lr_list):
            for optimizer_params in tqdm(args.optim_list):
                model = MCBPR(d=args.d, beta=args.beta_list[0],
                                        rd_seed=args.rd_seed,
                                        channels=channels, n_user=m, n_item=n)
                model.fit(lr=lr, reg_params=reg_params, n_epochs=args.n_epochs,train_loader = train_loader, optimizer_params = optimizer_params)

                res_filename = datetime.strftime(datetime.now(), '%Y%m%d_bpr_multi_')
                res_filename = res_filename +str(optimizer_params)+'_'+str(neg_sampling_mode) + '_lr_' + str(lr) +'_reg_'+str(reg_params['u'])+ '.txt'
                res_filepath = os.path.join(args.results_path, res_filename)
                model.save_user_item_embedding(user_hasher, item_hasher, res_filepath)
                print("finish saving embedding")
                total_precision, total_rec, total_ndcg = model.evaluate(args.test_data_path, args.k, user_hasher, item_hasher) 

                print("Start writing file...")
                with open(args.eval_results_path+res_filename, 'w+') as fw:
                    fw.writelines(['=================================\n',
                        'File: ',
                        str(res_filename),
                        '\n Data path: ',
                        str(args.data_path),
                        '\n Test Data path: ',
                        str(args.test_data_path),
                        '\n evaluated users: ',
                        str(m),
                        '\n--------------------------------'])
                    for i in range(len(args.k)):
                        fw.writelines(['\n Precision@{}: '.format(args.k[i]),
                            str(total_precision[i]),
                            '\n recall@{}: '.format(args.k[i]),
                            str(total_rec[i]),
                            '\n NDCG@{}: '.format(args.k[i]),
                            str(total_ndcg[i]),
                            '\n--------------------------------'])
                print('Finished!')



def run():
    """Entry point for console_scripts
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
