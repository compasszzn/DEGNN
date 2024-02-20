import torch
import argparse
import os
import numpy as np
import torch.multiprocessing as mp


import random


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Run parameters
    parser.add_argument('--epochs', type=int, default=5000,
                        help='number of epochs')

    parser.add_argument('--lmax_h', type=int, default=2,
                        help='max degree of hidden rep')
    parser.add_argument('--lmax_attr', type=int, default=3,
                        help='max degree of geometric attribute embedding')
    parser.add_argument('--norm', type=str, default="batch",
                        help='Normalisation type [instance, batch]')
    parser.add_argument('--weight_decay', type=float, default=1e-12,
                        help='weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Num workers in dataloader')
    parser.add_argument('--time_exp', type=bool, default=False,
                        help='Flag for timing experiment')
    parser.add_argument('--test_interval', type=int, default=5,
                        help='Test every test_interval epochs')
    
    parser.add_argument('--nbody_name', type=str, default="nbody_small")
    # Model parameters
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size. Does not scale with number of gpus.')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='learning rate')
    parser.add_argument('--model', type=str, default="DEGNN",
                        help='Model name')
    parser.add_argument('--dataset', type=str, default="nbody_charged_5_4_3",
                        help='Name of dataset [nbody_charged_5_5_5,nbody_charged_5_4_4,nbody_charged_4_4_4,nbody_charged_5_4_3]')
    parser.add_argument('--dataset_size', type=int, default=4200,
                        help='Data set')
    parser.add_argument('--dataset_segment', type=str, default='1,10,10',
                        help='Data set')
    parser.add_argument('--gpus_num', type=str, default="0",
                        help='Model name')
    parser.add_argument('--pool_method', type=str, default="self_attn",
                        help='pool_method')

 
    
    parser.add_argument('--hidden_features', type=int, default=64,
                        help='max degree of hidden rep')
    parser.add_argument('--layers', type=int, default=4,
                        help='Number of message passing layers')
    # Parallel computing stuff
    parser.add_argument('-g', '--gpus', default=True, type=bool,
                        help='number of gpus to use (assumes all are on one node)')
    parser.add_argument('--seed', type=int, default=-1, metavar='N',
                        help='the rand seed')

    args = parser.parse_args()

    if args.seed < 0:
        seed = random.randint(0,1000)
    else:
        seed = args.seed
    torch.manual_seed(seed)


    from nbody.trainnbody import train
    train(args)

