from __future__ import print_function

from my_args_parser import get_args
import os

import torch
import torch.multiprocessing as mp

import my_optim
from envs import create_atari_env
from model import ActorCritic
from test import test
from train import train
from logger import Logger


if __name__ == '__main__':
    args = get_args()

    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    mp.set_start_method("spawn")

    logger = Logger(args.log_dir, dir_pre=args.log_dir_pre)

    torch.manual_seed(args.seed)
    env = create_atari_env(args.env_name)
    shared_model = ActorCritic(env.observation_space.shape[0], env.action_space)
    shared_model.share_memory()

    if args.no_shared:
        optimizer = None
    else:
        optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=args.lr)
        optimizer.share_memory()

    processes = []

    counter = mp.Value('i', 0)
    lock = mp.Lock()

    p = mp.Process(target=test,
                   args=(args.num_processes, args, shared_model, counter, logger))
    p.start()
    processes.append(p)

    for rank in range(0, args.num_processes):
        p = mp.Process(target=train,
                       args=(rank, args, shared_model, counter, lock, logger, optimizer))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
