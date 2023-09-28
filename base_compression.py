import os
import torch
import random
import logger
import subprocess
import numpy as np
from argparse import ArgumentParser

def sh(cmd, input=""): # https://stackoverflow.com/a/56215593/3865166 to solve the issue that cannot get the stdout of os.system, although a naive method is to expand the cmd with '> tmp.txt' ...
    rst = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, input=input.encode("utf-8"))
    assert rst.returncode == 0, rst.stderr.decode("utf-8")
    return rst.stdout.decode("utf-8")


if __name__ == "__main__":
    parser = ArgumentParser(description='LSRN-PCGC')
    parser.add_argument('--seed', type=int, default=19920517)
    parser.add_argument('-rn', '--randomness', action='store_true',
                        help='Allow randomness during training?')
    parser.add_argument('-pc', '--dataset', default='redandblack', type=str,
                        help='point cloud path or point cloud seq. dir')
    parser.add_argument('-pqs', '--pqs', type=int, default=2,
                        help=' (default: 2)')
    parser.add_argument('-model', '--model', default='LSRN', type=str,
                        help='LSRN')
    parser.add_argument('-act', '--activation', default='Sine', type=str,
                        help='Sine')
    parser.add_argument('-bc', '--base_channel', type=int, default=16,
                        help='base channel (default: 16)')
    parser.add_argument('-nl', '--num_layers', type=int, default=1,
                        help='Number of layers (default: 1)')
    parser.add_argument('-K', '--K', type=int, default=2,
                        help='neighbors (2K+1)^3-1')
    parser.add_argument('-precision', '--precision', type=int, default=16,
                        help=' (default: 16)')
    parser.add_argument('-fsr', '--frame_sampling_rate', type=int, default=1,
                        help='#frame sampling rate (default: 1)')
    parser.add_argument('-eval', '--evaluate', action='store_true',
                        help='Evaluate only?')
    parser.add_argument('-lr', '--lr', type=float, default=1e-3,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('-bs', '--batch_size', type=int, default=2048,
                        help='batch size (default: 2048)')
    parser.add_argument('-e', '--epochs', type=int, default=150,
                        help='number of epochs to train (default: 150)')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('-lrd', '--lr_decay', type=float, default=0.1,
                        help='lr decay (default: 0.1)')
    parser.add_argument('-olrd', '--overall_lr_decay', type=float, default=0.01,
                        help='overall lr decay (default: 0.01)')
    args = parser.parse_args()
    args.dir = '../' # 
    if not args.randomness:
        torch.manual_seed(args.seed)  #
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)
        random.seed(args.seed)
    torch.utils.backcompat.broadcast_warning.enabled = True

    fs = '{}_{}_bc{}_nl{}_K{}_p{}_lr{}_fsr{}_bs{}_e{}_{}_pqs{}'
    args.f_str = fs.format(args.model, args.activation, args.base_channel, args.num_layers, args.K, 
                           args.precision, args.lr, args.frame_sampling_rate, args.batch_size, args.epochs, 
                           args.dataset, args.pqs)
    args.output_path = args.dir + 'outputs/' + args.f_str
    logger.create_logger(args.dir + 'logs_base_compression_by_OctAttention', args.f_str, False)
    logger.log.info(args)
    cmd = 'cd OctAttention; python encoder.py {}; cd ..'.format(args.output_path)
    r = sh(cmd) 
    logger.log.info(r)
    