import os
import glob
import fpzip
import torch
import random
import logger
import datetime
import subprocess
import numpy as np
from PCSRloss import PCSRLoss
from ignite.engine import Events
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from PCSRmodel import PCSRModelSiren
from PCSRdataset import PCSRDataset
from PCSRperformance import PCSRPerformance
from modified_ignite_engine import create_supervised_validator, create_supervised_trainer
from bitstream import *


metrics_printed = ['mAcc']
def writer_add_scalar(writer, status, dataset, scalars, iter):
    for metric in metrics_printed:
        writer.add_scalar('{}/{}/{}'.format(status, dataset, metric), scalars[metric], iter)

        
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    activation = torch.nn.ReLU() if args.activation == 'ReLU' else None # None -> default Sine
    model = PCSRModelSiren(dim_in=(2*args.D+1)**3-1, 
                           dim_hidden=args.base_channel, 
                           num_layers=args.num_layers,
                           activation=activation)
    model = model.to(device)
    logger.log.info(model)
    for param_tensor in model.state_dict():
        logger.log.info('{}\t {}'.format(param_tensor, model.state_dict()[param_tensor].size()))
    total_params = sum(p.numel() for p in model.parameters())
    logger.log.info('total_params: {}'.format(total_params))

    nscale = np.ceil(np.log2(args.pqs/2)).astype(int) # nscale
    
    train_dataset = PCSRDataset(args, status='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                num_workers=32, pin_memory=True)  
    val_loader = DataLoader(train_dataset, batch_size=5*args.batch_size, 
                            num_workers=32, pin_memory=True)    
    
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) 
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay)
    loss_func = PCSRLoss()
    trainer = create_supervised_trainer(model, optimizer, loss_func, device=device)
    evaluator = create_supervised_validator(model, metrics={'PCSR_performance': PCSRPerformance()}, device=device)
    current_time = datetime.datetime.now().strftime('%I:%M%p on %B %d, %Y')
    writer = SummaryWriter(log_dir='runs/{}-{}'.format(args.f_str, current_time))
    global best_val_criterion, best_epoch
    best_val_criterion, best_epoch = 0., -1 
    @trainer.on(Events.ITERATION_COMPLETED)
    def iter_event_function(engine):
        writer.add_scalar('train/loss', engine.state.output, engine.state.iteration)
    @trainer.on(Events.EPOCH_COMPLETED)
    def epoch_event_function(engine):
        scheduler.step()
        if engine.state.epoch % 5 == 0: # True # 
            evaluator.run(val_loader)
            performance = evaluator.state.metrics
            writer_add_scalar(writer, 'val', args.dataset, performance, engine.state.epoch)
            val_criterion = performance['mAcc']
            global best_val_criterion, best_epoch
            if val_criterion > best_val_criterion: 
                torch.save(model.state_dict(), args.trained_model_file)
                best_val_criterion = val_criterion
                best_epoch = engine.state.epoch
                logger.log.info('Save current best val model (mAcc: {:.5f}) @epoch {}'
                                .format(best_val_criterion, best_epoch))
            else:
                logger.log.info('Model is not updated (mAcc: {:.5f}) @epoch: {}'
                                .format(val_criterion, engine.state.epoch))           
    @trainer.on(Events.COMPLETED)
    def final_testing_results(engine):
        writer.close ()  
        logger.log.info('best epoch: {}'.format(best_epoch))
        model.load_state_dict(torch.load(args.trained_model_file))
        params = None
        for param_tensor in model.state_dict(): #
            if params is None:
                params = model.state_dict()[param_tensor].data.to('cpu').numpy().reshape(-1)
            else:
                params = np.concatenate((params, model.state_dict()[param_tensor].data.to('cpu').numpy().reshape(-1)))
        compressed_bytes = fpzip.compress(params, precision=args.precision, order='C')
        f = open(args.trained_model_file+'_cbytes.bin', 'wb')
        f.write(compressed_bytes)
        f.close()
        logger.log.info('network parameter bitstream: {} bits'.format(8*os.path.getsize(args.trained_model_file+'_cbytes.bin')))
    trainer.run(train_loader, max_epochs=args.epochs)


def sh(cmd, input=''): # Solve the issue that logging cannot get the stdout of os.system
    rst = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, 
                         stderr=subprocess.PIPE, input=input.encode('utf-8'))
    assert rst.returncode == 0, rst.stderr.decode('utf-8')
    return rst.stdout.decode('utf-8')

def encode_pc(args):
    # process org point cloud to base point cloud
    dynamic_pc = False
    if '.ply' in args.dataset: # static pc
        dynamic_pc = False
    else:  # dynamic pc
        args.evaluate = True
        dynamic_pc = True

    if dynamic_pc:
        test_dataset = PCSRDataset(args, status='test')
        test_loader = DataLoader(test_dataset, num_workers=32, pin_memory=True)   
        for idx, (neighs, (childs, dist_points, name)) in enumerate(test_loader):
            print()

    activation, D, base_channel, num_layers, pqs = args.activation, args.D, args.base_channel, args.num_layers, args.pqs
    trained_model_file = 'checkpoints/' + args.f_str + '_cbytes.bin'
    
    # compress base point cloud
    list_basefile = glob.glob('{}/*base.ply'.format(os.path.abspath(args.output_path)))
    list_basefile.sort()
    bin_sizes = []
    for base_pc in list_basefile:
        bin = base_pc[:-4]+'.bin'
        enc = base_pc[:-4]+'_enc.ply'
        tmc3 = 'tmc3v22' # 'tmc3', or other base compressors
        cmd_encode = './'+tmc3+' --config=cfg_base/encoder.cfg --uncompressedDataPath='+base_pc+' --reconstructedDataPath='+enc+' --compressedStreamPath='+bin+' --disableAttributeCoding=1'
        r = sh(cmd_encode) 
        logger.log.info(r)

        bin_size = os.path.getsize(bin)
        bin_sizes.append(bin_size)
    
    # write header 
    model_size = os.path.getsize(trained_model_file)
    fs = '{}_bc{}_nl{}_D{}_p{}_lr{}_fsr{}_bs{}_e{}_{}_ppqs{}_pqs{}.bs'
    bs_str = fs.format(args.dataset, args.base_channel, args.num_layers, args.D, 
                           args.precision, args.lr, args.frame_sampling_rate, args.batch_size, args.epochs, 
                           args.activation, args.ppqs, args.pqs)
    ppqs = args.ppqs
    bitstream = os.path.join(args.outputstream, bs_str)
    with open(bitstream, 'wb') as fout:
        write_ints(fout, (len(activation),))
        write_uchars(fout, activation.encode('ascii'))
        write_floats(fout, (ppqs,))
        write_ints(fout, (D, base_channel, num_layers, pqs, model_size, len(bin_sizes)))
        write_ints(fout, bin_sizes)

    # write file data
    subprocess.call(f'cat {trained_model_file} >> {bitstream}', shell=True)

    list_basefile = glob.glob('{}/*base.bin'.format(os.path.abspath(args.output_path)))
    list_basefile.sort()
    for base_pc in list_basefile:
        bin = base_pc[:-4]+'.bin'
        subprocess.call(f'cat {bin} >> {bitstream}', shell=True)

    return


if __name__ == '__main__':
    parser = ArgumentParser(description='LSRN-PCGC')
    parser.add_argument('--seed', type=int, default=19920517)
    parser.add_argument('-rn', '--randomness', action='store_true',
                        help='Allow randomness during training?')
    parser.add_argument('-pc', '--pointcloud', default='redandblack', type=str,
                        help='point cloud full path or point cloud seq. full dir')
    parser.add_argument('-os', '--outputstream', default='outputs', type=str,
                        help='output bitstream file dir full path')
    parser.add_argument('-ppqs', '--ppqs', type=float, default=1,
                        help='pre pqs (default: 1), not excute pre pqs')
    parser.add_argument('-pqs', '--pqs', type=int, default=2,
                        help=' (default: 2)')
    parser.add_argument('-model', '--model', default='LSRN', type=str,
                        help='LSRN')
    parser.add_argument('-act', '--activation', default='Sine', type=str,
                        help='Sine')
    parser.add_argument('-bc', '--base_channel', type=int, default=32,
                        help='base channel (default: 32)')
    parser.add_argument('-nl', '--num_layers', type=int, default=1,
                        help='Number of layers (default: 1)')
    parser.add_argument('-D', '--D', type=int, default=2,
                        help='neighbors (2D+1)^3-1')
    parser.add_argument('-prec', '--precision', type=int, default=16,
                        help=' (default: 16)')
    parser.add_argument('-fsr', '--frame_sampling_rate', type=int, default=10,
                        help='#frame sampling rate (default: 10)')
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
    args.dataset = os.path.basename(args.pointcloud)
    if 'vox12' in args.dataset:
        args.vox = 12
    else:
        args.vox = 11 if 'vox11' in args.dataset else 10 #
    if args.lr_decay == 1 or args.epochs < 3:  # no lr decay
        args.lr_decay_step = args.epochs
    else:  # 
        args.lr_decay_step = int(args.epochs/(1+np.log(args.overall_lr_decay)/np.log(args.lr_decay)))
    if not args.randomness:
        torch.manual_seed(args.seed)  #
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)
        random.seed(args.seed)
    torch.utils.backcompat.broadcast_warning.enabled = True
    fs = '{}_{}_bc{}_nl{}_D{}_p{}_lr{}_fsr{}_bs{}_e{}_{}_ppqs{}_pqs{}'
    args.f_str = fs.format(args.model, args.activation, args.base_channel, args.num_layers, args.D, 
                           args.precision, args.lr, args.frame_sampling_rate, args.batch_size, args.epochs, 
                           args.dataset, args.ppqs, args.pqs)
    if not os.path.exists('checkpoints'): os.makedirs('checkpoints')
    args.trained_model_file = 'checkpoints/' + args.f_str
    args.output_path = 'outputs/' + args.f_str
    if not os.path.exists(args.output_path): os.makedirs(args.output_path)

    logger.create_logger('logs', args.f_str)
    logger.log.info(args)
    train(args)
    encode_pc(args)
    