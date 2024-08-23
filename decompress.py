import os
import glob
import tempfile
import fpzip
import torch
import logger
import subprocess
import numpy as np

from argparse import ArgumentParser
from PCSRmodel import PCSRModelSiren
from PCSRdataset import process2neighs
from bitstream import *

def sh(cmd, input=''): # Solve the issue that logging cannot get the stdout of os.system
    rst = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, 
                         stderr=subprocess.PIPE, input=input.encode('utf-8'))
    assert rst.returncode == 0, rst.stderr.decode('utf-8')
    return rst.stdout.decode('utf-8')

def inference(model, x, points, device):
    x = torch.from_numpy(x).to(device)
    model.eval()
    with torch.no_grad():
        y_pred = model(x)

    idx1 = y_pred.max(dim=1)[0]>=0.5
    idx2 = y_pred.max(dim=1)[0]<0.5
    y_pred[idx1] = (y_pred[idx1]>=0.5).float() 
    y_pred[idx2] = (y_pred[idx2]>=y_pred.max()).float() #
    y_pred = y_pred.reshape(-1).to('cpu').numpy()
    dxyz = np.asarray([[[0,0,0],[0,0,1],[0,1,0],[0,1,1],
                        [1,0,0],[1,0,1],[1,1,0],[1,1,1]]])-1        
    all_points = 2*np.repeat(points, 8, axis=0)+np.repeat(dxyz, len(points), axis=0).reshape(-1,3)
    points = all_points[y_pred==1]
    points = np.unique(points, axis=0) #    
    return points

def gen_tmpfile_name():
    tmpfile_name = ''
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as temp_file:
        tmpfile_name = temp_file.name
    return tmpfile_name

def decompress(args):   
    # 
    ret_value = ''
    trained_model_file = gen_tmpfile_name()
    
    bin_filenames = []
    len_bin_sizes = 0
    with open(args.inputstream, 'rb') as fin:  
        act_sz = read_ints(fin, 1)[0]
        activation = bytes(read_uchars(fin, act_sz)).decode('ascii')
        ppqs = read_floats(fin, 1)[0]
        D, base_channel, num_layers, pqs, model_size, len_bin_sizes = read_ints(fin, 6) 
        bin_sizes = read_ints(fin, len_bin_sizes) 

        with open(trained_model_file, 'wb') as fout:
            fout.write(fin.read(model_size))

        for i in range(len_bin_sizes):
            bin_size = bin_sizes[i]
            bin_file = gen_tmpfile_name()
            bin_filenames.append(bin_file)
            with open(bin_file, 'wb') as fout:
                fout.write(fin.read(bin_size))

    logger.log.info('network parameter bitstream: {} bits'.format(8*os.path.getsize(trained_model_file)))
    
    # 1. load compressed model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    activation = torch.nn.ReLU() if activation == 'ReLU' else None # None -> default Sine
    model = PCSRModelSiren(dim_in=(2*D+1)**3-1, 
                           dim_hidden=base_channel, 
                           num_layers=num_layers,
                           activation=activation)
    model = model.to(device)

    with open(trained_model_file,'rb') as f: compressed_bytes = f.read()
    params = fpzip.decompress(compressed_bytes, order='C')[0][0][0]
    k = 0
    state_dict = {}
    for param_tensor in model.state_dict():
        values = params[k:k+model.state_dict()[param_tensor].numel()].reshape(model.state_dict()[param_tensor].size())
        state_dict[param_tensor] = torch.from_numpy(values)
        k = k + model.state_dict()[param_tensor].numel()
    model.load_state_dict(state_dict)    

    for i in range(len_bin_sizes):
        bin_file = bin_filenames[i]
        dec = gen_tmpfile_name()
        # 2. decode base point cloud
        tmc3 = 'tmc3v22' # 'tmc3', or other base compressors
        cmd_decode = './'+tmc3+' --config=cfg_base/decoder.cfg --compressedStreamPath='+bin_file+' --reconstructedDataPath='+dec
        r = sh(cmd_decode) 
        logger.log.info(r)    

        # 3. load base point, process to neighs
        points = load_ply(dec)
        neighs = process2neighs(points, D)

        # 4. inference and upscale
        nscale = np.ceil(np.log2(pqs/2)).astype(int) # nscale
        points = inference(model, neighs, points, device)
        # if nscale:
        points = np.round(points*(pqs/2)).astype(int)
        if False:
            neighs = process2neighs(points, D)
            points = inference(model, neighs, points, device)
            scale = pqs / 4
            if scale > 0:
                points = np.round(points*scale+1e-6).astype(int)
                points = np.unique(points, axis=0) 

        # 5. execute pre pqs
        if ppqs > 1.0:
            points = np.round(points*ppqs+1e-6).astype(int)
        # 6. save points to ply
        rec_file = os.path.splitext(os.path.basename(args.inputstream))[0]
        if len_bin_sizes <= 1:  # static pc
            rec_file = rec_file  + '.ply'
            ret_value = os.path.join(args.reconstruct, rec_file)
        else:                   # dynamic pc
            ret_value = os.path.join(args.reconstruct, rec_file)
            os.makedirs(ret_value, exist_ok=True)
            rec_file = rec_file  + f'/frame_{i:0>5}.ply'
        
        rec_file = os.path.join(args.reconstruct, rec_file)
        save_ply(points, rec_file) 

        #7 . clear tmp file 
        subprocess.call(f'rm -f {dec}', shell=True)
        subprocess.call(f'rm -f {bin_file}', shell=True)

    subprocess.call(f'rm -f {trained_model_file}', shell=True)

    return ret_value

if __name__ == '__main__':
    parser = ArgumentParser(description='LSRN-PCGC')
    parser.add_argument('-is', '--inputstream', default='compressed.bs', type=str,
                        help='bit stream file full path')
    parser.add_argument('-rec', '--reconstruct', default='outputs/rec', type=str,
                        help='reconstruct point cloud dir full path')    
    parser.add_argument('-org', '--pointcloud', default='', type=str,
                        help='original point cloud full path')       

    args = parser.parse_args()

    torch.utils.backcompat.broadcast_warning.enabled = True

    fs = 'decompress_{}.log'
    args.f_str = fs.format(os.path.splitext(os.path.basename(args.inputstream))[0])

    logger.create_logger('logs', args.f_str)
    logger.log.info(args)

    rec = decompress(args)

    if (args.pointcloud):
        vox = 10
        if 'vox12' in args.pointcloud:
            vox = 12
        else:
            vox = 11 if 'vox11' in args.pointcloud else 10 #        
        res = 2**vox-1
        if '.ply' in rec:   # static pc
            cmd = './pc_error -a '+'{}'.format(args.pointcloud)+' -b '+'{}'.format(rec)+' -c 1 -l 1 -d 1 --nbThreads=10 --dropdups=2  --neighborsProc=1 -r '+str(res)
            print(cmd)
            r = sh(cmd) 
            logger.log.info(r)
        else :              # dynamic pc
            org_files = glob.glob('{}/*.ply'.format(os.path.abspath(args.pointcloud)))
            org_files.sort()
            rec_files = glob.glob('{}/*.ply'.format(os.path.abspath(rec)))
            rec_files.sort()    
            for i in range(len(rec_files)):
                org_file = org_files[i]
                rec_file = rec_files[i]
                cmd = './pc_error -a '+'{}'.format(org_file)+' -b '+'{}'.format(rec_file)+' -c 1 -l 1 -d 1 --nbThreads=10 --dropdups=2  --neighborsProc=1 -r '+str(res)
                r = sh(cmd) 
                logger.log.info(r)                