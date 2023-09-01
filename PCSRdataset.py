import os
import glob
import torch
import psutil
import numpy as np
import pandas as pd
import multiprocessing
from pyntcloud import PyntCloud
from torch.utils.data import Dataset


def process(arg):
    path, pqs, K, output_path, core_id = arg
    p = psutil.Process()
    p.cpu_affinity([core_id])
    ori_pc = PyntCloud.from_file(path)
    ori_points = ori_pc.points.values[:,:3].astype(int)
    if pqs > 2:
        ori_points = np.round(ori_points/(pqs/2)+1e-6).astype(int) # downsample 
        ori_points = np.unique(ori_points, axis=0) # remove duplicated points
        dist_points = np.round(ori_points/2+1e-6).astype(int) # downsample 2
        dist_points = np.unique(dist_points, axis=0) # remove duplicated points
    else: # pqs <= 2
        dist_points = np.round(ori_points/pqs+1e-6).astype(int) # downsample <=2
        dist_points = np.unique(dist_points, axis=0) # remove duplicated points
    res_m = np.min(ori_points, axis=0).astype(int)
    dres_m = np.min(dist_points, axis=0).astype(int)
    res = (np.max(ori_points, axis=0)-res_m+3).astype(int)
    dres = (np.max(dist_points, axis=0)-dres_m+2*K+1).astype(int) 
    ori_voxels = np.zeros(res, dtype=np.int8)
    down_voxels = np.zeros(dres, dtype=np.int8)
    for i in range(len(ori_points)):
        ori_voxels[ori_points[i][0]+1-res_m[0], 
                   ori_points[i][1]+1-res_m[1], 
                   ori_points[i][2]+1-res_m[2]] = 1
    for i in range(len(dist_points)):
        down_voxels[dist_points[i][0]+K-dres_m[0], 
                    dist_points[i][1]+K-dres_m[1], 
                    dist_points[i][2]+K-dres_m[2]] = 1 
    neighs = np.zeros((len(dist_points), (2*K+1)**3-1))
    childs = np.zeros((len(dist_points), 8))
    for i in range(len(dist_points)):
        [x, y, z] = [dist_points[i][j] for j in range(3)]
        tmp_neighs = down_voxels[x-dres_m[0]:x+2*K+1-dres_m[0],
                                 y-dres_m[1]:y+2*K+1-dres_m[1],
                                 z-dres_m[2]:z+2*K+1-dres_m[2]].reshape(-1)
        neighs[i] = np.delete(tmp_neighs, (2*K+1)**3//2).reshape(-1) # remove the occupied center
        childs[i] = ori_voxels[2*x-res_m[0]:2*x+2-res_m[0],
                                2*y-res_m[1]:2*y+2-res_m[1],
                                2*z-res_m[2]:2*z+2-res_m[2]].reshape(-1)
    cloud = PyntCloud(pd.DataFrame(data=dist_points.astype(float), columns=['x', 'y', 'z']))
    name = os.path.splitext(os.path.split(path)[1])[0]
    if not os.path.exists('{}/{}_base.ply'.format(output_path, name)):
        cloud.to_file('{}/{}_base.ply'.format(output_path, name), as_text=True)
    return neighs, childs


class PCSRDataset(Dataset):
    def __init__(self, args, status='train'):
        self.K = args.K
        self.pqs = args.pqs # 
        self.status = status
        self.output_path = args.output_path
        if '.ply' in args.dataset: # static pc
            self.paths = ['data/{}'.format(args.dataset)]
        else: # dynamic pc
            self.paths = glob.glob('data/{}/*.ply'.format(args.dataset))
            self.paths.sort()
        if args.dataset in ['basketball_player_vox11', 'dancer_vox11']:
            self.paths = self.paths[:64] # V-PCC CTC
        if not args.evaluate:
            self.paths = [self.paths[i] for i in range(0, len(self.paths), args.frame_sampling_rate)]
        if self.status == 'train':
            self.neighs = [None] * len(self.paths)
            self.childs = [None] * len(self.paths)
            num_cores = psutil.cpu_count(logical=False)
            if num_cores>len(self.paths): num_cores = len(self.paths)
            zip_args = list(zip(self.paths, 
                                [self.pqs]*len(self.paths),
                                [self.K]*len(self.paths),
                                [self.output_path]*len(self.paths),
                                range(num_cores)
                                ))
            pool = multiprocessing.Pool(processes=num_cores)
            neighschilds = pool.map(process, zip_args)
            self.neighs = [data[0] for data in neighschilds]
            self.childs = [data[1] for data in neighschilds]
            pool.close()
            pool.join()
            self.neighs = torch.from_numpy(np.concatenate(self.neighs)).to(torch.float)
            self.childs = torch.from_numpy(np.concatenate(self.childs)).to(torch.float)

    def __len__(self):
        return len(self.neighs) if self.status == 'train' else len(self.paths)
    
    def __getitem__(self, idx):  
        if self.status == 'train':
            neighs = self.neighs[idx]
            childs = self.childs[idx]
            return neighs, childs
        else:
            ori_pc = PyntCloud.from_file(self.paths[idx])
            ori_points = ori_pc.points.values[:,:3].astype(int)
            if self.pqs > 2:
                ori_points = np.round(ori_points/(self.pqs/2)+1e-6).astype(int)
                ori_points = np.unique(ori_points, axis=0) 
                dist_points = np.round(ori_points/2+1e-6).astype(int) 
                dist_points = np.unique(dist_points, axis=0)
            else: # self.pqs <= 2
                dist_points = np.round(ori_points/self.pqs+1e-6).astype(int) 
                dist_points = np.unique(dist_points, axis=0)
            res_m = np.min(ori_points, axis=0).astype(int)
            dres_m = np.min(dist_points, axis=0).astype(int)
            res = (np.max(ori_points, axis=0)-res_m+3).astype(int)
            dres = (np.max(dist_points, axis=0)-dres_m+2*self.K+1).astype(int) 
            ori_voxels = np.zeros(res, dtype=np.int8)
            down_voxels = np.zeros(dres, dtype=np.int8)
            for i in range(len(ori_points)):
                ori_voxels[ori_points[i][0]+1-res_m[0], 
                           ori_points[i][1]+1-res_m[1], 
                           ori_points[i][2]+1-res_m[2]] = 1
            for i in range(len(dist_points)):
                down_voxels[dist_points[i][0]+self.K-dres_m[0], 
                            dist_points[i][1]+self.K-dres_m[1], 
                            dist_points[i][2]+self.K-dres_m[2]] = 1 
            neighs = np.zeros((len(dist_points), (2*self.K+1)**3-1))
            childs = np.zeros((len(dist_points), 8))
            for i in range(len(dist_points)):
                [x, y, z] = [dist_points[i][j] for j in range(3)]
                tmp_neighs = down_voxels[x-dres_m[0]:x+2*self.K+1-dres_m[0],
                                         y-dres_m[1]:y+2*self.K+1-dres_m[1],
                                         z-dres_m[2]:z+2*self.K+1-dres_m[2]].reshape(-1)
                neighs[i] = np.delete(tmp_neighs, (2*self.K+1)**3//2).reshape(-1) 
                childs[i] = ori_voxels[2*x-res_m[0]:2*x+2-res_m[0],
                                       2*y-res_m[1]:2*y+2-res_m[1],
                                       2*z-res_m[2]:2*z+2-res_m[2]].reshape(-1)
            cloud = PyntCloud(pd.DataFrame(data=dist_points.astype(float), columns=['x', 'y', 'z']))
            name = os.path.splitext(os.path.split(self.paths[idx])[1])[0]
            if not os.path.exists('{}/{}_base.ply'.format(self.output_path, name)):
                cloud.to_file('{}/{}_base.ply'.format(self.output_path, name), as_text=True)
            return neighs.astype(np.float32), (childs.astype(np.float32), dist_points, name)
    

class PCSRfDataset(Dataset):
    def __init__(self, args, nscale):
        self.K = args.K
        self.pqs = args.pqs
        self.nscale = nscale
        self.output_path = args.output_path
        if '.ply' in args.dataset: # static pc
            self.paths = ['data/{}'.format(args.dataset)]
        else: # dynamic pc
            self.paths = glob.glob('data/{}/*.ply'.format(args.dataset))
            self.paths.sort()
        if args.dataset in ['basketball_player_vox11', 'dancer_vox11']:
            self.paths = self.paths[:64] # V-PCC CTC
        if not args.evaluate:
            self.paths = [self.paths[i] for i in range(0, len(self.paths), args.frame_sampling_rate)]

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):   
        ori_pc = PyntCloud.from_file(self.paths[idx])
        ori_points = ori_pc.points.values[:,:3].astype(int)
        if self.nscale:
            ori_points = np.round(ori_points/2**self.nscale+1e-6).astype(int) 
            ori_points = np.unique(ori_points, axis=0)
        name = os.path.splitext(os.path.split(self.paths[idx])[1])[0]
        path = '{}/{}_preinv_output_scale{}.ply'.format(self.output_path, name, self.nscale+1)
        dist_pc = PyntCloud.from_file(path)
        # os.system('rm -r ' + '{}/{}_preinv_output_scale{}.ply'.format(self.output_path, name, self.nscale+1)) #
        dist_points = dist_pc.points.values[:,:3].astype(int)
        res_m = np.min(ori_points, axis=0).astype(int)
        dres_m = np.min(dist_points, axis=0).astype(int)
        res = (np.max(ori_points, axis=0)-res_m+3).astype(int)
        dres = (np.max(dist_points, axis=0)-dres_m+2*self.K+1).astype(int) 
        ori_voxels = np.zeros(res, dtype=np.int8)
        down_voxels = np.zeros(dres, dtype=np.int8)
        for i in range(len(ori_points)):
            ori_voxels[ori_points[i][0]+1-res_m[0], 
                       ori_points[i][1]+1-res_m[1], 
                       ori_points[i][2]+1-res_m[2]] = 1
        for i in range(len(dist_points)):
            down_voxels[dist_points[i][0]+self.K-dres_m[0], 
                        dist_points[i][1]+self.K-dres_m[1], 
                        dist_points[i][2]+self.K-dres_m[2]] = 1 
        neighs = np.zeros((len(dist_points), (2*self.K+1)**3-1))
        childs = np.zeros((len(dist_points), 8))
        for i in range(len(dist_points)):
            [x, y, z] = [dist_points[i][j] for j in range(3)]
            tmp_neighs = down_voxels[x-dres_m[0]:x+2*self.K+1-dres_m[0],
                                     y-dres_m[1]:y+2*self.K+1-dres_m[1],
                                     z-dres_m[2]:z+2*self.K+1-dres_m[2]].reshape(-1)
            neighs[i] = np.delete(tmp_neighs, (2*self.K+1)**3//2).reshape(-1) 
            tmp_childs = ori_voxels[2*x-res_m[0]:2*x+2-res_m[0],
                                    2*y-res_m[1]:2*y+2-res_m[1],
                                    2*z-res_m[2]:2*z+2-res_m[2]].reshape(-1)
            if len(tmp_childs)==8: childs[i] = tmp_childs # 0, 2, 4 cases -> all zeros ...
        return neighs.astype(np.float32), (childs.astype(np.float32), dist_points, name)
