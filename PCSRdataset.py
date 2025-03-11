import os
import glob
import torch
import psutil
import numpy as np
import pandas as pd
import multiprocessing
from pyntcloud import PyntCloud
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

cls_masks = {
    0: [0, 0, 0, 0, 0, 0, 0, 0],  # 类型0，
    1: [0, 0, 0, 1, 0, 0, 0, 1],  # 类型1，2位有效
    2: [0, 0, 0, 0, 0, 1, 0, 1],  # 类型2，2位有效
    4: [0, 0, 0, 0, 0, 0, 1, 1],  # 类型3，2位有效
    3: [0, 1, 0, 1, 0, 1, 0, 1],  # 类型4，4位有效
    5: [0, 0, 1, 1, 0, 0, 1, 1],  # 类型5，4位有效
    6: [0, 0, 0, 0, 1, 1, 1, 1],  # 类型6，4位有效
    7: [1, 1, 1, 1, 1, 1, 1, 1],  # 类型7，全部有效
}

def fill_data(data, mask):
    result = [0] * len(mask)
    positions = [i for i, val in enumerate(mask) if val == 1]
    
    if len(data) != len(positions):
        raise ValueError("Data length must match the number of 1s in mask")
    
    for idx, pos in enumerate(positions):
        result[pos] = data[idx]
    
    return result

def get_divisor(pqs):
    # 1 < s < 2
    p = pqs[0]
    q = pqs[1]

    if p >= 2*q :
        p, q = 2, 1

    x = np.arange(p)
    xd = np.round(x * q / p).astype(int)
    hist = np.zeros(q + 1, dtype=int)

    for value in xd:
        hist[value] += 1

    dup = np.where(hist == 2)[0]

    x_ch = np.zeros(2 * len(dup), dtype=int)
    idx = 0
    for i in range(p):
        if hist[xd[i]] == 2:
            x_ch[idx] = x[i] - round(xd[i] * p / q)
            idx += 1

    return p, q, dup, x_ch

def get_child_cls(q, dup, x_ch, point):
    child = np.zeros(3, dtype=int)   # child in each direction
    for k in range(3):
        r = int(point[k]) % q
        if r in dup:
            j = np.where(dup == r)[0][0]
            child[k] = x_ch[2 * j] + x_ch[2 * j + 1]
        else:
            child[k] = 0

    point_cls = abs(child[0]) + 2 * abs(child[1]) + 4 * abs(child[2])
    
    return child, point_cls

def process(arg):
    path, ppqs, pqs, D, output_path, core_id = arg
    p, q, dup, x_ch = get_divisor(pqs)
    pqs = pqs[0] / pqs[1]
    p = psutil.Process()
    p.cpu_affinity([core_id])
    ori_pc = PyntCloud.from_file(path)
    ori_points = ori_pc.points.values[:,:3].astype(int)
    if ppqs > 1.0:
        ori_points = np.round(ori_points/ppqs+1e-6).astype(int)
        ori_points = np.unique(ori_points, axis=0)     
    if pqs > 2:
        ori_points = np.round(ori_points/(pqs/2)+1e-6).astype(int) # downsample 
        ori_points = np.unique(ori_points, axis=0) # remove duplicated points
        dist_points = np.round(ori_points/2+1e-6).astype(int) # downsample 2
        dist_points = np.unique(dist_points, axis=0) # remove duplicated points
        pqs = 2
    else: # pqs <= 2
        dist_points = np.round(ori_points/pqs+1e-6).astype(int) # downsample <=2
        dist_points = np.unique(dist_points, axis=0) # remove duplicated points
    res_m = np.min(ori_points, axis=0).astype(int)
    dres_m = np.min(dist_points, axis=0).astype(int)
    res = (np.max(ori_points, axis=0)-res_m+3).astype(int)
    dres = (np.max(dist_points, axis=0)-dres_m+2*D+1).astype(int) 
    ori_voxels = np.zeros(res, dtype=np.int8)
    down_voxels = np.zeros(dres, dtype=np.int8)
    for i in range(len(ori_points)):
        ori_voxels[ori_points[i][0]+1-res_m[0], 
                   ori_points[i][1]+1-res_m[1], 
                   ori_points[i][2]+1-res_m[2]] = 1
    for i in range(len(dist_points)):
        down_voxels[dist_points[i][0]+D-dres_m[0], 
                    dist_points[i][1]+D-dres_m[1], 
                    dist_points[i][2]+D-dres_m[2]] = 1 

    neighs = []
    childs = []
    masks = []
    for i in range(len(dist_points)):
        [x, y, z] = [dist_points[i][j] for j in range(3)]
        ori_x, ori_y, ori_z = np.round(x*pqs+1e-6).astype(int), np.round(y*pqs+1e-6).astype(int), np.round(z*pqs+1e-6).astype(int)
        child, point_cls = get_child_cls(q, dup, x_ch, [x, y, z])
        if point_cls == 0:
            continue
        if point_cls != 7:   # 单独训练某类 
            continue

        tmp_neighs = np.zeros((1, (2*D+1)**3-1))
        tmp_neighs = down_voxels[x-dres_m[0]:x+2*D+1-dres_m[0],
                                 y-dres_m[1]:y+2*D+1-dres_m[1],
                                 z-dres_m[2]:z+2*D+1-dres_m[2]].reshape(-1)
        tmp_neighs = np.delete(tmp_neighs, (2*D+1)**3//2).reshape(-1) # remove the occupied center
        neighs.append(tmp_neighs)  

        tmp_childs = np.zeros((1, 8))
        tmp_childs = ori_voxels[ori_x-res_m[0]+1-abs(child[0]):ori_x+2-res_m[0],
                                ori_y-res_m[1]+1-abs(child[1]):ori_y+2-res_m[1],
                                ori_z-res_m[2]+1-abs(child[2]):ori_z+2-res_m[2]].reshape(-1)             
        childs.append(fill_data(tmp_childs, cls_masks[point_cls]))

        masks.append(cls_masks[point_cls])
    cloud = PyntCloud(pd.DataFrame(data=dist_points.astype(float), columns=['x', 'y', 'z']))
    name = os.path.splitext(os.path.split(path)[1])[0]
    if not os.path.exists('{}/{}_base.ply'.format(output_path, name)):
        cloud.to_file('{}/{}_base.ply'.format(output_path, name), as_text=True)

    return neighs, childs, masks


class PCSRDataset(Dataset):
    def __init__(self, args, status='train'):
        self.cls = 7     # non-linear down sample, child model class (0~7) 
        self.D = args.D
        self.ppqs = args.ppqs # pre pqs
        self.pqs = args.pqs # 
        self.status = status
        self.output_path = args.output_path
        self.neighs = []
        self.childs = []
        self.masks = []
        if '.ply' in args.dataset: # static pc
            # self.paths = ['data/{}'.format(args.dataset)]
            self.paths = ['{}'.format(args.pointcloud)]
        else: # dynamic pc
            # self.paths = glob.glob('data/{}/*.ply'.format(args.dataset))
            self.paths = glob.glob('{}/*.ply'.format(args.pointcloud))
            self.paths.sort()
        if args.dataset in ['basketball_player_vox11', 'dancer_vox11']:
            self.paths = self.paths[:64] # V-PCC CTC
        if not args.evaluate:
            self.paths = [self.paths[i] for i in range(0, len(self.paths), args.frame_sampling_rate)]
        if self.status == 'train':
            neighs = []
            childs = []
            masks = []
            num_cores = psutil.cpu_count(logical=False)
            if num_cores>len(self.paths): num_cores = len(self.paths)
            zip_args = list(zip(self.paths, 
                                [self.ppqs]*len(self.paths),
                                [self.pqs]*len(self.paths),
                                [self.D]*len(self.paths),
                                [self.output_path]*len(self.paths),
                                range(num_cores)
                                ))
            pool = multiprocessing.Pool(processes=num_cores)
            neighschilds = pool.map(process, zip_args)
            neighs.extend(neighschilds[0][0])
            childs.extend(neighschilds[0][1])
            masks.extend(neighschilds[0][2])
            pool.close()
            pool.join()
            neighs_tensors = []
            if neighs: 
                neighs_tensors = [torch.tensor(item, dtype=torch.float32) for item in neighs]
            else:  
                neighs_tensors = torch.zeros(1, 1)
            childs_tensors = []
            if childs: 
                childs_tensors = [torch.tensor(item, dtype=torch.float32) for item in childs]
            else:  
                childs_tensors = torch.zeros(1, 1)
            masks_tensors = []
            if childs: 
                masks_tensors = [torch.tensor(item, dtype=torch.float32) for item in masks]
            else:  
                masks_tensors = torch.zeros(1, 1)                

            self.neighs = neighs_tensors
            self.childs = childs_tensors
            self.masks = masks_tensors

    def __len__(self):
        return len(self.neighs) if self.status == 'train' else len(self.paths)
        # return len(self.neighs[self.cls]) if self.status == 'train' else len(self.paths)
    
    def __getitem__(self, idx):  
        if self.status == 'train':
            neighs = self.neighs[idx]
            childs = self.childs[idx]
            masks = self.masks[idx]
            return neighs, childs, masks
        else:
            if len(self.neighs) <= 0:
                self.neighs, self.childs = process((self.paths[idx], self.ppqs, self.pqs, self.D, self.output_path, 1))
            neighs = self.neighs[idx]
            childs = self.childs[idx]
            masks = self.masks[idx]
            return neighs.astype(np.float32), childs.astype(np.float32), masks.astype(np.float32)
    

def process2neighs(base_points, D, pqs):
    p, q, dup, x_ch = get_divisor(pqs)
    dres_m = np.min(base_points, axis=0).astype(int)
    dres = (np.max(base_points, axis=0)-dres_m+2*D+1).astype(int) 
    down_voxels = np.zeros(dres, dtype=np.int8)
    for i in range(len(base_points)):
        down_voxels[base_points[i][0]+D-dres_m[0], 
                    base_points[i][1]+D-dres_m[1], 
                    base_points[i][2]+D-dres_m[2]] = 1 
    # neighs = np.zeros((len(base_points), (2*D+1)**3-1))
    neighs = [[] for _ in range(8)]
    points = [[] for _ in range(8)]
    zero_neigh = np.zeros((1, (2*D+1)**3-1), dtype=np.float32) 
    zero_point = np.zeros((1, 3), dtype=np.float32) 
    for i in range(len(base_points)):
        [x, y, z] = [base_points[i][j] for j in range(3)]
        child, point_cls = get_child_cls(q, dup, x_ch, [x, y, z])
        tmp_neighs = down_voxels[x-dres_m[0]:x+2*D+1-dres_m[0],
                                    y-dres_m[1]:y+2*D+1-dres_m[1],
                                    z-dres_m[2]:z+2*D+1-dres_m[2]].reshape(-1)
        tmp_neighs = np.delete(tmp_neighs, (2*D+1)**3//2).reshape(-1) 

        neighs[point_cls].append(np.array(tmp_neighs))
        points[point_cls].append(np.array([x, y, z]))

    neighs = [np.vstack(item).astype(np.float32) if item else zero_neigh for item in neighs]
    points = [np.vstack(item).astype(np.float32) if item else zero_point for item in points]

    return neighs, points
