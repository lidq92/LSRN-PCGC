import logger
import subprocess
import numpy as np
import pandas as pd
from pyntcloud import PyntCloud
from ignite.metrics.metric import Metric


class PCSRPerformance(Metric):
    def __init__(self):
        super(PCSRPerformance, self).__init__()

    def reset(self):
        self._count = 0
        self._n     = 0

    def update(self, output):
        y_pred, y = output
        idx1 = y_pred.max(dim=1)[0]>=0.5
        idx2 = y_pred.max(dim=1)[0]<0.5
        y_pred[idx1] = (y_pred[idx1]>=0.5).float() 
        y_pred[idx2] = (y_pred[idx2] >= y_pred[idx2].max(dim=1, keepdim=True)[0]).float() # At least one interpolated point
        self._count += (y_pred*y+(1-y_pred)*(1-y)).sum().item()
        self._n     +=  y.numel()
        
    def compute(self):
        return {'mAcc': np.asarray(self._count/self._n)}


class PCSRPerformance1(Metric):
    def __init__(self, args, nscale, computePSNR=True):
        super(PCSRPerformance1, self).__init__()
        if '.ply' in args.dataset: # static pc
            self.ori_path = 'data'
        else: # dynamic pc
            self.ori_path = 'data/{}'.format(args.dataset)
        self.output_path = args.output_path
        self.res = 2**args.vox-1
        self.nscale = nscale
        self.computePSNR = computePSNR

    def reset(self):
        self._acc  = []

    def update(self, output):
        y_pred, (y, points, name) = output
        idx1 = y_pred.max(dim=1)[0]>=0.5
        idx2 = y_pred.max(dim=1)[0]<0.5
        y_pred[idx1] = (y_pred[idx1]>=0.5).float() 
        y_pred[idx2] = (y_pred[idx2] >= y_pred[idx2].max(dim=1, keepdim=True)[0]).float() # At least one interpolated point
        acc = (y_pred*y+(1-y_pred)*(1-y)).sum().item()/y.numel()
        self._acc.append(acc)
        y_pred = y_pred.reshape(-1).to('cpu').numpy()
        dxyz = np.asarray([[[0,0,0],[0,0,1],[0,1,0],[0,1,1],
                           [1,0,0],[1,0,1],[1,1,0],[1,1,1]]])-1        
        points = points.to('cpu').numpy() #
        all_points = 2*np.repeat(points, 8, axis=0)+np.repeat(dxyz, len(points), axis=0).reshape(-1,3)
        points = all_points[y_pred==1]
        points = np.unique(points, axis=0) #
        cloud = PyntCloud(pd.DataFrame(data=points.astype(float), columns=['x', 'y', 'z']))
        if self.nscale: cloud.to_file('{}/{}_preinv_output_scale{}.ply'.format(self.output_path, name, self.nscale), as_text=True)
        if self.computePSNR:
            if self.nscale: points = points*(2**self.nscale)
            cloud = PyntCloud(pd.DataFrame(data=points.astype(float), columns=['x', 'y', 'z']))
            cloud.to_file('{}/{}_output_scale{}.ply'.format(self.output_path, name, self.nscale), as_text=True)
            cmd = './pc_error -a '+'{}/{}.ply'.format(self.ori_path, name)+' -b '+'{}/{}_output_scale{}.ply'.format(self.output_path, name, self.nscale)+' -c 1 -l 1 -d 1 --nbThreads=10 --dropdups=2  --neighborsProc=1 -r '+str(self.res)
            r = sh(cmd) 
            logger.log.info(r)
            # sh('rm -r ' + '{}/{}_output_scale{}.ply'.format(self.output_path, name, self.nscale))
        
    def compute(self):
        return {'mAcc': np.asarray(self._acc).mean(),
                'acc': np.asarray(self._acc)}
    
    
def sh(cmd, input=''): # https://stackoverflow.com/a/56215593/3865166 to solve the issue that cannot get the stdout of os.system, although a naive method is to expand the cmd with '> tmp.txt'.
    rst = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, 
                         stderr=subprocess.PIPE, input=input.encode('utf-8'))
    assert rst.returncode == 0, rst.stderr.decode('utf-8')
    return rst.stdout.decode('utf-8')
