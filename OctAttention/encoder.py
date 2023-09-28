'''
Author: fuchy@stu.pku.edu.cn
Description: this file encodes point cloud
FilePath: /compression/encoder.py
All rights reserved.
'''
from numpy import mod
from Preparedata.data import dataPrepare
from encoderTool import main
from networkTool import reload,CPrintl,expName,device
from octAttention import model
import glob,datetime,os
import pt as pointCloud
import glob
import sys
############## warning ###############
## decoder.py relys on this model here
## do not move this lines to somewhere else
model = model.to(device)
saveDic = reload(None,'modelsave/obj/encoder_epoch_00800093.pth')
model.load_state_dict(saveDic['encoder'])

###########Objct##############
dataset = sys.argv[1] if len(sys.argv)>1 else '../redandblack'
list_orifile = glob.glob('{}/*base.ply'.format(os.path.abspath(dataset)))
list_orifile.sort()
if 'basketball' in dataset or 'dancer' in dataset:
    list_orifile = list_orifile[:64] # V-PCC CTC
if __name__=="__main__":
    printl = CPrintl(expName+'/{}_encoderPLY.txt'.format(dataset))
    printl('_'*50,'OctAttention V0.4','_'*50)
    printl(datetime.datetime.now().strftime('%Y-%m-%d:%H:%M:%S'))
    printl('load checkpoint', saveDic['path'])
    for oriFile in list_orifile:
        printl(oriFile)
#         if (os.path.getsize(oriFile)>300*(1024**2)):#300M
#             printl('too large!')
#             continue
        ptName = os.path.splitext(os.path.basename(oriFile))[0] 
        for qs in [1]:
            ptNamePrefix = ptName
            matFile,DQpt,refPt = dataPrepare(oriFile,saveMatDir='./Data/testPly',qs=qs,ptNamePrefix='',rotation=False)
            # please set `rotation=True` in the `dataPrepare` function when processing MVUB data
            main(matFile,model,actualcode=True,printl =printl) # actualcode=False: bin file will not be generated
#             print('_'*50,'pc_error','_'*50)
#             pointCloud.pcerror(refPt,DQpt,None,'-r 1023',None).wait()