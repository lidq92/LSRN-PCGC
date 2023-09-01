# LSRN-PCGC
## Data
```bash
mkdir data
cd data
ln -s ori_path data_name #
```
## Requirements
- PyTorch+Ignite

## Experiments
### 
```bash

# cd Workspace/LSRN-PCGC/
# conda activate pcc

# basketball: AIX7590 [single program, time]
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc basketball_player_vox11 -fsr 10000
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc basketball_player_vox11 -pqs 4 -fsr 10000
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc basketball_player_vox11 -pqs 8 -fsr 10000
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc basketball_player_vox11 -pqs 16 -fsr 10000 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc basketball_player_vox11 -pqs 32 -fsr 10000
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc basketball_player_vox11 -pqs 64 -fsr 10000
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc basketball_player_vox11 -eval -fsr 10000
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc basketball_player_vox11 -pqs 4 -eval -fsr 10000
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc basketball_player_vox11 -pqs 8 -eval -fsr 10000
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc basketball_player_vox11 -pqs 16 -eval -fsr 10000
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc basketball_player_vox11 -pqs 32 -eval -fsr 10000
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc basketball_player_vox11 -pqs 64 -eval -fsr 10000


# others [multiple program running simultaneously]
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc dancer_vox11 -pqs 64 -fsr 10000
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc dancer_vox11 -pqs 32 -fsr 10000
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc dancer_vox11 -pqs 16 -fsr 10000 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc dancer_vox11 -pqs 8 -fsr 10000
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc dancer_vox11 -pqs 4 -fsr 10000 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc dancer_vox11 -fsr 10000 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc dancer_vox11 -eval -fsr 10000
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc dancer_vox11 -pqs 4 -eval -fsr 10000 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc dancer_vox11 -pqs 8 -eval -fsr 10000
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc dancer_vox11 -pqs 16 -eval -fsr 10000 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc dancer_vox11 -pqs 32 -eval -fsr 10000
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc dancer_vox11 -pqs 64 -eval -fsr 10000


CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc longdress -pqs 64 -fsr 10000
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc longdress -pqs 32 -fsr 10000
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc longdress -pqs 16 -fsr 10000 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc longdress -pqs 8 -fsr 10000
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc longdress -pqs 4 -fsr 10000 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc longdress -fsr 10000 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc longdress -eval -fsr 10000
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc longdress -pqs 4 -eval -fsr 10000 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc longdress -pqs 8 -eval -fsr 10000
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc longdress -pqs 16 -eval -fsr 10000 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc longdress -pqs 32 -eval -fsr 10000
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc longdress -pqs 64 -eval -fsr 10000


CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc loot -pqs 64 -fsr 10000
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc loot -pqs 32 -fsr 10000
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc loot -pqs 16 -fsr 10000 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc loot -pqs 8 -fsr 10000
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc loot -pqs 4 -fsr 10000 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc loot -fsr 10000 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc loot -eval -fsr 10000
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc loot -pqs 4 -eval -fsr 10000 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc loot -pqs 8 -eval -fsr 10000
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc loot -pqs 16 -eval -fsr 10000 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc loot -pqs 32 -eval -fsr 10000
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc loot -pqs 64 -eval -fsr 10000


CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc soldier -pqs 64 -fsr 10000
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc soldier -pqs 32 -fsr 10000
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc soldier -pqs 16 -fsr 10000 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc soldier -pqs 8 -fsr 10000
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc soldier -pqs 4 -fsr 10000 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc soldier -fsr 10000 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc soldier -eval -fsr 10000
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc soldier -pqs 4 -eval -fsr 10000 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc soldier -pqs 8 -eval -fsr 10000
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc soldier -pqs 16 -eval -fsr 10000 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc soldier -pqs 32 -eval -fsr 10000
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc soldier -pqs 64 -eval -fsr 10000


CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc redandblack -pqs 64 -fsr 10000
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc redandblack -pqs 32 -fsr 10000
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc redandblack -pqs 16 -fsr 10000 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc redandblack -pqs 8 -fsr 10000
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc redandblack -pqs 4 -fsr 10000 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc redandblack -fsr 10000 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc redandblack -eval -fsr 10000
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc redandblack -pqs 4 -eval -fsr 10000 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc redandblack -pqs 8 -eval -fsr 10000
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc redandblack -pqs 16 -eval -fsr 10000 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc redandblack -pqs 32 -eval -fsr 10000
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc redandblack -pqs 64 -eval -fsr 10000


CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc queen -pqs 64 -fsr 10000
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc queen -pqs 32 -fsr 10000
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc queen -pqs 16 -fsr 10000 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc queen -pqs 8 -fsr 10000
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc queen -pqs 4 -fsr 10000 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc queen -fsr 10000 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc queen -eval -fsr 10000
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc queen -pqs 4 -eval -fsr 10000 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc queen -pqs 8 -eval -fsr 10000
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc queen -pqs 16 -eval -fsr 10000 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc queen -pqs 32 -eval -fsr 10000
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc queen -pqs 64 -eval -fsr 10000



```
