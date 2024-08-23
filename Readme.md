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

# cd ~/Workspace/LSRN-PCGC/
# conda activate pcc

# Trainning
# MPEG Cat1 (solid)
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/basketball_player_vox11_00000200.ply -ppqs 1.0 -pqs 2 -bc 32 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/basketball_player_vox11_00000200.ply -ppqs 1.0 -pqs 4 -bc 16 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/basketball_player_vox11_00000200.ply -ppqs 1.0 -pqs 8 -bc 8 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/basketball_player_vox11_00000200.ply -ppqs 1.0 -pqs 16 -bc 4 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/basketball_player_vox11_00000200.ply -ppqs 1.0 -pqs 32 -bc 2 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/basketball_player_vox11_00000200.ply -ppqs 1.0 -pqs 64 -bc 1 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/basketball_player_vox11_00000200.ply -ppqs 1.0 -pqs 2 -bc 32 -eval -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/basketball_player_vox11_00000200.ply -ppqs 1.0 -pqs 4 -bc 16 -eval -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/basketball_player_vox11_00000200.ply -ppqs 1.0 -pqs 8 -bc 8 -eval -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/basketball_player_vox11_00000200.ply -ppqs 1.0 -pqs 16 -bc 4 -eval -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/basketball_player_vox11_00000200.ply -ppqs 1.0 -pqs 32 -bc 2 -eval -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/basketball_player_vox11_00000200.ply -ppqs 1.0 -pqs 64 -bc 1 -eval -D 1 -fsr 1

CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/dancer_vox11_00000001.ply -ppqs 1.0 -pqs 2 -bc 32 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/dancer_vox11_00000001.ply -ppqs 1.0 -pqs 4 -bc 16 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/dancer_vox11_00000001.ply -ppqs 1.0 -pqs 8 -bc 8 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/dancer_vox11_00000001.ply -ppqs 1.0 -pqs 16 -bc 4 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/dancer_vox11_00000001.ply -ppqs 1.0 -pqs 32 -bc 2 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/dancer_vox11_00000001.ply -ppqs 1.0 -pqs 64 -bc 1 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/dancer_vox11_00000001.ply -ppqs 1.0 -pqs 2 -bc 32 -eval -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/dancer_vox11_00000001.ply -ppqs 1.0 -pqs 4 -bc 16 -eval -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/dancer_vox11_00000001.ply -ppqs 1.0 -pqs 8 -bc 8 -eval -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/dancer_vox11_00000001.ply -ppqs 1.0 -pqs 16 -bc 4 -eval -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/dancer_vox11_00000001.ply -ppqs 1.0 -pqs 32 -bc 2 -eval -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/dancer_vox11_00000001.ply -ppqs 1.0 -pqs 64 -bc 1 -eval -D 1 -fsr 1

CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/Facade_00064_vox11.ply -ppqs 1.0 -pqs 2 -bc 32 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/Facade_00064_vox11.ply -ppqs 1.0 -pqs 4 -bc 16 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/Facade_00064_vox11.ply -ppqs 1.0 -pqs 8 -bc 8 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/Facade_00064_vox11.ply -ppqs 1.0 -pqs 16 -bc 4 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/Facade_00064_vox11.ply -ppqs 1.0 -pqs 32 -bc 2 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/Facade_00064_vox11.ply -ppqs 1.0 -pqs 64 -bc 1 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/Facade_00064_vox11.ply -ppqs 1.0 -pqs 2 -bc 32 -eval -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/Facade_00064_vox11.ply -ppqs 1.0 -pqs 4 -bc 16 -eval -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/Facade_00064_vox11.ply -ppqs 1.0 -pqs 8 -bc 8 -eval -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/Facade_00064_vox11.ply -ppqs 1.0 -pqs 16 -bc 4 -eval -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/Facade_00064_vox11.ply -ppqs 1.0 -pqs 32 -bc 2 -eval -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/Facade_00064_vox11.ply -ppqs 1.0 -pqs 64 -bc 1 -eval -D 1 -fsr 1

CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/longdress_vox10_1300_n.ply -ppqs 1.0 -pqs 2 -bc 32 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/longdress_vox10_1300_n.ply -ppqs 1.0 -pqs 4 -bc 16 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/longdress_vox10_1300_n.ply -ppqs 1.0 -pqs 8 -bc 8 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/longdress_vox10_1300_n.ply -ppqs 1.0 -pqs 16 -bc 4 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/longdress_vox10_1300_n.ply -ppqs 1.0 -pqs 32 -bc 2 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/longdress_vox10_1300_n.ply -ppqs 1.0 -pqs 64 -bc 1 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/longdress_vox10_1300_n.ply -ppqs 1.0 -pqs 2 -bc 32 -eval -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/longdress_vox10_1300_n.ply -ppqs 1.0 -pqs 4 -bc 16 -eval -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/longdress_vox10_1300_n.ply -ppqs 1.0 -pqs 8 -bc 8 -eval -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/longdress_vox10_1300_n.ply -ppqs 1.0 -pqs 16 -bc 4 -eval -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/longdress_vox10_1300_n.ply -ppqs 1.0 -pqs 32 -bc 2 -eval -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/longdress_vox10_1300_n.ply -ppqs 1.0 -pqs 64 -bc 1 -eval -D 1 -fsr 1

CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/loot_vox10_1200_n.ply -ppqs 1.0 -pqs 2 -bc 32 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/loot_vox10_1200_n.ply -ppqs 1.0 -pqs 4 -bc 16 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/loot_vox10_1200_n.ply -ppqs 1.0 -pqs 8 -bc 8 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/loot_vox10_1200_n.ply -ppqs 1.0 -pqs 16 -bc 4 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/loot_vox10_1200_n.ply -ppqs 1.0 -pqs 32 -bc 2 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/loot_vox10_1200_n.ply -ppqs 1.0 -pqs 64 -bc 1 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/loot_vox10_1200_n.ply -ppqs 1.0 -pqs 2 -bc 32 -eval -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/loot_vox10_1200_n.ply -ppqs 1.0 -pqs 4 -bc 16 -eval -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/loot_vox10_1200_n.ply -ppqs 1.0 -pqs 8 -bc 8 -eval -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/loot_vox10_1200_n.ply -ppqs 1.0 -pqs 16 -bc 4 -eval -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/loot_vox10_1200_n.ply -ppqs 1.0 -pqs 32 -bc 2 -eval -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/loot_vox10_1200_n.ply -ppqs 1.0 -pqs 64 -bc 1 -eval -D 1 -fsr 1

CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/queen_0200_n.ply -ppqs 1.0 -pqs 2 -bc 32 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/queen_0200_n.ply -ppqs 1.0 -pqs 4 -bc 16 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/queen_0200_n.ply -ppqs 1.0 -pqs 8 -bc 8 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/queen_0200_n.ply -ppqs 1.0 -pqs 16 -bc 4 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/queen_0200_n.ply -ppqs 1.0 -pqs 32 -bc 2 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/queen_0200_n.ply -ppqs 1.0 -pqs 64 -bc 1 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/queen_0200_n.ply -ppqs 1.0 -pqs 2 -bc 32 -eval -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/queen_0200_n.ply -ppqs 1.0 -pqs 4 -bc 16 -eval -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/queen_0200_n.ply -ppqs 1.0 -pqs 8 -bc 8 -eval -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/queen_0200_n.ply -ppqs 1.0 -pqs 16 -bc 4 -eval -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/queen_0200_n.ply -ppqs 1.0 -pqs 32 -bc 2 -eval -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/queen_0200_n.ply -ppqs 1.0 -pqs 64 -bc 1 -eval -D 1 -fsr 1

CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/redandblack_vox10_1550_n.ply -ppqs 1.0 -pqs 2 -bc 32 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/redandblack_vox10_1550_n.ply -ppqs 1.0 -pqs 4 -bc 16 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/redandblack_vox10_1550_n.ply -ppqs 1.0 -pqs 8 -bc 8 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/redandblack_vox10_1550_n.ply -ppqs 1.0 -pqs 16 -bc 4 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/redandblack_vox10_1550_n.ply -ppqs 1.0 -pqs 32 -bc 2 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/redandblack_vox10_1550_n.ply -ppqs 1.0 -pqs 64 -bc 1 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/redandblack_vox10_1550_n.ply -ppqs 1.0 -pqs 2 -bc 32 -eval -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/redandblack_vox10_1550_n.ply -ppqs 1.0 -pqs 4 -bc 16 -eval -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/redandblack_vox10_1550_n.ply -ppqs 1.0 -pqs 8 -bc 8 -eval -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/redandblack_vox10_1550_n.ply -ppqs 1.0 -pqs 16 -bc 4 -eval -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/redandblack_vox10_1550_n.ply -ppqs 1.0 -pqs 32 -bc 2 -eval -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/redandblack_vox10_1550_n.ply -ppqs 1.0 -pqs 64 -bc 1 -eval -D 1 -fsr 1

CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/soldier_vox10_0690_n.ply -ppqs 1.0 -pqs 2 -bc 32 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/soldier_vox10_0690_n.ply -ppqs 1.0 -pqs 4 -bc 16 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/soldier_vox10_0690_n.ply -ppqs 1.0 -pqs 8 -bc 8 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/soldier_vox10_0690_n.ply -ppqs 1.0 -pqs 16 -bc 4 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/soldier_vox10_0690_n.ply -ppqs 1.0 -pqs 32 -bc 2 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/soldier_vox10_0690_n.ply -ppqs 1.0 -pqs 64 -bc 1 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/soldier_vox10_0690_n.ply -ppqs 1.0 -pqs 2 -bc 32 -eval -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/soldier_vox10_0690_n.ply -ppqs 1.0 -pqs 4 -bc 16 -eval -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/soldier_vox10_0690_n.ply -ppqs 1.0 -pqs 8 -bc 8 -eval -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/soldier_vox10_0690_n.ply -ppqs 1.0 -pqs 16 -bc 4 -eval -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/soldier_vox10_0690_n.ply -ppqs 1.0 -pqs 32 -bc 2 -eval -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/soldier_vox10_0690_n.ply -ppqs 1.0 -pqs 64 -bc 1 -eval -D 1 -fsr 1

CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/Thaidancer_viewdep_vox12.ply -ppqs 1.0 -pqs 2 -bc 32 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/Thaidancer_viewdep_vox12.ply -ppqs 1.0 -pqs 4 -bc 16 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/Thaidancer_viewdep_vox12.ply -ppqs 1.0 -pqs 8 -bc 8 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/Thaidancer_viewdep_vox12.ply -ppqs 1.0 -pqs 16 -bc 4 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/Thaidancer_viewdep_vox12.ply -ppqs 1.0 -pqs 32 -bc 2 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/Thaidancer_viewdep_vox12.ply -ppqs 1.0 -pqs 64 -bc 1 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/Thaidancer_viewdep_vox12.ply -ppqs 1.0 -pqs 2 -bc 32 -eval -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/Thaidancer_viewdep_vox12.ply -ppqs 1.0 -pqs 4 -bc 16 -eval -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/Thaidancer_viewdep_vox12.ply -ppqs 1.0 -pqs 8 -bc 8 -eval -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/Thaidancer_viewdep_vox12.ply -ppqs 1.0 -pqs 16 -bc 4 -eval -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/Thaidancer_viewdep_vox12.ply -ppqs 1.0 -pqs 32 -bc 2 -eval -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/Thaidancer_viewdep_vox12.ply -ppqs 1.0 -pqs 64 -bc 1 -eval -D 1 -fsr 1

# MPEG Cat2
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/basketball_player_vox11.ply+xyz+n+rgb_bin/basketball_player_vox11 -ppqs 1.0 -pqs 2
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/basketball_player_vox11.ply+xyz+n+rgb_bin/basketball_player_vox11 -ppqs 1.0 -pqs 4
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/basketball_player_vox11.ply+xyz+n+rgb_bin/basketball_player_vox11 -ppqs 1.0 -pqs 8
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/basketball_player_vox11.ply+xyz+n+rgb_bin/basketball_player_vox11 -ppqs 1.0 -pqs 16 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/basketball_player_vox11.ply+xyz+n+rgb_bin/basketball_player_vox11 -ppqs 1.0 -pqs 32
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/basketball_player_vox11.ply+xyz+n+rgb_bin/basketball_player_vox11 -ppqs 1.0 -pqs 64
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/basketball_player_vox11.ply+xyz+n+rgb_bin/basketball_player_vox11 -ppqs 1.0 -pqs 2 -eval
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/basketball_player_vox11.ply+xyz+n+rgb_bin/basketball_player_vox11 -ppqs 1.0 -pqs 4 -eval
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/basketball_player_vox11.ply+xyz+n+rgb_bin/basketball_player_vox11 -ppqs 1.0 -pqs 8 -eval
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/basketball_player_vox11.ply+xyz+n+rgb_bin/basketball_player_vox11 -ppqs 1.0 -pqs 16 -eval
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/basketball_player_vox11.ply+xyz+n+rgb_bin/basketball_player_vox11 -ppqs 1.0 -pqs 32 -eval
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/basketball_player_vox11.ply+xyz+n+rgb_bin/basketball_player_vox11 -ppqs 1.0 -pqs 64 -eval

CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/dancer_vox11.ply+xyz+n+rgb_bin/dancer_vox11 -ppqs 1.0 -pqs 64
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/dancer_vox11.ply+xyz+n+rgb_bin/dancer_vox11 -ppqs 1.0 -pqs 32
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/dancer_vox11.ply+xyz+n+rgb_bin/dancer_vox11 -ppqs 1.0 -pqs 16 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/dancer_vox11.ply+xyz+n+rgb_bin/dancer_vox11 -ppqs 1.0 -pqs 8
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/dancer_vox11.ply+xyz+n+rgb_bin/dancer_vox11 -ppqs 1.0 -pqs 4 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/dancer_vox11.ply+xyz+n+rgb_bin/dancer_vox11 -ppqs 1.0 -pqs 2 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/dancer_vox11.ply+xyz+n+rgb_bin/dancer_vox11 -ppqs 1.0 -pqs 2 -eval
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/dancer_vox11.ply+xyz+n+rgb_bin/dancer_vox11 -ppqs 1.0 -pqs 4 -eval 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/dancer_vox11.ply+xyz+n+rgb_bin/dancer_vox11 -ppqs 1.0 -pqs 8 -eval
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/dancer_vox11.ply+xyz+n+rgb_bin/dancer_vox11 -ppqs 1.0 -pqs 16 -eval 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/dancer_vox11.ply+xyz+n+rgb_bin/dancer_vox11 -ppqs 1.0 -pqs 32 -eval
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/dancer_vox11.ply+xyz+n+rgb_bin/dancer_vox11 -ppqs 1.0 -pqs 64 -eval

CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/longdress.ply+xyz+n_bin/longdress -ppqs 1.0 -pqs 64
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/longdress.ply+xyz+n_bin/longdress -ppqs 1.0 -pqs 32
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/longdress.ply+xyz+n_bin/longdress -ppqs 1.0 -pqs 16 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/longdress.ply+xyz+n_bin/longdress -ppqs 1.0 -pqs 8
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/longdress.ply+xyz+n_bin/longdress -ppqs 1.0 -pqs 4 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/longdress.ply+xyz+n_bin/longdress -ppqs 1.0 -pqs 2 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/longdress.ply+xyz+n_bin/longdress -ppqs 1.0 -pqs 2 -eval
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/longdress.ply+xyz+n_bin/longdress -ppqs 1.0 -pqs 4 -eval 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/longdress.ply+xyz+n_bin/longdress -ppqs 1.0 -pqs 8 -eval
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/longdress.ply+xyz+n_bin/longdress -ppqs 1.0 -pqs 16 -eval 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/longdress.ply+xyz+n_bin/longdress -ppqs 1.0 -pqs 32 -eval
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/longdress.ply+xyz+n_bin/longdress -ppqs 1.0 -pqs 64 -eval

CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/loot.ply+xyz+n_bin/loot -ppqs 1.0 -pqs 64
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/loot.ply+xyz+n_bin/loot -ppqs 1.0 -pqs 32
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/loot.ply+xyz+n_bin/loot -ppqs 1.0 -pqs 16 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/loot.ply+xyz+n_bin/loot -ppqs 1.0 -pqs 8
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/loot.ply+xyz+n_bin/loot -ppqs 1.0 -pqs 4 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/loot.ply+xyz+n_bin/loot -ppqs 1.0 -pqs 2 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/loot.ply+xyz+n_bin/loot -ppqs 1.0 -pqs 2 -eval
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/loot.ply+xyz+n_bin/loot -ppqs 1.0 -pqs 4 -eval 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/loot.ply+xyz+n_bin/loot -ppqs 1.0 -pqs 8 -eval
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/loot.ply+xyz+n_bin/loot -ppqs 1.0 -pqs 16 -eval 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/loot.ply+xyz+n_bin/loot -ppqs 1.0 -pqs 32 -eval
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/loot.ply+xyz+n_bin/loot -ppqs 1.0 -pqs 64 -eval

CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/soldier.ply+xyz+n_bin/soldier -ppqs 1.0 -pqs 64
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/soldier.ply+xyz+n_bin/soldier -ppqs 1.0 -pqs 32
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/soldier.ply+xyz+n_bin/soldier -ppqs 1.0 -pqs 16 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/soldier.ply+xyz+n_bin/soldier -ppqs 1.0 -pqs 8
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/soldier.ply+xyz+n_bin/soldier -ppqs 1.0 -pqs 4 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/soldier.ply+xyz+n_bin/soldier -ppqs 1.0 -pqs 2 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/soldier.ply+xyz+n_bin/soldier -ppqs 1.0 -pqs 2 -eval
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/soldier.ply+xyz+n_bin/soldier -ppqs 1.0 -pqs 4 -eval 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/soldier.ply+xyz+n_bin/soldier -ppqs 1.0 -pqs 8 -eval
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/soldier.ply+xyz+n_bin/soldier -ppqs 1.0 -pqs 16 -eval 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/soldier.ply+xyz+n_bin/soldier -ppqs 1.0 -pqs 32 -eval
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/soldier.ply+xyz+n_bin/soldier -ppqs 1.0 -pqs 64 -eval

CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/redandblack.ply+xyz+n_bin/redandblack -ppqs 1.0 -pqs 64
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/redandblack.ply+xyz+n_bin/redandblack -ppqs 1.0 -pqs 32
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/redandblack.ply+xyz+n_bin/redandblack -ppqs 1.0 -pqs 16 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/redandblack.ply+xyz+n_bin/redandblack -ppqs 1.0 -pqs 8
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/redandblack.ply+xyz+n_bin/redandblack -ppqs 1.0 -pqs 4 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/redandblack.ply+xyz+n_bin/redandblack -ppqs 1.0 -pqs 2 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/redandblack.ply+xyz+n_bin/redandblack -ppqs 1.0 -pqs 2 -eval
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/redandblack.ply+xyz+n_bin/redandblack -ppqs 1.0 -pqs 4 -eval 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/redandblack.ply+xyz+n_bin/redandblack -ppqs 1.0 -pqs 8 -eval
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/redandblack.ply+xyz+n_bin/redandblack -ppqs 1.0 -pqs 16 -eval 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/redandblack.ply+xyz+n_bin/redandblack -ppqs 1.0 -pqs 32 -eval
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/redandblack.ply+xyz+n_bin/redandblack -ppqs 1.0 -pqs 64 -eval

CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/queen.ply+xyz+n_bin/queen -ppqs 1.0 -pqs 64
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/queen.ply+xyz+n_bin/queen -ppqs 1.0 -pqs 32
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/queen.ply+xyz+n_bin/queen -ppqs 1.0 -pqs 16 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/queen.ply+xyz+n_bin/queen -ppqs 1.0 -pqs 8
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/queen.ply+xyz+n_bin/queen -ppqs 1.0 -pqs 4 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/queen.ply+xyz+n_bin/queen -ppqs 1.0 -pqs 2 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/queen.ply+xyz+n_bin/queen -ppqs 1.0 -pqs 2 -eval
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/queen.ply+xyz+n_bin/queen -ppqs 1.0 -pqs 4 -eval 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/queen.ply+xyz+n_bin/queen -ppqs 1.0 -pqs 8 -eval
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/queen.ply+xyz+n_bin/queen -ppqs 1.0 -pqs 16 -eval 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/queen.ply+xyz+n_bin/queen -ppqs 1.0 -pqs 32 -eval
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc /home/john/work/datasets/PCC/MPEG/Cat2/queen.ply+xyz+n_bin/queen -ppqs 1.0 -pqs 64 -eval


# compress
CUDA_VISIBLE_DEVICES=7 python compress.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/basketball_player_vox11_00000200.ply -pqs 2 -bc 32 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=7 python compress.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/basketball_player_vox11_00000200.ply -pqs 4 -bc 16 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=7 python compress.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/basketball_player_vox11_00000200.ply -pqs 8 -bc 8 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=7 python compress.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/basketball_player_vox11_00000200.ply -pqs 16 -bc 4 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=7 python compress.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/basketball_player_vox11_00000200.ply -pqs 32 -bc 2 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=7 python compress.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/basketball_player_vox11_00000200.ply -pqs 64 -bc 1 -D 1 -fsr 1

CUDA_VISIBLE_DEVICES=7 python compress.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/dancer_vox11_00000001.ply -pqs 2 -bc 32 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=7 python compress.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/dancer_vox11_00000001.ply -pqs 4 -bc 16 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=7 python compress.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/dancer_vox11_00000001.ply -pqs 8 -bc 8 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=7 python compress.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/dancer_vox11_00000001.ply -pqs 16 -bc 4 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=7 python compress.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/dancer_vox11_00000001.ply -pqs 32 -bc 2 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=7 python compress.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/dancer_vox11_00000001.ply -pqs 64 -bc 1 -D 1 -fsr 1

CUDA_VISIBLE_DEVICES=7 python compress.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/Facade_00064_vox11.ply -pqs 2 -bc 32 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=7 python compress.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/Facade_00064_vox11.ply -pqs 4 -bc 16 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=7 python compress.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/Facade_00064_vox11.ply -pqs 8 -bc 8 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=7 python compress.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/Facade_00064_vox11.ply -pqs 16 -bc 4 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=7 python compress.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/Facade_00064_vox11.ply -pqs 32 -bc 2 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=7 python compress.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/Facade_00064_vox11.ply -pqs 64 -bc 1 -D 1 -fsr 1

CUDA_VISIBLE_DEVICES=7 python compress.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/longdress_vox10_1300_n.ply -pqs 2 -bc 32 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=7 python compress.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/longdress_vox10_1300_n.ply -pqs 4 -bc 16 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=7 python compress.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/longdress_vox10_1300_n.ply -pqs 8 -bc 8 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=7 python compress.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/longdress_vox10_1300_n.ply -pqs 16 -bc 4 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=7 python compress.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/longdress_vox10_1300_n.ply -pqs 32 -bc 2 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=7 python compress.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/longdress_vox10_1300_n.ply -pqs 64 -bc 1 -D 1 -fsr 1

CUDA_VISIBLE_DEVICES=7 python compress.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/loot_vox10_1200_n.ply -pqs 2 -bc 32 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=7 python compress.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/loot_vox10_1200_n.ply -pqs 4 -bc 16 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=7 python compress.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/loot_vox10_1200_n.ply -pqs 8 -bc 8 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=7 python compress.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/loot_vox10_1200_n.ply -pqs 16 -bc 4 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=7 python compress.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/loot_vox10_1200_n.ply -pqs 32 -bc 2 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=7 python compress.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/loot_vox10_1200_n.ply -pqs 64 -bc 1 -D 1 -fsr 1

CUDA_VISIBLE_DEVICES=7 python compress.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/queen_0200_n.ply -pqs 2 -bc 32 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=7 python compress.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/queen_0200_n.ply -pqs 4 -bc 16 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=7 python compress.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/queen_0200_n.ply -pqs 8 -bc 8 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=7 python compress.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/queen_0200_n.ply -pqs 16 -bc 4 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=7 python compress.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/queen_0200_n.ply -pqs 32 -bc 2 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=7 python compress.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/queen_0200_n.ply -pqs 64 -bc 1 -D 1 -fsr 1

CUDA_VISIBLE_DEVICES=7 python compress.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/redandblack_vox10_1550_n.ply -pqs 2 -bc 32 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=7 python compress.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/redandblack_vox10_1550_n.ply -pqs 4 -bc 16 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=7 python compress.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/redandblack_vox10_1550_n.ply -pqs 8 -bc 8 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=7 python compress.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/redandblack_vox10_1550_n.ply -pqs 16 -bc 4 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=7 python compress.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/redandblack_vox10_1550_n.ply -pqs 32 -bc 2 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=7 python compress.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/redandblack_vox10_1550_n.ply -pqs 64 -bc 1 -D 1 -fsr 1

CUDA_VISIBLE_DEVICES=7 python compress.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/soldier_vox10_0690_n.ply -pqs 2 -bc 32 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=7 python compress.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/soldier_vox10_0690_n.ply -pqs 4 -bc 16 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=7 python compress.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/soldier_vox10_0690_n.ply -pqs 8 -bc 8 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=7 python compress.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/soldier_vox10_0690_n.ply -pqs 16 -bc 4 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=7 python compress.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/soldier_vox10_0690_n.ply -pqs 32 -bc 2 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=7 python compress.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/soldier_vox10_0690_n.ply -pqs 64 -bc 1 -D 1 -fsr 1

CUDA_VISIBLE_DEVICES=7 python compress.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/Thaidancer_viewdep_vox12.ply -pqs 2 -bc 32 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=7 python compress.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/Thaidancer_viewdep_vox12.ply -pqs 4 -bc 16 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=7 python compress.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/Thaidancer_viewdep_vox12.ply -pqs 8 -bc 8 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=7 python compress.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/Thaidancer_viewdep_vox12.ply -pqs 16 -bc 4 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=7 python compress.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/Thaidancer_viewdep_vox12.ply -pqs 32 -bc 2 -D 1 -fsr 1
CUDA_VISIBLE_DEVICES=7 python compress.py -pc /home/john/work/datasets/PCC/MPEG/Cat1A/Thaidancer_viewdep_vox12.ply -pqs 64 -bc 1 -D 1 -fsr 1

CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/basketball_player_vox11.ply+xyz+n+rgb_bin/basketball_player_vox11 -pqs 2
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/basketball_player_vox11.ply+xyz+n+rgb_bin/basketball_player_vox11 -pqs 4
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/basketball_player_vox11.ply+xyz+n+rgb_bin/basketball_player_vox11 -pqs 8
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/basketball_player_vox11.ply+xyz+n+rgb_bin/basketball_player_vox11 -pqs 16 
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/basketball_player_vox11.ply+xyz+n+rgb_bin/basketball_player_vox11 -pqs 32
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/basketball_player_vox11.ply+xyz+n+rgb_bin/basketball_player_vox11 -pqs 64

CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/dancer_vox11.ply+xyz+n+rgb_bin/dancer_vox11 -pqs 64
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/dancer_vox11.ply+xyz+n+rgb_bin/dancer_vox11 -pqs 32
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/dancer_vox11.ply+xyz+n+rgb_bin/dancer_vox11 -pqs 16 
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/dancer_vox11.ply+xyz+n+rgb_bin/dancer_vox11 -pqs 8
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/dancer_vox11.ply+xyz+n+rgb_bin/dancer_vox11 -pqs 4 
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/dancer_vox11.ply+xyz+n+rgb_bin/dancer_vox11 -pqs 2 

CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/longdress.ply+xyz+n_bin/longdress -pqs 64
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/longdress.ply+xyz+n_bin/longdress -pqs 32
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/longdress.ply+xyz+n_bin/longdress -pqs 16 
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/longdress.ply+xyz+n_bin/longdress -pqs 8
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/longdress.ply+xyz+n_bin/longdress -pqs 4 
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/longdress.ply+xyz+n_bin/longdress -pqs 2 

CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/loot.ply+xyz+n_bin/loot -pqs 64
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/loot.ply+xyz+n_bin/loot -pqs 32
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/loot.ply+xyz+n_bin/loot -pqs 16 
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/loot.ply+xyz+n_bin/loot -pqs 8
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/loot.ply+xyz+n_bin/loot -pqs 4 
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/loot.ply+xyz+n_bin/loot -pqs 2 

CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/soldier.ply+xyz+n_bin/soldier -pqs 64
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/soldier.ply+xyz+n_bin/soldier -pqs 32
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/soldier.ply+xyz+n_bin/soldier -pqs 16 
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/soldier.ply+xyz+n_bin/soldier -pqs 8
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/soldier.ply+xyz+n_bin/soldier -pqs 4 
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/soldier.ply+xyz+n_bin/soldier -pqs 2 

CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/redandblack.ply+xyz+n_bin/redandblack -pqs 64
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/redandblack.ply+xyz+n_bin/redandblack -pqs 32
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/redandblack.ply+xyz+n_bin/redandblack -pqs 16 
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/redandblack.ply+xyz+n_bin/redandblack -pqs 8
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/redandblack.ply+xyz+n_bin/redandblack -pqs 4 
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/redandblack.ply+xyz+n_bin/redandblack -pqs 2 

CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/queen.ply+xyz+n_bin/queen -pqs 64
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/queen.ply+xyz+n_bin/queen -pqs 32
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/queen.ply+xyz+n_bin/queen -pqs 16 
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/queen.ply+xyz+n_bin/queen -pqs 8
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/queen.ply+xyz+n_bin/queen -pqs 4 
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/queen.ply+xyz+n_bin/queen -pqs 2 


# decompress
CUDA_VISIBLE_DEVICES=7 python decompress.py -is outputs/basketball_player_vox11_00000200.ply_bc32_nl1_D1_p16_lr0.001_fsr1_bs2048_e150_Sine_pqs2.bs -org /home/john/work/datasets/PCC/MPEG/Cat1A/basketball_player_vox11_00000200.ply
CUDA_VISIBLE_DEVICES=7 python decompress.py -is outputs/basketball_player_vox11_00000200.ply_bc16_nl1_D1_p16_lr0.001_fsr1_bs2048_e150_Sine_pqs4.bs -org /home/john/work/datasets/PCC/MPEG/Cat1A/basketball_player_vox11_00000200.ply
CUDA_VISIBLE_DEVICES=7 python decompress.py -is outputs/basketball_player_vox11_00000200.ply_bc8_nl1_D1_p16_lr0.001_fsr1_bs2048_e150_Sine_pqs8.bs -org /home/john/work/datasets/PCC/MPEG/Cat1A/basketball_player_vox11_00000200.ply
CUDA_VISIBLE_DEVICES=7 python decompress.py -is outputs/basketball_player_vox11_00000200.ply_bc4_nl1_D1_p16_lr0.001_fsr1_bs2048_e150_Sine_pqs16.bs -org /home/john/work/datasets/PCC/MPEG/Cat1A/basketball_player_vox11_00000200.ply
CUDA_VISIBLE_DEVICES=7 python decompress.py -is outputs/basketball_player_vox11_00000200.ply_bc2_nl1_D1_p16_lr0.001_fsr1_bs2048_e150_Sine_pqs32.bs -org /home/john/work/datasets/PCC/MPEG/Cat1A/basketball_player_vox11_00000200.ply
CUDA_VISIBLE_DEVICES=7 python decompress.py -is outputs/basketball_player_vox11_00000200.ply_bc1_nl1_D1_p16_lr0.001_fsr1_bs2048_e150_Sine_pqs64.bs -org /home/john/work/datasets/PCC/MPEG/Cat1A/basketball_player_vox11_00000200.ply

CUDA_VISIBLE_DEVICES=7 python decompress.py -is outputs/dancer_vox11_00000001.ply_bc32_nl1_D1_p16_lr0.001_fsr1_bs2048_e150_Sine_pqs2.bs -org /home/john/work/datasets/PCC/MPEG/Cat1A/dancer_vox11_00000001.ply
CUDA_VISIBLE_DEVICES=7 python decompress.py -is outputs/dancer_vox11_00000001.ply_bc16_nl1_D1_p16_lr0.001_fsr1_bs2048_e150_Sine_pqs4.bs -org /home/john/work/datasets/PCC/MPEG/Cat1A/dancer_vox11_00000001.ply
CUDA_VISIBLE_DEVICES=7 python decompress.py -is outputs/dancer_vox11_00000001.ply_bc8_nl1_D1_p16_lr0.001_fsr1_bs2048_e150_Sine_pqs8.bs -org /home/john/work/datasets/PCC/MPEG/Cat1A/dancer_vox11_00000001.ply
CUDA_VISIBLE_DEVICES=7 python decompress.py -is outputs/dancer_vox11_00000001.ply_bc4_nl1_D1_p16_lr0.001_fsr1_bs2048_e150_Sine_pqs16.bs -org /home/john/work/datasets/PCC/MPEG/Cat1A/dancer_vox11_00000001.ply
CUDA_VISIBLE_DEVICES=7 python decompress.py -is outputs/dancer_vox11_00000001.ply_bc2_nl1_D1_p16_lr0.001_fsr1_bs2048_e150_Sine_pqs32.bs -org /home/john/work/datasets/PCC/MPEG/Cat1A/dancer_vox11_00000001.ply
CUDA_VISIBLE_DEVICES=7 python decompress.py -is outputs/dancer_vox11_00000001.ply_bc1_nl1_D1_p16_lr0.001_fsr1_bs2048_e150_Sine_pqs64.bs -org /home/john/work/datasets/PCC/MPEG/Cat1A/dancer_vox11_00000001.ply

CUDA_VISIBLE_DEVICES=7 python decompress.py -is outputs/Facade_00064_vox11.ply_bc32_nl1_D1_p16_lr0.001_fsr1_bs2048_e150_Sine_pqs2.bs -org /home/john/work/datasets/PCC/MPEG/Cat1A/Facade_00064_vox11.ply
CUDA_VISIBLE_DEVICES=7 python decompress.py -is outputs/Facade_00064_vox11.ply_bc16_nl1_D1_p16_lr0.001_fsr1_bs2048_e150_Sine_pqs4.bs -org /home/john/work/datasets/PCC/MPEG/Cat1A/Facade_00064_vox11.ply
CUDA_VISIBLE_DEVICES=7 python decompress.py -is outputs/Facade_00064_vox11.ply_bc8_nl1_D1_p16_lr0.001_fsr1_bs2048_e150_Sine_pqs8.bs -org /home/john/work/datasets/PCC/MPEG/Cat1A/Facade_00064_vox11.ply
CUDA_VISIBLE_DEVICES=7 python decompress.py -is outputs/Facade_00064_vox11.ply_bc4_nl1_D1_p16_lr0.001_fsr1_bs2048_e150_Sine_pqs16.bs -org /home/john/work/datasets/PCC/MPEG/Cat1A/Facade_00064_vox11.ply
CUDA_VISIBLE_DEVICES=7 python decompress.py -is outputs/Facade_00064_vox11.ply_bc2_nl1_D1_p16_lr0.001_fsr1_bs2048_e150_Sine_pqs32.bs -org /home/john/work/datasets/PCC/MPEG/Cat1A/Facade_00064_vox11.ply
CUDA_VISIBLE_DEVICES=7 python decompress.py -is outputs/Facade_00064_vox11.ply_bc1_nl1_D1_p16_lr0.001_fsr1_bs2048_e150_Sine_pqs64.bs -org /home/john/work/datasets/PCC/MPEG/Cat1A/Facade_00064_vox11.ply

CUDA_VISIBLE_DEVICES=7 python decompress.py -is outputs/longdress_vox10_1300_n.ply_bc32_nl1_D1_p16_lr0.001_fsr1_bs2048_e150_Sine_pqs2.bs -org /home/john/work/datasets/PCC/MPEG/Cat1A/longdress_vox10_1300_n.ply
CUDA_VISIBLE_DEVICES=7 python decompress.py -is outputs/longdress_vox10_1300_n.ply_bc16_nl1_D1_p16_lr0.001_fsr1_bs2048_e150_Sine_pqs4.bs -org /home/john/work/datasets/PCC/MPEG/Cat1A/longdress_vox10_1300_n.ply
CUDA_VISIBLE_DEVICES=7 python decompress.py -is outputs/longdress_vox10_1300_n.ply_bc8_nl1_D1_p16_lr0.001_fsr1_bs2048_e150_Sine_pqs8.bs -org /home/john/work/datasets/PCC/MPEG/Cat1A/longdress_vox10_1300_n.ply
CUDA_VISIBLE_DEVICES=7 python decompress.py -is outputs/longdress_vox10_1300_n.ply_bc4_nl1_D1_p16_lr0.001_fsr1_bs2048_e150_Sine_pqs16.bs -org /home/john/work/datasets/PCC/MPEG/Cat1A/longdress_vox10_1300_n.ply
CUDA_VISIBLE_DEVICES=7 python decompress.py -is outputs/longdress_vox10_1300_n.ply_bc2_nl1_D1_p16_lr0.001_fsr1_bs2048_e150_Sine_pqs32.bs -org /home/john/work/datasets/PCC/MPEG/Cat1A/longdress_vox10_1300_n.ply
CUDA_VISIBLE_DEVICES=7 python decompress.py -is outputs/longdress_vox10_1300_n.ply_bc1_nl1_D1_p16_lr0.001_fsr1_bs2048_e150_Sine_pqs64.bs -org /home/john/work/datasets/PCC/MPEG/Cat1A/longdress_vox10_1300_n.ply

CUDA_VISIBLE_DEVICES=7 python decompress.py -is outputs/loot_vox10_1200_n.ply_bc32_nl1_D1_p16_lr0.001_fsr1_bs2048_e150_Sine_pqs2.bs -org /home/john/work/datasets/PCC/MPEG/Cat1A/loot_vox10_1200_n.ply
CUDA_VISIBLE_DEVICES=7 python decompress.py -is outputs/loot_vox10_1200_n.ply_bc16_nl1_D1_p16_lr0.001_fsr1_bs2048_e150_Sine_pqs4.bs -org /home/john/work/datasets/PCC/MPEG/Cat1A/loot_vox10_1200_n.ply
CUDA_VISIBLE_DEVICES=7 python decompress.py -is outputs/loot_vox10_1200_n.ply_bc8_nl1_D1_p16_lr0.001_fsr1_bs2048_e150_Sine_pqs8.bs -org /home/john/work/datasets/PCC/MPEG/Cat1A/loot_vox10_1200_n.ply
CUDA_VISIBLE_DEVICES=7 python decompress.py -is outputs/loot_vox10_1200_n.ply_bc4_nl1_D1_p16_lr0.001_fsr1_bs2048_e150_Sine_pqs16.bs -org /home/john/work/datasets/PCC/MPEG/Cat1A/loot_vox10_1200_n.ply
CUDA_VISIBLE_DEVICES=7 python decompress.py -is outputs/loot_vox10_1200_n.ply_bc2_nl1_D1_p16_lr0.001_fsr1_bs2048_e150_Sine_pqs32.bs -org /home/john/work/datasets/PCC/MPEG/Cat1A/loot_vox10_1200_n.ply
CUDA_VISIBLE_DEVICES=7 python decompress.py -is outputs/loot_vox10_1200_n.ply_bc1_nl1_D1_p16_lr0.001_fsr1_bs2048_e150_Sine_pqs64.bs -org /home/john/work/datasets/PCC/MPEG/Cat1A/loot_vox10_1200_n.ply

CUDA_VISIBLE_DEVICES=7 python decompress.py -is outputs/queen_0200_n.ply_bc32_nl1_D1_p16_lr0.001_fsr1_bs2048_e150_Sine_pqs2.bs -org /home/john/work/datasets/PCC/MPEG/Cat1A/queen_0200_n.ply
CUDA_VISIBLE_DEVICES=7 python decompress.py -is outputs/queen_0200_n.ply_bc16_nl1_D1_p16_lr0.001_fsr1_bs2048_e150_Sine_pqs4.bs -org /home/john/work/datasets/PCC/MPEG/Cat1A/queen_0200_n.ply
CUDA_VISIBLE_DEVICES=7 python decompress.py -is outputs/queen_0200_n.ply_bc8_nl1_D1_p16_lr0.001_fsr1_bs2048_e150_Sine_pqs8.bs -org /home/john/work/datasets/PCC/MPEG/Cat1A/queen_0200_n.ply
CUDA_VISIBLE_DEVICES=7 python decompress.py -is outputs/queen_0200_n.ply_bc4_nl1_D1_p16_lr0.001_fsr1_bs2048_e150_Sine_pqs16.bs -org /home/john/work/datasets/PCC/MPEG/Cat1A/queen_0200_n.ply
CUDA_VISIBLE_DEVICES=7 python decompress.py -is outputs/queen_0200_n.ply_bc2_nl1_D1_p16_lr0.001_fsr1_bs2048_e150_Sine_pqs32.bs -org /home/john/work/datasets/PCC/MPEG/Cat1A/queen_0200_n.ply
CUDA_VISIBLE_DEVICES=7 python decompress.py -is outputs/queen_0200_n.ply_bc1_nl1_D1_p16_lr0.001_fsr1_bs2048_e150_Sine_pqs64.bs -org /home/john/work/datasets/PCC/MPEG/Cat1A/queen_0200_n.ply

CUDA_VISIBLE_DEVICES=7 python decompress.py -is outputs/redandblack_vox10_1550_n.ply_bc32_nl1_D1_p16_lr0.001_fsr1_bs2048_e150_Sine_pqs2.bs -org /home/john/work/datasets/PCC/MPEG/Cat1A/redandblack_vox10_1550_n.ply
CUDA_VISIBLE_DEVICES=7 python decompress.py -is outputs/redandblack_vox10_1550_n.ply_bc16_nl1_D1_p16_lr0.001_fsr1_bs2048_e150_Sine_pqs4.bs -org /home/john/work/datasets/PCC/MPEG/Cat1A/redandblack_vox10_1550_n.ply
CUDA_VISIBLE_DEVICES=7 python decompress.py -is outputs/redandblack_vox10_1550_n.ply_bc8_nl1_D1_p16_lr0.001_fsr1_bs2048_e150_Sine_pqs8.bs -org /home/john/work/datasets/PCC/MPEG/Cat1A/redandblack_vox10_1550_n.ply
CUDA_VISIBLE_DEVICES=7 python decompress.py -is outputs/redandblack_vox10_1550_n.ply_bc4_nl1_D1_p16_lr0.001_fsr1_bs2048_e150_Sine_pqs16.bs -org /home/john/work/datasets/PCC/MPEG/Cat1A/redandblack_vox10_1550_n.ply
CUDA_VISIBLE_DEVICES=7 python decompress.py -is outputs/redandblack_vox10_1550_n.ply_bc2_nl1_D1_p16_lr0.001_fsr1_bs2048_e150_Sine_pqs32.bs -org /home/john/work/datasets/PCC/MPEG/Cat1A/redandblack_vox10_1550_n.ply
CUDA_VISIBLE_DEVICES=7 python decompress.py -is outputs/redandblack_vox10_1550_n.ply_bc1_nl1_D1_p16_lr0.001_fsr1_bs2048_e150_Sine_pqs64.bs -org /home/john/work/datasets/PCC/MPEG/Cat1A/redandblack_vox10_1550_n.ply

CUDA_VISIBLE_DEVICES=7 python decompress.py -is outputs/soldier_vox10_0690_n.ply_bc32_nl1_D1_p16_lr0.001_fsr1_bs2048_e150_Sine_pqs2.bs -org /home/john/work/datasets/PCC/MPEG/Cat1A/soldier_vox10_0690_n.ply
CUDA_VISIBLE_DEVICES=7 python decompress.py -is outputs/soldier_vox10_0690_n.ply_bc16_nl1_D1_p16_lr0.001_fsr1_bs2048_e150_Sine_pqs4.bs -org /home/john/work/datasets/PCC/MPEG/Cat1A/soldier_vox10_0690_n.ply
CUDA_VISIBLE_DEVICES=7 python decompress.py -is outputs/soldier_vox10_0690_n.ply_bc8_nl1_D1_p16_lr0.001_fsr1_bs2048_e150_Sine_pqs8.bs -org /home/john/work/datasets/PCC/MPEG/Cat1A/soldier_vox10_0690_n.ply
CUDA_VISIBLE_DEVICES=7 python decompress.py -is outputs/soldier_vox10_0690_n.ply_bc4_nl1_D1_p16_lr0.001_fsr1_bs2048_e150_Sine_pqs16.bs -org /home/john/work/datasets/PCC/MPEG/Cat1A/soldier_vox10_0690_n.ply
CUDA_VISIBLE_DEVICES=7 python decompress.py -is outputs/soldier_vox10_0690_n.ply_bc2_nl1_D1_p16_lr0.001_fsr1_bs2048_e150_Sine_pqs32.bs -org /home/john/work/datasets/PCC/MPEG/Cat1A/soldier_vox10_0690_n.ply
CUDA_VISIBLE_DEVICES=7 python decompress.py -is outputs/soldier_vox10_0690_n.ply_bc1_nl1_D1_p16_lr0.001_fsr1_bs2048_e150_Sine_pqs64.bs -org /home/john/work/datasets/PCC/MPEG/Cat1A/soldier_vox10_0690_n.ply

CUDA_VISIBLE_DEVICES=7 python decompress.py -is outputs/Thaidancer_viewdep_vox12.ply_bc32_nl1_D1_p16_lr0.001_fsr1_bs2048_e150_Sine_pqs2.bs -org /home/john/work/datasets/PCC/MPEG/Cat1A/Thaidancer_viewdep_vox12.ply
CUDA_VISIBLE_DEVICES=7 python decompress.py -is outputs/Thaidancer_viewdep_vox12.ply_bc16_nl1_D1_p16_lr0.001_fsr1_bs2048_e150_Sine_pqs4.bs -org /home/john/work/datasets/PCC/MPEG/Cat1A/Thaidancer_viewdep_vox12.ply
CUDA_VISIBLE_DEVICES=7 python decompress.py -is outputs/Thaidancer_viewdep_vox12.ply_bc8_nl1_D1_p16_lr0.001_fsr1_bs2048_e150_Sine_pqs8.bs -org /home/john/work/datasets/PCC/MPEG/Cat1A/Thaidancer_viewdep_vox12.ply
CUDA_VISIBLE_DEVICES=7 python decompress.py -is outputs/Thaidancer_viewdep_vox12.ply_bc4_nl1_D1_p16_lr0.001_fsr1_bs2048_e150_Sine_pqs16.bs -org /home/john/work/datasets/PCC/MPEG/Cat1A/Thaidancer_viewdep_vox12.ply
CUDA_VISIBLE_DEVICES=7 python decompress.py -is outputs/Thaidancer_viewdep_vox12.ply_bc2_nl1_D1_p16_lr0.001_fsr1_bs2048_e150_Sine_pqs32.bs -org /home/john/work/datasets/PCC/MPEG/Cat1A/Thaidancer_viewdep_vox12.ply
CUDA_VISIBLE_DEVICES=7 python decompress.py -is outputs/Thaidancer_viewdep_vox12.ply_bc1_nl1_D1_p16_lr0.001_fsr1_bs2048_e150_Sine_pqs64.bs -org /home/john/work/datasets/PCC/MPEG/Cat1A/Thaidancer_viewdep_vox12.ply

CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/basketball_player_vox11.ply+xyz+n+rgb_bin/basketball_player_vox11 -pqs 2
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/basketball_player_vox11.ply+xyz+n+rgb_bin/basketball_player_vox11 -pqs 4
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/basketball_player_vox11.ply+xyz+n+rgb_bin/basketball_player_vox11 -pqs 8
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/basketball_player_vox11.ply+xyz+n+rgb_bin/basketball_player_vox11 -pqs 16 
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/basketball_player_vox11.ply+xyz+n+rgb_bin/basketball_player_vox11 -pqs 32
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/basketball_player_vox11.ply+xyz+n+rgb_bin/basketball_player_vox11 -pqs 64

CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/dancer_vox11.ply+xyz+n+rgb_bin/dancer_vox11 -pqs 64
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/dancer_vox11.ply+xyz+n+rgb_bin/dancer_vox11 -pqs 32
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/dancer_vox11.ply+xyz+n+rgb_bin/dancer_vox11 -pqs 16 
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/dancer_vox11.ply+xyz+n+rgb_bin/dancer_vox11 -pqs 8
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/dancer_vox11.ply+xyz+n+rgb_bin/dancer_vox11 -pqs 4 
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/dancer_vox11.ply+xyz+n+rgb_bin/dancer_vox11 -pqs 2 

CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/longdress.ply+xyz+n_bin/longdress -pqs 64
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/longdress.ply+xyz+n_bin/longdress -pqs 32
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/longdress.ply+xyz+n_bin/longdress -pqs 16 
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/longdress.ply+xyz+n_bin/longdress -pqs 8
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/longdress.ply+xyz+n_bin/longdress -pqs 4 
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/longdress.ply+xyz+n_bin/longdress -pqs 2 

CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/loot.ply+xyz+n_bin/loot -pqs 64
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/loot.ply+xyz+n_bin/loot -pqs 32
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/loot.ply+xyz+n_bin/loot -pqs 16 
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/loot.ply+xyz+n_bin/loot -pqs 8
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/loot.ply+xyz+n_bin/loot -pqs 4 
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/loot.ply+xyz+n_bin/loot -pqs 2 

CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/soldier.ply+xyz+n_bin/soldier -pqs 64
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/soldier.ply+xyz+n_bin/soldier -pqs 32
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/soldier.ply+xyz+n_bin/soldier -pqs 16 
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/soldier.ply+xyz+n_bin/soldier -pqs 8
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/soldier.ply+xyz+n_bin/soldier -pqs 4 
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/soldier.ply+xyz+n_bin/soldier -pqs 2 

CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/redandblack.ply+xyz+n_bin/redandblack -pqs 64
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/redandblack.ply+xyz+n_bin/redandblack -pqs 32
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/redandblack.ply+xyz+n_bin/redandblack -pqs 16 
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/redandblack.ply+xyz+n_bin/redandblack -pqs 8
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/redandblack.ply+xyz+n_bin/redandblack -pqs 4 
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/redandblack.ply+xyz+n_bin/redandblack -pqs 2 

CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/queen.ply+xyz+n_bin/queen -pqs 64
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/queen.ply+xyz+n_bin/queen -pqs 32
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/queen.ply+xyz+n_bin/queen -pqs 16 
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/queen.ply+xyz+n_bin/queen -pqs 8
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/queen.ply+xyz+n_bin/queen -pqs 4 
CUDA_VISIBLE_DEVICES=6 python compress.py  -pc /home/john/work/datasets/PCC/MPEG/Cat2/queen.ply+xyz+n_bin/queen -pqs 2 

```

## FAQ
1. Re-build `mpeg-pcc-tmc13` to get the `tmc` program if the following error occurs (this problem may be solved by installing the latest libstdc++).
```bash
AssertionError: ./tmc3v22: /lib/x86_64-linux-gnu/libm.so.6: version `GLIBC_2.29' not found (required by ./tmc3v22)
./tmc3v22: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.26' not found (required by ./tmc3v22)
```