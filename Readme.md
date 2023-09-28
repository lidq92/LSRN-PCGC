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

# MPEG Cat1 (solid)
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc basketball_player_vox11_00000200.ply -pqs 2 -bc 32 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc basketball_player_vox11_00000200.ply -pqs 4 -bc 16 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc basketball_player_vox11_00000200.ply -pqs 8 -bc 8 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc basketball_player_vox11_00000200.ply -pqs 16 -bc 4 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc basketball_player_vox11_00000200.ply -pqs 32 -bc 2 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc basketball_player_vox11_00000200.ply -pqs 64 -bc 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc basketball_player_vox11_00000200.ply -pqs 2 -bc 32 -eval -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc basketball_player_vox11_00000200.ply -pqs 4 -bc 16 -eval -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc basketball_player_vox11_00000200.ply -pqs 8 -bc 8 -eval -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc basketball_player_vox11_00000200.ply -pqs 16 -bc 4 -eval -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc basketball_player_vox11_00000200.ply -pqs 32 -bc 2 -eval -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc basketball_player_vox11_00000200.ply -pqs 64 -bc 1 -eval -fsr 1

CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc dancer_vox11_00000001.ply -pqs 2 -bc 32 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc dancer_vox11_00000001.ply -pqs 4 -bc 16 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc dancer_vox11_00000001.ply -pqs 8 -bc 8 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc dancer_vox11_00000001.ply -pqs 16 -bc 4 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc dancer_vox11_00000001.ply -pqs 32 -bc 2 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc dancer_vox11_00000001.ply -pqs 64 -bc 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc dancer_vox11_00000001.ply -pqs 2 -bc 32 -eval -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc dancer_vox11_00000001.ply -pqs 4 -bc 16 -eval -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc dancer_vox11_00000001.ply -pqs 8 -bc 8 -eval -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc dancer_vox11_00000001.ply -pqs 16 -bc 4 -eval -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc dancer_vox11_00000001.ply -pqs 32 -bc 2 -eval -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc dancer_vox11_00000001.ply -pqs 64 -bc 1 -eval -fsr 1

CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc Facade_00064_vox11.ply -pqs 2 -bc 32 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc Facade_00064_vox11.ply -pqs 4 -bc 16 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc Facade_00064_vox11.ply -pqs 8 -bc 8 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc Facade_00064_vox11.ply -pqs 16 -bc 4 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc Facade_00064_vox11.ply -pqs 32 -bc 2 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc Facade_00064_vox11.ply -pqs 64 -bc 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc Facade_00064_vox11.ply -pqs 2 -bc 32 -eval -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc Facade_00064_vox11.ply -pqs 4 -bc 16 -eval -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc Facade_00064_vox11.ply -pqs 8 -bc 8 -eval -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc Facade_00064_vox11.ply -pqs 16 -bc 4 -eval -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc Facade_00064_vox11.ply -pqs 32 -bc 2 -eval -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc Facade_00064_vox11.ply -pqs 64 -bc 1 -eval -fsr 1

CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc longdress_vox10_1300_n.ply -pqs 2 -bc 32 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc longdress_vox10_1300_n.ply -pqs 4 -bc 16 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc longdress_vox10_1300_n.ply -pqs 8 -bc 8 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc longdress_vox10_1300_n.ply -pqs 16 -bc 4 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc longdress_vox10_1300_n.ply -pqs 32 -bc 2 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc longdress_vox10_1300_n.ply -pqs 64 -bc 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc longdress_vox10_1300_n.ply -pqs 2 -bc 32 -eval -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc longdress_vox10_1300_n.ply -pqs 4 -bc 16 -eval -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc longdress_vox10_1300_n.ply -pqs 8 -bc 8 -eval -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc longdress_vox10_1300_n.ply -pqs 16 -bc 4 -eval -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc longdress_vox10_1300_n.ply -pqs 32 -bc 2 -eval -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc longdress_vox10_1300_n.ply -pqs 64 -bc 1 -eval -fsr 1

CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc loot_vox10_1200_n.ply -pqs 2 -bc 32 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc loot_vox10_1200_n.ply -pqs 4 -bc 16 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc loot_vox10_1200_n.ply -pqs 8 -bc 8 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc loot_vox10_1200_n.ply -pqs 16 -bc 4 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc loot_vox10_1200_n.ply -pqs 32 -bc 2 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc loot_vox10_1200_n.ply -pqs 64 -bc 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc loot_vox10_1200_n.ply -pqs 2 -bc 32 -eval -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc loot_vox10_1200_n.ply -pqs 4 -bc 16 -eval -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc loot_vox10_1200_n.ply -pqs 8 -bc 8 -eval -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc loot_vox10_1200_n.ply -pqs 16 -bc 4 -eval -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc loot_vox10_1200_n.ply -pqs 32 -bc 2 -eval -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc loot_vox10_1200_n.ply -pqs 64 -bc 1 -eval -fsr 1

CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc queen_0200_n.ply -pqs 2 -bc 32 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc queen_0200_n.ply -pqs 4 -bc 16 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc queen_0200_n.ply -pqs 8 -bc 8 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc queen_0200_n.ply -pqs 16 -bc 4 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc queen_0200_n.ply -pqs 32 -bc 2 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc queen_0200_n.ply -pqs 64 -bc 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc queen_0200_n.ply -pqs 2 -bc 32 -eval -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc queen_0200_n.ply -pqs 4 -bc 16 -eval -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc queen_0200_n.ply -pqs 8 -bc 8 -eval -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc queen_0200_n.ply -pqs 16 -bc 4 -eval -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc queen_0200_n.ply -pqs 32 -bc 2 -eval -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc queen_0200_n.ply -pqs 64 -bc 1 -eval -fsr 1

CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc redandblack_vox10_1550_n.ply -pqs 2 -bc 32 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc redandblack_vox10_1550_n.ply -pqs 4 -bc 16 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc redandblack_vox10_1550_n.ply -pqs 8 -bc 8 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc redandblack_vox10_1550_n.ply -pqs 16 -bc 4 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc redandblack_vox10_1550_n.ply -pqs 32 -bc 2 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc redandblack_vox10_1550_n.ply -pqs 64 -bc 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc redandblack_vox10_1550_n.ply -pqs 2 -bc 32 -eval -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc redandblack_vox10_1550_n.ply -pqs 4 -bc 16 -eval -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc redandblack_vox10_1550_n.ply -pqs 8 -bc 8 -eval -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc redandblack_vox10_1550_n.ply -pqs 16 -bc 4 -eval -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc redandblack_vox10_1550_n.ply -pqs 32 -bc 2 -eval -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc redandblack_vox10_1550_n.ply -pqs 64 -bc 1 -eval -fsr 1

CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc soldier_vox10_0690_n.ply -pqs 2 -bc 32 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc soldier_vox10_0690_n.ply -pqs 4 -bc 16 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc soldier_vox10_0690_n.ply -pqs 8 -bc 8 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc soldier_vox10_0690_n.ply -pqs 16 -bc 4 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc soldier_vox10_0690_n.ply -pqs 32 -bc 2 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc soldier_vox10_0690_n.ply -pqs 64 -bc 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc soldier_vox10_0690_n.ply -pqs 2 -bc 32 -eval -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc soldier_vox10_0690_n.ply -pqs 4 -bc 16 -eval -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc soldier_vox10_0690_n.ply -pqs 8 -bc 8 -eval -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc soldier_vox10_0690_n.ply -pqs 16 -bc 4 -eval -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc soldier_vox10_0690_n.ply -pqs 32 -bc 2 -eval -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc soldier_vox10_0690_n.ply -pqs 64 -bc 1 -eval -fsr 1

CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc Thaidancer_viewdep_vox12.ply -pqs 2 -bc 32 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc Thaidancer_viewdep_vox12.ply -pqs 4 -bc 16 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc Thaidancer_viewdep_vox12.ply -pqs 8 -bc 8 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc Thaidancer_viewdep_vox12.ply -pqs 16 -bc 4 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc Thaidancer_viewdep_vox12.ply -pqs 32 -bc 2 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc Thaidancer_viewdep_vox12.ply -pqs 64 -bc 1 -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc Thaidancer_viewdep_vox12.ply -pqs 2 -bc 32 -eval -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc Thaidancer_viewdep_vox12.ply -pqs 4 -bc 16 -eval -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc Thaidancer_viewdep_vox12.ply -pqs 8 -bc 8 -eval -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc Thaidancer_viewdep_vox12.ply -pqs 16 -bc 4 -eval -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc Thaidancer_viewdep_vox12.ply -pqs 32 -bc 2 -eval -fsr 1
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc Thaidancer_viewdep_vox12.ply -pqs 64 -bc 1 -eval -fsr 1

# MPEG Cat2
# basketball: AIX7590 [single program, time]
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc basketball_player_vox11 -pqs 2 -fsr 10
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc basketball_player_vox11 -pqs 4 -fsr 10
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc basketball_player_vox11 -pqs 8 -fsr 10
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc basketball_player_vox11 -pqs 16 -fsr 10 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc basketball_player_vox11 -pqs 32 -fsr 10
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc basketball_player_vox11 -pqs 64 -fsr 10
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc basketball_player_vox11 -pqs 2 -eval -fsr 10
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc basketball_player_vox11 -pqs 4 -eval -fsr 10
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc basketball_player_vox11 -pqs 8 -eval -fsr 10
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc basketball_player_vox11 -pqs 16 -eval -fsr 10
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc basketball_player_vox11 -pqs 32 -eval -fsr 10
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc basketball_player_vox11 -pqs 64 -eval -fsr 10
# others [multiple program running simultaneously]
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc dancer_vox11 -pqs 64 -fsr 10
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc dancer_vox11 -pqs 32 -fsr 10
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc dancer_vox11 -pqs 16 -fsr 10 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc dancer_vox11 -pqs 8 -fsr 10
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc dancer_vox11 -pqs 4 -fsr 10 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc dancer_vox11 -pqs 2 -fsr 10 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc dancer_vox11 -pqs 2 -eval -fsr 10
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc dancer_vox11 -pqs 4 -eval -fsr 10 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc dancer_vox11 -pqs 8 -eval -fsr 10
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc dancer_vox11 -pqs 16 -eval -fsr 10 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc dancer_vox11 -pqs 32 -eval -fsr 10
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc dancer_vox11 -pqs 64 -eval -fsr 10


CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc longdress -pqs 64 -fsr 10
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc longdress -pqs 32 -fsr 10
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc longdress -pqs 16 -fsr 10 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc longdress -pqs 8 -fsr 10
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc longdress -pqs 4 -fsr 10 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc longdress -pqs 2 -fsr 10 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc longdress -pqs 2 -eval -fsr 10
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc longdress -pqs 4 -eval -fsr 10 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc longdress -pqs 8 -eval -fsr 10
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc longdress -pqs 16 -eval -fsr 10 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc longdress -pqs 32 -eval -fsr 10
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc longdress -pqs 64 -eval -fsr 10


CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc loot -pqs 64 -fsr 10
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc loot -pqs 32 -fsr 10
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc loot -pqs 16 -fsr 10 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc loot -pqs 8 -fsr 10
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc loot -pqs 4 -fsr 10 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc loot -pqs 2 -fsr 10 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc loot -pqs 2 -eval -fsr 10
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc loot -pqs 4 -eval -fsr 10 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc loot -pqs 8 -eval -fsr 10
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc loot -pqs 16 -eval -fsr 10 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc loot -pqs 32 -eval -fsr 10
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc loot -pqs 64 -eval -fsr 10


CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc soldier -pqs 64 -fsr 10
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc soldier -pqs 32 -fsr 10
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc soldier -pqs 16 -fsr 10 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc soldier -pqs 8 -fsr 10
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc soldier -pqs 4 -fsr 10 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc soldier -pqs 2 -fsr 10 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc soldier -pqs 2 -eval -fsr 10
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc soldier -pqs 4 -eval -fsr 10 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc soldier -pqs 8 -eval -fsr 10
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc soldier -pqs 16 -eval -fsr 10 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc soldier -pqs 32 -eval -fsr 10
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc soldier -pqs 64 -eval -fsr 10


CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc redandblack -pqs 64 -fsr 10
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc redandblack -pqs 32 -fsr 10
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc redandblack -pqs 16 -fsr 10 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc redandblack -pqs 8 -fsr 10
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc redandblack -pqs 4 -fsr 10 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc redandblack -pqs 2 -fsr 10 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc redandblack -pqs 2 -eval -fsr 10
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc redandblack -pqs 4 -eval -fsr 10 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc redandblack -pqs 8 -eval -fsr 10
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc redandblack -pqs 16 -eval -fsr 10 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc redandblack -pqs 32 -eval -fsr 10
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc redandblack -pqs 64 -eval -fsr 10


CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc queen -pqs 64 -fsr 10
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc queen -pqs 32 -fsr 10
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc queen -pqs 16 -fsr 10 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc queen -pqs 8 -fsr 10
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc queen -pqs 4 -fsr 10 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc queen -pqs 2 -fsr 10 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc queen -pqs 2 -eval -fsr 10
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc queen -pqs 4 -eval -fsr 10 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc queen -pqs 8 -eval -fsr 10
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc queen -pqs 16 -eval -fsr 10 
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc queen -pqs 32 -eval -fsr 10
CUDA_VISIBLE_DEVICES=0 python PCSRmain.py -pc queen -pqs 64 -eval -fsr 10



```

## FAQ
1. Re-build `mpeg-pcc-tmc13` to get the `tmc` program if the following error occurs (this problem may be solved by installing the latest libstdc++).
```bash
AssertionError: ./tmc3v22: /lib/x86_64-linux-gnu/libm.so.6: version `GLIBC_2.29' not found (required by ./tmc3v22)
./tmc3v22: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.26' not found (required by ./tmc3v22)
```