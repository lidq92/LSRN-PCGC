import xlwt
from argparse import ArgumentParser


parser = ArgumentParser(description='LSRN-PCGC summarize results')
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
parser.add_argument('-lr', '--lr', type=float, default=1e-3,
                    help='learning rate (default: 1e-3)')
parser.add_argument('-bs', '--batch_size', type=int, default=2048,
                    help='batch size (default: 2048)')
parser.add_argument('-e', '--epochs', type=int, default=150,
                    help='number of epochs to train (default: 150)')
args = parser.parse_args()
logpath = 'logs_eval/' # eval
fs_base = '{}_{}_bc{}_nl{}_K{}_p{}_lr{}_fsr{}_bs{}_e{}'
format_str  = fs_base.format(args.model, args.activation, args.base_channel, args.num_layers, args.K, 
                        args.precision, args.lr, args.frame_sampling_rate, args.batch_size, args.epochs)
excel_file = '{}.xls'.format(format_str)
datasetname_lst = list([
    'basketball_player_vox11_00000200.ply',
    'dancer_vox11_00000001.ply',
    'Facade_00064_vox11.ply',
    'longdress_vox10_1300_n.ply',
    'loot_vox10_1200_n.ply',
    'queen_frame_0200_n.ply',
    'redandblack_vox10_1550.ply',
    'soldier_vox10_0690_n.ply',
    'Thaidancer_viewdep_vox12.ply',
])
pqs_lst = list([
	64,
	32,
	16,
	8,
	4,
	2,
])
wbk = xlwt.Workbook()
sheet = wbk.add_sheet('C2 lossyG,lossyA,intra', cell_overwrite_ok=True)        
output = open('{}.txt'.format(format_str), 'w')
file_lst = list()
for name in datasetname_lst:
    for pqs in pqs_lst:
        file_lst.append(logpath+format_str+'_{}_pqs{}'.format(name, pqs))
row = 0
col = 0
missed = 0
for k, file_name in enumerate(file_lst):
    print(file_name)
    record = list()
    netparam = 0
    base = 0
    points = 0
    D1 = 0
    D2 = 0
    nF = 0
    try:
        reader = open(file_name, 'r')
        for line in reader:
            words = line.split()
            if ('network' in words) :
                netparam = int(words[-2])
            if ('scaling' in words):
                points += int(words[-2][:-1])
                nF += 1
            if (('mseF,PSNR' in words) and ('(p2point):' in words)):
                D1 += float(words[2])
            if (('mseF,PSNR' in words) and ('(p2plane):' in words)):
                D2 += float(words[2])
            if (('Total' in words) and ('bitstream' in words)) :
                base += int(words[3]) * 8
        reader.close()
        D1 = D1 / nF
        D2 = D2 / nF
    except IOError:
        print('No files found!')
    base = int(base / 2) # encoder+decoder
    geom = base + netparam  
    total = geom
    print(points, geom, base, netparam, D1, D2)
    if base != 0:
        if points != 0:
            record.append(str(points)) 
            sheet.write(row, col, points)
        if total != 0:
            record.append(str(total))
            sheet.write(row, col+1, total)
        if geom != 0:
            record.append(str(base))
            sheet.write(row, col+2, base)
            record.append(str(netparam)) 
            sheet.write(row, col+3, netparam)
        if D1 != 0:
            record.append(str(D1)) 
            sheet.write(row, col+5, D1)
        if D2 != 0:
            record.append(str(D2)) 
            sheet.write(row, col+6, D2)            
        record_str = ''
        row = row + 1
    else:
        missed += 1
        record_str = ''
    if (k+1)%6==0: # 6 rate points / pc
        row = row + missed
        missed = 0
    for word in record:
        record_str = record_str + str(word) + ' '
    record_str = record_str + '\n'
    output.write(record_str)
output.close()
wbk.save(excel_file)
