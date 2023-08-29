import xlwt


logpath = 'logs_eval' # 
format_str = '/Siren_bc16_K2_lr0.001_nF10_bs2048_e150_{}_pqs{}'
excel_file = 'LSRN-PCGC.xls'

datasetname_lst = list([
	'loot',
	'redandblack',
	'soldier',
	'queen',
	'longdress',
	'basketball_player_vox11',
	'dancer_vox11',
])

pqs_lst = list([
	64,
	32,
	16,
	8,
	4,
	2,
])

pqs = {}

wbk = xlwt.Workbook()
sheet = wbk.add_sheet('C2 lossyG,lossyA,intra', cell_overwrite_ok=True)        
output = open('result.txt', 'w')

file_lst = list()
for name in datasetname_lst:
    for pqs in pqs_lst:
        file_lst.append(logpath + format_str.format(name, pqs))

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
            if (('positions' in words) and ('bitstream' in words)) :
                base += int(words[3]) * 8

        reader.close()
        D1 = D1 / nF
        D2 = D2 / nF
    except IOError:
        print('No files found!')
    geom = base + netparam  
    total = geom
    print(points, geom, base, netparam, D1, D2)
    if base != 0:
        if points != 0:
            record.append(str(points)) 
            sheet.write(row, col, points)
        if total != 0:
            record.append(str(total))
            sheet.write(row, col+3, total)
        if geom != 0:
            record.append(str(base))
            sheet.write(row, col+4, base)
            record.append(str(netparam)) 
            sheet.write(row, col+5, netparam)
        if D1 != 0:
            record.append(str(D1)) 
            sheet.write(row, col+7, D1)
        if D2 != 0:
            record.append(str(D2)) 
            sheet.write(row, col+8, D2)
            
        record_str = ''
        row = row + 1
    else:
        missed += 1
        record_str = ''
    if (k+1)%5==0:
        row = row + missed
        missed = 0

    for word in record:
        record_str = record_str + str(word) + ' '
    record_str = record_str + '\n'
    output.write(record_str)

output.close()
wbk.save(excel_file)