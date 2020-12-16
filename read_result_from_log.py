import os
import numpy as np

# used for collect results from over-fitting testing
workdir = '/home/wangcheng/maskrcnn-benchmark/log_results/'
filelist = os.listdir(workdir)
filelist.sort()
result = []
for fil in filelist:
    recall = []
    logfile = os.path.join(workdir, fil)
    print(logfile)
    iter_num = logfile[logfile.find('step')+ 4: logfile.find('.pth')]
    with open(logfile) as f:
        lines = f.readlines()

    iter_num = fil.split('.')[0]
    iter_num = iter_num.split('_')[1]

    res_lines = lines[-10:]
    for idx, line in enumerate(res_lines):

        if 'Task: bbox' in line and 'AP, AP50, AP75, APs, APm, APl' in res_lines[idx + 1]:
            ap50 = res_lines[idx + 2][8:14]
            result.append([iter_num, ap50])
            continue

print('model iter, AP50:')
result.sort()
result = np.array(result)
result = result.transpose()

print('\t'.join(list(result[0])))
print('\t'.join(list(result[1])))
print('\n\n')
