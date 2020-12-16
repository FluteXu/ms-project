import os
import pandas as pd
import sys
sys.path.insert(0, '../')

from LGAIresult import LGAIresult
from utils.common_utils import save_split_txt, load_split_txt, write_excel

def out_excel(ann_list, time_list, pid_list, save_dir):
    time_dict = get_dict(time_list)
    pid_dict = get_dict(pid_list, '\t')
    
    cols = ['patient_id', 'create_time', '气胸', '气胸位置', '胸腔积液', '积液位置','肋骨骨折', '骨折位置', '结节',\
            '条索影', '网格影', '实变影', '磨玻璃密度影', '肺大疱', '肺气肿', '胸膜增厚']
    excel = []
    for ann_dict in ann_list:
        cols_dict = get_excel_info_for_bounds(ann_dict['bounds_info'], cols)
        cols_dict['patient_id'] = pid_dict[ann_dict['sub_dir'].split('/')[0]]
        cols_dict['create_time'] = time_dict[ann_dict['sub_dir']]
        current = []
        for key, value in cols_dict.items():
            current.append(value)
        excel.append(current)
    out_df = pd.DataFrame(excel, columns=cols)
    write_excel(out_df, save_dir, file_name='njjz.xlsx')

def get_dict(list_path, parse_str=' '):
    with open(list_path) as f:
        lines = f.readlines()
    time_dict = {}
    for line in lines:
        time_dict[line.split(parse_str)[0]] = line.strip().split(parse_str)[1]
    return time_dict

def get_excel_info_for_bounds(bounds_info, cols):
    cols_dict = {item: 0 for item in cols}
    mapping = {"right": "R", "left": "L"}
    location = {'胸腔积液': [], '气胸': [], '肋骨骨折': []}
    for bound in bounds_info:
        category = bound[0]['category']
        cols_dict[category] = 1
        if category in ['胸腔积液', '气胸', '肋骨骨折']:
            location[category].append(bound[0]['location'])
    cols_dict['积液位置'] = ''.join(set(location['胸腔积液']))
    cols_dict['气胸位置'] = ''.join(set(location['气胸']))
    cols_dict['骨折位置'] = ', '.join(set(location['肋骨骨折']))
    return cols_dict


if __name__ == "__main__":
    root_dir = '/data/shuzhang/tmp_data/njjz_nm/'
    sub_dir_list = '/data/shuzhang/tmp_data/njjz_sub_dirs.txt'
    time_list = '/data/shuzhang/tmp_data/njjz_sub_dirs_w_dates.txt'
    pid_list = '/data/shuzhang/tmp_data/pid.txt'
    save_dir = '/data/shuzhang/tmp_data/'
    lg_ai = LGAIresult()
    lg_ai.init(root_dir, sub_dir_list)
    lg_ai.get_ann_list()
    #print(lg_ai.ann_list[0]['bounds_info'][0])
    out_excel(lg_ai.ann_list, time_list, pid_list, save_dir)



