import os
import json
import pandas as pd
import os.path as osp

label_index = {u'心包积液': 0,
               u'淋巴结肿大': 1,
               u'心包增厚': 2,
               u'纵隔肿块': 3
               }


def main():

    # process anonymous pid json
    pid_dict = {}
    json_file = json.load(open(json_path, 'r'))
    for key, value in json_file.items():
        new_pid = key.split('_')[0]
        old_pid = str(value.keys())
        old_pid = old_pid.split('/')[1]
        old_pid = old_pid.split(']')[0]
        old_pid = old_pid[:-1]

        if old_pid not in pid_dict.keys():
            pid_dict[old_pid] = new_pid

    # process dataset label excel
    label_dict = {}
    df = pd.read_excel(excel_path, skiprows=1, index_col=3)
    dft = df.T
    df_dict = dft.to_dict(orient='list')
    for key, value in df_dict.items():
        try:
            label_dict[pid_dict[str(key)]] = value[-4:]
        except:
            print('key:', key)

    df_label = pd.DataFrame(label_dict)
    dft_label = df_label.T
    dft_label.to_excel(save_path, sheet_name='pid_label')


if __name__ == '__main__':
    root = '/data/ms_data/dicom/dataset_info'
    # anonymous pid json
    json_path = osp.join(root, 'anonymous_pid.json')
    # dataset label excel
    excel_path = osp.join(root, 'qilu_dataset_label.xls')
    save_path = osp.join(root, 'pid_label.xlsx')
    main()
