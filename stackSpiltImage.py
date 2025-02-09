#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time: 12/8/20196:04 PM
#@Author: AnguliaYang
#@File : cv_data_split.py

import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
import os
import numpy as np

def stackSplitImage():
    n_splits = 5
    csv_dir_2019 = 'F:/wei/NN-MOBILENET/dataset/APOTS/crop/train2.csv'
    train_2019 = 'F:/wei/NN-MOBILENET/dataset/APOTS/crop/train_images2'
    save_path = 'F:/wei/NN-MOBILENET/dataset/APOTS/stackcrop/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    df_2019 = pd.read_csv(csv_dir_2019)
    df_2019['path'] = df_2019['id_code'].map(lambda x: os.path.join(train_2019, '{}.png'.format(x)))
    skf2 = StratifiedKFold(n_splits=n_splits)

    print('Processing 2019 data...')
    X19 = df_2019['id_code']
    y19 = df_2019['diagnosis']
    c2=0
    tmpData = skf2.split(X19,y19)
    for train_idx2,val_idx2 in tmpData:
        is_valid_2019 = np.zeros(len(df_2019), dtype=bool)
        print(len(val_idx2), val_idx2)
        c2+=1
        is_valid_2019[val_idx2] = True
        df_2019['is_valid'+str(c2)] = is_valid_2019
        df_2019.to_csv(save_path+'df_2019_cv.csv')


def saveStackImage():
    csv_dir_2019 = 'F:/wei/NN-MOBILENET/dataset/APOTS/crop/train2.csv'
    save_path = 'F:/wei/NN-MOBILENET/dataset/APOTS/stackcrop/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    df_2019 = pd.read_csv(save_path + 'df_2019_cv.csv')
    for i in range(1, 6):
        is_valid_col = f'is_valid{i}'
        df_false = df_2019[df_2019[is_valid_col] == False]

        # 重置索引，使索引从0开始
        # df_false = df_false.reset_index(drop=True)
        df_false = df_false.drop(columns=df_false.columns[0])

        # 保存到CSV文件，不保留索引列
        df_false.to_csv(save_path + f'train2_false_{i}.csv', index=False)

# import os
# import pandas as pd
#
# def stackSplitImage():
#     n_splits = 5
#     csv_dir_2019 = 'F:/wei/NN-MOBILENET/dataset/APOTS/crop/train2.csv'
#     train_2019 = 'F:/wei/NN-MOBILENET/dataset/APOTS/crop/train_images2'
#     save_path = 'F:/wei/NN-MOBILENET/dataset/APOTS/stackcrop/'
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#
#     df_2019 = pd.read_csv(csv_dir_2019)
#     df_2019['path'] = df_2019['id_code'].map(lambda x: os.path.join(train_2019, '{}.png'.format(x)))
#
#     print('Processing 2019 data...')
#     X19 = df_2019['id_code']
#     y19 = df_2019['diagnosis']
#
#     # 分割数据并保存为5个CSV文件
#     for i in range(1, n_splits + 1):
#         is_valid_col = f'is_valid{i}'
#         df_2019[is_valid_col] = False
#         df_2019.loc[df_2019.index % n_splits == (i - 1), is_valid_col] = True
#
#         # 筛选出当前列is_valid为False的数据
#         df_false = df_2019[df_2019[is_valid_col] == False]
#         df_false.to_csv(os.path.join(save_path, f'train2_false_{i}.csv'), index=False)
#
#         # 清除当前列的is_valid标记，避免影响后续列的判断
#         df_2019.drop(columns=[is_valid_col], inplace=True)

if __name__ == '__main__':
    stackSplitImage()
    saveStackImage()

