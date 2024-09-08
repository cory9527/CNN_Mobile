import csv
import os
import shutil
import random

import pandas as pd
from typing import List, Dict, Tuple


def readCsv(label_dir: str) -> Dict[int, List[Tuple[str, int]]]:
    # 读取CSV文件
    data = pd.read_csv(label_dir)

    # 初始化一个字典，用于存储分组数据
    no_class = []
    grouped_data = {}

    # 遍历数据中的每行
    for index, row in data.iterrows():
        image_name = row[0]  # 图像名称
        label = row[1]  # 标签值
        if label not in grouped_data:
            grouped_data[label] = []
            no_class.append(label)
        grouped_data[label].append((image_name, label))

    return grouped_data

def split(grouped_data, image_dir, target_image_dir, target_label_dir):

    train_data_total = []
    validation_data_total = []
    for label, value in grouped_data.items():
        # # 乱序列表
        # random.shuffle(value)
        # 计算80%的大小
        # split_index = int(len(value) * 0.8)

        split_index = int(len(value) - 4)
        # 按80%:20%划分列表
        train_data = value[:split_index]
        validation_data = value[split_index:]
        train_data_total.extend(train_data)
        validation_data_total.extend(validation_data)
    random.shuffle(train_data_total)
    random.shuffle(validation_data_total)

    copyBathImage(train_data_total, image_dir, target_image_dir + "train_images2")
    copyBathImage(validation_data_total, image_dir, target_image_dir + "test_images2")
    write_to_csv(train_data_total, target_label_dir + "train2.csv")
    write_to_csv(validation_data_total, target_label_dir + "test2.csv")

def write_to_csv(data: List[Tuple[str, int]], filename: str):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # 写入标题行
        writer.writerow(['id_code', 'diagnosis'])
        for row in data:
            writer.writerow(row)

def copyBathImage(infos, image_dir, target_image_dir):
    for image_name, _ in infos:
        image_path = image_dir + image_name + ".png"
        copyImageToTarget(image_path, target_image_dir)

def copyImageToTarget(image_dir, target_dir):
    # 检查目标目录是否存在，如果不存在则创建
    if not os.path.exists(target_dir):
        # os.makedirs(target_dir)
        print("not found")
        return

    # 获取源图像文件的文件名
    image_name = os.path.basename(image_dir)

    # 构造目标文件的完整路径
    target_path = os.path.join(target_dir, image_name)

    # 复制文件
    shutil.copy(image_dir, target_path)
    print(f"Copied {image_dir} to {target_path}")




def split2(grouped_data: Dict[int, List[Tuple[str, int]]]) -> Dict[int, Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]]:
    split_data = {}
    for label, value in grouped_data.items():
        # 乱序列表
        random.shuffle(value)
        # 计算80%的大小
        split_index = int(len(value) * 0.8)
        # 按80%:20%划分列表
        train_data = value[:split_index]
        validation_data = value[split_index:]
        # 存储划分后的数据
        split_data[label] = (train_data, validation_data)
    return split_data

# # 示例用法
# label_dir = 'path_to_your_csv_file.csv'
# grouped_images = readCsv(label_dir)
# splitted_data = split(grouped_images)
# print(splitted_data)


# 示例用法
label_dir = 'F:/wei/NN-MOBILENET/dataset/APOTS/train.csv'
source_image_dir = "F:/wei/NN-MOBILENET/dataset/APOTS/train_images/"
target_image_dir = "F:/wei/NN-MOBILENET/dataset/APOTS/crop/"
target_label_dir = "F:/wei/NN-MOBILENET/dataset/APOTS/crop/"
grouped_images = readCsv(label_dir)
split(grouped_images, source_image_dir, target_image_dir,target_label_dir)
# print(grouped_images.keys())
# print(grouped_images)