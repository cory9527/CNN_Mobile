import os

import pandas as pd


def getFileName(image_dir):
    # 获取目录下的所有文件和文件夹名
    entries = os.listdir(image_dir)

    # 过滤出所有文件，排除文件夹
    files = [entry for entry in entries if os.path.isfile(os.path.join(image_dir, entry))]
    return files

def readCsv(label_dir: str):
    # 读取CSV文件
    data = pd.read_csv(label_dir)

    # 初始化一个字典，用于存储分组数据
    imageInfos = []

    # 遍历数据中的每行
    for index, row in data.iterrows():
        image_name = row[0]  # 图像名称
        image_name = image_name + ".png"
        imageInfos.append(image_name)
    return imageInfos

if __name__ == '__main__':
    # 指定目录路径
    directory_path = '../dataset/APOTS/crop/train_images3'
    infos = getFileName(directory_path)
    print(len(infos))

    csv_path = '../dataset/APOTS/crop/train3.csv'
    imageInfos = readCsv(csv_path)
    print(len(imageInfos))
