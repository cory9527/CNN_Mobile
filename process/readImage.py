import os
import shutil


def copyBathImage(infos, image_dir, target_image_dir):
    for image_name in infos:
        image_path = image_dir + image_name
        copyImageToTarget(image_path, target_image_dir)

def copyImageToTarget(image_dir, target_dir):
    # 检查目标目录是否存在，如果不存在则创建
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print("not found")
        return

    # 获取源图像文件的文件名
    image_name = os.path.basename(image_dir)

    # 构造目标文件的完整路径
    target_path = os.path.join(target_dir, image_name)
    # 复制文件
    shutil.copy(image_dir, target_path)
    print(f"Copied {image_dir} to {target_path}")

def getFileName(image_dir):
    # 获取目录下的所有文件和文件夹名
    entries = os.listdir(image_dir)

    # 过滤出所有文件，排除文件夹
    files = [entry for entry in entries if os.path.isfile(os.path.join(image_dir, entry))]
    return files





if __name__ == '__main__':
    # 指定目录路径
    directory_path = 'F:/wei/NN-MOBILENET/dataset/APOTS/crop/train_images3'
    infos = getFileName(directory_path);
    source_image_dir = 'F:/wei/NN-MOBILENET/dataset/APOTS/train_images/'
    target_image_dir = 'F:/wei/NN-MOBILENET/dataset/APOTS/crop/train_images4/'
    copyBathImage(infos, source_image_dir, target_image_dir)


    directory_path = 'F:/wei/NN-MOBILENET/dataset/APOTS/crop/test_images3'
    infos = getFileName(directory_path);
    source_image_dir = 'F:/wei/NN-MOBILENET/dataset/APOTS/train_images/'
    target_image_dir = 'F:/wei/NN-MOBILENET/dataset/APOTS/crop/test_images4/'
    copyBathImage(infos, source_image_dir, target_image_dir)
