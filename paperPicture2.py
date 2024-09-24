import matplotlib.pyplot as plt
import torch
from PIL import Image
from timm.data import create_transform
from timm.data.mixup import Mixup

from process.processImage2 import circle_crop_final3

input_size = 512

color_jitter = 0.4
auto_augment = 'rand-m9-mstd0.5-inc1'
interpolation = 'bicubic'
re_prob = 0.25
re_mode = 'pixel'
re_count = 1
mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)

# RandomResizedCropAndInterpolation(size=(224, 224), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=bicubic) - 随机地对图像进行缩放、裁剪和插值
# RandomHorizontalFlip(p=0.5) -随机地对图像进行水平翻转

# n=2：表示每次增强时从操作集合中随机选择2个操作来应用。
# AutoContrast：自动对比度调整。
# Equalize：直方图均衡化。
# Invert：图像反色。
# Rotate：图像旋转。
# PosterizeIncreasing：增加图像的色调分离。
# SolarizeIncreasing：增加图像的曝光。
# SolarizeAdd：在图像中添加曝光效果。
# ColorIncreasing：增加图像的色彩饱和度。
# ContrastIncreasing：增加图像的对比度。
# BrightnessIncreasing：增加图像的亮度。
# SharpnessIncreasing：增加图像的锐度。
# ShearX：在X轴方向上应用剪切变换。
# ShearY：在Y轴方向上应用剪切变换。
# TranslateXRel：相对X轴的平移。
# TranslateYRel：相对Y轴的平移

# RandAugment(n=2, ops=
# 	AugmentOp(name=AutoContrast, p=0.5, m=9, mstd=0.5)
# 	AugmentOp(name=Equalize, p=0.5, m=9, mstd=0.5)
# 	AugmentOp(name=Invert, p=0.5, m=9, mstd=0.5)
# 	AugmentOp(name=Rotate, p=0.5, m=9, mstd=0.5)
# 	AugmentOp(name=PosterizeIncreasing, p=0.5, m=9, mstd=0.5)
# 	AugmentOp(name=SolarizeIncreasing, p=0.5, m=9, mstd=0.5)
# 	AugmentOp(name=SolarizeAdd, p=0.5, m=9, mstd=0.5)
# 	AugmentOp(name=ColorIncreasing, p=0.5, m=9, mstd=0.5)
# 	AugmentOp(name=ContrastIncreasing, p=0.5, m=9, mstd=0.5)
# 	AugmentOp(name=BrightnessIncreasing, p=0.5, m=9, mstd=0.5)
# 	AugmentOp(name=SharpnessIncreasing, p=0.5, m=9, mstd=0.5)
# 	AugmentOp(name=ShearX, p=0.5, m=9, mstd=0.5)
# 	AugmentOp(name=ShearY, p=0.5, m=9, mstd=0.5)
# 	AugmentOp(name=TranslateXRel, p=0.5, m=9, mstd=0.5)
# 	AugmentOp(name=TranslateYRel, p=0.5, m=9, mstd=0.5))

# MaybeToTensor()

# Normalize(mean=tensor([0.5000, 0.5000, 0.5000]), std=tensor([0.5000, 0.5000, 0.5000]))将图像数据的每个通道标准化到一个统一的分布
# RandomErasing(p=0.25, mode=pixel, count=(1, 1)) 对图像进行随机擦除操作
transformList = create_transform(
    input_size=input_size,
    is_training=True,
    color_jitter=color_jitter,
    auto_augment=auto_augment,
    interpolation=interpolation,
    re_prob=re_prob,
    re_mode=re_mode,
    re_count=re_count,
    mean=mean,
    std=std,
)


# # 创建一个虚拟的图像（例如，一张白色的图片）
# dummy_image = Image.new('RGB', (input_size, input_size), color='white')

# 定义转换序列
# transform = transforms.Compose([
#     RandomResizedCropAndInterpolation(size=(input_size, input_size), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=transforms.InterpolationMode.BICUBIC),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.ColorJitter(brightness=color_jitter, contrast=color_jitter, saturation=color_jitter, hue=color_jitter),
#     transforms.AutoAugment(auto_augment),
#     transforms.RandomApply([transforms.RandomErasing(p=re_prob, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)], p=re_prob),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
# ])

# 将张量转换回PIL图像的函数
# def tensor_to_pil(tensor):
#     # 逆转Normalize操作
#     unnormalize = transforms.Normalize(
#         mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
#         std=[1 / 0.5, 1 / 0.5, 1 / 0.5]
#     )
#     tensor_image_unnormalized = unnormalize(tensor)
#
#     # 将张量值缩放到0-255并转换为整数
#     tensor_image_unnormalized = (tensor_image_unnormalized * 255).type(torch.uint8)
#
#     # 将张量转换为PIL图像
#     to_pil = transforms.ToPILImage()
#     return to_pil(tensor_image_unnormalized)

def tensor_to_pil(tensor):
    # 将张量转换为PIL图像
    copy = tensor.clone()
    return Image.fromarray(((copy.numpy().transpose(1, 2, 0) * 255).clip(0, 255)).astype('uint8'))


def show_process(image_path, index):
    image = Image.open(image_path)

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image ' + str(index))
    plt.axis('off')

    process_image = circle_crop_final3(image_path)
    process_image = Image.fromarray(process_image)
    plt.subplot(1, 2, 2)
    plt.imshow(process_image)
    plt.title('Process Image ' + str(index))
    plt.axis('off')
    plt.show()
    return process_image

# 应用转换并显示每一步的结果
def show_transforms(image_path, transform, index):
    # image = Image.open(image_path)

    image = show_process(image_path, index)

    plt.figure(figsize=(15, 10))
    total = len(transform.transforms)
    figure_index = 0;
    fontsize = 8

    # 显示原始图像
    figure_index += 1
    plt.subplot(1, total, figure_index)
    plt.imshow(image)
    plt.title('Original Image', fontsize=fontsize)
    plt.xticks([]), plt.yticks([])  # 隐藏坐标轴

    for i, t in enumerate(transform.transforms):
        image = t(image)
        if isinstance(image, torch.Tensor):
            if type(t).__name__ == 'MaybeToTensor':
                continue
            copy_image = tensor_to_pil(image)
            figure_index += 1
            plt.subplot(1, total, figure_index)
            plt.imshow(copy_image)
            plt.title(type(t).__name__, fontsize=fontsize)
            plt.xticks([]), plt.yticks([])  # 隐藏坐标轴
            continue
        figure_index += 1
        plt.subplot(1, total, figure_index)
        plt.imshow(image)
        plt.title(type(t).__name__, fontsize=fontsize)
        plt.xticks([]), plt.yticks([])  # 隐藏坐标轴
    plt.show()
    return image


def do_transforms(image_path, transform):
    image = Image.open(image_path)
    for i, t in enumerate(transform.transforms):
        image = t(image)
    return image


mixup = 0.8
cutmix = 1.0
cutmix_minmax = None
mixup_prob = 1.0
mixup_switch_prob = 0.5
mixup_mode = 'batch'
smoothing = 0.1
nb_classes = 5
mixup_fn = Mixup(
    mixup_alpha=mixup, cutmix_alpha=cutmix, cutmix_minmax=cutmix_minmax,
    prob=mixup_prob, switch_prob=mixup_switch_prob, mode=mixup_mode,
    label_smoothing=smoothing, num_classes=nb_classes)


def show_mixup(image_path, target_path, labels, mixup_fn):
    image1 = show_transforms(image_path, transformList, 1)
    image2 = show_transforms(target_path, transformList, 2)

    mixup_image1 = image1.clone()
    mixup_image2 = image2.clone()

    # 显示原始图像1
    plt.subplot(1, 3, 1)
    plt.imshow(tensor_to_pil(image1))
    plt.title('Original Image 1')
    plt.axis('off')

    # 显示原始图像2
    plt.subplot(1, 3, 2)
    plt.imshow(tensor_to_pil(image2))
    plt.title('Original Image 2')
    plt.axis('off')

    mixup_image1 = mixup_image1.unsqueeze(0)  # 添加批次维度
    mixup_image2 = mixup_image2.unsqueeze(0)

    # 将两个图像和标签组合成一个批次
    images_batch = torch.cat((mixup_image1, mixup_image2), dim=0)
    labels_batch = torch.tensor(labels)

    # 应用mixup操作
    mixed_image, mixed_target = mixup_fn(images_batch, labels_batch)

    # 显示混合后的图像
    plt.subplot(1, 3, 3)
    plt.imshow(tensor_to_pil(mixed_image[0]))
    plt.title('Mixed Image 1')
    plt.axis('off')

    plt.show()
    return mixed_image, mixed_target


if __name__ == '__main__':
    # 测试方法
    # 使用Pillow库读取图像
    image_path = f"F:/wei/NN-MOBILENET/dataset/APOTS/crop/train_images/0a4e1a29ffff.png"
    target_path = f"F:/wei/NN-MOBILENET/dataset/APOTS/crop/train_images/0a9ec1e99ce4.png"
    # show_transforms(image_path, transformList)
    show_mixup(image_path, target_path, [0, 2], mixup_fn)
