import torch
import cv2
import os
import glob
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset


class DRIVE_Loader(Dataset):
    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        # 读取原图路径
        self.imgs_path = glob.glob(os.path.join(data_path, 'images/*.tif'))

    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip

    def __getitem__(self, index):
        image_path = self.imgs_path[index]
        base_name = os.path.splitext(os.path.basename(image_path))[0].replace('_training', '')
        label_path = os.path.join(self.data_path, '1st_manual', f'{base_name}_manual1.gif')

        image = cv2.imread(image_path)

        # 使用 PIL 读取标签图像 cv2读不了gif
        label = Image.open(label_path).convert('L')
        label = np.array(label)

        # 转灰度
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 随机翻转增强
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)


        # reshape 成 1×H×W
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])

        # 归一化标签
        if label.max() > 1:
            label = label / 255.0

        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)


if __name__ == "__main__":
    drive_dataset = DRIVE_Loader("data/train")
    print("数据个数：", len(drive_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=drive_dataset,
                                               batch_size=4,
                                               shuffle=True)
    for image, label in train_loader:
        print(image.shape)
