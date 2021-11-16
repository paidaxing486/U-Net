import os
import numpy as np
from PIL import Image, ImageDraw
import cv2
import torch
import torch.utils.data as data

root_path = 'C:/Users/114a/Desktop/denoising/data/train/'
class ALLDataset(data.Dataset):
    def __init__(self,data_choose = 'train'):
        self.root = root_path
        self.data_choose = data_choose
        self.img_ids = list()

        #根据data_choose选择是制作训练集还是数据集
        if self.data_choose == 'train':
            img_list = os.path.join(self.root, 'train.txt')
        elif self.data_choose == 'test':
            img_list = os.path.join(self.root, 'test.txt')
        else:
            print('Forget to choose data_choose')
            exit()

        #将数据制作成列表
        with open(img_list, 'r') as f:
            lines = f.readlines()
            for line in lines:
                fname = int(line.strip()[:-4])
                self.img_ids.append(
                    fname,
                )
    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        face_idx = self.img_ids[index]
        #输入数据和标签数据路径拼接
        input_path = os.path.join(self.root, 'inputs', str(face_idx) + '.jpg')
        label_path = os.path.join(self.root, 'labels', str(face_idx) + '.jpg')
        #读取全息图
        input = Image.open(input_path).convert('L')
        label = Image.open(label_path).convert('L')
        input = input.resize((512,512), Image.BICUBIC)
        label = label.resize((512,512), Image.BICUBIC)
        #获得频谱图
        f1 = np.fft.fft2(input)
        f1 = np.fft.fftshift(f1)
        #频谱图可视化图
        imorig2 = np.log(np.abs(f1))
        inputs = np.array(input)
        labels = np.array(label)
        #可能存在的预处理步骤

        return inputs,labels

