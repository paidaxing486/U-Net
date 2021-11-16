import cv2
import numpy as np
import math
import torch


def psnr(img1, img2):
   mse = np.mean((img1/1.0 - img2/1.0) ** 2 )
   if mse < 1.0e-10:
      return 100
   return 10 * math.log10(255.0**2/mse)

#负PCC皮尔森相关系数计算
def Pearson(img1,img2):
   #先计算标准差
   std1 = np.std(img1)
   std2 = np.std(img2)
   #计算img1和img2之间的协方差
   covariance = np.mean(np.multiply((img1-np.mean(img1)),(img2-np.mean(img2))))
   #皮尔森相关系数计算
   r = covariance/(-std1*std2)
   return r

#以PCC为基础的损失函数定义
def NPCC(outputs,target):
   criterion1 = 0.5*Pearson(outputs,target)
   #转频谱图
   f1 = np.fft.fft2(outputs)
   f1 = np.fft.fftshift(f1)
   f1 = np.abs(f1)
   f2 = np.fft.fft2(target)
   f2 = np.fft.fftshift(f2)
   f2 = np.abs(f2)
   criterion2 = 0.5*Pearson(f1,f2)
   loss = criterion1 + criterion2
   return loss

