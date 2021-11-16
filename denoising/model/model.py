import os
import torch
import torch.nn as nn

#模型参数初始化
def weight_init(m):
    class_name = m.__class__.__name__
    if class_name.find('conv')!=-1:
        m.weight.data.normal_(0,0.02)
    if class_name.find('norm')!=-1:
        m.weight.data.normal_(1.0,0.02)

#残差块
class ResBlock(nn.Module):
    def __init__(self, dim,dam):
        super(ResBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dam, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dam),
            nn.ReLU()
        )

    def forward(self, x):
        out = x + self.layers(x)
        return out

#定义第一个下采样模块
class Downsampling1(nn.Module):
    def __init__(self):
        super(Downsampling1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=5,padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
    def forward(self,x):
        out = self.conv1(x)
        return out
#定义后续的下采样模块
class Downsampling(nn.Module):
    def __init__(self,in_channle,out_channle):
        super(Downsampling, self).__init__()
        self.conv1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channle,out_channle,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channle),
            nn.ReLU(),
            nn.Conv2d(out_channle, out_channle, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channle),
            nn.ReLU()
        )
    def forward(self,x):
        out = self.conv1(x)
        return out

#定义上采样模块
class Upsampling(nn.Module):
    def __init__(self,in_channle,out_channle,y):
        super(Upsampling, self).__init__()
        self.conv1 = nn.Sequential(nn.ConvTranspose2d(in_channle, out_channle, 2, stride=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channle,out_channle,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channle),
            nn.ReLU(),
            nn.Conv2d(out_channle, out_channle, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channle),
            nn.ReLU()
        )
        self.y = y
    def forward(self,x):
        print(x.size())
        x = self.conv1(x)
        x = torch.cat((x,self.y),dim=1)
        out = self.conv2(x)
        print(out.size())
        return out

#定义最后一次上采样
class Upsampling1(nn.Module):
    def __init__(self,y):
        super(Upsampling1, self).__init__()
        self.conv1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 2, stride=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.y = y
        self.conv3 = nn.Conv2d(32,1,kernel_size=3,padding=1)
    def forward(self,x):
        x = self.conv1(x)
        x = torch.cat((x,self.y),dim=1)
        x = self.conv2(x)
        out = self.conv3(x)

        return out

#定义U-Net网络
class U_Net(nn.Module):
    def __init__(self):
        super(U_Net, self).__init__()

        self.down1 = Downsampling1()
        self.down2 = Downsampling(32,64)
        self.down3 = Downsampling(64,128)
        self.down4 = Downsampling(128,256)
        self.down5 = Downsampling(256,512)
        self.down6 = Downsampling(512,1024)
    def forward(self,x):
        #下采样
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)
        Up5 = Upsampling(1024,512,x5)
        y5 = Up5(x6)
        Up4 = Upsampling(512,256,x4)
        y4 = Up4(y5)
        Up3 = Upsampling(256,128,x3)
        y3 = Up3(y4)
        Up2 = Upsampling(128,64,x2)
        y2 = Up2(y3)
        Up1 = Upsampling1(x1)
        out = Up1(y2)
        return out



#定义整个池化层函数
class AvgPool(nn.Module):
    def __init__(self):
        super(AvgPool, self).__init__()
        self.layers1 = nn.Sequential(
            nn.AvgPool2d(2,2),
            nn.ReLU()
        )
#编码器
#最终输出为1×128×8×8的张量
class enconder(nn.Module):
    def __init__(self,nc,ndf):
        super(enconder,self).__init__()
        self.layers = nn.Sequential(
            ResBlock(3,32),
            AvgPool(),
            ResBlock(32,64),
            AvgPool(),
            ResBlock(64,128),
            AvgPool()
                                    )
    def forward(self,x):
        out = self.layers(x)
        return out