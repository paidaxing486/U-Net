from model.model import U_Net
import torch
import torch.nn as nn
import numpy as np
#from skimage.metrics import structural_similarity
import os
from data.celeb import CelebDataset
from data.dataset import ALLDataset
import torch.optim as optim
from datetime import datetime
from utils import NPCC
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim
import argparse
#读取基础参数
parser = argparse.ArgumentParser()
parser.add_argument('--num_epoches', type=int, default=100, help='训练的轮次')
parser.add_argument('--train_batchsizes', type=int, default=5, help='训练集的batchsize大小')
parser.add_argument('--test_batchsizes', type=int, default=1, help='测试集的batchsize大小')
parser.add_argument('--lr', type=float, default=0.001, help='学习率大小')
args = parser.parse_args()
#决定是否使用CUDA
use_CUDA = torch.cuda.is_available()

dtype = torch.cuda.FloatTensor if use_CUDA else torch.FloatTensor
itype = torch.cuda.LongTensor if use_CUDA else torch.LongTensor

#加载数据集
trn_dataset = ALLDataset(data_choose='train')
val_dataset = ALLDataset(data_choose='test')
#trn_dataset = CelebDataset(mode='train')
#val_dataset = CelebDataset(mode='test')
trn_dloader = torch.utils.data.DataLoader(dataset=trn_dataset, batch_size=args.train_batchsizes, shuffle=True)
val_dloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.test_batchsizes, shuffle=False)

#定义模型，优化器
if use_CUDA:
    net = U_Net().cuda()
else:
    net = U_Net()
optimizer = optim.SGD(net.parameters(),lr = args.lr,momentum=0.9)


#创建参数保存目录
output_dir = './outputs_{:%Y-%m-%d-%H-%M-%S}/'.format(datetime.now())
checkp_dir = os.path.join(output_dir, '_checkpoints')
logtxt_dir = os.path.join(output_dir, 'log.txt')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(checkp_dir, exist_ok=True)
#开始训练
num_epoches = args.num_epoches

for epoch_idx in range(num_epoches):
    loss = []
    for batch_idx, (inputs,target) in enumerate(trn_dloader,start=1):
        #将数据转化为张量
        inputs = np.array(inputs)
        target = np.array(target)
        inputs = torch.from_numpy(inputs).float()
        target = torch.from_numpy(target).float()
        # inputs,target = inputs.clone().detach(),target.clone().detach()
        # inputs.requires_grad = True
        if use_CUDA:
            inputs, target = inputs.cuda(), target.cuda()
            inputs,target = inputs.unsqueeze(1),target.unsqueeze(1)
        else:
            inputs, target = inputs.unsqueeze(1),target.unsqueeze(1)
        outputs = net(inputs)
        h = outputs[0]
        print(h.size())
        loss = NPCC(outputs,target)#计算损失值
        optimizer.zero_grad()#清空梯度值
        loss.backward()#反向传播
        optimizer.step()#随机梯度下降


        loss.append(loss.data)#将损失之加入到列表中
        if batch_idx % 100 == 0:
            _loss = sum(loss) / len(loss)
            log_str = ''
            log_str += '[%3d/%3d]' % (epoch_idx, num_epoches)
            log_str += '[%5d/%5d]' % (batch_idx, len(trn_dloader))
            log_str += '\t%.4f' % (_loss)
            print(log_str)
            loss = []
    _dir = os.path.join(output_dir, '%03d' % epoch_idx)
    outputs.save(os.path.join(_dir, '%d_pred.jpg' % batch_idx))
    psnr = compare_psnr(target, outputs)
    #对于新版本的skimage
    #PSNR = peak_signal_noise_ratio(target, outputs)
    ssim = compare_ssim(target, outputs, multichannel=True)

    dict_save_path = os.path.join(checkp_dir, '%03d_psnr%.2f.pkl' % (epoch_idx,psnr))
    torch.save(net.state_dict(),dict_save_path)