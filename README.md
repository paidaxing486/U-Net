# U-Net
train.py 训练文件 /n
utils.py 自定义函数文件
model/model.py 模型
data/dataset 数据集制作
data/train 数据集
test.py 测试文件

当前问题：
Traceback (most recent call last):
  File "train.py", line 69, in <module>
    outputs = net(inputs)
  File "E:\Anaconda3\lib\site-packages\torch\nn\modules\module.py", line 722, in _call_impl
    result = self.forward(*input, **kwargs)
  File "C:\Users\114a\Desktop\denoising\model\model.py", line 125, in forward
    y5 = Up5(x6)
  File "E:\Anaconda3\lib\site-packages\torch\nn\modules\module.py", line 722, in _call_impl
    result = self.forward(*input, **kwargs)
  File "C:\Users\114a\Desktop\denoising\model\model.py", line 76, in forward
    x = self.conv1(x)
  File "E:\Anaconda3\lib\site-packages\torch\nn\modules\module.py", line 722, in _call_impl
    result = self.forward(*input, **kwargs)
  File "E:\Anaconda3\lib\site-packages\torch\nn\modules\container.py", line 117, in forward
    input = module(input)
  File "E:\Anaconda3\lib\site-packages\torch\nn\modules\module.py", line 722, in _call_impl
    result = self.forward(*input, **kwargs)
  File "E:\Anaconda3\lib\site-packages\torch\nn\modules\conv.py", line 907, in forward
    output_padding, self.groups, self.dilation)
RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same
  
  但我在train.py中将net传入到GPU了(net.cuda()).
