"""
GAM 注意力机制：对CBAM注意力进行改进
先通道注意力，再空间注意力
"""
from datetime import datetime
import torch
import torch.nn as nn

# 通道注意力
class Channel_Attention(nn.Module):
    def __init__(self, in_channel, out_channel, ratio=4):
        super(Channel_Attention, self).__init__()
        self.fc1 = nn.Linear(in_channel, in_channel // ratio)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channel // ratio, in_channel)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # b, c, h, w = x.size()
        input = x.permute(0, 3, 2, 1)
        output = self.fc2(self.relu(self.fc1(input)))
        output = output.permute(0, 3, 2, 1)
        return output * x


# 空间注意力
class Spatial(nn.Module):
    def __init__(self, in_channel, out_channel, ratio, kernel_size=7):
        super(Spatial, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(
            in_channel, in_channel // ratio, kernel_size=7, padding=padding
        )
        self.bn = nn.BatchNorm2d(in_channel // ratio)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channel // ratio, in_channel, kernel_size=kernel_size, padding=padding
        )
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        conv1 = self.act(self.bn(self.conv1(x)))
        conv2 = self.bn1(self.conv2(conv1))
        output = self.sig(conv2)
        return x * output

# https://cloud.tencent.com/developer/article/2398288
# https://blog.csdn.net/zqx951102/article/details/127750927
class GAM(nn.Module):
    def __init__(self,in_channel, out_channel, ratio = 4, kernel_size = 7):
        super(GAM, self).__init__()
        self.channel_attention = Channel_Attention(in_channel,out_channel,ratio)
        self.spatial_attention = Spatial(in_channel,out_channel,ratio,kernel_size)

    def forward(self, x):
        start_time = datetime.now()
        input = self.channel_attention(x)
        output= self.spatial_attention(input)
        end_time = datetime.now()
        print(f"size: {x.size()}, cost: {end_time - start_time}")
        return output

if __name__ == '__main__':
    input = torch.randn(2, 3120, 7, 7)
    model = GAM(3120, 3120)
    output = model(input)
    # print(output)
    print(output.size())  #torch.Size([1, 4, 24, 24])
    # 20220928
