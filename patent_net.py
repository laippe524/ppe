import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor
from typing import List, Tuple
import numpy as np
import torch.jit


#定义SRM层
class SRMlayer(nn.Module):
    def __init__(self):
        super(SRMlayer, self).__init__()
        #使用三种滤波器作为卷积核
        q = [4.0, 12.0, 2.0]
        filter1 = [[0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 2, -4, 2, 0],
                   [0, -1, 2, -1, 0],
                   [0, 0, 0, 0, 0]]
        filter2 = [[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]]
        filter3 = [[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 1, -2, 1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]
        filter1 = np.asarray(filter1, dtype=float) / q[0]
        filter2 = np.asarray(filter2, dtype=float) / q[1]
        filter3 = np.asarray(filter3, dtype=float) / q[2]
        filters = np.asarray(
            [
                [filter1],
                [filter2],
                [filter3]
            ]
        )#shape=(3,1,5,5)
        filters = torch.tensor(filters)#转化为tensor
        filters = filters.float()
        self.weight = nn.Parameter(data=filters, requires_grad=False)#自定义卷积核，不需要训练权重参数
    def forward(self,inputs):
        #输入的图像是三通道的tensor
        x1 = inputs[:, 0]#第一个通道的tensor
        x2 = inputs[:, 1]
        x3 = inputs[:, 2]
        x1 = F.conv2d(x1.unsqueeze(1), self.weight, stride=1, padding=2)#output is 1x3xHxW
        x1 = torch.sum(x1, dim=1).unsqueeze(1)
        x2 = F.conv2d(x2.unsqueeze(1), self.weight, stride=1, padding=2)
        x2 = torch.sum(x2, dim=1).unsqueeze(1)
        x3 = F.conv2d(x3.unsqueeze(1), self.weight, stride=1, padding=2)
        x3 = torch.sum(x3, dim=1).unsqueeze(1)
        outputs = torch.cat([x1, x2, x3], dim=1)#每个通道进行卷积运算后按照通道连接起来
        outputs = torch.cat([outputs, inputs], dim=1)#使用srm滤波器提取残差信息后再concatate原始图像
        return outputs

#定义denseblock中的每一层
class _DenseLayer(nn.Module):
    def __init__(
        self,
        num_input_features: int,#输入denselayer的通道数
        growth_rate: int,#增长率：每一层输出的通道数
        bn_size: int,#1*1卷积的channel是growth rate*bn_size，用于控制传递过程中的channels数
        #由于需要连接前面所有层输出的特征图，输入参数会非常大，因而在每一层的输入前加入1x1的卷积（瓶颈层），改变输出的通道数为growth rate*bn_size
        drop_rate: float,
        memory_efficient: bool = False
    ) -> None:
        super(_DenseLayer, self).__init__()
        self.norm1: nn.BatchNorm2d
        self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.relu1: nn.ReLU
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.conv1: nn.Conv2d
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False))
        self.norm2: nn.BatchNorm2d
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.relu2: nn.ReLU
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.conv2: nn.Conv2d
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False))
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

        # 定义瓶颈层输出的函数
    def bn_function(self, inputs: List[Tensor]) -> Tensor:
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output


    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input: List[Tensor]) -> bool:
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False


    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input: List[Tensor]) -> Tensor:
        def closure(*inputs):
            return self.bn_function(inputs)
        return cp.checkpoint(closure, *input)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input: List[Tensor]) -> Tensor:
            pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input: Tensor) -> Tensor:
            pass

        # torchscript does not yet support *args, so we overload method
        # allowing it to take either a List[Tensor] or single Tensor

    def forward(self, inputs: Tensor) -> Tensor:  # noqa: F811
        if isinstance(inputs, Tensor):
            prev_features = [inputs]
        else:
            prev_features = inputs

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
        memory_efficient: bool = False
    ) -> None:
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features=num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features: Tensor) -> Tensor:
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)#记录每一层的输出
            features.append(new_features)
        return torch.cat(features, 1)#将每一层的输出按照通道连接起来

#下采样过渡层
class _Transition(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


#上采样的过渡层没有pooling，需要增大featuremap的大小
class deconv_transition(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super(deconv_transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('deconv', nn.ConvTranspose2d(in_channels=num_output_features,
                                                     out_channels=num_output_features,
                                                     kernel_size=4, padding=1, stride=2))

#上采样的过渡层的第一个过渡层需要充分利用全连接层的输出，因此不需要1x1卷积
class first_deconv_transition(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super(first_deconv_transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('deconv', nn.ConvTranspose2d(in_channels=num_input_features,
                                                     out_channels=num_output_features,
                                                     kernel_size=4, padding=1, stride=2))


#整个网络框架
class Net(nn.Module):
    def __init__(
        self,
        growth_rate: int = 15,#每一层输出的通道数，通过设置1x1卷积的通道数
        num_init_features: int = 64,#预处理层中7x7卷积输出通道数，决定了进入网络中的通道数
        bn_size: int = 4,
        #compression_rate=0.5,
        drop_rate: float = 0,
        #num_classes: int = 2,
        memory_efficient: bool = False
    ) -> None:

        super(Net, self).__init__()
        self.srm_layer: SRMlayer
        self.add_module('srm_layer', SRMlayer())
        self.pre_conv0: nn.Conv2d
        self.add_module('pre_conv0', nn.Conv2d(6, num_init_features, kernel_size=7, stride=2, padding=3, bias=False))
        self.pre_norm0: nn.BatchNorm2d
        self.add_module('pre_norm0', nn.BatchNorm2d(num_init_features))
        self.pre_relu0: nn.ReLU
        self.add_module('pre_relu0', nn.ReLU(inplace=True))#inplace=True,表示将计算得到的值覆盖原始值
        self.max_pooling: nn.MaxPool2d
        self.add_module('max_pooling', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        num_features = num_init_features
        self.down_block1: _DenseBlock
        self.add_module('down_block1', _DenseBlock(num_layers=5, num_input_features=num_features,
                                                   bn_size=bn_size, growth_rate=growth_rate,
                                                   drop_rate=drop_rate,
                                                   memory_efficient=memory_efficient))
        num_features += 5*growth_rate#由瓶颈层数目为5的block，5层输出通道数5x15，通道数=瓶颈层数×增长率
        y3_num = num_features
        self.deconv_2x_y3: nn.ConvTranspose2d
        self.add_module('deconv_2x_y3', nn.ConvTranspose2d(in_channels=num_features, out_channels=15,
                                                           kernel_size=4, stride=2, padding=1))
        self.deconv_4x_y3: nn.ConvTranspose2d
        self.add_module('deconv_4x_y3', nn.ConvTranspose2d(in_channels=num_features, out_channels=15,
                                                           kernel_size=8, stride=4, padding=2))
        self.down_transition1: _Transition
        self.add_module('down_transition1', _Transition(num_input_features=num_features,
                                                        num_output_features=num_features//2))

        num_features = num_features//2#经过transitionlayer，通道数减半

        self.down_block2: _DenseBlock
        self.add_module('down_block2', _DenseBlock(num_layers=10, num_input_features=num_features,
                                                   bn_size=bn_size, growth_rate=growth_rate,
                                                   drop_rate=drop_rate,
                                                   memory_efficient=memory_efficient))
        num_features += (10*growth_rate)
        y4_num = num_features
        self.deconv_2x_y4: nn.ConvTranspose2d
        self.add_module('deconv_2x_y4', nn.ConvTranspose2d(in_channels=num_features, out_channels=15,
                                                             kernel_size=4, stride=2, padding=1))
        self.deconv_4x_y4: nn.ConvTranspose2d
        self.add_module('deconv_4x_y4', nn.ConvTranspose2d(in_channels=num_features, out_channels=15,
                                                            kernel_size=8, stride=4, padding=2))

        self.down_transition2: _Transition
        self.add_module('down_transition2', _Transition(num_input_features=num_features,
                                                        num_output_features=num_features//2))
        num_features = num_features//2
        self.down_block3: _DenseBlock
        self.add_module('down_block3', _DenseBlock(num_layers=20, num_input_features=num_features,
                                                   bn_size=bn_size, growth_rate=growth_rate,
                                                   drop_rate=drop_rate,
                                                   memory_efficient=memory_efficient))
        num_features += (20*growth_rate)



        self.linear: nn.Linear
        self.add_module('linear', nn.Linear(128*num_features*2*2, 128*10))


        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)


    def forward(self, inputs: Tensor) -> Tensor:
        y1 = self.srm_layer(inputs)#size=x*x(x为输入图像的高与宽）
        y2 = self.pre_conv0(y1)#size=x/2 * x/2
        y2 = self.pre_norm0(y2)
        y2 = self.pre_relu0(y2)

        y3 = self.max_pooling(y2)
        y3 = self.down_block1(y3)#size=x/4 * x/4


        y4 = self.down_transition1(y3)
        y4 = self.down_block2(y4)#size=x/8

        y5 = self.down_transition2(y4)
        feature = self.down_block3(y5)#size=x/16
        output = torch.flatten(feature)
        output = self.linear(output)
        output = output.reshape(128, 10)
        return output


