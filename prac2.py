import torch
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from torch import Tensor
from typing import Any, List, Tuple
from SRMlay import SRMLayer

#srm and concatation
class srmcon(nn.Module):
    def __init__(self):
        super(srmcon, self).__init__()
    def forward(self,inputfeatures:Tensor)->Tensor:
        init_features=SRMLayer(inputfeatures)
        init_features=torch.cat((init_features,inputfeatures))
        return init_features


class downcon(nn.Module):
    def __init__(self):
        super(downcon,self).__init__()
    def forward(self,inputfeature:Tensor)->Tensor:
        conv=nn.Conv2d(inputfeature.size(0),2,kernel_size=1,stride=1,bias=False)
        out=conv(inputfeature)
        out=torch.cat((out,inputfeature))
        return out
concat=downcon()


class _DenseLayer(nn.Module):
    def __init__(
        self,
        num_input_features: int,
        growth_rate: int,
        bn_size: int,
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
    def call_checkpoint_bottleneck(self, input: List[Tensor])->Tensor:
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

    def forward(self, input: Tensor) -> Tensor:  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

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
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features: Tensor) -> Tensor:
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class first_Transition(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super(first_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        #self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
        #                                  kernel_size=1, stride=1, bias=False))
        #self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class _deTransition(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super(_deTransition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        #self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))




class DenseNet(nn.Module):
    def __init__(
        self,
        growth_rate: int = 15,
        block_configdown: Tuple[int, int, int, int] = (5, 10, 20, 12),
        block_configup:Tuple[int,int,int]=(12,6,3),
        num_init_features: int = 64,
        bn_size: int = 4,
        compression_rate=0.5,
        drop_rate: float = 0,
        num_classes: int = 2,
        memory_efficient: bool = False
    ) -> None:

        super(DenseNet, self).__init__()
        #srmlayer
        self.srmlayer=srmcon()
        #First convolution
        self.pretreatment=nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
        ]))
        self.maxp=nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Each downsample denseblock
#block1+transition1
        num_features = num_init_features
        self.denseblock1=_DenseBlock(
                num_layers=5,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
        num_features = num_features + 5 * growth_rate
        self.transitionlayer1=_Transition(num_input_features=num_features,
                                    num_output_features=int(num_features*compression_rate))
        num_features = int(num_features * compression_rate)
###block2+transition2
        self.denseblock2 = _DenseBlock(
            num_layers=10,
            num_input_features=num_features,
            bn_size=bn_size,
            growth_rate=growth_rate,
            drop_rate=drop_rate,
            memory_efficient=memory_efficient
        )
        num_features = num_features + 10 * growth_rate
        self.transitionlayer2 = _Transition(num_input_features=num_features,
                                            num_output_features=int(num_features * compression_rate))
        num_features = int(num_features * compression_rate)
##block3+transition3+block4
        self.denseblock3 = _DenseBlock(
            num_layers=20,
            num_input_features=num_features,
            bn_size=bn_size,
            growth_rate=growth_rate,
            drop_rate=drop_rate,
            memory_efficient=memory_efficient
        )
        num_features = num_features + 20 * growth_rate
        self.transitionlayer3 = _Transition(num_input_features=num_features,
                                            num_output_features=int(num_features * compression_rate))
        num_features = int(num_features * compression_rate)
        self.denseblock4 = _DenseBlock(
            num_layers=12,
            num_input_features=num_features,
            bn_size=bn_size,
            growth_rate=growth_rate,
            drop_rate=drop_rate,
            memory_efficient=memory_efficient
        )
        num_features = num_features + 12 * growth_rate
        self.outblock = nn.Sequential()
        self.outblock.add_module('denseblock output norm',nn.BatchNorm2d(num_features))
        self.outblock.add_module('denseblock output relu',nn.ReLU(inplace=True))


#fully conv layer
        self.fullyconvlayer=nn.Conv2d(num_features,2,kernel_size=1,stride=1,bias=False)
#update the number of features(because of the concatation with fullyconvlayer(used 2 1x1 conv)
        num_features = num_features + 2
#first deconv-transition layer(no 1x1 conv,no pooling)
        self.detransitionlayer1=first_Transition(num_input_features=num_features,
                                    num_output_features=int(num_features * compression_rate))
        num_features = int(num_features * compression_rate)
#first upsample denseblock
        self.denseblock5=_DenseBlock(
                num_layers=12,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
        num_features = num_features + 12 * growth_rate
#second deconv-transition layer(with 1x1 conv,no pooling)+second denseblock
        self.detransitionlayer2=_deTransition(num_input_features=num_features,
                                    num_output_features=int(num_features * compression_rate))
        num_features = int(num_features * compression_rate)
        self.denseblock6=_DenseBlock(
                num_layers=6,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
        num_features = num_features + 6 * growth_rate
# third deconv-transition layer(with 1x1 conv,no pooling)+third denseblock
        self.detransitionlayer3=_deTransition(num_input_features=num_features,
                                    num_output_features=int(num_features * compression_rate))
        num_features = int(num_features * compression_rate)
        self.denseblock7=_DenseBlock(
                num_layers=3,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
        num_features = num_features + 3 * growth_rate

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        s1=self.srmlayer(x)
        contop=torch.cat((s1,x))#the output of the srmlayer concatenate with the original input
        precon=self.pretreatment(contop)
        prepooling=self.maxp(precon)
        block1=self.denseblock1(prepooling)
        tran1=self.transitionlayer1(block1)
        block2=self.denseblock2(tran1)
        tran2=self.transitionlayer2(block2)
        block3=self.denseblock3(tran2)
        tran3=self.transitionlayer3(block3)
        block4=self.denseblock4(tran3)
        outblock=self.outblock(block4)
        fullyconv=self.fullyconvlayer(outblock)
        con=torch.cat((fullyconv,block4))#the output of final denseblock  concatenate with the output of fully conv layer
        detran1=self.detransitionlayer1(con)
        block5=self.denseblock5(detran1)
        con1=torch.cat((block5,fullyconv,block3))#three output concatation

        return con1

a=DenseNet()
print(a)
