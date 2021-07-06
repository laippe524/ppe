import numpy
import cv2
import torch
from patent_net import Net


a = numpy.load("/home/lailie/Downloads/数据集/patent/prcv2021-patent-design.npy")
net = Net()


for index, image in enumerate(a, 1):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image.transpose((2, 0, 1)))
    image.unsqueeze_(0)
    if (not (index % 129)) or index == 1:
        images = image
        continue
    images = torch.cat([images, image], dim=0)
    if not (images.shape[0] % 128):
        images = images.type(torch.FloatTensor)
        out = net(images)
        print(out.shape)






