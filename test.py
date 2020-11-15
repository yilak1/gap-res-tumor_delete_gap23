#test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model

author liudunsheng
"""
import numpy as np
import matplotlib.pyplot
import argparse
#from dataset import *

#from skimage import io
from matplotlib import pyplot as plt

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

from conf import settings
import matplotlib.pyplot as plt
from utils import get_network, get_test_dataloader
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 展示错误分类图片
def show_false_classification_image(images, predicted, label):
    print("predicted: %d and label: %d"%(predicted, label))
    images = images.to('cpu')
    images = images.numpy().transpose((1, 2, 0)).squeeze()
    print(images.shape)
    plt.imshow(images, cmap='gray')
    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    args = parser.parse_args()

    net = get_network(args)

    tumor_test_loader = get_test_dataloader(
        #settings.CIFAR100_PATH,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )

    net.load_state_dict(torch.load(args.weights), args.gpu)
    print(net)
    net.eval()

    # correct_1 = 0.0
    correct =0.0
    total = 0.0
    # 计算总损失
    for n_iter, (images, labels) in enumerate(tumor_test_loader):
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        output = net(images)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print()
    print("Accuracy of network on the test images: %.3f%%"%(100*correct/total))
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))

    # 计算每个类别损失
    classes = ['0', '1', '2']
    class_correct = list(0. for i in range(3))
    class_total = list(0. for i in range(3))
    false_image = {}
    with torch.no_grad():
        for (images, labels) in tumor_test_loader:
            images = Variable(images).cuda()
            labels = Variable(labels).cuda()
            output = net(images)
            _, predicted = torch.max(output, 1)
            # print(predicted)
            c = (predicted == labels).squeeze()
            # print(c)
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

            # 查找分类错误的图片
            for i, label_false  in enumerate(c):
                if label_false == 0:
                    plt.figure()
                    #print("images size", images[i].size())
                    #show_false_classification_image(images[i], predicted[i], labels[i])

    # 3个类别
    for i in range(3):
        print('Accuracy of %s : %.3f %%' % (classes[i], 100 * class_correct[i]/ class_total[i]))


