# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""


import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader
import os
from loss.circle_loss import convert_label_to_similarity, CircleLoss
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def loss_function1(outputs, labels):
    # print("type(outputs):",type(outputs))
    # print("type(labels):",type(labels))
    y_hat = F.softmax(outputs,dim=1)
    # print("y_hat:", y_hat.size())
    max_l = torch.pow(F.relu(0.9 - y_hat),2)
    # print("max_l:", max_l.size())
    # max_l = np.reshape(max_l, shape=(-1, 3))
    max_r = torch.pow(F.relu( y_hat - 0.1),2)
    # print("max_r:",max_r.size())
    # max_r = np.reshape(max_r, shape=(-1, 3))
    labels=torch.unsqueeze(labels,dim=1).cuda()
    # print(labels.size())
    # print(labels.size(0))
    t_c = torch.zeros(outputs.size(0),3).cuda().scatter_(1, labels, 1)
    # print(t_c)
    m_loss = t_c * max_l + 0.5 * (1. - t_c) * max_r
    loss_sum = torch.sum(m_loss, dim=1)
    loss = torch.mean(loss_sum)
    # print("loss:",loss.size())
    return loss


def train(epoch):

    net.train()
    for batch_index, (images, labels) in enumerate(tumor_training_loader):
        train_batch_correct = 0.0
        if epoch <= args.warm:
            optimizer.step()
        images = Variable(images)
        labels = Variable(labels)

        labels = labels.cuda()
        images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        _, preds = outputs.max(1)
        train_batch_correct += preds.eq(labels).sum()

        loss = loss_function(*convert_label_to_similarity(outputs, labels))  + loss_function1(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(tumor_training_loader) + batch_index + 1

        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)
        if batch_index % 50 == 49:
            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                loss.item(),
                optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_index * args.b + len(images),
                total_samples=len(tumor_training_loader.dataset)
            ))
        # batch_size_train acc
        writer.add_scalar("Train/acc", train_batch_correct.float()/args.b, n_iter)
        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

def eval_training(epoch):
    net.eval()

    test_loss = 0.0 # cost function error
    train_loss = 0.0
    train_correct = 0.0
    correct = 0.0
    # train set
    for (images, labels) in tumor_training_loader:
        images = Variable(images)
        labels = Variable(labels)

        images = images.cuda()
        labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(*convert_label_to_similarity(outputs, labels)) + loss_function1(outputs, labels)
        train_loss += loss.item()
        _, preds = outputs.max(1)
        train_correct += preds.eq(labels).sum()

    print('Train set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        train_loss / len(tumor_training_loader.dataset),
        train_correct.float() / len(tumor_training_loader.dataset)
    ))
    print()
    # test_set
    for (images, labels) in tumor_test_loader:
        images = Variable(images)
        labels = Variable(labels)

        images = images.cuda()
        labels = labels.cuda()

        outputs = net(images)
        print(outputs.size())
        print(labels.size())

        loss = loss_function(*convert_label_to_similarity(outputs, labels)) + loss_function1(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        test_loss / len(tumor_test_loader.dataset),
        correct.float() / len(tumor_test_loader.dataset)
    ))
    print()

    #add informations to tensorboard
    writer.add_scalar('Test/Average loss', test_loss / len(tumor_test_loader.dataset), epoch)
    writer.add_scalar('Test/Accuracy', correct.float() / len(tumor_test_loader.dataset), epoch)

    writer.add_scalar('Train/Average loss', train_loss / len(tumor_training_loader.dataset), epoch)
    writer.add_scalar('Train/Accuracy', train_correct.float() / len(tumor_training_loader.dataset), epoch)
    return correct.float() / len(tumor_test_loader.dataset)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=1, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=25, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate')
    args = parser.parse_args()

    net = get_network(args, use_gpu=args.gpu)
        
    #data preprocessing:
    tumor_training_loader = get_training_dataloader(
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )
    
    tumor_test_loader = get_test_dataloader(
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )
    
    # loss_function = nn.CrossEntropyLoss()
    loss_function = CircleLoss(m = settings.M, gamma=settings.GAMMA)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=np.sqrt(0.1))
    #train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(tumor_training_loader)
    #warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    writer = SummaryWriter(logdir=os.path.join(
            settings.LOG_DIR, args.net, settings.TIME_NOW))
    # input_tensor = torch.Tensor(20, 1, 512, 512).cuda()
    # with writer:
    #     writer.add_graph(net, Variable(input_tensor, requires_grad=True))

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    for epoch in range(1, settings.EPOCH):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        train(epoch)
        acc = eval_training(epoch)

        #start to save best performance model after learning rate decay to 0.01 
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))

    writer.close()
