import torch
import torchvision
import torch.nn as nn
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

def get_dataloader_workers():
    """使用4个进程来读取数据"""
    return 4


def load_data_fashion_mnist(batch_size,resize=None):
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0,transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data",train=True,transform=trans,download=True
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data",train=False,transform=trans,download=True
    )
    return (data.DataLoader(mnist_train,batch_size,shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test,batch_size,shuffle=True,
                            num_workers=get_dataloader_workers()))

def train_epochs(net,train_iter,loss,trainer):
    "训练模型一个迭代周期"
    if isinstance(net,nn.Module):
        net.train()
    metric = d2l.Accumulator(3)
    for X,y in train_iter:
        y_hat = net(X)
        l = loss(y_hat,y)
        trainer.zero_grad()
        l.mean().backward()
        trainer.step()
        metric.add(float(l.sum().detach()),d2l.accuracy(y_hat,y),y.numel())
    return metric[0]/metric[2],metric[1]/metric[2]

def evaluate_loss(net,data_iter,loss):
    """评估给定数据集上的模型损失"""
    metric = d2l.Accumulator(2)
    for X,y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out,y)
        metric.add(l.sum().detach(),l.numel)
    return metric[0]/metric[1]