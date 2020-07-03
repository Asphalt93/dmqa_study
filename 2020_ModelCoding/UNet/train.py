##라이브러리 추가하기
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms, datasets

from model import UNet
from utils import Dataset, ToTensor, Normalization, RandomFlip, saver, loader


## 트레이닝 파라메터 설정하기
lr = 1e-3
batch_size = 4
num_epoch = 100

data_dir = './data'
ckpt_dir = './checkpoint'
log_dir = './log'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Call Transform
transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])

## 데이터셋 생성
dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)

dataset_val = Dataset(data_dir=os.path.join(data_dir, 'val'), transform=transform)
loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=4)

## 네트워크 생성
net = UNet().to(device)

## 손실함수 정의
fn_loss = nn.BCEWithLogitsLoss().to(device)

## Optimizer 설정하기
optim = torch.optim.Adam(net.parameters(), lr=lr)

## 그 밖에 부수적인 variables 설정하기
num_data_train = len(dataset_train)
num_data_val = len(dataset_val)

num_batch_train = np.ceil(num_data_train / batch_size)
num_batch_val = np.ceil(num_data_val / batch_size)

## 그 밖에 부수적인 functions 설정하기
fn_tonumpy = lambda x: x.to('cpu').detach().numpy.transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x*std) + mean
fn_class = lambda x: 1.0 * (x > 0.5)

## Tensorboard를 사용하기 위한 SummaryWriter 설정
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))


## 네트워크 학습시키기
st_epoch = 0
net, optim, st_epoch = loader(ckpt_dir=ckpt_dir, net=net, optim=optim)

for epoch in range(st_epoch + 1, num_epoch + 1):
    net.train()
    loss_arr = []

    for batch, data in enumerate(loader_train, 1):
        #forward pass
        label = data['label'].to(device)
        image = data['image'].to(device)

        output = net(image)

        #backward pass
        optim.zero_grad()

        loss = fn_loss(output, label)
        loss.backward()

        optim.step()

        # 손실함수 표시
        loss_arr += [loss.item()]

        print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f"%(epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))

        # Tensorboard 저장하기
        label = fn_tonumpy(label)
        image = fn_tonumpy(fn_denorm(image, mean=0.5, std=0.5))
        output = fn_tonumpy(fn_class(output))

        writer_train.add_image('label', label, num_batch_train * (epoch -1) + batch, dataformats='NHWC')
        writer_train.add_image('image', image, num_batch_train * (epoch -1) + batch, dataformats='NHWC')
        writer_train.add_image('output', output, num_batch_train * (epoch -1) + batch, dataformats='NHWC')
    # Tensorboard 저장하기 (손실함수 저장)
    writer_train.add_scalar('loss', np.mean(loss_arr), epoch)

    with torch.no_grad():
        net.eval()
        loss_arr = []

        for batch, data in enumerate(loader_val, 1):
            # forward pass
            label = data['label'].to(device)
            image = data['image'].to(device)

            output = net(image)

            # 손실함수 계산하기
            loss = fn_loss(output, label)

            loss_arr += [loss.item()]

            print("VALID: EPOCH %04d / %04d | BATCH % 04d / %04d | LOSS %.4f"% (epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr)))

        # Tensorbard 저장하기
        label = fn_tonumpy(label)
        image = fn_tonumpy(fn_denorm(image))
        output = fn_tonumpy(fn_class(output))

        writer_val.add_image('label', label, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
        writer_val.add_image('image', image, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
        writer_val.add_image('output', output, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
    # Tensorboard 저장하기 (손실함수 저장)
    writer_val.add_scalar('loss', np.mean(loss_arr), epoch)

    if epoch // 5 == 0:
        saver (ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)
     

writer_train.close()
writer_val.close()





## 데이터 시각화
data = dataset_train.__getitem__(0)

image = data['image']
label = data['label']

plt.subplot(121)
plt.imshow(image.squeeze()) # squeeze를 하지 않으면 데이터의 shape가 (512, 512, 1)이 된다.
                            # squeeze로 차원 하나를 제거하여 (512,512)로 만든다.
plt.subplot(122)
plt.imshow(label.squeeze())

plt.show()

