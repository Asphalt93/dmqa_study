import os
import torch
import numpy as np
from torchvision import transforms, datasets


## Data Loader 구현
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        lst_data = os.listdir(self.data_dir)

        lst_label = [f for f in lst_data if f.startswith('label')]
        lst_image = [f for f in lst_data if f.startswith('input')]

        lst_label.sort()
        lst_image.sort()

        self.lst_label = lst_label
        self.lst_image = lst_image
    
    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self, index):
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        image = np.load(os.path.join(self.data_dir, self.lst_image[index]))

        label = label/255.0
        image = image/255.0

        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if image.ndim == 2:
            image = image[:, :, np.newaxis]

        data = {'image':image, 'label':label}

        if self.transform:
            data = self.transform(data)

        return data


'''
이미지의 numpy  차원 = (Y, X, CH)
이미지의 tensor 차원 = (CH, Y, X)
따라서 Transpose가 필요하다.
'''

class ToTensor(object):
    def __call__(self, data):
        label, image = data['label'], data['image']

        label = label.transpose((2, 0, 1)).astype(np.float32)
        image = image.transpose((2, 0, 1)).astype(np.float32)

        data = {'label': torch.from_numpy(label),
                'image': torch.from_numpy(image)       
            }

        return data


class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, image = data['label'], data['image']

        image = (image - self.mean) / self.std

        data = {'label': label,
                'image': image       
            }

        return data


class RandomFlip(object):
    def __call__(self, data):
        label, image = data['label'], data['image']
        
        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            image = np.filplr(image)

        if np.random.rand() > 0.5:
            label = np.flipud(label)
            image = np.flipud(image)

        data = {'label':label, 'image':image}

        return data



## 네트워크 저장하기
def saver(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    
    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
    "./%s/model_epoch%d.pth"  % (ckpt_dir, epoch))


## 네트워크 불러오기
def loader(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch
    
    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('./%s/%s'%(ckpt_dir, ckpt_lst[-1]))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return net, optim, epoch