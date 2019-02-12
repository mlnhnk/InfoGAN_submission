import os

import torch
from torch.utils.data import DataLoader
import utils

from torchvision import datasets
from torchvision import transforms


def get_emoji_loader(data, opts):
    """Creates training and test data loaders.
    """
    transform = transforms.Compose([
                    transforms.Scale(opts.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

    train_path = os.path.join('./data', emoji_type)
    test_path = os.path.join('./emdataojis', 'Test_{}'.format(emoji_type))

    train_dataset = datasets.ImageFolder(train_path, transform)
    test_dataset = datasets.ImageFolder(test_path, transform)

    train_dloader = DataLoader(dataset=train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers)
    test_dloader = DataLoader(dataset=test_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers)

    return train_dloader, test_dloader

def get_celeba_loader(opts):
    transform = transforms.Compose([
                    transforms.Scale(32),
                    transforms.ToTensor()
                ])

#    train_path = os.path.join('./data/', emoji_type)
    train_dataset = datasets.ImageFolder('celeba/', transform)
    train_dloader = DataLoader(dataset=train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers)

    return train_dloader

def get_mnist_data(opts):
    # Check if data exists and load it:
    data_path = utils.check_mnist_dataset_exists()
    
    train_data=torch.load(data_path+'mnist/train_data.pt').unsqueeze(1)
    train_label=torch.load(data_path+'mnist/train_label.pt').float().unsqueeze(1)
    test_data=torch.load(data_path+'mnist/test_data.pt').unsqueeze(1)
    test_label=torch.load(data_path+'mnist/test_label.pt').float().unsqueeze(1)
    
    print(train_data.size())
    print(test_data.size())
    
    train = []
    for i in range(train_data.size()[0]):
        train.append((train_data[i], train_label[i]))
        
    test = []
    for i in range(test_data.size()[0]):
        test.append((test_data[i], test_label[i]))
    
    train_dloader = DataLoader(dataset=train, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers)
    test_dloader = DataLoader(dataset=test, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers)
    
    return train_dloader, test_dloader
