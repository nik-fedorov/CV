import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import numpy as np

from tools.constants import (
    N_CLASSES,
    TRAINVAL_PATH,
    TEST_PATH,
    TRAINVAL_LABELS_PATH,
    TEST_PREDICTIONS_PATH,
)
from tools.dataset import DatasetBHW
from tools.train_test_funcs import train_with_wandb, make_predictions
from tools.utils import get_scripts


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(in_channels, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += self.bn3(self.conv3(x))

        out = F.relu(out)

        return out


class StridedBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += self.bn3(self.conv3(x))

        out = F.relu(out)

        return out


class BigNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.blocks = nn.Sequential(
            BasicBlock(3, 32),
            StridedBasicBlock(32, 64),
            BasicBlock(64, 64),
            StridedBasicBlock(64, 128),
            BasicBlock(128, 128),
            StridedBasicBlock(128, 254),
            BasicBlock(254, 254),
            StridedBasicBlock(254, 512)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(512, N_CLASSES)

    def forward(self, x):
        out = self.blocks(x)
        out = self.avgpool(out)
        out = out.squeeze(-1).squeeze(-1)
        out = self.linear(out)
        return out


if __name__ == '__main__':
    scripts_contents = get_scripts('bignet.py', 
                                   'tools/dataset.py',
                                   'tools/constants.py',
                                   'tools/train_test_funcs.py',
                                   'tools/utils.py')
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    dataloader_num_workers = 6
    batch_size = 32
    
    train_transform = T.Compose([
        T.RandomChoice([
            T.RandomChoice([
                T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
                T.AutoAugment(T.AutoAugmentPolicy.IMAGENET),
                T.AutoAugment(T.AutoAugmentPolicy.SVHN),
            ]),
            
            T.TrivialAugmentWide(),
            
            T.RandomChoice([
                T.RandAugment(num_ops=5),
                T.RandAugment(num_ops=6),
                T.RandAugment(num_ops=7),
            ]),
            
            T.Compose([
                T.RandomResizedCrop(64, scale=(0.6, 1.0)),
                T.RandomApply([T.RandomChoice([
                    T.ColorJitter(brightness=.5, hue=.3),
                    T.Grayscale(num_output_channels=3),
                ])], p=0.7),
                T.RandomChoice([
                    T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
                    T.AugMix(),
                ]),
                T.RandomChoice([
                    T.RandomPerspective(),
                    T.RandomRotation(30)
                ]),
            ])
        ]),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_set = DatasetBHW(TRAINVAL_PATH, TRAINVAL_LABELS_PATH, train=True, transform=train_transform)
    val_set = DatasetBHW(TRAINVAL_PATH, TRAINVAL_LABELS_PATH, train=False, transform=test_transform)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=dataloader_num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=dataloader_num_workers)
    
    # ========== START Edit ==========
    gamma = 0.98
    net = BigNet().to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    num_epochs = 300

    wandb_init_data = {
        'project': 'intro-to-dl-bhw-01',
        'name': f'BigNet without last basic block',
        'config': {
            'model': {
                'name': 'BigNet'
            },
            'optimizer': {
                'name': 'SGD',
                'init_lr': 0.1,
                'momentum': 0.9
            },
            'scheduler': {
                'name': 'ExponentialLR',
                'gamma': gamma
            },
            'augmentations': train_transform,

            'dataset': 'bhw',
            'num_epochs': num_epochs,
            'train_loader_batch_size': batch_size,
            'dataloader_num_workers': dataloader_num_workers,
            'scripts': scripts_contents
        }
    }

    train_with_wandb(net, optimizer, num_epochs, train_loader, val_loader, device,
                     scheduler=scheduler, wandb_init_data=wandb_init_data)
    
#     torch.save({
#             'epochs': num_epochs,
#             'model_state_dict': net.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None
#         }, 'bignet.pt')
    
#     make_predictions(TEST_PATH, test_transform, net, device, 'bignet_' + TEST_PREDICTIONS_PATH)
