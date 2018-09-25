import glob
import random
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class ImageDataset(Dataset):
    def __init__(self, args, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.args = args
        self.unaligned = unaligned
        self.files_X = sorted(glob.glob(os.path.join(root, '%sA' % mode) + '/*.*'))
        self.files_Y = sorted(glob.glob(os.path.join(root, '%sB' % mode) + '/*.*'))


    def __getitem__(self, index):

        img_X = Image.open(self.files_X[index % len(self.files_X)])
        if self.unaligned:
            img_Y = Image.open(self.files_Y[random.randint(0, len(self.files_Y)-1)])
        else:
            img_Y = Image.open(self.files_Y[index % len(self.files_Y)] )

        img_X = self.transform(img_X)
        img_Y = self.transform(img_Y)

        if self.args.input_nc_A == 1:  # RGB to gray
            img_X = img_X.convert('L')

        if self.args.input_nc_B == 1:  # RGB to gray
            img_Y = img_Y.convert('L')

        return {'X': img_X, 'Y': img_Y}

    def __len__(self):
        return max(len(self.files_X), len(self.files_Y))


# Configure dataloaders
def Get_dataloader(args):
    # Image transformations
    transforms_ = [ transforms.Resize(int(args.img_height*1.12), Image.BICUBIC),
                transforms.RandomCrop((args.img_height, args.img_width)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

    train_dataloader = DataLoader(ImageDataset(args, "%s/%s" % (args.data_root, args.dataset_name), transforms_=transforms_,unaligned=True,mode='train'),
                        batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu//2, drop_last=True)

    test_dataloader = DataLoader(ImageDataset(args, "%s/%s" % (args.data_root, args.dataset_name), transforms_=transforms_, unaligned=True,mode='test'),
                            batch_size=4, shuffle=True, num_workers=1, drop_last=True)

    val_dataloader = DataLoader(ImageDataset(args, "%s/%s" % (args.data_root, args.dataset_name), transforms_=transforms_, unaligned=True, mode='val'),
                            batch_size=4, shuffle=True, num_workers=1, drop_last=True)

    return train_dataloader, test_dataloader, val_dataloader
