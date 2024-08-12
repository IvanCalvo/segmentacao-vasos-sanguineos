import random
import json
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transf
from torchvision import tv_tensors

class Subset(Dataset):

    def __init__(self, ds, indices, transform=None):
        self.ds = ds
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):

        img, target = self.ds[self.indices[idx]]
        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.indices)

class VessMAP(Dataset):

    def __init__(self, root, transforms=None, ignore_val=2):

        root = Path(root)
        images_folder = root / "images"
        segs_folder = root / "annotator1/labels"
        anns_file = root / "annotator1/measures.json"

        images = []
        segs = []

        ann_file = open(anns_file)
        dados = json.load(ann_file)
        names = dados.keys()
        for name in names:
            images.append(images_folder/f'{name}.tiff')
            segs.append(segs_folder/f'{name}.png')
            
        self.images = images
        self.segs = segs
        self.transforms = transforms
        self.ignore_val = ignore_val

    def __getitem__(self, idx, apply_transform=True):
        
        image = Image.open(self.images[idx]).convert("RGB")
        target_or = Image.open(self.segs[idx])

        target_np = np.array(target_or)
        target_np[target_np==255] = 1

        target = Image.fromarray(target_np, mode="L")
        
        if self.transforms and apply_transform:
            image, target = self.transforms(image, target)

        return image, target
    
    def __len__(self):
        return len(self.images)

class TransformsTrain:

    def __init__(self, resize_size=256):
    
        transforms = transf.Compose([
            transf.PILToTensor(),
            transf.Grayscale(num_output_channels=3),
            transf.RandomResizedCrop(size=(resize_size,resize_size), scale=(0.9,1.), 
                                     ratio=(0.9,1.1), antialias=True),
            transf.RandomHorizontalFlip(),
            transf.RandomVerticalFlip(),
            transf.ToDtype({tv_tensors.Image: torch.float32, tv_tensors.Mask: torch.int64}),
            transf.Normalize(mean=(33.3, 33.3, 33.3), std=(10.9, 10.9, 10.9))
        ])

        self.transforms = transforms

    def __call__(self, img, target):
        img = tv_tensors.Image(img)
        target = tv_tensors.Mask(target)
        img, target = self.transforms(img, target)
        img = img.data
        target = target.data
        target = target.squeeze()
        return img, target

class TransformsEval:

    def __init__(self, resize_size=256):

        transforms = transf.Compose([
            transf.PILToTensor(),   
            transf.Grayscale(num_output_channels=3),
            transf.Resize(size=resize_size, antialias=True),
            transf.ToDtype({tv_tensors.Image: torch.float32, tv_tensors.Mask: torch.int64}),
            transf.Normalize(mean=(33.3, 33.3, 33.3), std=(10.9, 10.9, 10.9))

        ])

        self.transforms = transforms

    def __call__(self, img, target):
        img = tv_tensors.Image(img)
        target = tv_tensors.Mask(target)
        img, target = self.transforms(img, target)
        img = img.data
        target = target.data
        target = target.squeeze()
        return img, target

def unormalize(img):
    img = img.permute(1, 2, 0)
    mean = torch.tensor([122.7, 114.6, 100.9])
    std = torch.tensor([59.2, 58.4, 59.0])
    img = img*std + mean
    img = img.to(torch.uint8)

    return img

def get_dataset(root, split=0.2, resize_size=384):
    class_weights = (0.33, 0.67)

    ds = VessMAP(root)
    n = len(ds)
    n_valid = int(n*split)

    indices = list(range(n))
    random.seed(42)
    random.shuffle(indices)
    
    ds_train = Subset(ds, indices[n_valid:], TransformsTrain(resize_size))
    ds_valid = Subset(ds, indices[:n_valid], TransformsEval(resize_size))

    return ds_train, ds_valid, class_weights