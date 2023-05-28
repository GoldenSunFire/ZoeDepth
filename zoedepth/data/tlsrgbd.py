import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ToTensor(object):
    def __init__(self):
        # self.normalize = transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.normalize = lambda x : x
	self.resize = transforms.Resize((480, 640))

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        image = self.to_tensor(image)
        image = self.normalize(image)
        depth = self.to_tensor(depth)

        return {'image': image, 'depth': depth, 'dataset': "tlsrgbd"}

    def to_tensor(self, pic):

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        #         # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img


class ToulouseRGBD(Dataset):
    def __init__(self, data_dir_root):
#        self.image_files = []
#        self.depth_files = []
#        with open(os.path.join(data_dir_root, "train_files_with_gt.txt"), "r") as file:
#            for line in file:
#                image_name, depth_name, _ = line.strip().split()
#                image_path = os.path.join(data_dir_root, "images", image_name)
#                depth_path = os.path.join(data_dir_root, "depth", depth_name)
#                self.image_files.append(image_path)
#                self.depth_files.append(depth_path)
        self.image_files = []
        self.depth_files = []
        image_dir = os.path.join(data_dir_root, "images")
        depth_dir = os.path.join(data_dir_root, "depth")
        for filename in os.listdir(image_dir):
            if filename.endswith(".png"):
                image_path = os.path.join(image_dir, filename)
                depth_filename = filename.replace("_rgb.png", "_gt.png")
                depth_path = os.path.join(depth_dir, depth_filename)
                self.image_files.append(image_path)
                self.depth_files.append(depth_path)
        self.transform = ToTensor()

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        depth_path = self.depth_files[idx]

        image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0
        depth = np.asarray(Image.open(depth_path), dtype='uint16') / 1000.0
        depth[depth > 8] = -1
        depth = depth[..., None]

        return self.transform({"image": image, "depth": depth})

    def __len__(self):
        return len(self.image_files)

def tlsrgbd_loader(data_dir_root, batch_size=1, **kwargs):
    dataset = MyDataset(data_dir_root)
    return DataLoader(dataset, batch_size, **kwargs)