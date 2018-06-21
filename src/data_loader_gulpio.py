import os
import sys
import time
import numpy as np
import torch
import torch.utils.data as data

from pprint import pprint
from torchvision.transforms import *
from gulpio import GulpDirectory
from data_parser import GulpDataset
from utils import save_images_for_debug


class VideoFolder(data.Dataset):

    def __init__(self, root, csv_file_input, csv_file_labels, clip_size,
                 nclips, step_size, is_val, transform=None,):

        self.dataset_object = GulpDataset(csv_file_input, csv_file_labels)

        self.csv_data = self.dataset_object.csv_data
        self.classes_dict = self.dataset_object.classes_dict
        self.classes = self.dataset_object.classes

        self.gulp_directory = GulpDirectory(root)
        self.merged_meta_dict = self.gulp_directory.merged_meta_dict

        self.transform = transform

        self.clip_size = clip_size
        self.nclips = nclips
        self.step_size = step_size
        self.is_val = is_val

    def __getitem__(self, index):
        item = self.csv_data[index]

        num_frames = len(self.merged_meta_dict[item.id]['frame_info'])
        target_idx = self.classes_dict[item.label]

        if self.nclips > -1:
            num_frames_necessary = self.clip_size * self.nclips * self.step_size
        else:
            num_frames_necessary = num_frames
        offset = 0
        if num_frames_necessary < num_frames:
            # If there are more frames, then sample starting offset.
            diff = (num_frames - num_frames_necessary)
            # temporal augmentation
            if not self.is_val:
                offset = np.random.randint(0, diff)

        slice_object = slice(
            offset, num_frames_necessary + offset, self.step_size)

        frames, meta = self.gulp_directory[item.id, slice_object]
        if len(frames) < (self.clip_size * self.nclips):
            frames.extend([frames[-1]] *
                          ((self.clip_size * self.nclips) - len(frames)))
        imgs = []
        for img in frames:
            img = self.transform(img)
            imgs.append(torch.unsqueeze(img, 0))

        # format data to torch
        data = torch.cat(imgs)
        data = data.permute(1, 0, 2, 3)

        return (data, target_idx)

    def __len__(self):
        return len(self.csv_data)


if __name__ == "__main__":
    transform = Compose([
                        ToPILImage(),
                        CenterCrop(84),
                        ToTensor(),
                        # Normalize(mean=[0.485, 0.456, 0.406],
                        #           std=[0.229, 0.224, 0.225])
                        ])
    loader = VideoFolder(root="/media/deepak/deepak/DeepLearning/MajorProject/DataSet/20bn-jester-v1-gulpio/",
                         csv_file_input="/media/deepak/deepak/DeepLearning/MajorProject/DataSet/csv_files/jester-v1-validation.csv",
                         csv_file_labels="/media/deepak/deepak/DeepLearning/MajorProject/DataSet/csv_files/jester-v1-labels.csv",
                         clip_size=18,
                         nclips=1,
                         step_size=2,
                         is_val=False,
                         transform=transform,
                         )

    # data, target_idx = loader[0]
    # save_images_for_debug("input_images", data.unsqueeze(0))

    train_loader = torch.utils.data.DataLoader(
        loader,
        batch_size=10, shuffle=False,
        num_workers=5, pin_memory=True)

    start = time.time()
    for i, a in enumerate(train_loader):
        if i == 49:
            break
    print("Size --> {}".format(a[0].size()))
    print(time.time() - start)
