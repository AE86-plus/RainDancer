import h5py
import numpy as np
import torch
import os
import sys
from data.spatial_transform import Random_crop, ToTorchFormatTensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
import time
import torch.distributed as dist

def make_dataset(source, mode):
    #root:'./datasets/hmdb51_frames'
    #source:'./datasets/settings/hmdb51/train_rgb_split1.txt'
    if not os.path.exists(source):
        print("Setting file %s for hmdb51 dataset doesn't exist." % (source))
        sys.exit()
    else:
        rgb_samples = []
        with open(source) as split_f:
            data = split_f.readlines()
            for line in data:
                line_info = line.split()[0]
                rgb_samples.append(line_info)

        print('{}: {} sequences have been loaded'.format(mode, len(rgb_samples)))
    return rgb_samples

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class SequenceDataset(Dataset):

    def __init__(self, h5_file, train_txt, seq_len):

        # self.h5 = h5py.File(h5_file, "r")

        self.h5_file = h5_file

        self.crop_size = 128

        self.seq_len = seq_len

        self.sequence_samples = make_dataset(train_txt, "train")

        self.num_sequences = len(self.sequence_samples)

        ###
        self.transform = torchvision.transforms.Compose([
                                Random_crop(self.crop_size, self.seq_len),
                                ToTorchFormatTensor(),
                                ])

        # print(self.gan_list)
        # import pdb
        # pdb.set_trace()

    def open_h5(self):

        self.h5 = h5py.File(self.h5_file, "r")
        self.h5_format = self._detect_h5_format()

    def _detect_h5_format(self):
        top_keys = set(self.h5.keys())
        if {"input", "processed"}.issubset(top_keys):
            return "rainsyn"
        return "legacy"

    def get_sequence(self, idx, transform=True):

        path_parts = self.sequence_samples[idx].split('/')

        rainy_frame, clean_frame, rainy_event, clean_event = [], [], [], []

        if self.h5_format == "rainsyn":
            if len(path_parts) != 3:
                raise ValueError("RainSyn list entry should be formatted as input/<scene>/<frame_idx>")
            _, scene_key, img_idx = path_parts
            start_idx = int(img_idx)

            for i in range(self.seq_len):
                rainy_frame.append(
                    self.h5["input/{}/{:05d}".format(scene_key, start_idx + i)][:][:, :, ::-1]
                )
                clean_frame.append(
                    self.h5["processed/{}/{:05d}".format(scene_key, start_idx + i)][:][:, :, ::-1]
                )

            for i in range(self.seq_len - 1):
                rainy_event.append(
                    self.h5["input/{}/voxel/{:05d}".format(scene_key, start_idx + i)][:]
                )
                clean_event.append(
                    self.h5["processed/{}/voxel/{:05d}".format(scene_key, start_idx + i)][:]
                )
        else:
            key, sub_key, img_idx = path_parts
            start_idx = int(img_idx)

            for i in range(self.seq_len):
                rainy_frame.append(
                    self.h5["{}/{}/{}/{:05d}".format(key, sub_key, "rainy", start_idx + i)][:][:, :, ::-1]
                )
                clean_frame.append(
                    self.h5["{}/{}/{:05d}".format(key, "gt", start_idx + i)][:][:, :, ::-1]
                )

            for i in range(self.seq_len - 1):
                rainy_event.append(
                    self.h5["{}/{}/{}/{:05d}".format(key, sub_key, "voxel", start_idx + i)][:]
                )
                clean_event.append(
                    self.h5["{}/{}/{:05d}".format(key, "clean_voxel", start_idx + i)][:]
                )

        item = {
            "rainy": rainy_frame,
            "gt": clean_frame,
            "rainy_events": rainy_event,
            "clean_events": clean_event
        }

        if transform:
            item = self.transform(item)

        return item

    def __len__(self):

        return self.num_sequences

    def __getitem__(self, idx):

        if not hasattr(self, "h5"):
            self.open_h5()

        sequence = self.get_sequence(idx)
        return sequence
