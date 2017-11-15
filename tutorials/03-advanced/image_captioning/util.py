from __future__ import division

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import argparse
import pprint
import os

import pandas as pd
import numpy as np

from PIL import Image
from scipy import misc
import sklearn.metrics
import pickle
import argparse

class Dataset(data.Dataset):
    def __init__(self, label_dir, vocab, toy=False):
        super(Dataset, self).__init__()
        self.vocab = vocab
        df = pd.read_csv(label_dir, sep = '|')
        if toy:
            df = df.sample(frac=0.01)

            # Remove any paths for which the image files do not exist
            df = df[df["Path"].apply(os.path.exists)]

        self.img_paths = df["Path"].tolist()
        # To convert from label vector to natural language labels,
        # use label2ind.txt (maps label to an index [line index])
        # the index of a label vector is 1 if that label appears, 0 otherwise
        label = [] 
        label.append(self.vocab('<start>'))
        for i, row in df.iterrows():
                row_label = row["Label"].split("|")
                for r in row_label:
                        label.append(self.vocab(r))
        label.append(self.vocab('<end>'))
        self.labels = label
        
        self.transform = transforms.Compose([
                            transforms.Scale([342, 2270]),
                            transforms.ToTensor(),
                        ])

    def __getitem__(self, index):
           img = Image.open(self.img_paths[index]).convert("RGB")
           label = self.labels[index]
                
           return self.transform(img), torch.Tensor(label)

    def __len__(self):
        return len(self.img_paths)
def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths

def load_data(args, vocab):
    train_dataset = Dataset("data/train.csv", vocab, toy=args.toy)
    train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers,  collate_fn=collate_fn)

    test_dataset = Dataset("data/test.csv",vocab, toy=args.toy)
    test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_fn)

    return train_loader, test_loader
