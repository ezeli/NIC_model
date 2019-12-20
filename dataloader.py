# coding:utf8
import torch
from torch.utils import data
import numpy as np


def create_collate_fn(pad_index, max_sql_len):
    def collate_fn(dataset):
        dataset.sort(key=lambda p: len(p[1]), reverse=True)
        fns, caps, fc_feats = zip(*dataset)
        fc_feats = torch.FloatTensor(np.array(fc_feats))

        lengths = [min(len(c), max_sql_len) for c in caps]
        caps_tensor = torch.LongTensor(len(caps), lengths[0]).fill_(pad_index)
        for i, c in enumerate(caps):
            end_cap = lengths[i]
            caps_tensor[i, :end_cap] = torch.LongTensor(c[:end_cap])
        lengths = [l-1 for l in lengths]
        return fns, fc_feats, (caps_tensor, lengths)

    return collate_fn


class CaptionDataset(data.Dataset):
    def __init__(self, img_feats, img_captions):
        self.feats = img_feats
        self.captions = []
        for fn, caps in img_captions.items():
            for cap in caps:
                self.captions.append((fn, cap))

    def __getitem__(self, index):
        fn, cap = self.captions[index]
        img = self.feats[fn][:]
        return fn, cap, np.array(img)

    def __len__(self):
        return len(self.captions)


def get_dataloader(img_feats, img_captions, pad_index, max_sql_len, batch_size, num_workers=0, shuffle=True):
    dataset = CaptionDataset(img_feats, img_captions)
    dataloader = data.DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 num_workers=num_workers,
                                 collate_fn=create_collate_fn(pad_index, max_sql_len))
    return dataloader
