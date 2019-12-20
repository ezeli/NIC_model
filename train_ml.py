# coding:utf8
import tqdm
import os
import h5py
import time
import json
import sys
import pdb
import traceback
from bdb import BdbQuit
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from dataloader import get_dataloader
from models.decoder import Decoder
from opts import parse_opt


def train():
    opt = parse_opt()

    idx2word = json.load(open(opt.idx2word, 'r'))
    captions = json.load(open(opt.captions, 'r'))
    f_fc = h5py.File(opt.img_feats, mode='r')

    pad_index = 0
    train_data = get_dataloader(f_fc, captions['train'], pad_index, opt.max_sql_len, opt.batch_size)
    val_data = get_dataloader(f_fc, captions['val'], pad_index, opt.max_sql_len, opt.batch_size, shuffle=False)

    # 模型
    decoder = Decoder(idx2word, opt)
    decoder.to(opt.device)
    lr = opt.learning_rate
    optimizer = decoder.get_optimizer(lr)
    if opt.resume:
        print("====> loading checkpoint '{}'".format(opt.resume))
        chkpoint = torch.load(opt.resume, map_location=lambda s, l: s)
        decoder.load_state_dict(chkpoint['model'])
        optimizer.load_state_dict(chkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        print("=> loaded checkpoint '{}' epoch {}"
              .format(opt.resume, chkpoint['epoch']))
    criterion = nn.CrossEntropyLoss()

    LOSS_LAMDA = 0.5
    def forward(data, training=True):
        decoder.train(training)
        loss_val = 0.0
        for img_feats, (img_captions, lengths) in tqdm.tqdm(data):
            img_feats = img_feats.to(opt.device)
            img_captions = img_captions.to(opt.device)
            pred = decoder(img_feats, img_captions, lengths)
            real1 = pack_padded_sequence(img_captions[:, 1:], lengths, batch_first=True)[0]
            loss1 = criterion(pred, real1)

            real2 = img_captions[:, 1:].new_zeros(img_captions[:, 1:].size())
            real2[:, :-1] = img_captions[:, 2:]
            real2 = pack_padded_sequence(real2, lengths, batch_first=True)[0]
            loss2 = criterion(pred, real2)

            real3 = img_captions[:, 1:].new_zeros(img_captions[:, 1:].size())
            real3[:, :-2] = img_captions[:, 3:]
            real3 = pack_padded_sequence(real3, lengths, batch_first=True)[0]
            loss3 = criterion(pred, real3)

            real4 = img_captions[:, 1:].new_zeros(img_captions[:, 1:].size())
            real4[:, :-3] = img_captions[:, 4:]
            real4 = pack_padded_sequence(real4, lengths, batch_first=True)[0]
            loss4 = criterion(pred, real4)

            loss = loss1 + LOSS_LAMDA * loss2 + LOSS_LAMDA**2 * loss3 + LOSS_LAMDA**3 * loss4
            loss_val += loss.item()
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return loss_val / len(data)

    previous_loss = None
    for epoch in range(opt.max_epochs):
        print('--------------------epoch: %d' % epoch)
        train_loss = forward(train_data)
        with torch.no_grad():
            val_loss = forward(val_data, training=False)

        if previous_loss is not None and val_loss >= previous_loss:
            lr = lr * 0.5
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        previous_loss = val_loss

        print('train_loss: %.2f, val_loss: %.2f' % (train_loss, val_loss))
        if epoch in [0, 5, 10, 15, 20, 25, 26, 28, 29, 30, 32, 34, 36, 37, 39]:
            chkpoint = {
                'epoch': epoch,
                'model': decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            checkpoint_path = os.path.join(opt.checkpoint, 'model_%d_%.4f_%.4f_%s.pth' % (
                epoch, train_loss, val_loss, time.strftime('%m%d-%H%M')))
            torch.save(chkpoint, checkpoint_path)


if __name__ == '__main__':
    try:
        train()
    except BdbQuit:
        sys.exit(1)
    except Exception:
        traceback.print_exc()
        print('')
        pdb.post_mortem()
        sys.exit(1)
