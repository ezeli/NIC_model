# coding:utf8
import tqdm
import os
import h5py
import time
import json
import numpy as np
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
from self_critical.utils import get_ciderd_scorer, get_self_critical_reward, RewardCriterion


def train():
    opt = parse_opt()

    idx2word = json.load(open(opt.idx2word, 'r'))
    captions = json.load(open(opt.captions, 'r'))
    f_fc = h5py.File(opt.img_feats, mode='r')

    # 模型
    decoder = Decoder(idx2word, opt.settings)
    decoder.to(opt.device)
    lr = opt.learning_rate
    optimizer = decoder.get_optimizer(lr)
    if opt.resume:
        print("====> loading checkpoint '{}'".format(opt.resume))
        chkpoint = torch.load(opt.resume, map_location=lambda s, l: s)
        decoder.load_state_dict(chkpoint['model'])
        optimizer.load_state_dict(chkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        assert opt.settings == chkpoint['settings'], \
            'opt.settings and resume model settings are different'
        assert idx2word == chkpoint['idx2word'], \
            'idx2word and resume model idx2word are different'
        print("=> loaded checkpoint '{}' epoch {}"
              .format(opt.resume, chkpoint['epoch']))

    train_data = get_dataloader(f_fc, captions['train'], decoder.pad_id, opt.max_sql_len+1, opt.batch_size)
    val_data = get_dataloader(f_fc, captions['val'], decoder.pad_id, opt.max_sql_len+1, opt.batch_size, shuffle=False)

    xe_criterion = nn.CrossEntropyLoss()
    train_mode = opt.train_mode
    if train_mode == 'rl':
        rl_criterion = RewardCriterion()
        ciderd_scorer = get_ciderd_scorer(captions, decoder.sos_id, decoder.eos_id)

    def forward(data, training=True):
        decoder.train(training)
        loss_val = 0.0
        reward_val = 0.0
        for fns, fc_feats, (caps_tensor, lengths) in tqdm.tqdm(data):
            fc_feats = fc_feats.to(opt.device)
            caps_tensor = caps_tensor.to(opt.device)

            if training and train_mode == 'rl':
                sample_captions, sample_logprobs = decoder(fc_feats, sample_max=0,
                                                           max_seq_length=opt.max_sql_len, mode=train_mode)
                decoder.eval()
                with torch.no_grad():
                    greedy_captions, _ = decoder(fc_feats, sample_max=1,
                                                 max_seq_length=opt.max_sql_len, mode=train_mode)
                decoder.train()
                reward = get_self_critical_reward(sample_captions, greedy_captions, fns, captions['train'],
                                                  decoder.sos_id, decoder.eos_id, ciderd_scorer)
                loss = rl_criterion(sample_captions, sample_logprobs, torch.from_numpy(reward).float().to(opt.device))
                reward_val += np.mean(reward[:, 0]).item()
            else:
                pred = decoder(fc_feats, caps_tensor, lengths)
                real = pack_padded_sequence(caps_tensor[:, 1:], lengths, batch_first=True)[0]
                loss = xe_criterion(pred, real)

            loss_val += loss.item()
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return loss_val / len(data), reward_val / len(data)

    previous_loss = None
    for epoch in range(opt.max_epochs):
        print('--------------------epoch: %d' % epoch)
        # torch.cuda.empty_cache()
        train_loss, avg_reward = forward(train_data)
        with torch.no_grad():
            val_loss, _ = forward(val_data, training=False)

        if previous_loss is not None and val_loss >= previous_loss:
            lr = lr * 0.5
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        previous_loss = val_loss

        print('train_loss: %.4f, val_loss: %.4f, avg_reward: %.4f' % (train_loss, val_loss, avg_reward))
        if epoch in [0, 5, 10, 15, 17, 19, 20, 21, 23, 25, 27, 29] or epoch > 30:
            chkpoint = {
                'epoch': epoch,
                'model': decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'settings': opt.settings,
                'idx2word': idx2word,
            }
            checkpoint_path = os.path.join(opt.checkpoint, 'model_%d_%.4f_%.4f_%.4f_%s.pth' % (
                epoch, train_loss, val_loss, avg_reward, time.strftime('%m%d-%H%M')))
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
