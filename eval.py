# coding:utf8
import torch
import h5py
import json
import tqdm

from opts import parse_opt
from models.decoder import Decoder

opt = parse_opt()
assert opt.eval_model, 'please input eval_model'
assert opt.result_file, 'please input result_file'


captions = json.load(open(opt.captions, 'r'))
f_fc = h5py.File(opt.img_feats, mode='r')
val_imgs = captions['test'].keys()

print("====> loading checkpoint '{}'".format(opt.eval_model))
chkpoint = torch.load(opt.eval_model, map_location=lambda s, l: s)
decoder = Decoder(chkpoint['idx2word'], chkpoint['settings'])
decoder.load_state_dict(chkpoint['model'])
print("====> loaded checkpoint '{}', epoch: {}, train_mode: {}".
      format(opt.eval_model, chkpoint['epoch'], chkpoint['train_mode']))
decoder.to(opt.device)
decoder.eval()

results = []
for fn in tqdm.tqdm(val_imgs):
    img_feat = f_fc[fn][:]
    img_feat = torch.FloatTensor(img_feat).to(opt.device)
    rest, _ = decoder.sample(img_feat, beam_size=opt.beam_size, max_seq_len=opt.max_sql_len)
    results.append({'image_id': fn, 'caption': rest[0]})

json.dump(results, open(opt.result_file, 'w'))
