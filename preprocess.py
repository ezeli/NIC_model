import argparse
import json
import tqdm
from collections import Counter


def get_coco_captions(opt):
    images = json.load(open(opt.dataset_coco, 'r'))['images']
    captions_word = {'train': {}, 'val': {}}
    for img in tqdm.tqdm(images):
        sents = []
        for sent in img['sentences']:
            sents.append(sent['tokens'])
        split = 'train'
        if img['split'] == 'val':
            split = 'val'
        captions_word[split][img['filename']] = sents
    json.dump(captions_word, open(opt.captions_word, 'w'))
    return captions_word


def build_vocab(opt, captions_word=None):
    if captions_word is None:
        captions_word = json.load(open(opt.captions_word, 'r'))

    tc = Counter()
    for split, caps in captions_word.items():
        for fn, sents in tqdm.tqdm(caps.items()):
            for sent in sents:
                tc.update(sent)
    tc = tc.most_common()
    idx2word = ['<PAD>', '<SOS>', '<EOS>', '<UNK>'] + [w[0] for w in tc if w[1] > 5]
    json.dump(idx2word, open(opt.idx2word, 'w'))
    return idx2word


def convert_captions(opt, captions_word=None, idx2word=None):
    if captions_word is None:
        captions_word = json.load(open(opt.captions_word, 'r'))
    if idx2word is None:
        idx2word = json.load(open(opt.idx2word, 'r'))

    w2i = {}
    for i, w in enumerate(idx2word):
        w2i[w] = i
    sos_id, eos_id, unk_id = w2i['<SOS>'], w2i['<EOS>'], w2i['<UNK>']

    captions_id = {}
    for split, caps in captions_word.items():
        captions_id[split] = {}
        for fn, sents in tqdm.tqdm(caps.items()):
            tmp_sents = []
            for sent in sents:
                tmp = [sos_id]
                tmp += [w2i.get(w, unk_id) for w in sent]
                tmp += [eos_id]
                tmp_sents.append(tmp)
            captions_id[split][fn] = tmp_sents
    json.dump(captions_id, open(opt.captions_id, 'w'))
    return captions_id


def get_annotation(opt):
    images = json.load(open(opt.dataset_coco, 'r'))['images']
    anno1 = {}
    anno2 = {}
    for img in tqdm.tqdm(images):
        if img['split'] == 'val':
            sents1 = []
            sents2 = []
            for sent in img['sentences']:
                sents1.append(sent['raw'])
                sents2.append(' '.join(sent['tokens']))
            anno1[img['filename']] = sents1
            anno2[img['filename']] = sents2
    json.dump(anno1, open('./data/captions/annotation_raw.json', 'w'))
    json.dump(anno2, open('./data/captions/annotation_tks.json', 'w'))
    return anno1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_coco', type=str, default='./data/captions/dataset_coco.json')
    parser.add_argument('--captions_word', type=str, default='./data/captions/captions_word.json')
    parser.add_argument('--idx2word', type=str, default='./data/captions/idx2word.json')
    parser.add_argument('--captions_id', type=str, default='./data/captions/captions_id.json')
    parser.add_argument('--annotation', type=str, default='./data/captions/annotation.json')

    opt = parser.parse_args()

    captions_word = get_coco_captions(opt)
    idx2word = build_vocab(opt, captions_word)
    convert_captions(opt, captions_word, idx2word)
