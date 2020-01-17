import numpy as np
import torch
import torch.nn as nn
import tqdm

from .cider.pyciderevalcap.ciderD.ciderD import CiderD
# from .bleu.bleu import Bleu
#
# Bleu_scorer = Bleu(4)


def array_to_str(arr, sos_token, eos_token):
    arr = list(arr)
    for i in range(len(arr)):
        arr[i] = int(arr[i])
    if arr[0] == sos_token:
        arr = arr[1:]
    if arr[-1] != eos_token:
        arr = arr + [eos_token]
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == eos_token:
            break
    return out.strip()


def get_ciderd_scorer(split_captions, sos_token, eos_token):
    print('====> get_ciderd_scorer begin')
    captions = split_captions['train'].copy()
    captions.update(split_captions['val'])

    refs_idxs = []
    for caps in tqdm.tqdm(captions.values()):
        ref_idxs = []
        for cap in caps:
            ref_idxs.append(array_to_str(cap, sos_token, eos_token))
        refs_idxs.append(ref_idxs)

    scorer = CiderD(refs=refs_idxs)
    print('====> get_ciderd_scorer end')
    return scorer


def get_self_critical_reward(sample_captions, greedy_captions, fns, ground_truth, sos_token, eos_token, ciderd_scorer):
    batch_size = len(fns)
    assert sample_captions.size(0) == greedy_captions.size(0) == batch_size
    sample_result = []
    greedy_result = []
    gts = {}
    for i, fn in enumerate(fns):
        sample_result.append({'image_id': fn, 'caption': [array_to_str(sample_captions[i], sos_token, eos_token)]})
        greedy_result.append({'image_id': fn, 'caption': [array_to_str(greedy_captions[i], sos_token, eos_token)]})
        caps = []
        for cap in ground_truth[fn]:
            caps.append(array_to_str(cap, sos_token, eos_token))
        gts[fn] = caps
    all_result = sample_result + greedy_result
    _, scores = ciderd_scorer.compute_score(gts, all_result)
    # _, scores = Bleu_scorer.compute_score(gts, all_result)
    # scores = np.array(scores[3])

    scores = scores[:batch_size] - scores[batch_size:]
    rewards = np.repeat(scores[:, np.newaxis], sample_captions.shape[1], 1)
    return rewards


class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, seq, seq_logprobs, reward):
        seq_logprobs = seq_logprobs.view(-1)
        reward = reward.view(-1)
        # mask = (seq > 0).float().view(-1)
        mask = (seq > 0).float()
        mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1).view(-1)
        output = - seq_logprobs * reward * mask
        output = torch.sum(output) / torch.sum(mask)

        return output
