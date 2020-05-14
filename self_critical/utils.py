import numpy as np
import torch
import torch.nn as nn
import tqdm

from .cider.pyciderevalcap.ciderD.ciderD import CiderD
from .bleu.bleu import Bleu


def array_to_str(arr, sos_token, eos_token):
    arr = list(arr)
    if arr[0] == sos_token:
        arr = arr[1:]
    # if arr[-1] != eos_token:
    #     arr = arr + [eos_token]
    out = ''
    for i in range(len(arr)):
        if arr[i] == eos_token:
            break
        out += str(arr[i]) + ' '
    return out.strip()


def get_ciderd_scorer(split_captions, sos_token, eos_token):
    print('====> get_ciderd_scorer begin')
    captions = {}
    for caps in split_captions.values():
        captions.update(caps)

    refs_idxs = []
    for caps in tqdm.tqdm(captions.values()):
        ref_idxs = []
        for cap in caps:
            ref_idxs.append(array_to_str(cap, sos_token, eos_token))
        refs_idxs.append(ref_idxs)

    scorer = CiderD(refs=refs_idxs)
    print('====> get_ciderd_scorer end')
    return scorer


def get_self_critical_reward(sample_captions, greedy_captions, fns, ground_truth,
                             sos_token, eos_token, scorer):
    batch_size = len(fns)
    sample_captions = sample_captions.cpu().numpy()
    greedy_captions = greedy_captions.cpu().numpy()
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
    if isinstance(scorer, CiderD):
        _, scores = scorer.compute_score(gts, all_result)
    elif isinstance(scorer, Bleu):
        _, scores = scorer.compute_score(gts, all_result)
        scores = np.array(scores[3])
    else:
        raise Exception('do not support this scorer: %s' % type(scorer))

    scores = scores[:batch_size] - scores[batch_size:]
    rewards = np.repeat(scores[:, np.newaxis], sample_captions.shape[1], 1)
    return rewards


class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, seq_logprobs, seq_masks, reward):
        seq_logprobs = seq_logprobs.view(-1)
        seq_masks = seq_masks.view(-1)
        reward = reward.view(-1)
        output = - seq_logprobs * seq_masks * reward
        output = torch.sum(output) / torch.sum(seq_masks)

        return output