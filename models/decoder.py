# coding:utf8
import torch
from torch import nn
from collections import namedtuple
import torch.nn.functional as F


BeamCandidate = namedtuple('BeamCandidate',
                           ['state', 'log_prob_sum', 'log_prob_seq', 'last_word_id', 'word_id_seq'])


class Decoder(nn.Module):
    def __init__(self, idx2word, settings):
        super(Decoder, self).__init__()
        self.idx2word = idx2word
        self.pad_id = idx2word.index('<PAD>')
        self.unk_id = idx2word.index('<UNK>')
        self.sos_id = idx2word.index('<SOS>') if '<SOS>' in idx2word else self.pad_id
        self.eos_id = idx2word.index('<EOS>') if '<EOS>' in idx2word else self.pad_id

        self.vocab_size = len(idx2word)
        self.word_embed = nn.Sequential(nn.Embedding(self.vocab_size, settings['emb_dim'],
                                                     padding_idx=self.pad_id),
                                        nn.ReLU(),
                                        nn.Dropout(settings['dropout_p']))
        self.fc_embed = nn.Sequential(nn.Linear(settings['fc_feat_dim'], settings['emb_dim']),
                                      nn.ReLU(),
                                      nn.Dropout(settings['dropout_p']))

        self.rnn = nn.LSTMCell(settings['emb_dim'], settings['rnn_hid_dim'])
        self.rnn_drop = nn.Dropout(settings['dropout_p'])

        self.classifier = nn.Linear(settings['rnn_hid_dim'], self.vocab_size)

    def forward(self, *args, **kwargs):
        mode = kwargs.get('mode', 'xe')
        if 'mode' in kwargs:
            del kwargs['mode']
        return getattr(self, 'forward_' + mode)(*args, **kwargs)

    def _forward_step(self, it, state):
        word_embs = self.word_embed(it)  # bs*word_emb
        output, state = self.rnn(word_embs, state)  # bs*rnn_hid
        output = self.rnn_drop(output)
        logprobs = F.log_softmax(self.classifier(output), dim=1)  # bs*vocab
        return logprobs, state

    def forward_xe(self, fc_feats, captions, ss_prob=0.0):
        batch_size = fc_feats.size(0)
        fc_feats = self.fc_embed(fc_feats)  # bz*emb_dim
        _, state = self.rnn(fc_feats)

        outputs = []
        for i in range(captions.size(1) - 1):
            if self.training and i >= 1 and ss_prob > 0.0:  # otherwise no need to sample
                sample_prob = fc_feats.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < ss_prob
                if sample_mask.sum() == 0:
                    it = captions[:, i].clone()  # bs
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = captions[:, i].clone()  # bs
                    prob_prev = outputs[i - 1].detach().exp()  # bs*vocab, fetch prev distribution
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
            else:
                it = captions[:, i].clone()  # bs
            logprobs, state = self._forward_step(it, state)
            outputs.append(logprobs)

        outputs = torch.stack(outputs, dim=1)  # [bs, max_len, vocab_size]
        return outputs

    def forward_rl(self, fc_feats, sample_max, max_seq_len):
        batch_size = fc_feats.size(0)
        fc_feats = self.fc_embed(fc_feats)  # bz*emb_dim
        _, state = self.rnn(fc_feats)

        seq = fc_feats.new_zeros((batch_size, max_seq_len), dtype=torch.long)
        seq_logprobs = fc_feats.new_zeros((batch_size, max_seq_len))
        seq_masks = fc_feats.new_zeros((batch_size, max_seq_len))
        it = fc_feats.new_zeros(batch_size, dtype=torch.long).fill_(self.sos_id)  # first input <SOS>
        unfinished = it == self.sos_id
        for t in range(max_seq_len):
            logprobs, state = self._forward_step(it, state)

            if sample_max:
                sample_logprobs, it = torch.max(logprobs, 1)
            else:
                prob_prev = torch.exp(logprobs)
                it = torch.multinomial(prob_prev, 1)
                sample_logprobs = logprobs.gather(1, it)  # gather the logprobs at sampled positions
            it = it.view(-1).long()
            sample_logprobs = sample_logprobs.view(-1)

            seq_masks[:, t] = unfinished
            it = it * unfinished.type_as(it)  # bs
            seq[:, t] = it
            seq_logprobs[:, t] = sample_logprobs

            unfinished = unfinished * (it != self.eos_id)
            if unfinished.sum() == 0:
                break

        return seq, seq_logprobs, seq_masks

    def sample(self, fc_feat, max_seq_len=16, beam_size=3, decoding_constraint=1):
        self.eval()
        fc_feat = fc_feat.view(1, -1)  # 1*2048
        fc_feat = self.fc_embed(fc_feat)  # 1*emb_dim
        _, state = self.rnn(fc_feat)

        # state, log_prob_sum, log_prob_seq, last_word_id, word_id_seq
        candidates = [BeamCandidate(state, 0., [], self.sos_id, [])]
        for t in range(max_seq_len):
            tmp_candidates = []
            end_flag = True
            for candidate in candidates:
                state, log_prob_sum, log_prob_seq, last_word_id, word_id_seq = candidate
                if t > 0 and last_word_id == self.eos_id:
                    tmp_candidates.append(candidate)
                else:
                    end_flag = False
                    it = fc_feat.new_tensor([last_word_id], dtype=torch.long)
                    logprobs, state = self._forward_step(it, state)  # [1, vocab_size]
                    logprobs = logprobs.squeeze(0)
                    if self.pad_id != self.eos_id:
                        logprobs[self.pad_id] += float('-inf')  # do not generate <PAD> and <SOS> and <UNK>
                        logprobs[self.sos_id] += float('-inf')
                        logprobs[self.unk_id] += float('-inf')
                    if decoding_constraint:  # do not generate last step word
                        logprobs[last_word_id] += float('-inf')

                    output_sorted, index_sorted = torch.sort(logprobs, descending=True)
                    for k in range(beam_size):
                        log_prob, word_id = output_sorted[k], index_sorted[k]  # tensor, tensor
                        log_prob = float(log_prob)
                        word_id = int(word_id)
                        tmp_candidates.append(BeamCandidate(state, log_prob_sum + log_prob,
                                                            log_prob_seq + [log_prob],
                                                            word_id, word_id_seq + [word_id]))
            candidates = sorted(tmp_candidates, key=lambda x: x.log_prob_sum, reverse=True)[:beam_size]
            if end_flag:
                break

        # captions, scores
        captions = [' '.join([self.idx2word[idx] for idx in candidate.word_id_seq if idx != self.eos_id])
                    for candidate in candidates]
        scores = [candidate.log_prob_sum for candidate in candidates]
        return captions, scores

    def get_optim_and_crit(self, lr, weight_decay=0):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay), \
               XECriterion()


class XECriterion(nn.Module):
    def __init__(self):
        super(XECriterion, self).__init__()

    def forward(self, pred, target, lengths):
        max_len = max(lengths)
        mask = pred.new_zeros(len(lengths), max_len)
        for i, l in enumerate(lengths):
            mask[i, :l] = 1

        loss = - pred.gather(2, target.unsqueeze(2)).squeeze(2) * mask
        loss = torch.sum(loss) / torch.sum(mask)

        return loss
