import argparse
import torch


def parse_opt():
    parser = argparse.ArgumentParser()

    # train settings
    parser.add_argument('--train_mode', type=str, default='xe', choices=['xe', 'rl'],
                        help='"xe" means Cross Entropy, "rl" means Reinforcement learning')
    parser.add_argument('--learning_rate', type=float, default=4e-4)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--max_epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=50)

    parser.add_argument('--scheduled_sampling_start', type=int, default=0)
    parser.add_argument('--scheduled_sampling_increase_every', type=int, default=5)
    parser.add_argument('--scheduled_sampling_increase_prob', type=float, default=0.05)
    parser.add_argument('--scheduled_sampling_max_prob', type=float, default=0.25)

    parser.add_argument('--idx2word', type=str, default='./data/captions/idx2word.json')
    parser.add_argument('--captions', type=str, default='./data/captions/captions_id.json')
    parser.add_argument('--img_feats', type=str, default='./data/features/coco_fc.h5')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint/')
    parser.add_argument('--max_sql_len', type=int, default=16)
    parser.add_argument('--grad_clip', type=float, default=0.1)

    # eval settings
    parser.add_argument('-e', '--eval_model', type=str, default='')
    parser.add_argument('-r', '--result_file', type=str, default='')
    parser.add_argument('--beam_size', type=int, default=3)

    # test setting
    parser.add_argument('-m', '--test_model', type=str, default='./checkpoint/model_0_3.2555_2.6799_0.0000_1219-2014.pth')
    parser.add_argument('-i', '--image_file', type=str, default='')
    # encoder settings
    parser.add_argument('--resnet101_file', type=str, default='./data/pre_models/resnet101.pth',
                        help='Pre-trained resnet101 network for extracting image features')

    args = parser.parse_args()

    # decoder settings
    settings = dict()
    settings['emb_dim'] = 512
    settings['fc_feat_dim'] = 2048
    settings['dropout_p'] = 0.5
    settings['rnn_hid_dim'] = 512

    args.settings = settings
    args.use_gpu = torch.cuda.is_available()
    args.device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    return args