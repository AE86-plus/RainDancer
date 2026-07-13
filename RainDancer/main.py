import argparse
import commentjson as json
import sys
from trainer.trainer import trainer
import torch
import os
import numpy as np
import random
import torch.backends.cudnn as cudnn
import torch.distributed as dist

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def init_seeds(seed=0, cuda_deterministic=True):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if cuda_deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.deterministic = False
        cudnn.benchmark = False

def main():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', default='./options/v1.json', type=str,
                               help='path to json config file')
    config_args, remaining_argv = config_parser.parse_known_args()

    with open(config_args.config, 'r') as f:
        args = json.load(f)

    for key, value in args.items():
        print('{:<25s}: {:s}'.format(key, str(value)))

    if True:

        parser = argparse.ArgumentParser(description="DCLGAN")

        parser.add_argument('--local-rank', default = -1, type = int, help = "node rank for distributed training")
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--batch_size', default = 8, type = int, help = "batch_size")
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'],
                            help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')

        ####PatchNCE
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--event_input_nc', type=int, default=20, help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--normG', type=str, default='instance', choices=['instance', 'batch', 'none'], help='instance normalization or batch normalization for G')
        parser.add_argument('--no_dropout', type=str2bool, nargs='?', const=True, default=True,
                            help='no dropout for the generator')

        ####Train strategy
        parser.add_argument('--lr_policy', type=str, default='cosine', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--warmup_epochs', type=int, default=4, help='number of epochs with the initial learning rate')
        parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=600, help='number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')
        parser.add_argument('--lr_min', type=float, default=0, help='initial learning rate for adam')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--gan_mode', type=str, default='hinge', help='the type of GAN objective. [vanilla| lsgan | wgangp| hinge]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')

        parser.set_defaults(pool_size=0)  # no image pooling

        opts = parser.parse_args(remaining_argv)

        opts.nprocs = torch.cuda.device_count()

    random_seed = 1234
    init_seeds(random_seed+opts.local_rank)

    print("Now process is {}".format(opts.local_rank))

    trainer_v1 = trainer(args, opts)
    trainer_v1.train(opts.local_rank, opts.nprocs)

if __name__ == "__main__":
    main()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
