import argparse
import configparser

parser = argparse.ArgumentParser()
parser.add_argument('--env', '-e', type=str, default=None)
parser.add_argument('--algo', '-a', type=str, default='ICM')
parser.add_argument('--seed', '-s', type=int, default=0)
parser.add_argument('--wandb_mode', '-w', type=str, default='online')
parser.add_argument('--out_dir', '-o', type=str, default='runs')
parser.add_argument('--ckpt_path', '-c', type=str, default='')
args = parser.parse_args()

config = configparser.ConfigParser()

if args.env is None:
    config.read('configs/config.conf')
else:
    config.read(f'configs/config_{args.env}_{args.algo}.conf')

print(f'Read config {args.env} {args.algo}')

# ---------------------------------
default = 'DEFAULT'
# ---------------------------------
default_config = config[default]