import yaml
import argparse
from trainer import train


def main():
    args = setup_parser().parse_args()
    configs = load_yaml(args.config)
    varargs = vars(args)  # Converting argparse Namespace to a dict.
    varargs.update(configs)  # Add parameters from yaml

    train(args)


def load_yaml(settings_path):
    with open(settings_path) as data_file:
        param = yaml.load(data_file, Loader=yaml.FullLoader)

    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='multiple continual learning algorithms.')
    parser.add_argument('--config', type=str, default='./exps/finetune.yaml',
                        help='Yaml file of settings.')
    parser.add_argument('--tb', type=bool, default=True,
                        help='enable or disable tensorboard logs')

    return parser


if __name__ == '__main__':
    main()
