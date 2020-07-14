import argparse
import configparser

config = configparser.ConfigParser()
config.read('cfg.ini')

# class Parser(argparse.ArgumentParser):
#     def __init__(self):
#         parser = argparse.ArgumentParser(description='Surrogate Model for Uncertainty Quatification')
#
#         # parser.add_argument('training_data_path', help='the path of training dataset')
#         #
#         # recommended_group = parser.add_argument_group('recommended settings', 'Recommended settings for different scenarios')
#         # recommended_type = recommended_group.add_mutually_exclusive_group()
#         # recommended_type.add_argument('--supervised', metavar='')
#
#         # args = parser.parse_args()
#         # if args.supervised is not None:
#         #     parser.set_defaults()
#
#         # if args.cuda:
#     def parse(self):
#         args = self.parse_args()
#
#         hparams = f'{args.dataset}_ntrain{args.ntrain}_run{args.run}_bs{args.batch_size}_lr{args.lr}_epochs{args.epochs}'
#         if args.debug:
#             hparams = 'debug/' + hparams
#         args.run_dir = args.exp_dir + '/' + args.exp_name + '/' + hparams
#         args.ckpt_dir = args.run_dir + '/checkpoints'
#         mkdirs(args.run_dir, args.ckpt_dir)
#
#         assert args.ntrain % args.batch_size == 0 and \
#                args.ntest % args.test_batch_size == 0
#
#         print('Arguments:')
#         pprint(vars(args))
#         with open(args.run_dir + "/args.txt", 'w') as args_file:
#             json.dump(vars(args), args_file, indent=4)
#
#         return args


def parse_args():
    parser = argparse.ArgumentParser(description='Surrogate Model for Uncertainty Quatification')
    # parser.add_argument('-d', '--dataset', default=)
    parser.add_argument('--seed', type=int, default=0, help='the random seed (defaults to 0)')
    parser.add_argument('--cuda', action='store_true', help='Use cuda')

    recommended_group = parser.add_argument_group('recommended settings', 'Recommended settings for different scenarios')
    recommended_type = recommended_group.add_mutually_exclusive_group()
    recommended_type.add_argument('--active_learning', '--AL', help='For active learning')
    recommended_type.add_argument('--neural_optimization', '--NO',
                                  help='For neural network optimizations with several methods,'
                                       ' e.g. Bayesian Optimization, hyperopt')

    init_group = parser.add_argument_group('Advanced initialization arguments')

    # active_learning_group = parser.add_argument_group('Advanced arguments for active learning')
    # active_learning_group.add_argument('')

    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    args = parse_args()
    config
