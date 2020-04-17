"""Parse flags and set up hyperparameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import random
import tensorflow as tf

from pba.augmentation_transforms_hp import NUM_HP_TRANSFORM


def create_parser(state):
    """Create arg parser for flags."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default=''
    )
    parser.add_argument(
        '--model_name',
        default='wrn',
        choices=('wrn_28_10', 'wrn_40_2', 'shake_shake_32', 'shake_shake_96',
                 'shake_shake_112', 'pyramid_net', 'resnet', 'ResNet', 'ResNet8', 'ResNet18', 'ResNet28', 'ResNet50', 'AlexNetCifar'))
    parser.add_argument(
        '--data_path',
        default='/tmp/datasets/',
        help='Directory where dataset is located.')
    parser.add_argument(
        '--dataset',
        default='cifar10',
        choices=('cifar10', 'cifar100', 'svhn', 'svhn-full', 'test'))
    parser.add_argument(
        '--recompute_dset_stats',
        action='store_true',
        help='Instead of using hardcoded mean/std, recompute from dataset.')
    parser.add_argument('--local_dir', type=str, default='/tmp/ray_results/', help='Ray directory.')
    parser.add_argument('--restore', type=str, default=None, help='If specified, tries to restore from given path.')
    parser.add_argument('--load', type=str, default=None, help='Load pretrained model.')
    parser.add_argument('--train_size', type=int, default=5000, help='Number of training examples.')
    parser.add_argument('--val_size', type=int, default=45000, help='Number of validation examples.')
    parser.add_argument('--checkpoint_freq', type=int, default=50, help='Checkpoint frequency in epoch.')
    parser.add_argument('--checkpoint_at_end', type=bool, default=True, help='whether save a checkpoint in the end.')
    parser.add_argument("--nofreeze", type=bool, default=False, help="whether to freeze teacher.")
    parser.add_argument(
        '--cpu', type=float, default=4, help='Allocated by Ray')
    parser.add_argument(
        '--gpu', type=float, default=1, help='Allocated by Ray')
    parser.add_argument(
        '--aug_policy',
        type=str,
        default='cifar10',
        help=
        'which augmentation policy to use (in augmentation_transforms_hp.py)')
    # search-use only
    parser.add_argument(
        '--explore',
        type=str,
        default='cifar10',
        help='which explore function to use')
    parser.add_argument(
        '--epochs',
        type=int,
        default=0,
        help='Number of epochs, or <=0 for default')
    parser.add_argument(
        '--no_cutout', action='store_true', help='turn off cutout')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--wd', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--bs', type=int, default=128, help='batch size')
    parser.add_argument('--test_bs', type=int, default=25, help='test batch size')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of Ray samples')
    parser.add_argument('--bit_a', type=int, default=32, help='number of bits for activations')
    parser.add_argument('--bit_w', type=int, default=32, help='number of bits for weights')
    parser.add_argument('--bit_g', type=int, default=32, help='number of bits for gradients')
    parser.add_argument('--opt', type=str, default='adam', help='optimizer to use')
    parser.add_argument('--lamb', type=float, default=1.0, help='kd dist lamb')

    ################### KD edited ###########################
    parser.add_argument(
        '--Distillation',
        default=None,
        help='KD way'
    )

    parser.add_argument(
        '--main_scope',
        default='Student',
        help='network scope'
    )
    parser.add_argument(
        '--init_epoch',
        default=0,
        help='init epoch'
    )
    parser.add_argument(
        '--train_dir',
        help='store models'
    )
    parser.add_argument(
        '--teacher',
        default='',
        help='specify teacher model'
    )

    if state == 'train':
        parser.add_argument(
            '--use_hp_policy',
            action='store_true',
            help='otherwise use autoaug policy')
        parser.add_argument(
            '--hp_policy',
            type=str,
            default=None,
            help='either a comma separated list of values or a file')
        parser.add_argument(
            '--hp_policy_epochs',
            type=int,
            default=200,
            help='number of epochs/iterations policy trained for')
        parser.add_argument(
            '--no_aug',
            action='store_true',
            help=
            'no additional augmentation at all (besides cutout if not toggled)'
        )
        parser.add_argument(
            '--flatten',
            action='store_true',
            help='randomly select aug policy from schedule')
        parser.add_argument('--name', type=str, default='autoaug')

    elif state == 'search':
        parser.add_argument('--perturbation_interval', type=int, default=10)
        parser.add_argument('--name', type=str, default='autoaug_pbt')
    else:
        raise ValueError('unknown state')
    args = parser.parse_args()
    tf.logging.info(str(args))
    return args


def create_hparams(state, FLAGS):  # pylint: disable=invalid-name
    """Creates hyperparameters to pass into Ray config.

  Different options depending on search or eval mode.

  Args:
    state: a string, 'train' or 'search'.
    FLAGS: parsed command line flags.

  Returns:
    tf.hparams object.
  """
    epochs = 0
    tf.logging.info('data path: {}'.format(FLAGS.data_path))
    hparams = tf.contrib.training.HParams(
        train_size=FLAGS.train_size,
        validation_size=FLAGS.val_size,
        dataset=FLAGS.dataset,
        data_path=FLAGS.data_path,
        batch_size=FLAGS.bs,
        gradient_clipping_by_global_norm=5.0,
        explore=FLAGS.explore,
        aug_policy=FLAGS.aug_policy,
        no_cutout=FLAGS.no_cutout,
        recompute_dset_stats=FLAGS.recompute_dset_stats,
        lr=FLAGS.lr,
        weight_decay_rate=FLAGS.wd,
        test_batch_size=FLAGS.test_bs)

    if state == 'train':
        tf.logging.info("if exists aug: {}".format(FLAGS.no_aug))
        hparams.add_hparam('no_aug', FLAGS.no_aug)
        hparams.add_hparam('use_hp_policy', FLAGS.use_hp_policy)
        if FLAGS.use_hp_policy:
            if FLAGS.hp_policy == 'random':
                tf.logging.info('RANDOM SEARCH')
                parsed_policy = []
                for i in range(NUM_HP_TRANSFORM * 4):
                    if i % 2 == 0:
                        parsed_policy.append(random.randint(0, 10))
                    else:
                        parsed_policy.append(random.randint(0, 9))
            elif FLAGS.hp_policy.endswith('.txt') or FLAGS.hp_policy.endswith(
                    '.p'):
                # will be loaded in in data_utils
                parsed_policy = FLAGS.hp_policy
            else:
                # parse input into a fixed augmentation policy
                parsed_policy = FLAGS.hp_policy.split(', ')
                parsed_policy = [int(p) for p in parsed_policy]
            hparams.add_hparam('hp_policy', parsed_policy)
            hparams.add_hparam('hp_policy_epochs', FLAGS.hp_policy_epochs)
            hparams.add_hparam('flatten', FLAGS.flatten)
    elif state == 'search':
        hparams.add_hparam('no_aug', False)
        hparams.add_hparam('use_hp_policy', True)
        # default start value of 0
        hparams.add_hparam('hp_policy',
                           [0 for _ in range(4 * NUM_HP_TRANSFORM)])
    else:
        raise ValueError('unknown state')

    if FLAGS.model_name == 'wrn_40_2':
        hparams.add_hparam('model_name', 'wrn')
        epochs = 200
        hparams.add_hparam('wrn_size', 32)
        hparams.add_hparam('wrn_depth', 40)
    elif FLAGS.model_name == 'wrn_28_10':
        hparams.add_hparam('model_name', 'wrn')
        epochs = 200
        hparams.add_hparam('wrn_size', 160)
        hparams.add_hparam('wrn_depth', 28)
    elif FLAGS.model_name == 'resnet':
        hparams.add_hparam('model_name', 'resnet')
        epochs = 200
        hparams.add_hparam('resnet_size', 20)
        hparams.add_hparam('num_filters', FLAGS.resnet_size)
    elif FLAGS.model_name == 'ResNet':
        hparams.add_hparam('model_name', 'ResNet')
        epochs = 200
    elif FLAGS.model_name == 'shake_shake_32':
        hparams.add_hparam('model_name', 'shake_shake')
        epochs = 1800
        hparams.add_hparam('shake_shake_widen_factor', 2)
    elif FLAGS.model_name == 'shake_shake_96':
        hparams.add_hparam('model_name', 'shake_shake')
        epochs = 1800
        hparams.add_hparam('shake_shake_widen_factor', 6)
    elif FLAGS.model_name == 'shake_shake_112':
        hparams.add_hparam('model_name', 'shake_shake')
        epochs = 1800
        hparams.add_hparam('shake_shake_widen_factor', 7)
    elif FLAGS.model_name == 'pyramid_net':
        hparams.add_hparam('model_name', 'pyramid_net')
        epochs = 1800
        hparams.set_hparam('batch_size', 64)
    elif FLAGS.model_name == 'ResNet8':
        hparams.add_hparam('model_name', 'ResNet8')
    elif FLAGS.model_name == 'ResNet18':
        hparams.add_hparam('model_name', 'ResNet18')
    elif FLAGS.model_name == 'ResNet50':
        hparams.add_hparam('model_name', 'ResNet50')
    elif FLAGS.model_name == 'ResNet28':
        hparams.add_hparam('model_name', 'ResNet28')
    elif FLAGS.model_name == 'AlexNetCifar':
        hparams.add_hparam('model_name', 'AlexNetCifar')
    else:
        raise ValueError('Not Valid Model Name: %s' % FLAGS.model_name)
    if FLAGS.epochs > 0:
        tf.logging.info('overwriting with custom epochs')
        epochs = FLAGS.epochs
    hparams.add_hparam('num_epochs', epochs)

    ############### KD ###################
    hparams.add_hparam('main_scope', FLAGS.main_scope)
    if FLAGS.Distillation == 'None':
        hparams.add_hparam('Distillation', None)
    else:
        hparams.add_hparam('Distillation', FLAGS.Distillation)
    hparams.add_hparam('init_epoch', FLAGS.init_epoch)
    hparams.add_hparam('train_dir', FLAGS.train_dir)
    hparams.add_hparam('teacher', FLAGS.teacher)
    hparams.add_hparam('opt', FLAGS.opt)
    hparams.add_hparam('lamb', FLAGS.lamb)
    hparams.add_hparam('nofreeze', FLAGS.nofreeze)

    ########### Quantization ##############
    hparams.add_hparam('bit_a', FLAGS.bit_a)
    hparams.add_hparam('bit_w', FLAGS.bit_w)
    hparams.add_hparam('bit_g', FLAGS.bit_g)

    hparams.add_hparam('pretrained_model', FLAGS.load)

    tf.logging.info('epochs: {}, lr: {}, wd: {}'.format(
        hparams.num_epochs, hparams.lr, hparams.weight_decay_rate))

    return hparams
