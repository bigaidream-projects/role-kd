from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import six

from . import Distillation as Dist
from . import Response, Multiple, Shared, Relation
from tensorpack.utils import logger
from tensorpack.tfutils.varreplace import remap_variables

if six.PY2:
    import functools32 as functools
else:
    import functools


def ResNet_arg_scope():
    with tf.contrib.framework.arg_scope([tf.contrib.layers.conv2d, tf.contrib.layers.fully_connected],
                                        weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                        biases_initializer=None, activation_fn=None,
                                        ):
        with tf.contrib.framework.arg_scope([tf.contrib.layers.batch_norm],
                                            scale=True, center=True, activation_fn=tf.nn.relu, decay=0.99, epsilon=1e-3,
                                            variables_collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                                   'BN_collection']) as arg_sc:
            return arg_sc


def AlexNet_arg_scope():
    with tf.contrib.framework.arg_scope([tf.contrib.layers.conv2d, tf.contrib.layers.fully_connected],
                                        weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                        biases_initializer=None, activation_fn=None,
                                        ):
        with tf.contrib.framework.arg_scope([tf.contrib.layers.batch_norm],
                                            scale=True, center=True, decay=0.99, epsilon=1e-3,
                                            variables_collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                                   'BN_collection']) as arg_sc:
            return arg_sc


def ResBlock(x, depth, stride, get_feat, is_training, reuse, name, scope='Teacher', func=None):
    with tf.variable_scope(name):
        if scope == 'Student':
            x = func(x)
        out = tf.contrib.layers.conv2d(x, depth, [3, 3], stride, scope='conv0', trainable=True, reuse=reuse)
        out = tf.contrib.layers.batch_norm(out, scope='bn0', trainable=True, is_training=is_training, reuse=reuse)
        out = tf.nn.relu(out)
        if scope == 'Student':
            out = func(out)
        out = tf.contrib.layers.conv2d(out, depth, [3, 3], 1, scope='conv1', trainable=True, reuse=reuse)
        out = tf.contrib.layers.batch_norm(out, scope='bn1', trainable=True, is_training=is_training, reuse=reuse,
                                           activation_fn=None)
        if stride > 1 or depth != x.get_shape().as_list()[-1]:
            x = tf.contrib.layers.conv2d(x, depth, [1, 1], stride, scope='conv2', trainable=True, reuse=reuse)
            x = tf.contrib.layers.batch_norm(x, scope='bn2', trainable=True, is_training=is_training, reuse=reuse,
                                             activation_fn=None)
        out = x + out
        if get_feat:
            tf.add_to_collection('feat_noact', out)
        out = tf.nn.relu(out)
        if get_feat:
            tf.add_to_collection('feat', out)
        return out


def NetworkBlock(x, block_func, nb_layers, depth, stride, func=None, is_training=False, reuse=False, name='',
                 scope='Teacher'):
    with tf.variable_scope(name):
        for i in range(nb_layers):
            x = block_func(x, depth, stride=stride if i == 0 else 1,
                           get_feat=True if i == nb_layers - 1 else False,
                           is_training=is_training, reuse=reuse, name='BasicBlock%d' % i, scope=scope, func=func)
        return x


def get_dorefa(bitW, bitA, bitG):
    """
    Return the three quantization functions fw, fa, fg, for weights, activations and gradients respectively
    It's unsafe to call this function multiple times with different parameters
    """

    def quantize(x, k):
        n = float(2 ** k - 1)

        @tf.custom_gradient
        def _quantize(x):
            return tf.round(x * n) / n, lambda dy: dy

        return _quantize(x)

    def fw(x):
        if bitW == 32:
            return x

        if bitW == 1:  # BWN
            E = tf.stop_gradient(tf.reduce_mean(tf.abs(x)))

            @tf.custom_gradient
            def _sign(x):
                return tf.where(tf.equal(x, 0), tf.ones_like(x), tf.sign(x / E)) * E, lambda dy: dy

            return _sign(x)

        x = tf.tanh(x)
        x = x / tf.reduce_max(tf.abs(x)) * 0.5 + 0.5
        return 2.0 * quantize(x, bitW) - 1

    def fa(x):
        if bitA == 32:
            return x
        return quantize(x, bitA)

    def fg(x):
        if bitG == 32:
            return x

        @tf.custom_gradient
        def _identity(input):
            def grad_fg(x):
                rank = x.get_shape().ndims
                assert rank is not None
                maxx = tf.reduce_max(tf.abs(x), list(range(1, rank)), keep_dims=True)
                x = x / maxx
                n = float(2 ** bitG - 1)
                x = x * 0.5 + 0.5 + tf.random_uniform(
                    tf.shape(x), minval=-0.5 / n, maxval=0.5 / n)
                x = tf.clip_by_value(x, 0.0, 1.0)
                x = quantize(x, bitG) - 0.5
                return x * maxx * 2

            return input, grad_fg

        return _identity(x)

    return fw, fa, fg


def ResNet8(image, label, scope, is_training, dataset='cifar', reuse=False, Distill=None, bit_a=32, bit_w=32,
             bit_g=32):
    end_points = {}

    nChannels = []
    if 'cifar' in dataset or 'svhn' in dataset:
        nChannels = [64, 64, 128, 256, 512]
    elif 'imagenet' in dataset:
        nChannels = [64, 256, 512, 1024, 2048]

    assert len(nChannels) > 0, "empty channels!!"

    stride = [1, 2, 2, 2]

    # 4 (stride) * n * 2 + 2. shortcut is not involved.
    # 32 -> 16 -> 8 -> 4
    if scope == 'Teacher':
        n = 2
        with tf.variable_scope(scope):
            std = tf.contrib.layers.conv2d(image, nChannels[0], [3, 3], 1, scope='base_conv', trainable=True,
                                           reuse=reuse)
            std = tf.contrib.layers.batch_norm(std, scope='bn0', trainable=True, is_training=is_training, reuse=reuse)
            std = tf.nn.relu(std)
            for i in range(len(stride)):
                std = NetworkBlock(std, ResBlock, n, nChannels[i + 1], stride[i], is_training=is_training, reuse=reuse,
                                   name='Resblock%d' % i)
            fc = tf.reduce_mean(std, [1, 2])
            logits = tf.contrib.layers.fully_connected(fc, label.get_shape().as_list()[-1],
                                                       weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                       biases_initializer=tf.zeros_initializer(),
                                                       trainable=True, scope='full', reuse=reuse)
            end_points['Logits'] = logits
    elif scope == 'Student':
        with tf.variable_scope(scope):
            n = 1
            std = tf.contrib.layers.conv2d(image, nChannels[0], [3, 3], 1, scope='base_conv', trainable=True,
                                           reuse=reuse)
            std = tf.contrib.layers.batch_norm(std, scope='bn0', trainable=True, is_training=is_training, reuse=reuse)
            std = tf.nn.relu(std)
            for i in range(len(stride)):
                std = NetworkBlock(std, ResBlock, n, nChannels[i + 1], stride[i], is_training=is_training, reuse=reuse,
                                   name='Resblock%d' % i)
            fc = tf.reduce_mean(std, [1, 2])
            logits = tf.contrib.layers.fully_connected(fc, label.get_shape().as_list()[-1],
                                                       weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                       biases_initializer=tf.zeros_initializer(),
                                                       trainable=True, scope='full', reuse=reuse)
            end_points['Logits'] = logits

    if Distill is not None:
        if Distill == 'DML':
            teacher_train = True
        else:
            is_training = False
            teacher_train = False
        with tf.variable_scope('Teacher'):
            with tf.contrib.framework.arg_scope([tf.contrib.layers.conv2d, tf.contrib.layers.fully_connected],
                                                variables_collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'Teacher']):
                with tf.contrib.framework.arg_scope([tf.contrib.layers.batch_norm],
                                                    variables_collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'Teacher']):
                    n = 2
                    tch = tf.contrib.layers.conv2d(image, nChannels[0], [3, 3], 1, scope='base_conv',
                                                   trainable=teacher_train, reuse=reuse)
                    tch = tf.contrib.layers.batch_norm(tch, scope='bn0', trainable=teacher_train,
                                                       is_training=is_training, reuse=reuse)
                    tch = tf.nn.relu(tch)
                    for i in range(len(stride)):
                        tch = NetworkBlock(tch, ResBlock, n, nChannels[i + 1], stride[i], is_training=is_training,
                                           reuse=reuse,
                                           name='Resblock%d' % i)
                    fc = tf.reduce_mean(tch, [1, 2])
                    logits_tch = tf.contrib.layers.fully_connected(fc, label.get_shape().as_list()[-1],
                                                                   weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                                   biases_initializer=tf.zeros_initializer(),
                                                                   trainable=teacher_train, scope='full', reuse=reuse)
                    end_points['Logits_tch'] = logits_tch

        with tf.variable_scope('Distillation'):
            feats = tf.get_collection('feat')
            student_feats = feats[:len(feats) // 2]
            teacher_feats = feats[len(feats) // 2:]
            feats_noact = tf.get_collection('feat_noact')
            student_feats_noact = feats[:len(feats_noact) // 2]
            teacher_feats_noact = feats[len(feats_noact) // 2:]

            if Distill == 'Soft_logits':
                tf.add_to_collection('dist', Response.Soft_logits(logits, logits_tch, 3))
            elif Distill == 'DML':
                tf.add_to_collection('dist', Response.DML(logits, logits_tch))
            elif Distill == 'FT':
                tf.add_to_collection('dist', Response.Factor_Transfer(student_feats_noact[-1], teacher_feats_noact[-1]))
            elif Distill == 'ClassKD':
                tf.add_to_collection('dist', Response.ClassKD(logits, logits_tch, 3))

            elif Distill == 'Comb':
                d = [Response.Soft_logits(logits, logits_tch, 3), Relation.RKD(logits, logits_tch, l=[5e1, 1e2]),
                     Relation.MHGD(student_feats, teacher_feats)]
                dist = tf.contrib.layers.fully_connected(d, 1, weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                         biases_initializer=tf.zeros_initializer())
                tf.add_to_collection('dist', dist)

            elif Distill == 'FitNet':
                tf.add_to_collection('dist', Multiple.FitNet(student_feats, teacher_feats))
            elif Distill == 'AT':
                tf.add_to_collection('dist', Multiple.Attention_transfer(student_feats, teacher_feats))
            elif Distill == 'AB':
                tf.add_to_collection('dist', Multiple.AB_distillation(student_feats, teacher_feats, 1., 3e-3))

            elif Distill == 'FSP':
                tf.add_to_collection('dist', Shared.FSP(student_feats, teacher_feats))
            elif Distill[:3] == 'KD-':
                tf.add_to_collection('dist', Shared.KD_SVD(student_feats, teacher_feats, Distill[-3:]))
            elif Distill == 'RKD-SVD':
                tf.add_to_collection('dist',
                                     Relation.RKD(logits, logits_tch, l=[5e1, 1e2]) + Shared.KD_SVD(student_feats, teacher_feats, "SVD"))

            elif Distill == 'RKD':
                tf.add_to_collection('dist', Relation.RKD(logits, logits_tch, l=[5e1, 1e2]))
            elif Distill == 'MHGD':
                tf.add_to_collection('dist', Relation.MHGD(student_feats, teacher_feats))
            elif Distill == 'MHGD-RKD':
                tf.add_to_collection('dist', Relation.MHGD(student_feats, teacher_feats) + Relation.RKD(logits, logits_tch, l=[5e1, 1e2]))
            elif Distill == 'MHGD-RKD-DML':
                tf.add_to_collection('dist',
                                     Relation.MHGD(student_feats, teacher_feats) + Relation.RKD(logits, logits_tch, l=[5e1, 1e2]) + Response.DML(
                                         logits, logits_tch))
            elif Distill == 'MHGD-RKD-SVD':
                tf.add_to_collection('dist',
                                     Relation.MHGD(student_feats, teacher_feats) + Relation.RKD(logits, logits_tch, l=[5e1, 1e2]) +
                                     Shared.KD_SVD(student_feats, teacher_feats, "SVD"))

    return end_points


def ResNet18(image, label, scope, is_training, dataset='cifar', reuse=False, Distill=None, bit_a=32, bit_w=32,
             bit_g=32):
    end_points = {}

    nChannels = []
    if 'cifar' in dataset or 'svhn' in dataset:
        nChannels = [64, 64, 128, 256, 512]
    elif 'imagenet' in dataset:
        nChannels = [64, 256, 512, 1024, 2048]

    assert len(nChannels) > 0, "empty channels!!"

    stride = [1, 2, 2, 2]

    # 4 (stride) * n * 2 + 2. shortcut is not involved.
    # 32 -> 16 -> 8 -> 4
    n = 2
    if scope == 'Teacher':
        with tf.variable_scope(scope):
            std = tf.contrib.layers.conv2d(image, nChannels[0], [3, 3], 1, scope='base_conv', trainable=True,
                                           reuse=reuse)
            std = tf.contrib.layers.batch_norm(std, scope='bn0', trainable=True, is_training=is_training, reuse=reuse)
            std = tf.nn.relu(std)
            for i in range(len(stride)):
                std = NetworkBlock(std, ResBlock, n, nChannels[i + 1], stride[i], is_training=is_training, reuse=reuse,
                                   name='Resblock%d' % i)
            fc = tf.reduce_mean(std, [1, 2])
            logits = tf.contrib.layers.fully_connected(fc, label.get_shape().as_list()[-1],
                                                       weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                       biases_initializer=tf.zeros_initializer(),
                                                       trainable=True, scope='full', reuse=reuse)
            end_points['Logits'] = logits
    elif scope == 'Student':
        fw, fa, fg = get_dorefa(bit_w, bit_a, bit_g)

        # monkey-patch tf.get_variable to apply fw
        def new_get_variable(v):
            name = v.op.name
            # don't binarize first and last layer
            if not name.endswith('weights') or 'base_conv' in name or 'full' in name:
                return v
            else:
                tf.logging.info("Quantizing weight {} at bits {}".format(v.op.name, bit_w))
                return fw(v)

        def nonlin(x):
            if bit_a == 32:
                return tf.nn.relu(x)  # still use relu for 32-bit cases
            return tf.clip_by_value(x, 0.0, 1.0)

        def activate(x):
            tf.logging.info("Quantizing activations {} at bits {}".format(x.name, bit_a))
            return fa(nonlin(x))

        with tf.variable_scope(scope), remap_variables(new_get_variable):
            std = tf.contrib.layers.conv2d(image, nChannels[0], [3, 3], 1, scope='base_conv', trainable=True,
                                           reuse=reuse)
            std = tf.contrib.layers.batch_norm(std, scope='bn0', trainable=True, is_training=is_training, reuse=reuse)
            for i in range(len(stride)):
                std = NetworkBlock(std, ResBlock, n, nChannels[i + 1], stride[i], activate, is_training=is_training,
                                   reuse=reuse,
                                   name='Resblock%d' % i, scope=scope)
            fc = tf.reduce_mean(std, [1, 2])
            logits = tf.contrib.layers.fully_connected(fc, label.get_shape().as_list()[-1],
                                                       weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                       biases_initializer=tf.zeros_initializer(),
                                                       trainable=True, scope='full', reuse=reuse)
            end_points['Logits'] = logits

    if Distill is not None:
        if Distill == 'DML':
            teacher_train = True
        else:
            is_training = False
            teacher_train = False
        with tf.variable_scope('Teacher'):
            with tf.contrib.framework.arg_scope([tf.contrib.layers.conv2d, tf.contrib.layers.fully_connected],
                                                variables_collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'Teacher']):
                with tf.contrib.framework.arg_scope([tf.contrib.layers.batch_norm],
                                                    variables_collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'Teacher']):
                    tch = tf.contrib.layers.conv2d(image, nChannels[0], [3, 3], 1, scope='base_conv',
                                                   trainable=teacher_train, reuse=reuse)
                    tch = tf.contrib.layers.batch_norm(tch, scope='bn0', trainable=teacher_train,
                                                       is_training=is_training, reuse=reuse)
                    tch = tf.nn.relu(tch)
                    for i in range(len(stride)):
                        tch = NetworkBlock(tch, ResBlock, n, nChannels[i + 1], stride[i], is_training=is_training,
                                           reuse=reuse,
                                           name='Resblock%d' % i)
                    fc = tf.reduce_mean(tch, [1, 2])
                    logits_tch = tf.contrib.layers.fully_connected(fc, label.get_shape().as_list()[-1],
                                                                   weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                                   biases_initializer=tf.zeros_initializer(),
                                                                   trainable=teacher_train, scope='full', reuse=reuse)
                    end_points['Logits_tch'] = logits_tch

        with tf.variable_scope('Distillation'):
            feats = tf.get_collection('feat')
            student_feats = feats[:len(feats) // 2]
            teacher_feats = feats[len(feats) // 2:]
            feats_noact = tf.get_collection('feat_noact')
            student_feats_noact = feats[:len(feats_noact) // 2]
            teacher_feats_noact = feats[len(feats_noact) // 2:]

            if Distill == 'Soft_logits':
                tf.add_to_collection('dist', Response.Soft_logits(logits, logits_tch, 3))
            elif Distill == 'DML':
                tf.add_to_collection('dist', Response.DML(logits, logits_tch))
            elif Distill == 'FT':
                tf.add_to_collection('dist', Response.Factor_Transfer(student_feats_noact[-1], teacher_feats_noact[-1]))

            elif Distill == 'FitNet':
                tf.add_to_collection('dist', Multiple.FitNet(student_feats, teacher_feats))
            elif Distill == 'AT':
                tf.add_to_collection('dist', Multiple.Attention_transfer(student_feats, teacher_feats))
            elif Distill == 'AB':
                tf.add_to_collection('dist', Multiple.AB_distillation(student_feats, teacher_feats, 1., 3e-3))

            elif Distill == 'FSP':
                tf.add_to_collection('dist', Shared.FSP(student_feats, teacher_feats))
            elif Distill[:3] == 'KD-':
                tf.add_to_collection('dist', Shared.KD_SVD(student_feats, teacher_feats, Distill[-3:]))

            elif Distill == 'RKD':
                tf.add_to_collection('dist', Relation.RKD(logits, logits_tch, l=[5e1, 1e2]))
            elif Distill == 'MHGD':
                tf.add_to_collection('dist', Relation.MHGD(student_feats, teacher_feats))
            elif Distill == 'MHGD-RKD':
                tf.add_to_collection('dist', Relation.MHGD(student_feats, teacher_feats) + Relation.RKD(logits, logits_tch, l=[5e1, 1e2]))
            elif Distill == 'MHGD-RKD-SVD':
                tf.add_to_collection('dist',
                                     Relation.MHGD(student_feats, teacher_feats) + Relation.RKD(logits, logits_tch, l=[5e1, 1e2]) +
                                     Shared.KD_SVD(student_feats, teacher_feats, "SVD"))

    return end_points


def AlexNetCifar(image, label, scope, is_training, dataset='cifar', reuse=False, Distill=None, bit_a=32, bit_w=32,
                 bit_g=32):
    end_points = {}

    if scope == 'Teacher':
        with tf.variable_scope(scope):
            image = tf.pad(image, [[0, 0], [5, 5], [5, 5], [0, 0]])
            std = tf.contrib.layers.conv2d(image, 64, [11, 11], 1, scope='conv0', padding='VALID', trainable=True,
                                           reuse=reuse)
            std = tf.contrib.layers.batch_norm(std, scope='bn0', trainable=True, is_training=is_training, reuse=reuse)
            std = tf.nn.relu(std)

            std = tf.layers.max_pooling2d(std, 2, strides=2, padding='SAME')
            std = tf.contrib.layers.conv2d(std, 192, [5, 5], padding='SAME', trainable=True, reuse=reuse)
            std = tf.contrib.layers.batch_norm(std, scope='bn1', trainable=True, is_training=is_training, reuse=reuse)
            std = tf.nn.relu(std)
            std = tf.layers.max_pooling2d(std, 2, strides=2, padding='SAME')
            tf.add_to_collection('feat', std)

            std = tf.contrib.layers.conv2d(std, 384, [3, 3], padding='SAME', trainable=True, reuse=reuse)
            std = tf.contrib.layers.batch_norm(std, scope='bn2', trainable=True, is_training=is_training, reuse=reuse)
            std = tf.nn.relu(std)
            tf.add_to_collection('feat', std)

            std = tf.contrib.layers.conv2d(std, 256, [3, 3], padding='SAME', trainable=True, reuse=reuse)
            std = tf.contrib.layers.batch_norm(std, scope='bn3', trainable=True, is_training=is_training, reuse=reuse)
            std = tf.nn.relu(std)
            tf.add_to_collection('feat', std)

            std = tf.contrib.layers.conv2d(std, 256, [3, 3], padding='SAME', trainable=True, reuse=reuse)
            std = tf.contrib.layers.batch_norm(std, scope='bn4', trainable=True, is_training=is_training, reuse=reuse)
            std = tf.nn.relu(std)
            std = tf.layers.max_pooling2d(std, 2, strides=2, padding='SAME')
            tf.add_to_collection('feat', std)

            fc = tf.layers.flatten(std, name='fc_flat')

            fc1 = tf.contrib.layers.fully_connected(fc, 4096, scope='fc0', trainable=True, reuse=reuse)
            fc1 = tf.contrib.layers.batch_norm(fc1, scope='bn_fc0', trainable=True, is_training=is_training,
                                               reuse=reuse)
            fc1 = tf.nn.relu(fc1)
            fc2 = tf.contrib.layers.fully_connected(fc1, 4096, scope='fc1', trainable=True, reuse=reuse)
            fc2 = tf.contrib.layers.batch_norm(fc2, scope='bn_fc1', trainable=True, is_training=is_training,
                                               reuse=reuse)
            fc2 = tf.nn.relu(fc2)
            logits = tf.contrib.layers.fully_connected(fc2, label.get_shape().as_list()[-1], scope='fct',
                                                       trainable=True, reuse=reuse)

            end_points['Logits'] = logits
    else:
        fw, fa, fg = get_dorefa(bit_w, bit_a, bit_g)

        def new_get_variable(v):
            name = v.op.name
            # don't binarize first and last layer
            if not name.endswith('weights') or 'conv0' in name or 'fct' in name:
                return v
            else:
                tf.logging.info("Quantizing weight {} at bits {}".format(v.op.name, bit_w))
                return fw(v)

        def nonlin(x):
            if bit_a == 32:
                return tf.nn.relu(x)  # still use relu for 32-bit cases
            return tf.clip_by_value(x, 0.0, 1.0)

        def activate(x):
            tf.logging.info("Quantizing activations {} at bits {}".format(x.name, bit_a))
            return fa(nonlin(x))

        with tf.variable_scope(scope), remap_variables(new_get_variable):
            image = tf.pad(image, [[0, 0], [5, 5], [5, 5], [0, 0]])
            std = tf.contrib.layers.conv2d(image, 64, [11, 11], 1, scope='conv0', padding='VALID', trainable=True,
                                           reuse=reuse)
            std = tf.contrib.layers.batch_norm(std, scope='bn0', trainable=True, is_training=is_training, reuse=reuse)
            std = tf.nn.relu(std)

            std = activate(std)

            std = tf.layers.max_pooling2d(std, 2, strides=2, padding='SAME')
            std = tf.contrib.layers.conv2d(std, 192, [5, 5], padding='SAME', trainable=True, reuse=reuse)
            std = tf.contrib.layers.batch_norm(std, scope='bn1', trainable=True, is_training=is_training, reuse=reuse)
            std = tf.nn.relu(std)

            std = activate(std)

            std = tf.layers.max_pooling2d(std, 2, strides=2, padding='SAME')
            tf.add_to_collection('feat', std)

            std = tf.contrib.layers.conv2d(std, 384, [3, 3], padding='SAME', trainable=True, reuse=reuse)
            std = tf.contrib.layers.batch_norm(std, scope='bn2', trainable=True, is_training=is_training, reuse=reuse)
            std = tf.nn.relu(std)

            std = activate(std)
            tf.add_to_collection('feat', std)

            std = tf.contrib.layers.conv2d(std, 256, [3, 3], padding='SAME', trainable=True, reuse=reuse)
            std = tf.contrib.layers.batch_norm(std, scope='bn3', trainable=True, is_training=is_training, reuse=reuse)
            std = tf.nn.relu(std)

            std = activate(std)
            tf.add_to_collection('feat', std)

            std = tf.contrib.layers.conv2d(std, 256, [3, 3], padding='SAME', trainable=True, reuse=reuse)
            std = tf.contrib.layers.batch_norm(std, scope='bn4', trainable=True, is_training=is_training, reuse=reuse)
            std = tf.nn.relu(std)

            std = activate(std)

            std = tf.layers.max_pooling2d(std, 2, strides=2, padding='SAME')
            tf.add_to_collection('feat', std)

            fc = tf.layers.flatten(std, name='fc_flat')

            fc1 = tf.contrib.layers.fully_connected(fc, 4096, scope='fc0', trainable=True, reuse=reuse)
            fc1 = tf.contrib.layers.batch_norm(fc1, scope='bn_fc0', trainable=True, is_training=is_training,
                                               reuse=reuse)
            fc1 = tf.nn.relu(fc1)
            fc1 = activate(fc1)

            fc2 = tf.contrib.layers.fully_connected(fc1, 4096, scope='fc1', trainable=True, reuse=reuse)
            fc2 = tf.contrib.layers.batch_norm(fc2, scope='bn_fc1', trainable=True, is_training=is_training,
                                               reuse=reuse)
            fc2 = tf.nn.relu(fc2)

            logits = tf.contrib.layers.fully_connected(fc2, label.get_shape().as_list()[-1], scope='fct',
                                                       trainable=True, reuse=reuse)

            end_points['Logits'] = logits

    if Distill is not None:
        if Distill == 'DML':
            teacher_train = True
        else:
            is_training = False
            teacher_train = False
        with tf.variable_scope('Teacher'):
            with tf.contrib.framework.arg_scope([tf.contrib.layers.conv2d, tf.contrib.layers.fully_connected],
                                                variables_collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'Teacher']):
                with tf.contrib.framework.arg_scope([tf.contrib.layers.batch_norm],
                                                    variables_collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'Teacher']):
                    std = tf.contrib.layers.conv2d(image, 64, [11, 11], 1, scope='conv0', padding='VALID',
                                                   trainable=True,
                                                   reuse=reuse)
                    std = tf.contrib.layers.batch_norm(std, scope='bn0', trainable=True, is_training=is_training,
                                                       reuse=reuse)
                    std = tf.nn.relu(std)

                    std = tf.layers.max_pooling2d(std, 2, strides=2, padding='SAME')
                    # tf.add_to_collection('feat', std)
                    std = tf.contrib.layers.conv2d(std, 192, [5, 5], padding='SAME', trainable=teacher_train,
                                                   reuse=reuse)
                    std = tf.contrib.layers.batch_norm(std, scope='bn1', trainable=teacher_train,
                                                       is_training=is_training,
                                                       reuse=reuse)
                    std = tf.nn.relu(std)

                    std = tf.layers.max_pooling2d(std, 2, strides=2, padding='SAME')
                    tf.add_to_collection('feat', std)
                    std = tf.contrib.layers.conv2d(std, 384, [3, 3], padding='SAME', trainable=teacher_train,
                                                   reuse=reuse)
                    std = tf.contrib.layers.batch_norm(std, scope='bn2', trainable=teacher_train,
                                                       is_training=is_training,
                                                       reuse=reuse)
                    std = tf.nn.relu(std)
                    tf.add_to_collection('feat', std)

                    std = tf.contrib.layers.conv2d(std, 256, [3, 3], padding='SAME', trainable=teacher_train,
                                                   reuse=reuse)
                    std = tf.contrib.layers.batch_norm(std, scope='bn3', trainable=teacher_train,
                                                       is_training=is_training,
                                                       reuse=reuse)
                    std = tf.nn.relu(std)
                    tf.add_to_collection('feat', std)

                    std = tf.contrib.layers.conv2d(std, 256, [3, 3], padding='SAME', trainable=teacher_train,
                                                   reuse=reuse)
                    std = tf.contrib.layers.batch_norm(std, scope='bn4', trainable=teacher_train,
                                                       is_training=is_training,
                                                       reuse=reuse)
                    std = tf.nn.relu(std)
                    std = tf.layers.max_pooling2d(std, 2, strides=2, padding='SAME')
                    tf.add_to_collection('feat', std)
                    fc_tch = tf.layers.flatten(std, name='fc_flat')

                    fc1 = tf.contrib.layers.fully_connected(fc_tch, 4096, scope='fc0', trainable=teacher_train, reuse=reuse)
                    fc1 = tf.contrib.layers.batch_norm(fc1, scope='bn_fc0', trainable=teacher_train,
                                                       is_training=is_training,
                                                       reuse=reuse)
                    fc1 = tf.nn.relu(fc1)
                    fc2 = tf.contrib.layers.fully_connected(fc1, 4096, scope='fc1', trainable=teacher_train,
                                                            reuse=reuse)
                    fc2 = tf.contrib.layers.batch_norm(fc2, scope='bn_fc1', trainable=teacher_train,
                                                       is_training=is_training,
                                                       reuse=reuse)
                    fc2 = tf.nn.relu(fc2)
                    logits_tch = tf.contrib.layers.fully_connected(fc2, label.get_shape().as_list()[-1], scope='fct',
                                                                   trainable=teacher_train, reuse=reuse)

                    end_points['Logits_tch'] = logits_tch

        with tf.variable_scope('Distillation'):
            feats = tf.get_collection('feat')
            student_feats = feats[:len(feats) // 2]
            teacher_feats = feats[len(feats) // 2:]
            feats_noact = tf.get_collection('feat_noact')
            student_feats_noact = feats[:len(feats_noact) // 2]
            teacher_feats_noact = feats[len(feats_noact) // 2:]

            if Distill == 'Soft_logits':
                tf.add_to_collection('dist', Response.Soft_logits(logits, logits_tch, 3))
            elif Distill == 'DML':
                tf.add_to_collection('dist', Response.DML(logits, logits_tch))
            elif Distill == 'FT':
                tf.add_to_collection('dist', Response.Factor_Transfer(student_feats_noact[-1], teacher_feats_noact[-1]))

            elif Distill == 'FitNet':
                tf.add_to_collection('dist', Multiple.FitNet(student_feats, teacher_feats))
            elif Distill == 'AT':
                tf.add_to_collection('dist', Multiple.Attention_transfer(student_feats, teacher_feats))
            elif Distill == 'AB':
                tf.add_to_collection('dist', Multiple.AB_distillation(student_feats, teacher_feats, 1., 3e-3))

            elif Distill == 'FSP':
                tf.add_to_collection('dist', Shared.FSP(student_feats, teacher_feats))
            elif Distill[:3] == 'KD-':
                tf.add_to_collection('dist', Shared.KD_SVD(student_feats, teacher_feats, Distill[-3:]))

            elif Distill == 'RKD':
                tf.add_to_collection('dist', Relation.RKD(logits, logits_tch, l=[5e1, 1e2]))
            elif Distill == 'MHGD':
                tf.add_to_collection('dist', Relation.MHGD(student_feats, teacher_feats))
            elif Distill == 'MHGD-RKD':
                tf.add_to_collection('dist', Relation.MHGD(student_feats, teacher_feats) + Relation.RKD(logits, logits_tch, l=[5e1, 1e2]))
            elif Distill == 'MHGD-RKD-SVD':
                tf.add_to_collection('dist',
                                     Relation.MHGD(student_feats, teacher_feats) + Relation.RKD(logits, logits_tch, l=[5e1, 1e2]) +
                                     Shared.KD_SVD(student_feats, teacher_feats, "SVD"))

    return end_points
