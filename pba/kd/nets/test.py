#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Xue Geng

import os
import tensorflow as tf
from tensorpack import *
from tensorpack.tfutils.summary import add_param_summary
from tensorpack.tfutils.varreplace import remap_variables
from tensorpack.dataflow import dataset
from tensorpack.utils.gpu import get_num_gpu

from imagenet_utils_mixed import ImageNetModel, eval_on_ILSVRC12, eval_on_ILSVRC12_conf, fbresnet_augmentor, get_imagenet_dataflow, \
    GoogleNetResize
from dorefa import get_dorefa, ternarize


class Model(ImageNetModel):
    def __init__(self, dataset_name, class_num, image_shape, train_schema, bit_a, bit_w, bit_g,
                 model_name, init_lr, data_format, weight_decay_pattern, weight_decay, confidence_threshold):
        super(Model, self).__init__()
        self.dataset_name = dataset_name
        self.class_num = class_num
        self.model_name = model_name
        self.bit_a, self.bit_w, self.bit_g = bit_a, bit_w, bit_g
        self.init_lr = init_lr
        self.image_shape = image_shape
        self.data_format = data_format
        self.image_dtype = tf.float32
        self.weight_decay_pattern = weight_decay_pattern
        self.weight_decay = weight_decay
        self.confidence_threshold = confidence_threshold

        if train_schema == 'quantized_train':
            self.float_train, self.quantized_train, self.confidence_train, self.channel_train = False, True, False, False
            # self.weight_decay = 5e-4
        elif train_schema == 'confidence_train':
            self.float_train, self.quantized_train, self.confidence_train, self.channel_train = False, False, True, False
            # self.weight_decay = 0
            self.weight_decay_pattern = 'fc_confidence/W'
        elif train_schema == 'channel_train':
            self.float_train, self.quantized_train, self.confidence_train, self.channel_train = False, False, False, True
        elif train_schema == 'float_train':
            # self.weight_decay = 5e-4
            self.float_train, self.quantized_train, self.confidence_train, self.channel_train = True, False, False, False

    def get_logits(self, image, label=None):
        if self.bit_w == 't':
            fw, fa, fg = get_dorefa(32, 32, 32)
            fw = ternarize
        else:
            fw, fa, fg = get_dorefa(self.bit_w, self.bit_a, self.bit_g)

        # monkey-patch tf.get_variable to apply fw
        def new_get_variable(v):
            if self.float_train:
                return v
            else:
                name = v.op.name
                # don't binarize first and last layer
                if model_name == 'alexnet':
                    if not name.endswith('W') or 'conv0' in name or 'fct' in name:
                        return v
                    else:
                        logger.info("Quantizing weight {}".format(v.op.name))
                        return fw(v)
                elif model_name == 'resnet18' and dataset_name == 'cifar':
                    if not name.endswith('kernel') or 'conv1_1' in name or 'dense' in name:
                        return v
                    else:
                        logger.info("Quantizing weight {}".format(v.op.name))
                        return fw(v)
                elif model_name == 'resnet18' and dataset_name == 'ImageNet':
                    if not name.endswith('W') or 'conv0' in name or 'linear' in name:
                        return v
                    else:
                        logger.info("Quantizing weight {}".format(v.op.name))
                        return fw(v)

        def nonlin(x):
            if BITA == 32:
                return tf.nn.relu(x)  # still use relu for 32-bit cases
            return tf.clip_by_value(x, 0.0, 1.0)

        def activate(x):
            if self.float_train:
                return x
            else:
                return fa(nonlin(x))

        def identity_block2d(input_tensor, kernel_size, filters, stage, block, is_training, reuse,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer()):
            filters1, filters2, filters3 = filters

            conv_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3'
            bn_name_2 = 'bn' + str(stage) + '_' + str(block) + '_3x3'
            x = tf.layers.conv2d(input_tensor, filters2, kernel_size, use_bias=False, padding='SAME',
                                 kernel_initializer=kernel_initializer, name=conv_name_2, reuse=reuse)
            x = tf.layers.batch_normalization(x, training=is_training, name=bn_name_2, reuse=reuse)
            x = activate(x)

            conv_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase'
            bn_name_3 = 'bn' + str(stage) + '_' + str(block) + '_1x1_increase'
            x = tf.layers.conv2d(x, filters3, (kernel_size, kernel_size), use_bias=False, padding='SAME',
                                 kernel_initializer=kernel_initializer, name=conv_name_3, reuse=reuse)
            x = tf.layers.batch_normalization(x, training=is_training, name=bn_name_3, reuse=reuse)

            x = tf.add(input_tensor, x)
            if block != '4b':
                x = activate(x)
            return x

        def conv_block_2d(input_tensor, kernel_size, filters, stage, block, is_training, reuse, strides=(2, 2),
                          kernel_initializer=tf.contrib.layers.variance_scaling_initializer()):
            filters1, filters2, filters3 = filters

            conv_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3'
            bn_name_2 = 'bn' + str(stage) + '_' + str(block) + '_3x3'
            x = tf.layers.conv2d(input_tensor, filters2, (kernel_size, kernel_size), use_bias=False, strides=strides,
                                 padding='SAME', kernel_initializer=kernel_initializer, name=conv_name_2, reuse=reuse)
            x = tf.layers.batch_normalization(x, training=is_training, name=bn_name_2, reuse=reuse)
            x = tf.nn.relu(x)

            conv_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase'
            bn_name_3 = 'bn' + str(stage) + '_' + str(block) + '_1x1_increase'
            x = tf.layers.conv2d(x, filters3, (kernel_size, kernel_size), use_bias=False, padding='SAME',
                                 kernel_initializer=kernel_initializer, name=conv_name_3, reuse=reuse)
            x = tf.layers.batch_normalization(x, training=is_training, name=bn_name_3, reuse=reuse)

            conv_name_4 = 'conv' + str(stage) + '_' + str(block) + '_1x1_shortcut'
            bn_name_4 = 'bn' + str(stage) + '_' + str(block) + '_1x1_shortcut'
            shortcut = tf.layers.conv2d(input_tensor, filters3, (kernel_size, kernel_size), use_bias=False,
                                        strides=strides, padding='SAME', kernel_initializer=kernel_initializer,
                                        name=conv_name_4, reuse=reuse)
            shortcut = tf.layers.batch_normalization(shortcut, training=is_training, name=bn_name_4, reuse=reuse)

            x = tf.add(shortcut, x)
            x = tf.nn.relu(x)
            return x

        def resnet18_cifar(input_tensor, is_training=True, pooling_and_fc=True, reuse=False,
                           kernel_initializer=tf.contrib.layers.variance_scaling_initializer()):
            with remap_variables(new_get_variable):
                x = tf.layers.conv2d(input_tensor, 64, (3, 3), strides=(1, 1), kernel_initializer=kernel_initializer,
                                     use_bias=False, padding='SAME', name='conv1_1/3x3_s1', reuse=reuse)
                x = tf.layers.batch_normalization(x, training=is_training, name='bn1_1/3x3_s1', reuse=reuse)
                x = tf.nn.relu(x)

                x1 = identity_block2d(x, 3, [48, 64, 64], stage=2, block='1b', is_training=is_training, reuse=reuse,
                                      kernel_initializer=kernel_initializer)
                x1 = identity_block2d(x1, 3, [48, 64, 64], stage=3, block='1c', is_training=is_training, reuse=reuse,
                                      kernel_initializer=kernel_initializer)

                x2 = conv_block_2d(x1, 3, [96, 128, 128], stage=3, block='2a', strides=(2, 2), is_training=is_training,
                                   reuse=reuse, kernel_initializer=kernel_initializer)
                x2 = activate(x2)
                x2 = identity_block2d(x2, 3, [96, 128, 128], stage=3, block='2b', is_training=is_training, reuse=reuse,
                                      kernel_initializer=kernel_initializer)

                x3 = conv_block_2d(x2, 3, [128, 256, 256], stage=4, block='3a', strides=(2, 2), is_training=is_training,
                                   reuse=reuse, kernel_initializer=kernel_initializer)
                x3 = activate(x3)
                x3 = identity_block2d(x3, 3, [128, 256, 256], stage=4, block='3b', is_training=is_training, reuse=reuse,
                                      kernel_initializer=kernel_initializer)

                x4 = conv_block_2d(x3, 3, [256, 512, 512], stage=5, block='4a', strides=(2, 2), is_training=is_training,
                                   reuse=reuse, kernel_initializer=kernel_initializer)
                x4 = activate(x4)
                x4 = identity_block2d(x4, 3, [256, 512, 512], stage=5, block='4b', is_training=is_training, reuse=reuse,
                                      kernel_initializer=kernel_initializer)

                print('before gap: ', x4)
                x4 = tf.reduce_mean(x4, [1, 2])
                print('after gap: ', x4)
                # flatten = tf.contrib.layers.flatten(x4)
                prob = tf.layers.dense(x4, self.class_num, reuse=reuse,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer())

                # tmp = tf.trainable_variables()
                # prob = tf.layers.batch_normalization(prob, training=is_training, name='fbn', reuse=reuse)
                print('prob', prob)

            return prob

        def resnet_group(name, l, block_func, features, count, stride):
            with tf.variable_scope(name):
                for i in range(0, count):
                    with tf.variable_scope('block{}'.format(i)):
                        l = block_func(l, features, stride if i == 0 else 1)
            return l

        def resnet_shortcut(l, n_out, stride, activation=tf.identity):
            # data_format = get_arg_scope()['Conv2D']['data_format']
            n_in = l.get_shape().as_list()[1 if self.data_format in ['NCHW', 'channels_first'] else 3]
            if n_in != n_out:  # change dimension when channel is not the same
                return activate(Conv2D('convshortcut', l, n_out, 1, strides=stride, activation=activation))
            else:
                return l

        def get_bn(zero_init=False):
            """
            Zero init gamma is good for resnet. See https://arxiv.org/abs/1706.02677.
            """
            if zero_init:
                return lambda x, name=None: BatchNorm('bn', x, gamma_initializer=tf.zeros_initializer())
            else:
                return lambda x, name=None: BatchNorm('bn', x)

        def resnet_basicblock(l, ch_out, stride):
            shortcut = l
            l = Conv2D('conv1', l, ch_out, 3, strides=stride, activation=BNReLU)
            l = activate(l)
            l = Conv2D('conv2', l, ch_out, 3, activation=get_bn(zero_init=True))
            l = activate(l)
            out = l + resnet_shortcut(shortcut, ch_out, stride, activation=get_bn(zero_init=False))
            return tf.nn.relu(out)

        def resnet18_imagenet(image):
            with remap_variables(new_get_variable), \
                 argscope(Conv2D, use_bias=False,
                          kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out')):
                # Note that this pads the image by [2, 3] instead of [3, 2].
                # Similar things happen in later stride=2 layers as well.
                l = Conv2D('conv0', image, 64, 7, strides=2, activation=BNReLU)
                l = MaxPooling('pool0', l, pool_size=3, strides=2, padding='SAME')
                l = resnet_group('group0', l, resnet_basicblock, 64, 2, 1)
                l = activate(l)
                l = resnet_group('group1', l, resnet_basicblock, 128, 2, 2)
                l = activate(l)
                l = resnet_group('group2', l, resnet_basicblock, 256, 2, 2)
                l = activate(l)
                l = resnet_group('group3', l, resnet_basicblock, 512, 2, 2)
                l = GlobalAvgPooling('gap', l)
                logits = FullyConnected('linear', l, 1000,
                                        kernel_initializer=tf.random_normal_initializer(stddev=0.01))

            # tmp = tf.trainable_variables()
            return logits

        def alexnet(image):
            with remap_variables(new_get_variable), \
                 argscope([Conv2D, BatchNorm, MaxPooling], data_format='channels_first'), \
                 argscope(BatchNorm, momentum=0.9, epsilon=1e-4), \
                 argscope(Conv2D, use_bias=False):
                logits = (LinearWrap(image)
                          .Conv2D('conv0', 96, 12, strides=4, padding='VALID', use_bias=True)
                          .apply(fg)
                          .Conv2D('conv1', 256, 5, padding='SAME', split=2)
                          .apply(fg)
                          .BatchNorm('bn1')
                          .MaxPooling('pool1', 3, 2, padding='SAME')
                          .apply(activate)

                          .Conv2D('conv2', 384, 3)
                          .apply(fg)
                          .BatchNorm('bn2')
                          .MaxPooling('pool2', 3, 2, padding='SAME')
                          .apply(activate)

                          .Conv2D('conv3', 384, 3, split=2)
                          .apply(fg)
                          .BatchNorm('bn3')
                          .apply(activate)

                          .Conv2D('conv4', 256, 3, split=2)
                          .apply(fg)
                          .BatchNorm('bn4')
                          .MaxPooling('pool4', 3, 2, padding='VALID')
                          .apply(activate)

                          .FullyConnected('fc0', 4096)
                          .apply(fg)
                          .BatchNorm('bnfc0')
                          .apply(activate)

                          .FullyConnected('fc1', 4096, use_bias=False)
                          .apply(fg)
                          .BatchNorm('bnfc1')
                          .apply(nonlin)
                          .FullyConnected('fct', self.class_num, use_bias=True)())

            return logits

        logits = None
        if self.model_name == 'alexnet':
            logits = alexnet(image)
        elif self.model_name == 'resnet18':
            if dataset_name == 'cifar':
                logits = resnet18_cifar(image, reuse=tf.AUTO_REUSE)
            elif dataset_name == 'ImageNet':
                logits = resnet18_imagenet(image)

        add_param_summary(('.*/W', ['histogram', 'rms']))
        tf.nn.softmax(logits, name='output')  # for prediction

        return logits

    def optimizer(self):
        logger.info("hocon config " + str(conf))
        lr = tf.get_variable('learning_rate', initializer=self.init_lr, trainable=False)
        tf.summary.scalar('learning_rate-summary', lr)
        if self.quantized_train:
            logger.info("adam")
            return tf.train.AdamOptimizer(lr, epsilon=1e-5)
        else:
            return tf.train.MomentumOptimizer(lr, momentum=0.9)

    def image_preprocess(self, image):
        if self.dataset_name == 'ImageNet':
            print 'ImageNet preprocessing.'
            return super(Model, self).image_preprocess(image)

        with tf.name_scope('image_preprocess'):
            if image.dtype.base_dtype != tf.float32:
                image = tf.cast(image, tf.float32)
            mean = [125.426445, 123.07675, 114.030174]  # rgb
            std = [51.5865, 50.847, 51.255]
            if self.image_bgr:
                mean = mean[::-1]
                std = std[::-1]
            image_mean = tf.constant(mean, dtype=tf.float32)
            image_std = tf.constant(std, dtype=tf.float32)
            image -= image_mean
            image /= image_std
            return image


def get_data_imagenet(data_dir, dataset_name, batch_size):
    isTrain = dataset_name == 'train'
    augmentors = fbresnet_augmentor(isTrain)
    return get_imagenet_dataflow(data_dir, dataset_name, batch_size, augmentors, parallel=16)


def get_data_cifar(train_or_test, cifar_classnum, image_shape, batch_size):
    from tensorpack.dataflow import dataset
    isTrain = train_or_test == 'train'
    if cifar_classnum == 10:
        ds = dataset.Cifar10(train_or_test)
    else:
        ds = dataset.Cifar100(train_or_test)

    if isTrain:
        import numpy as np
        import cv2
        augmentors = [
            GoogleNetResize(target_shape=image_shape),
            imgaug.RandomOrderAug(
                [imgaug.BrightnessScale((0.6, 1.4), clip=False),
                 imgaug.Contrast((0.6, 1.4), clip=False),
                 imgaug.Saturation(0.4, rgb=False),
                 # rgb-bgr conversion for the constants copied from fb.resnet.torch
                 imgaug.Lighting(0.1,
                                 eigval=np.asarray(
                                     [0.2175, 0.0188, 0.0045][::-1]) * 255.0,
                                 eigvec=np.array(
                                     [[-0.5675, 0.7192, 0.4009],
                                      [-0.5808, -0.0045, -0.8140],
                                      [-0.5836, -0.6948, 0.4203]],
                                     dtype='float32')[::-1, ::-1]
                                 )]),
            imgaug.Flip(horiz=True),
        ]
    else:
        import cv2
        re_size = 256
        if image_shape == 32:
            re_size = 40
        augmentors = [
            imgaug.ResizeShortestEdge(re_size, cv2.INTER_CUBIC),
            imgaug.CenterCrop((image_shape, image_shape)),
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, batch_size, remainder=not isTrain)
    if isTrain:
        ds = PrefetchDataZMQ(ds, 5)
    return ds


def get_config(dataset_name, class_num, image_shape, data_format, data_dir, train_schema, BITW, BITA, BITG, init_lr, lr,
               epoches, total_data, total_batch_size, num_gpu, model_name, weight_decay_pattern, weight_decay,
               confidence_threshold):
    dataset_train, dataset_test = None, None
    if dataset_name == 'cifar':
        dataset_train = get_data_cifar('train', class_num, image_shape, total_batch_size // num_gpu)
        dataset_test = get_data_cifar('test', class_num, image_shape, total_batch_size // num_gpu)
    elif dataset_name == 'ImageNet':
        dataset_train = get_data_imagenet(data_dir, 'train', total_batch_size // num_gpu)
        dataset_test = get_data_imagenet(data_dir, 'val', total_batch_size // num_gpu)

    summary = [ClassificationError('wrong-top1', 'val-error-top1'), ClassificationError('wrong-top5', 'val-error-top5')]
    if train_schema == 'confidence_train':
        # lr = [(4, 1e-2), (6, 5e-3), (8, 1e-3), (10, 5e-4)]
        # epoches = 12
        summary.append(ClassificationError('conf-correct-top1', 'val-conf-correct-top1'))

    return TrainConfig(
        model=Model(dataset_name, class_num, image_shape, train_schema, BITW, BITA, BITG,
                    model_name, init_lr, data_format, weight_decay_pattern, weight_decay, confidence_threshold),
        data=QueueInput(dataset_train),
        callbacks=[
            ModelSaver(),
            InferenceRunner(dataset_test, summary),
            ScheduledHyperParamSetter('learning_rate', lr),
        ],
        steps_per_epoch=total_data // total_batch_size,
        max_epoch=epoches,
    )


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    from pyhocon import ConfigFactory
    import sys

    config_path = sys.argv[1]
    conf = ConfigFactory.parse_file(config_path)
    gpus = conf['gpus']
    dataset_name = conf['dataset']
    dorefa = conf['dorefa'].split(',')
    BITW, BITA, BITG = map(int, dorefa)
    model_path = conf.get_string('load')
    class_num = conf.get_int('class_num')
    image_shape = conf.get_int('image_shape')
    data_format = conf.get_string('data_format')
    train_schema = conf['train_schema']
    lr_string = conf['lr']
    model_name = conf.get_string("model_name")
    weight_decay_pattern = conf.get_string("weight_decay_pattern")
    weight_decay = conf['weight_decay']
    init_lr = conf['init_lr']

    confidence_threshold = 0.5
    if 'confidence_threshold' in conf:
        confidence_threshold = conf.get_float('confidence_threshold')

    eval = False
    if 'eval' in conf:
        eval = conf.get_int('eval')

    lr = []
    i = 0
    while i < len(lr_string):
        if i % 2 == 0:
            lr.append((int(lr_string[i][1:]), float(lr_string[i + 1][:-1])))
            i += 2

    epoches = conf.get_int('epoches')
    total_data = conf.get_int('total_data')
    total_batch_size = conf.get_int('total_batch_size')
    data_dir = conf.get_string("data_dir")

    os.environ['CUDA_VISIBLE_DEVICES'] = gpus

    num_gpu = len(gpus.split(','))

    if eval:
        model = Model(dataset_name, class_num, image_shape, train_schema, BITW, BITA, BITG, model_name, init_lr,
                      data_format, weight_decay_pattern, weight_decay, confidence_threshold)
        ds = None
        if dataset_name == 'cifar' and train_schema == 'confidence_train':
            ds = get_data_cifar('test', class_num, image_shape, total_batch_size // num_gpu)
            eval_on_ILSVRC12_conf(model, get_model_loader(model_path), ds)
    else:
        with tf.Graph().as_default():
            if train_schema == 'float_train':
                logger.set_logger_dir(
                    os.path.join('train_log', dataset_name + str(class_num) + '_' + model_name + '_' + train_schema))
            else:
                logger.set_logger_dir(os.path.join('train_log',
                                                   dataset_name + str(class_num) + '_' + model_name + '_' + conf[
                                                       'dorefa'] + '_' + train_schema))
            config = get_config(dataset_name, class_num, image_shape, data_format, data_dir, train_schema, BITW, BITA,
                                BITG, init_lr, lr, epoches, total_data, total_batch_size, num_gpu, model_name,
                                weight_decay_pattern, weight_decay, confidence_threshold)
            if model_path:
                config.session_init = get_model_loader(model_path)

            num_gpu = get_num_gpu()
            trainer = SimpleTrainer() if num_gpu <= 1 \
                else SyncMultiGPUTrainerParameterServer(num_gpu)
            launch_train_with_config(config, trainer)
