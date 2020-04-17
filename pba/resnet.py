# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains definitions for Residual Networks.

Residual networks ('v1' ResNets) were originally proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

The full preactivation 'v2' ResNet variant was introduced by:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027

The key difference of the full preactivation 'v2' variant compared to the
'v1' variant in [1] is the use of batch normalization before every weight layer
rather than after.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
DEFAULT_VERSION = 2
DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16, )
ALLOWED_TYPES = (DEFAULT_DTYPE, ) + CASTABLE_TYPES


################################################################################
# Convenience functions for building the ResNet model.
################################################################################
def batch_norm(inputs, training, data_format):
    """Performs a batch normalization using a standard set of parameters."""
    # We set fused=True for a significant performance boost. See
    # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
    return tf.layers.batch_normalization(
        inputs=inputs,
        axis=1 if data_format == 'channels_first' else 3,
        momentum=_BATCH_NORM_DECAY,
        epsilon=_BATCH_NORM_EPSILON,
        center=True,
        scale=True,
        training=training,
        fused=True)


def fixed_padding(inputs, kernel_size, data_format):
    """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or [batch,
      height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
      Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
  """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if data_format == 'channels_first':
        padded_inputs = tf.pad(
            inputs, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(
            inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
    """Strided 2-D convolution with explicit padding."""
    # The padding is consistent and is based only on `kernel_size`, not on the
    # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)

    return tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=('SAME' if strides == 1 else 'VALID'),
        use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer(),
        data_format=data_format)


################################################################################
# ResNet block definitions.
################################################################################
def _building_block_v1(inputs, filters, training, projection_shortcut, strides,
                       data_format):
    """A single block for ResNet v1, without a bottleneck.

  Convolution then batch normalization then ReLU as described by:
    Deep Residual Learning for Image Recognition
    https://arxiv.org/pdf/1512.03385.pdf
    by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or [batch,
      height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    training: A Boolean for whether the model is in training or inference mode.
      Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts (typically
      a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block; shape should match inputs.
  """
    shortcut = inputs

    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)
        shortcut = batch_norm(
            inputs=shortcut, training=training, data_format=data_format)

    inputs = conv2d_fixed_padding(
        inputs=inputs,
        filters=filters,
        kernel_size=3,
        strides=strides,
        data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)

    inputs = conv2d_fixed_padding(
        inputs=inputs,
        filters=filters,
        kernel_size=3,
        strides=1,
        data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs += shortcut
    inputs = tf.nn.relu(inputs)

    return inputs


def _building_block_v2(inputs, filters, training, projection_shortcut, strides,
                       data_format):
    """A single block for ResNet v2, without a bottleneck.

  Batch normalization then ReLu then convolution as described by:
    Identity Mappings in Deep Residual Networks
    https://arxiv.org/pdf/1603.05027.pdf
    by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or [batch,
      height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    training: A Boolean for whether the model is in training or inference mode.
      Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts (typically
      a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block; shape should match inputs.
  """
    shortcut = inputs
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = conv2d_fixed_padding(
        inputs=inputs,
        filters=filters,
        kernel_size=3,
        strides=strides,
        data_format=data_format)

    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_fixed_padding(
        inputs=inputs,
        filters=filters,
        kernel_size=3,
        strides=1,
        data_format=data_format)

    return inputs + shortcut


def _bottleneck_block_v1(inputs, filters, training, projection_shortcut,
                         strides, data_format):
    """A single block for ResNet v1, with a bottleneck.

  Similar to _building_block_v1(), except using the "bottleneck" blocks
  described in:
    Convolution then batch normalization then ReLU as described by:
      Deep Residual Learning for Image Recognition
      https://arxiv.org/pdf/1512.03385.pdf
      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or [batch,
      height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    training: A Boolean for whether the model is in training or inference mode.
      Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts (typically
      a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block; shape should match inputs.
  """
    shortcut = inputs

    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)
        shortcut = batch_norm(
            inputs=shortcut, training=training, data_format=data_format)

    inputs = conv2d_fixed_padding(
        inputs=inputs,
        filters=filters,
        kernel_size=1,
        strides=1,
        data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)

    inputs = conv2d_fixed_padding(
        inputs=inputs,
        filters=filters,
        kernel_size=3,
        strides=strides,
        data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)

    inputs = conv2d_fixed_padding(
        inputs=inputs,
        filters=4 * filters,
        kernel_size=1,
        strides=1,
        data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs += shortcut
    inputs = tf.nn.relu(inputs)

    return inputs


def _bottleneck_block_v2(inputs, filters, training, projection_shortcut,
                         strides, data_format):
    """A single block for ResNet v2, with a bottleneck.

  Similar to _building_block_v2(), except using the "bottleneck" blocks
  described in:
    Convolution then batch normalization then ReLU as described by:
      Deep Residual Learning for Image Recognition
      https://arxiv.org/pdf/1512.03385.pdf
      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

  Adapted to the ordering conventions of:
    Batch normalization then ReLu then convolution as described by:
      Identity Mappings in Deep Residual Networks
      https://arxiv.org/pdf/1603.05027.pdf
      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or [batch,
      height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    training: A Boolean for whether the model is in training or inference mode.
      Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts (typically
      a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block; shape should match inputs.
  """
    shortcut = inputs
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = conv2d_fixed_padding(
        inputs=inputs,
        filters=filters,
        kernel_size=1,
        strides=1,
        data_format=data_format)

    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_fixed_padding(
        inputs=inputs,
        filters=filters,
        kernel_size=3,
        strides=strides,
        data_format=data_format)

    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_fixed_padding(
        inputs=inputs,
        filters=4 * filters,
        kernel_size=1,
        strides=1,
        data_format=data_format)

    return inputs + shortcut


def block_layer(inputs, filters, bottleneck, block_fn, blocks, strides,
                training, name, data_format):
    """Creates one layer of blocks for the ResNet model.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or [batch,
      height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first convolution of the layer.
    bottleneck: Is the block created a bottleneck block.
    block_fn: The block to use within the model, either `building_block` or
      `bottleneck_block`.
    blocks: The number of blocks contained in the layer.
    strides: The stride to use for the first convolution of the layer. If
      greater than 1, this layer will ultimately downsample the input.
    training: Either True or False, whether we are currently training the model.
      Needed for batch norm.
    name: A string name for the tensor output of the block layer.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block layer.
  """

    # Bottleneck blocks end with 4x the number of filters as they start with
    filters_out = filters * 4 if bottleneck else filters

    def projection_shortcut(inputs):
        return conv2d_fixed_padding(
            inputs=inputs,
            filters=filters_out,
            kernel_size=1,
            strides=strides,
            data_format=data_format)

    # Only the first block per block_layer uses projection_shortcut and strides
    inputs = block_fn(inputs, filters, training, projection_shortcut, strides,
                      data_format)

    for _ in range(1, blocks):
        inputs = block_fn(inputs, filters, training, None, 1, data_format)

    return tf.identity(inputs, name)


def build_resnet_model(inputs, num_classes, hparams, training):
    """Creates a model for classifying an image.

  Args:
    inputs: Input tensor.
    num_classes: The number of classes used as labels.
    hparams: tf.hparam object.
    training: Training or evaluation.

  Raises:
    ValueError: if invalid version is selected.

  Returns:
    Output tensor.
  """
    resnet_size = hparams.resnet_size
    if resnet_size % 6 != 2:
        raise ValueError('resnet_size must be 6n + 2:', resnet_size)

    # Kernel size is 3 and conv stride is 1.
    bottleneck = False
    num_filters = hparams.num_filters
    first_pool_size = None
    first_pool_stride = None
    num_blocks = (resnet_size - 2) // 6
    block_sizes = [num_blocks * 3]
    block_strides = [1, 2, 2]
    data_format = 'channels_last'
    resnet_version = DEFAULT_VERSION
    dtype = DEFAULT_DTYPE
    if resnet_version not in (1, 2):
        raise ValueError(
            'Resnet version should be 1 or 2. See README for citations.')
    if bottleneck:
        if resnet_version == 1:
            block_fn = _bottleneck_block_v1
        else:
            block_fn = _bottleneck_block_v2
    else:
        if resnet_version == 1:
            block_fn = _building_block_v1
        else:
            block_fn = _building_block_v2

    if dtype not in ALLOWED_TYPES:
        raise ValueError('dtype must be one of: {}'.format(ALLOWED_TYPES))

    pre_activation = resnet_version == 2

    # We do not include batch normalization or activation functions in V2
    # for the initial conv1 because the first ResNet unit will perform these
    # for both the shortcut and non-shortcut paths as part of the first
    # block's projection. Cf. Appendix of [2].
    if resnet_version == 1:
        inputs = batch_norm(inputs, training, data_format)
        inputs = tf.nn.relu(inputs)

    if first_pool_size:
        inputs = tf.layers.max_pooling2d(
            inputs=inputs,
            pool_size=first_pool_size,
            strides=first_pool_stride,
            padding='SAME',
            data_format=data_format)
        inputs = tf.identity(inputs, 'initial_max_pool')

    for i, num_blocks in enumerate(block_sizes):
        num_filters = num_filters * (2**i)
        inputs = block_layer(
            inputs=inputs,
            filters=num_filters,
            bottleneck=bottleneck,
            block_fn=block_fn,
            blocks=num_blocks,
            strides=block_strides[i],
            training=training,
            name='block_layer{}'.format(i + 1),
            data_format=data_format)

    # Only apply the BN and ReLU for model that does pre_activation in each
    # building/bottleneck block, eg resnet V2.
    if pre_activation:
        inputs = batch_norm(inputs, training, data_format)
        inputs = tf.nn.relu(inputs)

    # The current top layer has shape
    # `batch_size x pool_size x pool_size x final_size`.
    # ResNet does an Average Pooling layer over pool_size,
    # but that is the same as doing a reduce_mean. We do a reduce_mean
    # here because it performs better than AveragePooling2D.
    axes = [2, 3] if data_format == 'channels_first' else [1, 2]
    inputs = tf.reduce_mean(inputs, axes, keepdims=True)
    inputs = tf.identity(inputs, 'final_reduce_mean')

    inputs = tf.squeeze(inputs, axes)
    inputs = tf.layers.dense(inputs=inputs, units=num_classes)
    inputs = tf.identity(inputs, 'final_dense')
    return inputs
