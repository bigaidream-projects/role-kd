import functools

import tensorflow as tf

from pba.kd.nets import ResNet

networks_map = {
    'ResNet8': ResNet.ResNet8,
    'ResNet18': ResNet.ResNet18,
    'AlexNetCifar': ResNet.AlexNetCifar
}

arg_scopes_map = {
    'ResNet8': ResNet.ResNet_arg_scope,
    'ResNet18': ResNet.ResNet_arg_scope,
    'AlexNetCifar': ResNet.AlexNet_arg_scope
}


def get_network_fn(name, dataset, bit_a=32, bit_w=32, bit_g=32):
    if name not in networks_map:
        raise ValueError('Name of network unknown %s' % name)

    arg_scope = arg_scopes_map[name]()
    func = networks_map[name]

    @functools.wraps(func)
    def network_fn(images, label, scope, is_training, reuse, Distill):
        with tf.contrib.framework.arg_scope(arg_scope):
            return func(images, label, dataset=dataset, scope=scope, is_training=is_training, reuse=reuse,
                        Distill=Distill, bit_a=bit_a, bit_w=bit_w, bit_g=bit_g)

    if hasattr(func, 'default_image_size'):
        network_fn.default_image_size = func.default_image_size

    return network_fn
