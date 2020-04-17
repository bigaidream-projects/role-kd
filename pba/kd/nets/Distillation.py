import tensorflow as tf
import pba.kd.nets.SVP


def Soft_logits(student, teacher, T=2):
    '''
    Geoffrey Hinton, Oriol Vinyals, and Jeff Dean.  
    Distilling the knowledge in a neural network.
    arXiv preprint arXiv:1503.02531, 2015.
    '''
    with tf.variable_scope('KD'):
        return tf.reduce_mean(tf.reduce_sum(
            tf.nn.softmax(teacher / T) * (tf.nn.log_softmax(teacher / T) - tf.nn.log_softmax(student / T)), 1))


def FitNet(student, teacher):
    '''
     Adriana Romero, Nicolas Ballas, Samira Ebrahimi Kahou, Antoine Chassang, Carlo Gatta,  and  Yoshua  Bengio.
     Fitnets:   Hints  for  thin  deep  nets.
     arXiv preprint arXiv:1412.6550, 2014.
    '''

    def Guided(source, target):
        with tf.variable_scope('Guided'):
            Ds = source.get_shape().as_list()[-1]
            Dt = target.get_shape().as_list()[-1]
            if Ds != Dt:
                with tf.variable_scope('Map'):
                    target = tf.contrib.layers.fully_connected(target, Ds, biases_initializer=None, trainable=True,
                                                               scope='fc')

            return tf.reduce_mean(tf.square(source - target))

    return tf.add_n([Guided(std, tch) for i, std, tch in zip(range(len(student)), student, teacher)])


def Attention_transfer(student, teacher, beta=1e3):
    '''
     Zagoruyko, Sergey and Komodakis, Nikos.
     Paying more attention to attention: Improving the performance of convolutional neural networks via attention transfer.
     arXiv preprint arXiv:1612.03928, 2016.
    '''

    def Attention(source, target):
        with tf.variable_scope('Attention'):
            B, _, _, Ds = source.get_shape().as_list()
            Dt = target.get_shape().as_list()[-1]
            if Ds != Dt:
                with tf.variable_scope('Map'):
                    source = tf.contrib.layers.fully_connected(source, Ds, biases_initializer=None, trainable=True,
                                                               scope='fc')

            Qt = tf.contrib.layers.flatten(tf.reduce_mean(tf.square(source), -1))
            Qt = tf.nn.l2_normalize(Qt, [1, 2])

            Qs = tf.contrib.layers.flatten(tf.reduce_mean(tf.square(target), -1))
            Qs = tf.nn.l2_normalize(Qs, [1, 2])

            return tf.reduce_mean(tf.square(Qt - Qs)) * beta / 2

    return tf.add_n([Attention(std, tch) for i, std, tch in zip(range(len(student)), student, teacher)])


def FSP(students, teachers, weight=1e-3):
    """
    Junho Yim, Donggyu Joo, Jihoon Bae, and Junmo Kim.
    A gift from knowledge distillation: Fast optimization, network minimization and transfer learning.
    """

    def Grammian(top, bot):
        with tf.variable_scope('Grammian'):
            t_sz = top.get_shape().as_list()
            b_sz = bot.get_shape().as_list()

            if t_sz[1] > b_sz[1]:
                top = tf.contrib.layers.max_pool2d(top, [2, 2], 2)

            top = tf.reshape(top, [-1, b_sz[1] * b_sz[2], t_sz[3]])
            bot = tf.reshape(bot, [-1, b_sz[1] * b_sz[2], b_sz[3]])

            Gram = tf.matmul(top, bot, transpose_a=True) / (b_sz[1] * b_sz[2])
            return Gram

    with tf.variable_scope('FSP'):
        Dist_loss = []
        for i in range(len(students) - 1):
            gs0 = Grammian(students[i], students[i + 1])
            gt0 = Grammian(teachers[i], teachers[i + 1])

            Dist_loss.append(tf.reduce_mean(tf.reduce_sum(tf.square(tf.stop_gradient(gt0) - gs0), [1, 2]) / 2))

        return tf.add_n(Dist_loss) * weight

def KD_SVD(student_feature_maps, teacher_feature_maps, dist_type='SVD'):
    """
    Seung Hyun Lee, Dae Ha Kim, and Byung Cheol Song.
    Self-supervised knowledge distillation using singular value decomposition. In
    """
    with tf.variable_scope('Distillation'):
        Transfer_losses = []
        K = 4
        V_Tb = V_Sb = None
        for i, sfm, tfm in zip(range(len(student_feature_maps)), student_feature_maps, teacher_feature_maps):
            with tf.variable_scope('Compress_feature_map%d' % i):
                if dist_type == 'SVD':
                    Sigma_T, U_T, V_T = SVP.SVD(tfm, K, name='TSVD%d' % i)
                    Sigma_S, U_S, V_S = SVP.SVD(sfm, K, name='SSVD%d' % i)
                    B, D, _ = V_S.get_shape().as_list()
                    V_S, V_T = SVP.Align_rsv(V_S, V_T)

                elif dist_type == 'EID':
                    # KD-EID is the lighter version of KD-SVD. It is faster than KD-SVD about twice!
                    Sigma_T, U_T, V_T = SVP.SVD_eid(tfm, K, name='TSVD%d' % i)
                    Sigma_S, U_S, V_S = SVP.SVD_eid(sfm, K, name='SSVD%d' % i)
                    B, D, _ = V_S.get_shape().as_list()
                    V_S, V_T = SVP.Align_rsv(V_S, V_T)

                Sigma_T = tf.expand_dims(Sigma_T, 1)
                V_T *= Sigma_T
                V_S *= Sigma_T

            if i > 0:
                with tf.variable_scope('RBF%d' % i):
                    S_rbf = tf.exp(-tf.square(tf.expand_dims(V_S, 2) - tf.expand_dims(V_Sb, 1)) / 8)
                    T_rbf = tf.exp(-tf.square(tf.expand_dims(V_T, 2) - tf.expand_dims(V_Tb, 1)) / 8)

                    l2loss = (S_rbf - tf.stop_gradient(T_rbf)) ** 2
                    l2loss = tf.where(tf.is_finite(l2loss), l2loss, tf.zeros_like(l2loss))
                    Transfer_losses.append(tf.reduce_sum(l2loss))
            V_Tb = V_T
            V_Sb = V_S

        transfer_loss = tf.add_n(Transfer_losses)

        return transfer_loss


def AB_distillation(student, teacher, margin=1., weight=3e-3):
    '''
    Byeongho Heo,  Minsik Lee,  Sangdoo Yun,  and Jin Young Choi.   
    Knowledge transfer via distillation of activation boundaries formed by hidden neurons.
    arXiv preprint arXiv:1811.03233, 2018.
    '''

    def criterion_alternative_L2(source, target, margin, num):
        with tf.variable_scope('criterion_alternative_L2'):
            Dt = target.get_shape().as_list()[-1]
            with tf.variable_scope('Map'):
                source = tf.contrib.layers.conv2d(source, Dt, [1, 1], biases_initializer=None, trainable=True,
                                                  scope='connector%d' % (num))
                source = tf.contrib.layers.batch_norm(source, scope='connector_bn%d' % (num), is_training=True,
                                                      trainable=True, activation_fn=None)

            loss = tf.square(source + margin) * tf.cast(tf.logical_and(source > -margin, target <= 0.), tf.float32) \
                   + tf.square(source - margin) * tf.cast(tf.logical_and(source <= margin, target > 0.), tf.float32)
            return tf.reduce_mean(tf.reduce_sum(tf.abs(loss), [1, 2, 3]))

    return tf.add_n([criterion_alternative_L2(std, tch, margin, i) / 2 ** (-i)
                     for i, std, tch in zip(range(len(student)), student, teacher)]) * weight


def RKD(source, target, l=[1e2, 2e2]):
    '''
    Wonpyo Park, Dongju Kim, Yan Lu, Minsu Cho.  
    relational knowledge distillation.
    arXiv preprint arXiv:1904.05068, 2019.
    '''
    with tf.variable_scope('Relational_Knowledge_distillation'):
        def Huber_loss(x, y):
            with tf.variable_scope('Huber_loss'):
                return tf.reduce_mean(tf.where(tf.less_equal(tf.abs(x - y), 1.),
                                               tf.square(x - y) / 2, tf.abs(x - y) - 1 / 2))

        def Distance_wise_potential(x):
            with tf.variable_scope('DwP'):
                x_square = tf.reduce_sum(tf.square(x), -1)
                prod = tf.matmul(x, x, transpose_b=True)
                distance = tf.sqrt(
                    tf.maximum(tf.expand_dims(x_square, 1) + tf.expand_dims(x_square, 0) - 2 * prod, 1e-12))
                mu = tf.reduce_sum(distance) / tf.reduce_sum(
                    tf.where(distance > 0., tf.ones_like(distance), tf.zeros_like(distance)))
                return distance / (mu + 1e-8)

        def Angle_wise_potential(x):
            with tf.variable_scope('AwP'):
                e = tf.expand_dims(x, 0) - tf.expand_dims(x, 1)
                e_norm = tf.nn.l2_normalize(e, 2)
            return tf.matmul(e_norm, e_norm, transpose_b=True)

        source = tf.nn.l2_normalize(source, 1)
        target = tf.nn.l2_normalize(target, 1)
        distance_loss = Huber_loss(Distance_wise_potential(source), Distance_wise_potential(target))
        angle_loss = Huber_loss(Angle_wise_potential(source), Angle_wise_potential(target))

        return distance_loss * l[0] + angle_loss * l[1]
