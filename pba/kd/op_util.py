import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from re import split


def Optimizer_w_Distillation(class_loss, LR, epoch, init_epoch, global_step, Distillation, weight_decay, model_name,
                             main_scope='Teacher', optimize='adam', lamb=1):
    with tf.variable_scope('Optimizer_w_Distillation'):
        ### TODO: no weight decay for teacher.
        # get variables and update operations
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        teacher_variables = tf.get_collection('Teacher')
        variables = list(set(variables) - set(teacher_variables))

        # make optimizer w/ learning rate scheduler
        if optimize == 'adam':
            optimizer = tf.train.MomentumOptimizer(LR, 0.9, use_nesterov=True)
        else:
            optimizer = tf.train.AdamOptimizer(LR, epsilon=1e-5)

        # . 1. Student. No teacher. student only last layer has quantization.
        # . 2. Student. teacher. teacher no decay.
        # . 3. Teacher. No student. all layers has quantization.

        # conv_var = []
        # if main_scope == 'Student':
        #     if 'ResNet' in model_name:
        #         conv_var = [var for var in variables if
        #                     ('base_conv' in var.name or 'full' in var.name) and 'weights' in var.name]
        #     elif 'AlexNet' in model_name:
        #         conv_var = [var for var in variables if
        #                     ('conv0' in var.name or 'fct' in var.name) and 'weights' in var.name]
        #
        #     for l in conv_var:
        #         tf.logging.info("student reqgularized layer name {}".format(l.name))
        #
        #     # conv_var = [var for var in variables if 'weights' in var.name]
        # elif main_scope == 'Teacher':
        #     # named scope in Teacher is not in Teacher collection. Only with main_scope=Student,
        #     # then in Teacher collection
        #     conv_var = [var for var in variables if 'weights' in var.name]

        # tf.logging.info("main scope {}".format(main_scope))
        # for var in conv_var:
        #     tf.logging.info("var name {}".format(var.name))
        #     tf.logging.info("var shape {}".format(var.shape))
        # tf.logging.info("weight decay {}".format(weight_decay))
        l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in variables]) * weight_decay
        total_loss = None
        gradients = None

        if Distillation is None:
            # training main-task
            total_loss = class_loss + l2_loss
            tf.summary.scalar('loss/total_loss', total_loss)
            gradients = optimizer.compute_gradients(total_loss, var_list=variables)

        elif Distillation == 'Soft_logits':
            # multi-task learning with alpha
            total_loss = class_loss * 0.7 + l2_loss + tf.get_collection('dist')[0] * 0.3
            tf.summary.scalar('loss/total_loss', total_loss)
            gradients = optimizer.compute_gradients(total_loss, var_list=variables)
        elif Distillation == 'AT' or Distillation == 'RKD' or Distillation == 'ClassKD':
            # simple multi-task learning
            total_loss = class_loss + l2_loss + lamb * tf.get_collection('dist')[0]
            tf.summary.scalar('loss/total_loss', total_loss)
            gradients = optimizer.compute_gradients(total_loss, var_list=variables)
        elif Distillation[:3] == 'KD-' or Distillation == 'RKD-SVD' or Distillation == 'MHGD-RKD-SVD':
            # multi-task learning w/ distillation gradients clipping
            # distillation gradients are clipped by norm of main-task gradients
            # reg_loss = tf.add_n(tf.losses.get_regularization_losses())
            distillation_loss = tf.get_collection('dist')[0]
            total_loss = class_loss + l2_loss + lamb * distillation_loss
            tf.summary.scalar('loss/total_loss', total_loss)
            tf.summary.scalar('loss/distillation_loss', distillation_loss)
            gradients = optimizer.compute_gradients(class_loss, var_list=variables)
            gradient_wdecay = optimizer.compute_gradients(l2_loss, var_list=variables)
            gradient_dist = optimizer.compute_gradients(distillation_loss, var_list=variables)

            with tf.variable_scope('clip_grad'):
                for i, (gc, gw, gd) in enumerate(zip(gradients, gradient_wdecay, gradient_dist)):
                    gw = 0. if gw[0] is None else gw[0]
                    if gd[0] is not None:
                        norm = tf.sqrt(tf.reduce_sum(tf.square(gc[0]))) * sigmoid(epoch, 0)
                        gradients[i] = (gc[0] + gw + tf.clip_by_norm(gd[0], norm), gc[1])
                    elif gc[0] is not None:
                        gradients[i] = (gc[0] + gw, gc[1])

            if Distillation[-3:] == 'SVP':
                gradient_dist += optimizer.compute_gradients(tf.add_n(tf.get_collection('basis_loss')), var_list=tf.get_collection('basises'))

        # merge update operators and make train operator
        update_ops.append(optimizer.apply_gradients(gradients, global_step=global_step))
        update_op = tf.group(*update_ops)
        train_op = control_flow_ops.with_dependencies([update_op], total_loss, name='train_op')
        # train_op_cls = control_flow_ops.with_dependencies([update_op], class_loss, name='train_op_cls')
        return train_op#, train_op_cls


def Optimizer_w_Initializer(class_loss, LR, epoch, init_epoch, global_step, weight_decay, model_name,
                             main_scope='Teacher', optimize='adam', lamb=1):
    with tf.variable_scope('Optimizer_w_Distillation'):
        # get variables and update operations
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        teacher_variables = tf.get_collection('Teacher')
        variables = list(set(variables) - set(teacher_variables))

        # make optimizer w/ learning rate scheduler
        if optimize == 'adam':
            optimizer = tf.train.MomentumOptimizer(LR, 0.9, use_nesterov=True)
        else:
            optimizer = tf.train.AdamOptimizer(LR, epsilon=1e-5)

        reg_loss = tf.add_n([tf.nn.l2_loss(var) for var in variables]) * weight_decay
        # initialization and fine-tuning
        distillation_loss = tf.get_collection('dist')[0]
        total_loss = class_loss + reg_loss
        tf.summary.scalar('loss/total_loss', total_loss)
        gradients = optimizer.compute_gradients(total_loss, var_list=variables)

        gradient_dist = optimizer.compute_gradients(distillation_loss, var_list=variables)
        gradient_wdecay = optimizer.compute_gradients(reg_loss, var_list=variables)
        with tf.variable_scope('clip_grad'):
            for i, (gw, gd) in enumerate(zip(gradient_wdecay, gradient_dist)):
                if gd[0] is not None:
                    gradient_dist[i] = (gw[0] + gd[0], gd[1])

        # merge update operators and make train operator
        update_ops.append(optimizer.apply_gradients(gradients, global_step=global_step))
        update_op = tf.group(*update_ops)
        train_op = control_flow_ops.with_dependencies([update_op], total_loss, name='train_op')

        update_ops_dist = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_ops_dist.append(optimizer.apply_gradients(gradient_dist, global_step=global_step))
        update_op_dist = tf.group(*update_ops_dist)
        train_op_dist = control_flow_ops.with_dependencies([update_op_dist], distillation_loss, name='train_op_dist')
        # train_op_cls = control_flow_ops.with_dependencies([update_op], class_loss, name='train_op_cls')
        return train_op, train_op_dist#, train_op_cls


def Optimizer_w_DML(class_loss, LR, global_step, weight_decay, model_name, main_scope='Teacher', optimize='adam', lamb=1):
    ### TODO: has weight decay for teacher.
    # 1. main_scope = Teacher. has weight decay
    # 2. main_scope = Student. has weight decay for teacher collection all layers.
    # but for student, only last fc layer.
    # with tf.variable_scope('Optimizer_w_Distillation'):
    #     # get variables and update operations
    #     teacher_variables = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if
    #                          v.name[:len('Teacher')] == 'Teacher']
    #     teacher_update_ops = [u for u in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if
    #                           u.name[:len('Teacher')] == 'Teacher']
    #     teacher_reg_loss = tf.add_n(
    #         [l for l in tf.losses.get_regularization_losses() if l.name[:len('Teacher')] == 'Teacher'])
    #
    #     student_variables = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if
    #                          v.name[:len('Teacher')] != 'Teacher']
    #     student_update_ops = [u for u in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if
    #                           u.name[:len('Teacher')] != 'Teacher']
        with tf.variable_scope('Optimizer_w_Distillation'):
            # get variables and update operations
            teacher_variables = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if split('/', v.name)[0] == 'Teacher']
            teacher_update_ops = [u for u in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if split('/', u.name)[0] == 'Teacher']
            teacher_conv_var = [var for var in teacher_variables if 'weights' in var.name]
            teacher_reg_loss = tf.add_n([tf.nn.l2_loss(var) for var in teacher_conv_var]) * weight_decay
            # teacher_reg_loss = tf.add_n([l for l in tf.losses.get_regularization_losses() if split('/', l.name)[0] == 'Teacher'])
            student_variables = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if split('/', v.name)[0] == 'Student']
            student_update_ops = [u for u in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if split('/', u.name)[0] == 'Student']
            student_conv_var = [var for var in student_variables if 'weights' in var.name]
            student_reg_loss = tf.add_n([tf.nn.l2_loss(var) for var in student_conv_var]) * weight_decay
            # student_reg_loss = tf.add_n([l for l in tf.losses.get_regularization_losses() if split('/', l.name)[0] == 'Student'])
            # student_reg_loss = None
            # if 'ResNet' in model_name:
            #     student_reg_loss = tf.add_n(
            #         [l for l in tf.losses.get_regularization_losses() if
            #          l.name[:len('Teacher')] != 'Teacher' and (
            #                  'base_conv' in l.name or 'full' in l.name) and 'weights' in l.name])
            # elif 'AlexNet' in model_name:
            #     student_reg_loss = tf.add_n(
            #         [l for l in tf.losses.get_regularization_losses() if
            #          l.name[:len('Teacher')] != 'Teacher' and (
            #                  'conv0' in l.name or 'fct' in l.name) and 'weights' in l.name])

            optimizer = None
            if optimize == 'sgd':
                optimizer = tf.train.MomentumOptimizer(LR, 0.9, use_nesterov=True)
            elif optimize == 'adam':
                optimizer = tf.train.AdamOptimizer(LR, epsilon=1e-5)

            teacher_loss = tf.get_collection('teacher_class_loss')[0] + teacher_reg_loss + tf.get_collection('dist')[0]
            student_loss = class_loss + student_reg_loss + lamb * tf.get_collection('dist')[0]

            tf.summary.scalar('loss/total_loss', student_loss)
            gradients_teacher = optimizer.compute_gradients(teacher_loss, var_list=teacher_variables)
            gradients_student = optimizer.compute_gradients(student_loss, var_list=student_variables)

            # merge update operators and make train operator
            teacher_update_ops.append(optimizer.apply_gradients(gradients_teacher))
            teacher_update_op = tf.group(*teacher_update_ops)
            teacher_train_op = control_flow_ops.with_dependencies([teacher_update_op], teacher_loss, name='teacher_train_op')

            student_update_ops.append(optimizer.apply_gradients(gradients_student, global_step=global_step))
            student_update_op = tf.group(*student_update_ops)
            student_train_op = control_flow_ops.with_dependencies([student_update_op], student_loss, name='student_train_op')
            # train_op_cls = control_flow_ops.with_dependencies([student_update_op], class_loss, name='train_op_cls')

            return teacher_train_op, student_train_op


def Optimizer_w_FT(class_loss, LR, epoch, init_epoch, global_step, weight_decay, model_name,
                             main_scope='Teacher', optimize='adam', lamb=1):
    with tf.variable_scope('Optimizer_w_Distillation'):
        # get variables and update operations
        variables_teacher = tf.get_collection('Teacher')
        variables_para = tf.get_collection('Para')
        variables = list(
            set(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)) - set(variables_teacher) - set(variables_para))

        # make optimizer w/ learning rate scheduler
        if optimize == 'Teacher':
            optimizer = tf.train.MomentumOptimizer(LR, 0.9, use_nesterov=True)
        else:
            optimizer = tf.train.AdamOptimizer(LR, epsilon=1e-5)

        # . 1. Student. No teacher. student only last layer has quantization.
        # . 2. Student. teacher. teacher no decay.
        # . 3. Teacher. No student. all layers has quantization.

        # conv_var = []
        # if main_scope == 'Student':
        #     if 'ResNet' in model_name:
        #         conv_var = [var for var in variables if
        #                     ('base_conv' in var.name or 'full' in var.name) and 'weights' in var.name]
        #     elif 'AlexNet' in model_name:
        #         conv_var = [var for var in variables if
        #                     ('conv0' in var.name or 'fct' in var.name) and 'weights' in var.name]
        #     for l in conv_var:
        #         tf.logging.info("student reqgularized layer name {}".format(l.name))
        #
        #     # conv_var = [var for var in variables if 'weights' in var.name]
        # elif main_scope == 'Teacher':
        #     # named scope in Teacher is not in Teacher collection. Only with main_scope=Student,
        #     # then in Teacher collection
        #     conv_var = [var for var in variables if 'weights' in var.name]

        reg_loss = tf.add_n([tf.nn.l2_loss(var) for var in variables]) * weight_decay

        distillation_loss = tf.add_n(tf.get_collection('dist')) * 5e2

        total_loss = lamb * distillation_loss + reg_loss + class_loss
        tf.summary.scalar('loss/total_loss', total_loss)
        tf.summary.scalar('loss/distillation_loss', distillation_loss)

        # optimize = tf.train.MomentumOptimizer(LR, 0.9, use_nesterov=True)
        gradients = optimizer.compute_gradients(total_loss, var_list=variables)

        # merge update operators and make train operator
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_ops.append(optimizer.apply_gradients(gradients, global_step=global_step))
        update_op = tf.group(*update_ops)
        train_op = control_flow_ops.with_dependencies([update_op], total_loss, name='train_op')

        para_loss = tf.add_n(tf.get_collection('Para_loss'))
        for v in variables_para:
            if split('/', v.name)[-1][0] == 'w':
                para_loss += tf.reduce_sum(tf.square(v)) * 5e-4

        gradients_para = optimizer.compute_gradients(para_loss, var_list=variables_para)
        update_ops_para = [optimizer.apply_gradients(gradients_para, global_step=global_step)]
        update_ops_para = tf.group(*update_ops_para)
        train_op_para = control_flow_ops.with_dependencies([update_ops_para], para_loss, name='train_op_para')
        # train_op_cls = control_flow_ops.with_dependencies([update_op], class_loss, name='train_op_cls')
        return train_op, train_op_para#, train_op_cls


def Optimizer_w_MHGD(class_loss, LR, epoch, init_epoch, global_step, weight_decay, model_name,
                             main_scope='Teacher', optimize='adam', lamb=1):
    with tf.variable_scope('Optimizer_w_Distillation'):
        # get variables and update operations
        variables_mha = tf.get_collection('MHA')
        variables = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if
                     split('/', v.name)[0] == 'Student']
        # make optimizer w/ learning rate scheduler
        optimizer = None
        if optimize == 'sgd':
            optimizer = tf.train.MomentumOptimizer(LR, 0.9, use_nesterov=True)
        elif optimize == 'adam':
            optimizer = tf.train.AdamOptimizer(LR, epsilon=1e-5)

        # . 1. Student. No teacher. student only last layer has quantization.
        # . 2. Student. teacher. teacher no decay.
        # . 3. Teacher. No student. all layers has quantization.

        conv_var = [var for var in variables if 'weights' in var.name]
        # conv_var = []
        # if main_scope == 'Student':
        #     if 'ResNet' in model_name:
        #         conv_var = [var for var in variables if
        #                     ('base_conv' in var.name or 'full' in var.name) and 'weights' in var.name]
        #     elif 'AlexNet' in model_name:
        #         conv_var = [var for var in variables if
        #                     ('conv0' in var.name or 'fct' in var.name) and 'weights' in var.name]
        #     for l in conv_var:
        #         tf.logging.info("student reqgularized layer name {}".format(l.name))
        #
        #     # conv_var = [var for var in variables if 'weights' in var.name]
        # elif main_scope == 'Teacher':
        #     # named scope in Teacher is not in Teacher collection. Only with main_scope=Student,
        #     # then in Teacher collection
        #     conv_var = [var for var in variables if 'weights' in var.name]

        reg_loss = tf.add_n([tf.nn.l2_loss(var) for var in conv_var]) * weight_decay
        distillation_loss = tf.get_collection('dist')[0]
        total_loss = lamb * distillation_loss + reg_loss + class_loss
        tf.summary.scalar('loss/class_loss', class_loss)
        tf.summary.scalar('loss/total_loss', total_loss)
        tf.summary.scalar('loss/distillation_loss', distillation_loss)

        # optimize = tf.train.MomentumOptimizer(LR, 0.9, use_nesterov=True)

        gradients = optimizer.compute_gradients(class_loss, var_list=variables)
        gradients_wdecay = optimizer.compute_gradients(reg_loss, var_list=variables)
        gradients_dist = optimizer.compute_gradients(distillation_loss, var_list=variables)

        with tf.variable_scope('clip_grad'):
            for i, (gc, gw, gd) in enumerate(zip(gradients, gradients_wdecay, gradients_dist)):
                gw = 0. if gw[0] is None else gw[0]
                if gd[0] is not None:
                    norm = tf.sqrt(tf.reduce_sum(tf.square(gc[0]))) * sigmoid(epoch - init_epoch, 0)
                    gd = tf.clip_by_norm(gd[0], norm)
                    gradients[i] = (gw + gc[0] + gd, gc[1])
                elif gc[0] is not None:
                    gradients[i] = (gw + gc[0], gc[1])

        # merge update operators and make train operator
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_ops_mha = [u for u in update_ops if split('/', u.name)[0] == 'Distillation']
        update_ops = [u for u in update_ops if split('/', u.name)[0] == 'Student']
        update_ops.append(optimizer.apply_gradients(gradients, global_step=global_step))
        update_op = tf.group(*update_ops)
        train_op = control_flow_ops.with_dependencies([update_op], total_loss, name='train_op')

        mha_loss = tf.add_n(tf.get_collection('MHA_loss'))
        tf.summary.scalar('loss/mha_loss', mha_loss)
        for v in variables_mha:
            if v.name.split('/')[-1][0] in {'g', 'w', 'b'}:
                mha_loss += tf.reduce_sum(tf.square(v)) * 5e-4
        gradients_mha = optimizer.compute_gradients(mha_loss, var_list=variables_mha)
        update_ops_mha.append(optimizer.apply_gradients(gradients_mha, global_step=global_step))
        update_op_mha = tf.group(*update_ops_mha)
        train_op_mha = control_flow_ops.with_dependencies([update_op_mha], mha_loss, name='train_op_mha')
        # train_op_cls = control_flow_ops.with_dependencies([update_op], class_loss, name='train_op_cls')

        return train_op, train_op_mha#, train_op_cls


def sigmoid(x, k, d=1):
    s = 1 / (1 + tf.exp(-(x - k) / d))
    s = tf.cond(tf.greater(s, 1 - 1e-8),
                lambda: 1.0, lambda: s)
    return s