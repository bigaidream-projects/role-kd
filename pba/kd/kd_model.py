import tensorflow as tf
from pba.kd.nets import nets_factory
from pba.model import Model, ModelTrainer
from pba.kd import op_util
import numpy as np
import scipy.io as sio
import time
import pba.data_utils as data_utils
import os


class KDModel(Model):
    def _build_model(self, inputs, label, is_training, hparams):
        network_fn = nets_factory.get_network_fn(hparams.model_name, dataset=hparams.dataset,
                                                 bit_a=hparams.bit_a, bit_g=hparams.bit_g, bit_w=hparams.bit_w)
        end_points = network_fn(inputs, label, hparams.main_scope, is_training=is_training, reuse=False,
                                Distill=hparams.Distillation)

        loss = tf.losses.softmax_cross_entropy(label, end_points['Logits'])
        if hparams.Distillation == 'DML':
            tf.add_to_collection('teacher_class_loss',
                                 tf.losses.softmax_cross_entropy(label, end_points['Logits_tch']))
        accuracy = tf.contrib.metrics.accuracy(tf.to_int32(tf.argmax(end_points['Logits'], 1)),
                                               tf.to_int32(tf.argmax(label, 1)))
        return end_points['Logits'], loss, accuracy

    def _learning_rate_scheduler(self, lr, epochs, decay_point, decay_rate):
        with tf.variable_scope('learning_rate_scheduler'):
            e, ie, te = epochs
            for i, dp in enumerate(decay_point):
                lr = tf.cond(tf.greater_equal(e, ie + int(te * dp)), lambda: lr * decay_rate,
                             lambda: lr)
            tf.summary.scalar('learning_rate', lr)
            return lr

    def _build_train_op_kd(self, hparams):
        teacher_train_op = None
        train_op_cls = None

        epoch = tf.floor_div(tf.cast(self.global_step, tf.float32) * self.batch_size, hparams.train_size)
        LR = self._learning_rate_scheduler(hparams.lr, [epoch, hparams.init_epoch, hparams.num_epochs],
                                           [0.3, 0.6, 0.8], 0.1)

        lamb = self._learning_rate_scheduler(hparams.lamb, [epoch, hparams.init_epoch, hparams.num_epochs],
                                             [0.3, 0.6, 0.8], 0.6)

        if hparams.Distillation == 'DML':
            teacher_train_op, train_op = op_util.Optimizer_w_DML(self.cost, LR, self.global_step,
                                                                 hparams.weight_decay_rate, hparams.model_name,
                                                                 hparams.main_scope, hparams.opt, lamb
                                                                 )
            # train_op, teacher_train_op = op_util.Optimizer_w_DML( class_loss, LR, epoch, init_epoch, global_step)
        elif hparams.Distillation in {'FitNet', 'FSP', 'AB'}:
            train_op, _ = op_util.Optimizer_w_Initializer(self.cost, LR, epoch, hparams.init_epoch,
                                                          self.global_step, hparams.weight_decay_rate,
                                                          hparams.model_name,
                                                          hparams.main_scope, hparams.opt, lamb)
        elif hparams.Distillation == 'MHGD' or hparams.Distillation == 'MHGD-RKD' or hparams.Distillation == 'MHGD-RKD-DML' or \
                hparams.Distillation == 'MHGD-RKD-SVD':
            train_op, _ = op_util.Optimizer_w_MHGD(self.cost, LR, epoch, hparams.init_epoch,
                                                   self.global_step,
                                                   hparams.weight_decay_rate, hparams.model_name,
                                                   hparams.main_scope, hparams.opt, lamb)
        elif hparams.Distillation == 'FT':
            train_op, _ = op_util.Optimizer_w_FT(self.cost, LR, epoch, hparams.init_epoch,
                                                 self.global_step,
                                                 hparams.weight_decay_rate, hparams.model_name,
                                                 hparams.main_scope, hparams.opt,
                                                 lamb)
        else:
            train_op = op_util.Optimizer_w_Distillation(self.cost, LR, epoch, hparams.init_epoch,
                                                        self.global_step, hparams.Distillation,
                                                        hparams.weight_decay_rate,
                                                        hparams.model_name,
                                                        hparams.main_scope,
                                                        hparams.opt,
                                                        lamb
                                                        )
        self.teacher_train_op = teacher_train_op
        self.train_op = train_op
        self.train_op_cls = train_op_cls

    def _build_graph(self, images, labels, mode):
        is_training = 'train' in mode
        if is_training:
            self.global_step = tf.train.get_or_create_global_step()

        self.predictions, self.cost, self.accuracy = self._build_model(images, labels, is_training, self.hparams)

        if is_training:
            self._build_train_op_kd(self.hparams)

        # Setup checkpointing for this child model
        # Keep 2 or more checkpoints around during training.
        with tf.device('/cpu:0'):
            self.saver = tf.train.Saver(max_to_keep=10)


class KDModelTrainer(ModelTrainer):
    def __init__(self, hparams):
        self._session = None
        self.hparams = hparams

        # Set the random seed to be sure the same validation set
        # is used for each model
        np.random.seed(0)
        self.data_loader = data_utils.DataSet(hparams)
        np.random.seed()  # Put the random seed back to random
        self.data_loader.reset()

        # extra stuff for ray
        self._build_models()
        self._new_session()
        self._setup_model(hparams)
        self._setup_teacher()
        if hparams.pretrained_model:
            tf.logging.info('load pretrained model')
            self.extract_model_spec(hparams.pretrained_model)
        self._session.__enter__()

    def _setup_model(self, hparams):
        if hparams.pretrained_model:
            with tf.device('/cpu:0'):
                if self.hparams.Distillation is not None:
                    self.var_list = [e for e in self.var_list if 'Student' in e.name]
                    tf.logging.info('Student params assigned {}'.format(len(self.var_list)))
                self._saver = tf.train.Saver(max_to_keep=10, var_list=self.var_list)

    def _setup_teacher(self):
        if self.hparams.Distillation is not None and self.hparams.Distillation != 'DML':
            global_variables = tf.get_collection('Teacher')
            teacher = sio.loadmat(self.hparams.teacher)
            n = 0
            for v in global_variables:
                if teacher.get(v.name[:-2]) is not None:
                    # print v.name, v.name[:-2], teacher[v.name[:-2]].shape, v.get_shape()
                    self.session.run(v.assign(teacher[v.name[:-2]].reshape(*v.get_shape().as_list())))
                    n += 1
            tf.logging.info('Teacher params assigned {}'.format(n))

    def _build_models(self):
        """Builds the image models for train and eval."""
        m = KDModel(self.hparams, self.data_loader.num_classes, self.data_loader.image_size)
        m.build('train')
        self._saver = m.saver
        self.m = m
        self.meval = m

    def save_model(self, checkpoint_dir, step=None):
        """Dumps model into the backup_dir.

        Args:
          step: If provided, creates a checkpoint with the given step
            number, instead of overwriting the existing checkpoints.
        """
        ## save variables to use for something
        # var = {}
        # variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) + tf.get_collection('BN_collection')
        # for v in variables:
        #     var[v.name[:-2]] = self.session.run(v)
        # sio.savemat(self.m.hparams.train_dir + '/train_params.mat', var)

        model_save_name = os.path.join(checkpoint_dir,
                                       'model.ckpt') + '-' + str(step)
        self.saver.save(self.session, model_save_name)
        tf.logging.info('Saved child model')

        return model_save_name

    def _run_training_loop(self, curr_epoch):
        steps_per_epoch = int(self.m.hparams.train_size / self.m.hparams.batch_size)
        tf.logging.info('steps per epoch: {}'.format(steps_per_epoch))
        curr_step = self.session.run(self.m.global_step)
        assert curr_step % steps_per_epoch == 0

        total_loss = []
        cls_loss = []
        correct, count = 0, 0
        start_time = time.time()
        for step in range(steps_per_epoch):
            train_images, train_labels = self.data_loader.next_batch(curr_epoch)
            if self.hparams.Distillation == 'DML':
                self.session.run([self.m.teacher_train_op],
                                 feed_dict={self.m.images: train_images,
                                            self.m.labels: np.squeeze(train_labels)})

            # add class loss as debugging info.
            # if self.hparams.Distillation == 'MHGD':
            #     tl, cls, train_pred, train_acc = self.session.run(
            #         [self.m.train_op, self.m.train_op_cls, self.m.predictions, self.m.accuracy],
            #         feed_dict={self.m.images: train_images,
            #                    self.m.labels: np.squeeze(train_labels)})
            # else:
            tl, train_pred, train_acc = self.session.run(
                [self.m.train_op, self.m.predictions, self.m.accuracy],
                feed_dict={self.m.images: train_images,
                           self.m.labels: np.squeeze(train_labels)})

            total_loss.append(tl)
            # cls_loss.append(cls)
            correct += np.sum(
                np.equal(np.argmax(train_labels, 1), np.argmax(train_pred, 1)))
            count += len(train_pred)
            # if step == 0:
            # tf.logging.info('step loss: %.4f, cls loss: %.4f', tl, cls)
            step += 1
        tf.logging.info('current epoch %s: loss = %.4f (%.3f sec/epoch)',  # , class loss = %.4f (%.3f sec/epoch)',
                        str(curr_epoch).rjust(6, '0'),
                        np.mean(total_loss),
                        # np.mean(cls_loss),
                        time.time() - start_time)
        train_acc = correct * 1.0 / count

        return train_acc

    def eval_child_model(self, model, data_loader, mode):
        """
        evaluate one epoch.
        :param model:
        :param data_loader:
        :param mode:
        :return:
        """
        if mode == 'val':
            images = data_loader.val_images
            labels = data_loader.val_labels
        elif mode == 'test':
            images = data_loader.test_images
            labels = data_loader.test_labels
        else:
            raise ValueError('Not valid eval mode')
        assert len(images) == len(labels)
        tf.logging.info('model.batch_size is {}'.format(model.batch_size))
        eval_batches = int(len(images) / model.batch_size)

        sum_val_accuracy = []
        for i in range(eval_batches):
            val_batch = images[i * model.batch_size:(i + 1) * model.batch_size]
            acc = self.session.run(self.m.accuracy,
                                   feed_dict={self.m.images: val_batch,
                                              self.m.labels:
                                                  np.squeeze(
                                                      labels[i * model.batch_size:(i + 1) * model.batch_size])})
            sum_val_accuracy.append(acc)

        sum_val_accuracy = np.mean(sum_val_accuracy)

        return sum_val_accuracy
