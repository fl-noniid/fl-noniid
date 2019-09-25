from __future__ import print_function
import tensorflow as tf
import cifar_input
import noniid
import numpy as np
import six
import svhn_input
from scipy.spatial import distance

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset', 'cifar10', """dataset : cifar10 or svhn""")
tf.app.flags.DEFINE_string('data_path', '/tmp/cifar10', """The directory containing the input data""")
tf.app.flags.DEFINE_integer('num_convs', 1, """# of conv groups; 1, 2, 3""")
tf.app.flags.DEFINE_integer('num_convs_width_multi', 4, """1, 2, 4""")
tf.app.flags.DEFINE_string('last_pooling', 'max', """a way of pooling after the last conv layer; max, avg""")
tf.app.flags.DEFINE_integer('last_stride', 2, """stride in the last max pooling; 2, 4""")
tf.app.flags.DEFINE_integer('num_fcs', 3, """# of fc layers; 1, 3""")
tf.app.flags.DEFINE_integer('num_fcs_width_multi', 2, """1, 2""")
tf.app.flags.DEFINE_integer('num_learners', 10, """1, 10""")
tf.app.flags.DEFINE_integer('num_classes_per_learner', 2, """10, 2""")
tf.app.flags.DEFINE_float('wd_init', 0.0, """weight decay""")
tf.app.flags.DEFINE_boolean('is_bn', False, """batch normalization""")
tf.app.flags.DEFINE_boolean('is_brn', False, """batch renormalization""")
tf.app.flags.DEFINE_boolean('is_aug', False, """data augmentation""")
tf.app.flags.DEFINE_boolean('is_dropout', False, """dropout""")
tf.app.flags.DEFINE_integer('batch_size', 50, """batch size""")
tf.app.flags.DEFINE_string('opt_algo', 'adam', """pure (Pure SGD Optimizer),
                                                  nmom-wb (Momentum Optimizer; averaging only weights and biases), 
                                                  nmom-a (Momentum Optimizer; averaging all the parameters), 
                                                  adam-wb (Adam Optimizer; averaging only weights and biases),
                                                  adam-a (Adam Optimizer; averaging all the parameters)""")

dataset = FLAGS.dataset
data_path = FLAGS.data_path

num_convs = FLAGS.num_convs
num_convs_width_multi = FLAGS.num_convs_width_multi
last_pooling = FLAGS.last_pooling
last_stride = FLAGS.last_stride
num_fcs = FLAGS.num_fcs
num_fcs_width_multi = FLAGS.num_fcs_width_multi

num_learners = FLAGS.num_learners
num_classes_per_learner = FLAGS.num_classes_per_learner

wd_init = FLAGS.wd_init
is_bn = FLAGS.is_bn
is_brn = FLAGS.is_brn
is_aug = FLAGS.is_aug
is_dropout = FLAGS.is_dropout

batch_size = FLAGS.batch_size
opt_algo = FLAGS.opt_algo

if num_learners == 1:
    training_rounds = 100
elif num_learners == 10 and num_classes_per_learner == 10:
    training_rounds = 200
elif num_learners == 10 and num_classes_per_learner == 2:
    training_rounds = 300
elif num_learners == 10 and num_classes_per_learner == 1:
    training_rounds = 300

if opt_algo == 'pure':
    lr_init = 0.05
elif opt_algo == 'nmom-wb' or opt_algo == 'nmom-a':
    lr_init = 0.01
elif opt_algo == 'adam-wb' or opt_algo == 'adam-a':
    lr_init = 0.001

num_training_data_examples = 50000
num_classes = 10
num_test_data_examples = 10000

num_training_data_examples_per_learner = int(num_training_data_examples / num_learners)

local_steps = num_training_data_examples_per_learner / batch_size

dropout_rate1 = 0.2
dropout_rate2 = 0.5

if is_aug == True:
    brightness = 0.2
    min_contrast = 0.0
    max_contrast = 1.5
    crop_size = 24


def image_augmentation(batch_x):
    out = tf.map_fn(lambda img: tf.image.random_brightness(img, brightness), batch_x)
    if dataset != 'svhn':
        out = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), out)
    out = tf.map_fn(lambda img: tf.image.random_contrast(img, min_contrast, max_contrast), out)
    out = tf.map_fn(lambda img: tf.image.per_image_standardization(img), out)
    out = tf.map_fn(lambda img: tf.random_crop(img, [crop_size, crop_size, 3]), out)
    out = tf.image.resize_image_with_crop_or_pad(out, 32, 32)
    return tf.identity(out)


def main():
    if dataset == 'cifar10':
        x_train, y_train = cifar_input.get_cifar_data(data_path, option='train')
        x_test, y_test = cifar_input.get_cifar_data(data_path, option='test')
    elif dataset == 'svhn':
        x_train, y_train, x_test, y_test = svhn_input.get_svhn_data('../datasets')

    model_x_train = []
    model_y_train = []

    sorted_x_train, sorted_y_train = noniid.generate_nonIID(x_train, y_train)
    temp_idx = [0] * num_classes

    cls_idx_ex = [] * 10
    cls_idx_ex.append([9, 7, 3, 6, 4, 8, 0, 5, 1, 2])
    cls_idx_ex.append([2, 0, 6, 7, 5, 8, 3, 4, 1, 9])
    cls_idx_ex.append([4, 3, 1, 0, 6, 9, 5, 8, 2, 7])
    cls_idx_ex.append([3, 0, 1, 4, 9, 5, 2, 8, 7, 6])
    cls_idx_ex.append([8, 4, 6, 1, 2, 9, 0, 5, 7, 3])
    cls_idx_ex.append([6, 9, 8, 4, 7, 2, 3, 5, 0, 1])
    cls_idx_ex.append([8, 6, 2, 4, 1, 7, 3, 0, 9, 5])
    cls_idx_ex.append([5, 1, 7, 6, 9, 3, 2, 0, 8, 4])
    cls_idx_ex.append([5, 6, 1, 3, 9, 0, 7, 2, 4, 8])
    cls_idx_ex.append([8, 0, 9, 1, 3, 4, 5, 7, 6, 2])

    temp_cls_idx = 0
    cls_idx = cls_idx_ex[temp_cls_idx]

    for i in range(num_learners):
        temp_x = np.concatenate([sorted_x_train[cls_idx[j]]
                                 [temp_idx[cls_idx[j]]:
                                  temp_idx[
                                      cls_idx[j]] + num_training_data_examples_per_learner / num_classes_per_learner]
                                 for j in range(num_classes_per_learner)])
        temp_y = np.concatenate([sorted_y_train[cls_idx[j]]
                                 [temp_idx[cls_idx[j]]:
                                  temp_idx[
                                      cls_idx[j]] + num_training_data_examples_per_learner / num_classes_per_learner]
                                 for j in range(num_classes_per_learner)])

        for j in range(num_classes_per_learner):
            temp_idx[cls_idx[j]] += num_training_data_examples_per_learner / num_classes_per_learner

        model_x_train.append(temp_x)
        model_y_train.append(temp_y)

        cls_idx = np.delete(cls_idx, np.s_[:num_classes_per_learner], axis=0)
        if len(cls_idx) == 0 and i < num_learners - 1:
            temp_cls_idx += 1
            cls_idx = cls_idx_ex[temp_cls_idx]

    del sorted_x_train
    del sorted_y_train

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False
    config.allow_soft_placement = True

    model_batch_x = []
    model_batch_y = []

    model_pred = []
    model_cost = []

    model_schedule = []
    model_opt = []
    model_train_op = []

    model_accuracy = []

    model_params = []
    update_ops = []
    variable_set = []
    moving_variances = []
    moving_mean_squares = []
    moving_square_means = []
    moving_means = []

    model_is_train = []

    num_conv_chs = [3, 16 * num_convs_width_multi, 128, 256]
    num_fc_outdims = [num_classes, 256, 256 * num_fcs_width_multi]

    for i in range(num_learners + 2):
        with tf.variable_scope('model{}'.format(i)):
            with tf.name_scope('input'):
                model_batch_x.insert(i, tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='x-input'))
                model_batch_y.insert(i, tf.placeholder(tf.float32, shape=[None, num_classes], name='y-input'))
                model_schedule.insert(i, tf.placeholder_with_default(0.0, shape=None))
                model_is_train.insert(i, tf.placeholder_with_default(False, shape=None, name="is_train"))

            if is_aug is True:
                out = tf.cond(model_is_train[i],
                              lambda: image_augmentation(model_batch_x[i]),
                              lambda: tf.map_fn(lambda img: tf.image.per_image_standardization(img), model_batch_x[i]))
            else:
                out = tf.map_fn(lambda img: tf.image.per_image_standardization(img), model_batch_x[i])

            with tf.variable_scope('conv_g1'):
                with tf.variable_scope('conv1'):
                    out = _conv(out, num_conv_chs[0], num_conv_chs[1], 1)
                    if is_bn is True: out = _batchnorm(out, model_is_train[i])
                    if is_brn is True: out = _batchrnorm(out, model_is_train[i])
                out = tf.nn.relu(out)
                if is_dropout is True: out = tf.layers.dropout(out, dropout_rate1, training=model_is_train[i])
                for j in six.moves.range(1, num_convs):
                    with tf.variable_scope('conv%d' % (j + 1)):
                        out = _conv(out, num_conv_chs[1], num_conv_chs[1], 1)
                        if is_bn is True: out = _batchnorm(out, model_is_train[i])
                        if is_brn is True: out = _batchrnorm(out, model_is_train[i])
                    out = tf.nn.relu(out)
                    if is_dropout is True: out = tf.layers.dropout(out, dropout_rate1, training=model_is_train[i])

            out = tf.nn.max_pool(out, ksize=(1, 3, 3, 1), strides=[1, 2, 2, 1], padding='SAME')

            with tf.variable_scope('conv_g2'):
                with tf.variable_scope('conv1'):
                    out = _conv(out, num_conv_chs[1], num_conv_chs[2], 2)
                    if is_bn is True: out = _batchnorm(out, model_is_train[i])
                    if is_brn is True: out = _batchrnorm(out, model_is_train[i])
                out = tf.nn.relu(out)
                if is_dropout is True: out = tf.layers.dropout(out, dropout_rate1, training=model_is_train[i])
                for j in six.moves.range(1, num_convs):
                    with tf.variable_scope('conv%d' % (j + 1)):
                        out = _conv(out, num_conv_chs[2], num_conv_chs[2], 2)
                        if is_bn is True: out = _batchnorm(out, model_is_train[i])
                        if is_brn is True: out = _batchrnorm(out, model_is_train[i])
                    out = tf.nn.relu(out)
                    if is_dropout is True: out = tf.layers.dropout(out, dropout_rate1, training=model_is_train[i])

            out = tf.nn.max_pool(out, ksize=(1, 3, 3, 1), strides=[1, 2, 2, 1], padding='SAME')

            with tf.variable_scope('conv_g3'):
                with tf.variable_scope('conv1'):
                    out = _conv(out, num_conv_chs[2], num_conv_chs[3], 3)
                    if is_bn is True: out = _batchnorm(out, model_is_train[i])
                    if is_brn is True: out = _batchrnorm(out, model_is_train[i])
                out = tf.nn.relu(out)
                if is_dropout is True: out = tf.layers.dropout(out, dropout_rate1, training=model_is_train[i])
                for j in six.moves.range(1, num_convs):
                    with tf.variable_scope('conv%d' % (j + 1)):
                        out = _conv(out, num_conv_chs[3], num_conv_chs[3], 3)
                        if is_bn is True: out = _batchnorm(out, model_is_train[i])
                        if is_brn is True: out = _batchrnorm(out, model_is_train[i])
                    out = tf.nn.relu(out)
                    if is_dropout is True: out = tf.layers.dropout(out, dropout_rate1, training=model_is_train[i])

            if last_pooling == 'max':
                out = tf.nn.max_pool(out, ksize=(1, 3, 3, 1), strides=[1, last_stride, last_stride, 1], padding='SAME')
            elif last_pooling == 'avg':
                out = _global_avg_pool(out)

            n = 1
            for j in range(1, len(out.get_shape().as_list())):
                n = n * out.get_shape().as_list()[j]
            out = tf.reshape(out, shape=[-1, n])

            for j in six.moves.range(1, num_fcs):
                temp_outdims = num_fc_outdims[2]
                if j == num_fcs - 1: temp_outdims = num_fc_outdims[1]
                with tf.variable_scope('fc%d' % (j)):
                    out = _fc(out, temp_outdims)
                    if is_bn is True: out = _batchnorm(out, model_is_train[i])
                    if is_brn is True: out = _batchrnorm(out, model_is_train[i])
                out = tf.nn.relu(out)
                if is_dropout is True: out = tf.layers.dropout(out, dropout_rate2, training=model_is_train[i])

            with tf.variable_scope('fc%d' % num_fcs):
                logits = _fc(out, num_fc_outdims[0])

            model_pred.insert(i, tf.nn.softmax(logits=logits))

            with tf.name_scope('cost'):
                xent1 = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                labels=model_batch_y[i])
                model_cost.insert(i, tf.reduce_mean(xent1))

            lr = lr_init * model_schedule[i]
            wd = wd_init * model_schedule[i]

            with tf.name_scope('opt'):
                if opt_algo == 'pure':
                    model_opt.insert(i, tf.contrib.opt.MomentumWOptimizer(weight_decay=wd,
                                                                          learning_rate=lr,
                                                                          momentum=0.0))
                elif opt_algo == 'nmom-wb' or opt_algo == 'nmom-a':
                    model_opt.insert(i, tf.contrib.opt.MomentumWOptimizer(weight_decay=wd,
                                                                          learning_rate=lr,
                                                                          momentum=0.9,
                                                                          use_nesterov=True))
                elif opt_algo == 'adam-wb' or opt_algo == 'adam-a':
                    model_opt.insert(i, tf.contrib.opt.AdamWOptimizer(weight_decay=wd,
                                                                      learning_rate=lr))

                update_ops.insert(i, tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='model{}'.format(i)))
                with tf.control_dependencies(update_ops[i]):
                    model_train_op.insert(i, model_opt[i].minimize(model_cost[i]))

            with tf.name_scope('Accuracy'):
                predictions = tf.argmax(model_pred[i], axis=1)
                model_accuracy.insert(i, tf.reduce_mean(
                    tf.to_float(tf.equal(predictions, tf.argmax(model_batch_y[i], axis=1)))))

        model_params.insert(i, tf.contrib.framework.get_trainable_variables(scope='model{}'.format(i)))
        variable_set.insert(i, tf.contrib.framework.get_variables(scope='model{}'.format(i),
                                                                  collection=tf.GraphKeys.VARIABLES))
        moving_means.insert(i, tf.contrib.framework.get_variables(scope='model{}'.format(i),
                                                                  suffix='moving_mean',
                                                                  collection=tf.GraphKeys.VARIABLES))
        moving_variances.insert(i, tf.contrib.framework.get_variables(scope='model{}'.format(i),
                                                                      suffix='moving_variance',
                                                                      collection=tf.GraphKeys.VARIABLES))

        moving_mean_squares.insert(i, [tf.math.square(v) for v in tf.contrib.framework.get_variables(scope='model{}'.format(i),
                                                                                                     suffix='moving_mean',
                                                                                                     collection=tf.GraphKeys.VARIABLES)])

        moving_square_means.insert(i, [var + msq  for var, msq in zip(moving_variances[i], moving_mean_squares[i])])

    init_params = []
    avg_assign_global = []
    avg_assign_wb = []
    avg_assign_a = []
    assign_var_params = []

    for i in range(num_learners + 2):
        init_params.insert(i, [])
        avg_assign_wb.insert(i, [])
        avg_assign_a.insert(i, [])
        for idx in range(len(model_params[0])):
            init_params[i].insert(idx, tf.assign(model_params[i][idx], model_params[0][idx]))

    variable_avg = []
    for idx in range(len(variable_set[0])):
        variable_avg.insert(idx, 0)
        for i in range(num_learners):
            variable_avg[idx] += variable_set[i][idx] / num_learners


    temp_idx = 0
    for idx in variable_set[0]:
        avg_assign_global.insert(temp_idx,
                                 tf.assign(variable_set[num_learners][temp_idx], variable_avg[temp_idx]))
        temp_idx += 1

    temp_idx = 0
    for idx in variable_set[0]:
        for i in range(num_learners + 2):
            if idx.op.name.find(r'MomentumW') == -1 and idx.op.name.find(r'AdamW') == -1:
                avg_assign_wb[i].insert(temp_idx, tf.assign(variable_set[i][temp_idx],
                                                            variable_set[num_learners][temp_idx]))
            else:
                avg_assign_wb[i].insert(temp_idx, tf.assign(variable_set[i][temp_idx],
                                                            variable_set[i][temp_idx]))
        temp_idx += 1

    temp_idx = 0
    for idx in variable_set[0]:
        for i in range(num_learners + 2):
            avg_assign_a[i].insert(temp_idx,
                                   tf.assign(variable_set[i][temp_idx],
                                             variable_set[num_learners][temp_idx]))
        temp_idx += 1

    moving_var_avg = []
    moving_mean_avg = []

    for idx in range(len(moving_means[0])):
        moving_mean_avg.insert(idx, 0)
        for i in range(num_learners):
            moving_mean_avg[idx] += moving_means[i][idx] / num_learners

    for idx in range(len(moving_variances[0])):
        moving_var_avg.insert(idx, 0)
        for i in range(num_learners):
            moving_var_avg[idx] += moving_square_means[i][idx] / num_learners
        moving_var_avg[idx] -= tf.math.square(moving_mean_avg[idx])

    temp_idx = 0
    for idx in moving_variances[0]:
        assign_var_params.insert(temp_idx,
                                 tf.assign(moving_variances[num_learners][temp_idx], moving_var_avg[temp_idx]))
        temp_idx += 1

    with tf.train.MonitoredTrainingSession(config=config) as sess:
        sess.run(init_params)
        training_loss = np.zeros([num_learners], dtype=np.float32)

        best_accuracy = 0.0
        best_accuracy_round = 0
        comm_round = 1
        while comm_round <= training_rounds:
            print('====================Round {0}===================='.format(comm_round))
            if num_learners == 1:
                if comm_round < 51:
                    schedule = 1.0
                elif comm_round >= 51 and comm_round < 76:
                    schedule = 0.1
                elif comm_round >= 76:
                    schedule = 0.01
            elif num_learners == 10 and num_classes_per_learner == 10:
                if comm_round < 101:
                    schedule = 1.0
                elif comm_round >= 101 and comm_round < 151:
                    schedule = 0.1
                elif comm_round >= 151:
                    schedule = 0.01
            elif num_learners == 10 and num_classes_per_learner == 2:
                if comm_round < 151:
                    schedule = 1.0
                elif comm_round >= 151 and comm_round < 226:
                    schedule = 0.1
                elif comm_round >= 226:
                    schedule = 0.01
            elif num_learners == 10 and num_classes_per_learner == 1:
                if comm_round < 151:
                    schedule = 1.0
                elif comm_round >= 151 and comm_round < 226:
                    schedule = 0.1
                elif comm_round >= 226:
                    schedule = 0.01

            training_loss.fill(0.0)

            shuffle_idx = [] * num_learners
            for i in range(num_learners):
                shuffle_idx.insert(i, np.arange(num_training_data_examples_per_learner, dtype=np.int32))
                np.random.shuffle(shuffle_idx[i])

            for j in range(local_steps):
                for i in range(num_learners):
                    batch_xs = model_x_train[i][shuffle_idx[i][:batch_size]]
                    batch_ys = model_y_train[i][shuffle_idx[i][:batch_size]]
                    shuffle_idx[i] = np.delete(shuffle_idx[i], np.s_[:batch_size], axis=0)
                    if len(shuffle_idx[i]) == 0:
                        shuffle_idx[i] = np.arange(num_training_data_examples_per_learner, dtype=np.int32)
                        np.random.shuffle(shuffle_idx[i])

                    _, loss = sess.run([model_train_op[i], model_cost[i]],
                                       feed_dict={model_batch_x[i]: batch_xs, model_batch_y[i]: batch_ys,
                                                  model_is_train[i]: True, model_schedule[i]: schedule})

                    training_loss[i] += loss / local_steps

            print('[1-1] Training Loss (Each Learner, Mean): {0}, {1}'.format(training_loss, np.mean(training_loss)))
            print(' ')

            if num_learners != 1:
                shuffle_idx = np.arange(num_training_data_examples, dtype=np.int32)
                np.random.shuffle(shuffle_idx)
                for idx in range(local_steps):
                    batch_xs = x_train[shuffle_idx[:batch_size]]
                    batch_ys = y_train[shuffle_idx[:batch_size]]
                    shuffle_idx = np.delete(shuffle_idx, np.s_[:batch_size], axis=0)
                    if len(shuffle_idx) == 0:
                        shuffle_idx = np.arange(num_training_data_examples)
                        np.random.shuffle(shuffle_idx)

                    sess.run(model_train_op[num_learners + 1],
                             feed_dict={model_batch_x[num_learners + 1]: batch_xs,
                                        model_batch_y[num_learners + 1]: batch_ys,
                                        model_is_train[num_learners + 1]: True,
                                        model_schedule[num_learners + 1]: schedule})

                model_params_per_layer = [] * (num_learners + 2)
                model_params_whole = [] * (num_learners + 2)
                for i in range(num_learners + 2):
                    temp_params = sess.run(model_params[i])

                    temp_params_per_layer = []
                    temp_params_whole = np.array((), dtype=float)

                    temp_idx = 0
                    for idx in model_params[0]:
                        if idx.op.name.find(r'weight') > 0:
                            temp_params_per_layer.append(temp_params[temp_idx].flatten())
                            temp_params_whole = np.append(temp_params_whole, temp_params[temp_idx].flatten())
                        temp_idx += 1

                    model_params_per_layer.insert(i, temp_params_per_layer)
                    model_params_whole.insert(i, temp_params_whole)

                dist_per_layer = [0.0] * len(model_params_per_layer[0])
                for i in range(num_learners):
                    for j in range(num_learners):
                        if i != j:
                            for idx in range(len(model_params_per_layer[0])):
                                dist_per_layer[idx] += distance.cosine(
                                    model_params_per_layer[i][idx], model_params_per_layer[j][idx]
                                ) / (num_learners * num_learners - num_learners)

                print('[2-1] PD_Ls (Each Layer): {0}'.format(dist_per_layer))

                dist_per_layer = [0.0] * len(model_params_per_layer[0])

                for i in range(num_learners):
                    for idx in range(len(model_params_per_layer[0])):
                        dist_per_layer[idx] += distance.cosine(
                            model_params_per_layer[i][idx], model_params_per_layer[num_learners + 1][idx]
                        ) / num_learners

                print('[2-2] PD_VL (Each Layer): {0}'.format(dist_per_layer))

            sess.run(avg_assign_global)
            sess.run(assign_var_params)

            test_accuracy = 0.0
            test_iters = 100
            test_batch_size = num_test_data_examples / test_iters
            for j in range(test_iters):
                temp_accuracy = sess.run(model_accuracy[num_learners],
                                         feed_dict={model_batch_x[num_learners]:
                                                        x_test[test_batch_size * j:test_batch_size * (j + 1)],
                                                    model_batch_y[num_learners]:
                                                        y_test[test_batch_size * j:test_batch_size * (j + 1)],
                                                    model_is_train[num_learners]: False})
                test_accuracy += temp_accuracy
            test_accuracy = test_accuracy / float(test_iters)

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_accuracy_round = comm_round

            print('[3-1] Test Accuracy (Global Model) - Current Round : {0}'.format(test_accuracy))
            print('[3-2] Test Accuracy (Global Model) - Best : {0}'.format(best_accuracy))
            print('[3-3] Round - Best Test Accuracy (Global Model) : {0}'.format(best_accuracy_round))
            print(' ')

            if opt_algo == 'pure' or opt_algo == 'nmom-wb' or opt_algo == 'adam-wb':
                for i in range(num_learners):
                    sess.run(avg_assign_wb[i])
                sess.run(avg_assign_wb[num_learners + 1])
            elif opt_algo == 'nmom-a' or opt_algo == 'adam-a':
                for i in range(num_learners):
                    sess.run(avg_assign_a[i])
                sess.run(avg_assign_a[num_learners + 1])

            comm_round += 1



    print('Done')
    print('Session closed cleanly')


def get_session(sess):
    session = sess
    while type(session).__name__ != 'Session':
        # pylint: disable=W0212
        session = session._sess
    return session


def _conv(out, in_channels, out_channels, conv_g):
    strides = [1, 1, 1, 1]

    if conv_g == 1:
        kernel = tf.get_variable('weight', shape=[3, 3, in_channels, out_channels], dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.05))
    else:
        kernel = tf.get_variable('weight', shape=[3, 3, in_channels, out_channels], dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(
                                     stddev=np.sqrt(2.0 / (3.0 * 3.0 * out_channels))))

    if is_bn is True or is_brn is True:
        conv = tf.nn.conv2d(out, kernel, strides, padding='SAME')
    else:
        bias = tf.get_variable('bias', shape=[out_channels], dtype=tf.float32,
                               initializer=tf.constant_initializer(value=0.0))
        conv = tf.nn.conv2d(out, kernel, strides, padding='SAME')
        conv = tf.nn.bias_add(conv, bias)
    return conv


def _global_avg_pool(out):
    assert out.get_shape().ndims == 4
    return tf.reduce_mean(out, [1, 2])


def _fc(out, out_dim):
    in_dim = out.get_shape()[1].value

    if out_dim != num_classes:
        weight = tf.get_variable('weight', shape=[out.get_shape()[1], out_dim], dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0 / in_dim)))
    else:
        weight = tf.get_variable('weight', shape=[out.get_shape()[1], out_dim], dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.05))

    if (is_bn is True or is_brn is True) and out_dim != num_classes:
        fc = tf.matmul(out, weight)
    else:
        bias = tf.get_variable('bias', shape=[out_dim], dtype=tf.float32,
                               initializer=tf.constant_initializer(value=0.0))
        fc = tf.nn.xw_plus_b(out, weights=weight, biases=bias)
    return fc


def _batchnorm(out, is_train):
    return tf.layers.batch_normalization(out, training=is_train)


def _batchrnorm(out, is_train):
    return tf.layers.batch_normalization(out,
                                         training=is_train,
                                         renorm=True,
                                         renorm_clipping={'rmax': 2.0, 'rmin': 1.0 / 2.0, 'dmax': 2.0})


if __name__ == '__main__':
    main()
