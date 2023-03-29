# coding=utf-8
"""Implementation of MixCam attack."""
# MixCam before scale

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import numpy as np
import cv2
import scipy.stats as st
from imageio import imread, imsave
from tensorflow.contrib.image import transform as images_transform
from tensorflow.contrib.image import rotate as images_rotate

from func import *
from cutmix import cutmix

from nets import inception_v3, inception_v4, inception_resnet_v2, resnet_v2

import random

import warnings

warnings.filterwarnings('ignore')

slim = tf.contrib.slim

tf.flags.DEFINE_integer('batch_size', 8, 'How many images process at one time.')

tf.flags.DEFINE_float('max_epsilon', 16.0, 'max epsilon.')

tf.flags.DEFINE_integer('num_iter', 10, 'max iteration.')

tf.flags.DEFINE_float('momentum', 1.0, 'momentum about the model.')

tf.flags.DEFINE_float('portion', 0.6, 'protion for the mixed image')

tf.flags.DEFINE_integer('size', 3, 'Number of randomly sampled images')

tf.flags.DEFINE_integer('image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer('image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_float('prob', 0.5, 'probability of using diverse inputs.')

tf.flags.DEFINE_integer('image_resize', 331, 'Height of each input images.')

tf.flags.DEFINE_string('checkpoint_path', './models', 'Path to checkpoint for pretained models.')

tf.flags.DEFINE_string('input_dir', './dev_data/val_rs', 'Input directory with images.')

tf.flags.DEFINE_string('output_dir', './outputs', 'Output directory with images.')

tf.flags.DEFINE_string('model_name', "inception_v3", "Name of the model")

tf.flags.DEFINE_string('attack_method', "", "Name of the model")

tf.flags.DEFINE_integer('percentile', 90, "Name of the model")

tf.flags.DEFINE_integer('sigma', 10, "Name of the model")

tf.flags.DEFINE_string('mix_op', 'mixup', 'Output directory with images.')



FLAGS = tf.flags.FLAGS
print(f'model_name = {FLAGS.model_name}')
np.random.seed(0)
tf.set_random_seed(0)
random.seed(0)

model_checkpoint_map = {
    'inception_v3': os.path.join(FLAGS.checkpoint_path, 'inception_v3.ckpt'),
    'adv_inception_v3': os.path.join(FLAGS.checkpoint_path, 'adv_inception_v3_rename.ckpt'),
    'ens3_adv_inception_v3': os.path.join(FLAGS.checkpoint_path, 'ens3_adv_inception_v3_rename.ckpt'),
    'ens4_adv_inception_v3': os.path.join(FLAGS.checkpoint_path, 'ens4_adv_inception_v3_rename.ckpt'),
    'inception_v4': os.path.join(FLAGS.checkpoint_path, 'inception_v4.ckpt'),
    'inception_resnet_v2': os.path.join(FLAGS.checkpoint_path, 'inception_resnet_v2_2016_08_30.ckpt'),
    'ens_adv_inception_resnet_v2': os.path.join(FLAGS.checkpoint_path, 'ens_adv_inception_resnet_v2_rename.ckpt'),
    'resnet_v2': os.path.join(FLAGS.checkpoint_path, 'resnet_v2_101.ckpt')}

_layer_names = {"resnet_v2": ["PrePool", "Predictions"],
                "inception_v3": ["PrePool", "Predictions"],
                "inception_v4": ["PrePool", "Predictions"],
                "inception_resnet_v2": ["PrePool", "Predictions"],
                 }

percentile_list = [i for i in range(1, 100) if (i % FLAGS.sigma) == 0]


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


kernel = gkern(7, 3).astype(np.float32)
stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
stack_kernel = np.expand_dims(stack_kernel, 3)


def load_images(input_dir, batch_shape):
    """Read png images from input directory in batches.
    Args:
      input_dir: input directory
      batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]
    Yields:
      filenames: list file names without path of each image
        Lenght of this list could be less than batch_size, in this case only
        first few images of the result are elements of the minibatch.
      images: array with all images from this batch
    """
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*')):
        with tf.gfile.Open(filepath, 'rb') as f:
            image = imread(f, pilmode='RGB').astype(np.float) / 255.0
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        # print(f'filepath={filepath}')
        # if filepath == './dev_data/val_rs/cat.jpg':
        #     print('*****************************************')
        #     image = cv2.resize(image, (299, 299))
        images[idx, :, :, :] = image * 2.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images


def save_images(images, filenames, output_dir):
    """Saves images to the output directory.

    Args:
        images: array with minibatch of images
        filenames: list of filenames without path
            If number of file names in this list less than number of images in
            the minibatch then only first len(filenames) images will be saved.
        output_dir: directory where to save images
    """
    for i, filename in enumerate(filenames):
        # Images for inception classifier are normalized to be in [-1, 1] interval,
        # so rescale them back to [0, 1].
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            # imsave(f, img_as_ubyte((images[i, :, :, :] + 1.0) * 0.5), format='png')
            imsave(f, (images[i, :, :, :] + 1.0) * 0.5, format='png')


def check_or_create_dir(directory):
    """Check if directory exists otherwise create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)

        
def grad_cam(end_points, predicted_class, nb_classes=1001, eval_image_size=FLAGS.image_height):
    _logits_name = "Logits"
    # Conv layer tensor [?,10,10,2048]
    layer_name = _layer_names[FLAGS.model_name][0]
    conv_layer = end_points[layer_name]
    predicted_class = tf.reshape(predicted_class, [-1])
    one_hot = tf.one_hot(predicted_class, nb_classes, 1.0)

    signal = tf.multiply(end_points[_logits_name], one_hot)
    loss = tf.reduce_mean(signal, axis=1)
    grads = tf.gradients(loss, conv_layer)[0]

    # Normalizing the gradients
    norm_grads = tf.divide(grads, tf.reshape(tf.sqrt(tf.reduce_mean(tf.square(grads), axis=[1, 2, 3])), [FLAGS.batch_size, 1, 1, 1]) + tf.constant(1e-5))

    output, grads_val = conv_layer, norm_grads # sess.run([conv_layer, norm_grads], feed_dict={x_input: images})
    weights = tf.reduce_mean(grads_val, axis=(1, 2)) 			 # [8, 2048]
    cam = tf.ones(output.shape[0: 3], dtype=tf.float32)	 # [10,10]
    # print(f'cam.shape={cam.shape}')
    # result = list()
    '''
    weights: [batch * 2048]
    output:  [batch * 10 * 10, 2048]
    return [batch * 299 * 299 * 1]
    '''
    # print(f'weight={weights.shape}')
    # print(f'output={output.shape}')
    cam += tf.einsum("bmnk, bkl->bmn", output, tf.reshape(weights, [weights.shape[0], weights.shape[1], 1]))
    cam = tf.maximum(cam, 0)
    # print(f'cam.shape={cam.shape}')
    # cam = tf.reshape(cam / tf.reduce_max(cam, axis=[1, 2]), [FLAGS.batch_size, -1])
    cam = tf.divide(cam, tf.reshape(tf.reduce_max(cam, axis=[1, 2]), [FLAGS.batch_size, 1, 1]))
    # cam = tf.reshape(cam, [FLAGS.batch_size, -1])
    # print(f'cam1={cam.shape}')
    cam = tf.image.resize_images(tf.expand_dims(cam, axis=-1), [eval_image_size, eval_image_size], method=0)
    #
    # cam = tf.reshape(cam, [FLAGS.batch_size, eval_image_size, eval_image_size])
    # percentile = tf.contrib.distributions.percentile(cam, FLAGS.percentile, axis=[1, 2])
    # # print(f'percentile={percentile.shape}')
    # cam = cam - tf.reshape(percentile, [FLAGS.batch_size, 1, 1])
    # cam = tf.maximum(tf.sign(cam), 0)
    return cam


def kl_for_probs(p, q):
    '''
    计算两个离散分布的KL散度
    :param p: 分布1
    :param q:分布2
    :return: KL散度值
    '''

    neg_ent = tf.reduce_sum(p * tf.log(p), axis=-1)
    neg_cross_ent = tf.reduce_sum(p * tf.log(q), axis=-1)
    kl = neg_ent - neg_cross_ent
    return kl


def js_for_probs(p, q):
    M = (p + q) / 2
    return 0.5 * (kl_for_probs(p, M) + kl_for_probs(q, M))


def psnr(p, q):
    p, q = p * 255, q * 255
    mse = tf.reduce_mean(tf.squared_difference(p, q), axis=-1)
    psnr_value = tf.subtract(20 * tf.log(255.) / tf.log(10.), 10 * tf.log(mse) / tf.log(10.))
    return psnr_value


def get_model_results(x, model_name=FLAGS.model_name, num_classes=1001):
    logits, end_points = None, None
    if model_name == 'resnet_v2':
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits, end_points = resnet_v2.resnet_v2_101(
                x, num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)
    elif model_name == 'inception_v3':
        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits, end_points = inception_v3.inception_v3(
                x, num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)
    elif model_name == 'inception_v4':
        with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
            logits, end_points = inception_v4.inception_v4(
                x, num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)
    elif model_name == 'inception_resnet_v2':
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            logits, end_points = inception_resnet_v2.inception_resnet_v2(
                x, num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)

    return logits, end_points


def percentile_value(x, down_p, step_v):
    '''
    :param x:  [batch_size, 299, 299, 1]
    :param down_p: 70
    :param upper_p: 90
    :return: [batch_size, 299, 299, 1]
    '''
    down_percentile = tf.contrib.distributions.percentile(x, down_p, axis=[1, 2, 3])
    # upper_percentile = tf.contrib.distributions.percentile(x, down_p + step_v, axis=[1, 2, 3])

    down_v = tf.minimum(tf.sign(x - tf.reshape(down_percentile, [FLAGS.batch_size, 1, 1, 1])), 0)
    # upper_v = -tf.minimum(tf.sign(x - tf.reshape(upper_percentile, [FLAGS.batch_size, 1, 1, 1])), 0)

    return x * down_v


def cam_augment(x, cam_value):
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    # percentile_list = [FLAGS.percentile - (i * 10) for i in range(FLAGS.size)]
    if FLAGS.mix_op == 'mixup':
        augment_value = [tf.random.uniform(x.get_shape().as_list(), minval=-eps, maxval=eps, dtype=tf.dtypes.float32)
                           + (1 - FLAGS.portion) * x
                           + FLAGS.portion * percentile_value(cam_value, down_p=dp, step_v=10)
                           for dp in percentile_list]
    elif FLAGS.mix_op == 'admix':
        augment_value = [tf.random.uniform(x.get_shape().as_list(), minval=-eps, maxval=eps, dtype=tf.dtypes.float32)
                         + x
                         + 0.2 * percentile_value(cam_value, down_p=dp, step_v=10)
                         for dp in percentile_list]
    elif FLAGS.mix_op == 'cutmix':
        # print(f'xxx={x.shape}, cam_value={cam_value.shape}')
        augment_value = [tf.random.uniform(x.get_shape().as_list(), minval=-eps, maxval=eps, dtype=tf.dtypes.float32)
                         + cutmix(x, tf.concat([percentile_value(cam_value, down_p=dp, step_v=10)] * 3, axis=-1))
                         for dp in percentile_list]
    return tf.concat(augment_value, axis=0)


def graph(x, y, i, x_max, x_min, grad):
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    num_iter = FLAGS.num_iter
    alpha = eps / num_iter
    momentum = FLAGS.momentum
    num_classes = 1001

    logits_v3, end_points_inc_v3 = get_model_results(x=x, model_name=FLAGS.model_name, num_classes=num_classes)

    cam_value = grad_cam(end_points=end_points_inc_v3, predicted_class=y, nb_classes=num_classes)

    x_enhanced = cam_augment(x, cam_value)
    x_batch = tf.concat([x_enhanced, x_enhanced / 2., x_enhanced / 4., x_enhanced / 8., x_enhanced / 16.], axis=0)
    num_size = len(percentile_list)

    '''input diversity'''
    x_input = x_batch
    if 'di' in FLAGS.attack_method.lower():
        x_input = input_diversity(x_batch)

    logits_v3, end_points_inc_v3 = get_model_results(x=x_input, model_name=FLAGS.model_name, num_classes=num_classes)

    one_hot = tf.concat([tf.one_hot(y, num_classes)] * 5 * num_size, axis=0)

    cross_entropy = tf.losses.softmax_cross_entropy(one_hot, logits_v3)

    noise = tf.reduce_mean(tf.split(tf.gradients(cross_entropy, x_batch, colocate_gradients_with_ops=True)[0], 5) * tf.constant([1, 1/2., 1/4., 1/8., 1/16.])[:, None, None, None, None], axis=0)
    noise = tf.reduce_sum(tf.split(noise, num_size), axis=0)
    '''TI'''
    if 'ti' in FLAGS.attack_method.lower():
        noise = tf.nn.depthwise_conv2d(noise, stack_kernel, strides=[1, 1, 1, 1], padding='SAME')

    noise = noise / tf.reduce_mean(tf.abs(noise), [1, 2, 3], keep_dims=True)
    noise = momentum * grad + noise
    x = x + alpha * tf.sign(noise)
    x = tf.clip_by_value(x, x_min, x_max)
    i = tf.add(i, 1)

    return x, y, i, x_max, x_min, noise


def normal(x: np.array):
    '''
    normal to [0, 1]
    :param x:
    :return:
    '''

    # x = x - np.mean(x)
    # x = x / np.max(np.abs(x))
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def stop(x, y, i, x_max, x_min, grad):
    num_iter = FLAGS.num_iter
    return i < num_iter


def image_augmentation(x):
    # img, noise
    one = tf.fill([tf.shape(x)[0], 1], 1.)
    zero = tf.fill([tf.shape(x)[0], 1], 0.)
    transforms = tf.concat([one, zero, zero, zero, one, zero, zero, zero], axis=1)
    rands = tf.concat([tf.truncated_normal([tf.shape(x)[0], 6], stddev=0.05), zero, zero], axis=1)
    return images_transform(x, transforms + rands, interpolation='BILINEAR')


def image_rotation(x):
    """ imgs, scale, scale is in radians """
    rands = tf.truncated_normal([tf.shape(x)[0]], stddev=0.05)
    return images_rotate(x, rands, interpolation='BILINEAR')


def input_diversity(input_tensor):
    rnd = tf.random_uniform((), FLAGS.image_width, FLAGS.image_resize, dtype=tf.int32)
    rescaled = tf.image.resize_images(input_tensor, [rnd, rnd], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    h_rem = FLAGS.image_resize - rnd
    w_rem = FLAGS.image_resize - rnd
    pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
    pad_bottom = h_rem - pad_top
    pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
    pad_right = w_rem - pad_left
    padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0.)
    padded.set_shape((input_tensor.shape[0], FLAGS.image_resize, FLAGS.image_resize, 3))
    ret = tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(FLAGS.prob), lambda: padded, lambda: input_tensor)
    ret = tf.image.resize_images(ret, [FLAGS.image_height, FLAGS.image_width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return ret


from skimage import io
from matplotlib import pyplot as plt
import cv2

model_variables_map = {"resnet_v2": ["resnet_v2", "resnet_v2"],
                       "inception_v3": ["InceptionV3", "inception_v3"],
                       "inception_v4": ["InceptionV4", "inception_v4"],
                       "inception_resnet_v2": ["InceptionResnetV2", "inception_resnet_v2"],
                      }

def main(_):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # eps is a difference between pixels so it should be in [0, 2] interval.
    # Renormalizing epsilon from [0, 255] to [0, 2].
    f2l = load_labels('./dev_data/val_rs.csv')
    eps = 2 * FLAGS.max_epsilon / 255.0

    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]

    tf.logging.set_verbosity(tf.logging.ERROR)

    save_path = os.path.join(FLAGS.output_dir,
                             FLAGS.mix_op,
                             f'model_{FLAGS.model_name}_method_{FLAGS.attack_method}_{FLAGS.percentile}_{FLAGS.size}_sigma_{FLAGS.sigma}_portion_{FLAGS.portion}')
    check_or_create_dir(save_path)

    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
        x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)

        # with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        #     logits_v3, end_points_v3 = resnet_v2.resnet_v2_101(
        #         x_input, num_classes=1001, is_training=False, reuse=tf.AUTO_REUSE)
        logits_v3, end_points_v3 = get_model_results(x=x_input, model_name=FLAGS.model_name, num_classes=1001)
        pred = tf.argmax(end_points_v3['Predictions'], 1)

        i = tf.constant(0, dtype=tf.float32)
        grad = tf.zeros(shape=batch_shape)
        # grad = tf.placeholder(tf.float32, shape=batch_shape)
        # init_cam = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, 100])
        cam_value = grad_cam(end_points=end_points_v3, predicted_class=pred)
        x_adv, _, _, _, _, _ = tf.while_loop(stop, graph, [x_input, pred, i, x_max, x_min, grad])

        # Run computation
        s = tf.train.Saver(slim.get_model_variables(scope=model_variables_map[FLAGS.model_name][0]))
        # s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
        # s2 = tf.train.Saver(slim.get_model_variables(scope='InceptionV4'))
        # s3 = tf.train.Saver(slim.get_model_variables(scope='InceptionResnetV2'))
        # s4 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2'))
        # s5 = tf.train.Saver(slim.get_model_variables(scope='Ens3AdvInceptionV3'))
        # s6 = tf.train.Saver(slim.get_model_variables(scope='Ens4AdvInceptionV3'))
        # s7 = tf.train.Saver(slim.get_model_variables(scope='EnsAdvInceptionResnetV2'))
        # s8 = tf.train.Saver(slim.get_model_variables(scope='AdvInceptionV3'))
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            s.restore(sess, model_checkpoint_map[model_variables_map[FLAGS.model_name][1]])
            # s1.restore(sess, model_checkpoint_map['inception_v3'])
            # s2.restore(sess, model_checkpoint_map['inception_v4'])
            # s3.restore(sess, model_checkpoint_map['inception_resnet_v2'])
            # s4.restore(sess, model_checkpoint_map['resnet_v2'])
            # s5.restore(sess, model_checkpoint_map['ens3_adv_inception_v3'])
            # s6.restore(sess, model_checkpoint_map['ens4_adv_inception_v3'])
            # s7.restore(sess, model_checkpoint_map['ens_adv_inception_resnet_v2'])
            # s8.restore(sess, model_checkpoint_map['adv_inception_v3'])
            idx = 0
            l2_diff = 0
            for filenames, images in load_images(FLAGS.input_dir, batch_shape):
                idx = idx + 1
                print("start the i={} attack".format(idx))
                # continue
                adv_images = sess.run(x_adv, feed_dict={x_input: images})
                save_images(adv_images, filenames, save_path)
                diff = (adv_images + 1) / 2 * 255 - (images + 1) / 2 * 255
                l2_diff += np.mean(np.linalg.norm(np.reshape(diff, [-1, 3]), axis=1))

            # print('{:.2f}'.format(l2_diff * FLAGS.batch_size / 1000))


def load_labels(file_name):
    import pandas as pd
    dev = pd.read_csv(file_name)
    f2l = {dev.iloc[i]['filename']: dev.iloc[i]['label'] for i in range(len(dev))}
    return f2l


if __name__ == '__main__':
    tf.app.run()
