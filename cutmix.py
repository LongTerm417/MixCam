# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np


def cutmix(img_batch, cut_batch, beta=1.0):
    batch_size = int(img_batch.shape[0])
    img_h, img_w, img_channels = tf.shape(img_batch)[1], tf.shape(img_batch)[2], tf.shape(img_batch)[3]
    # num_classes = tf.shape(label_batch)[1]
    imgs, labs = [], []
    # CHOOSE RANDOM LOCATION
    cut_xs = tf.cast(tf.random.uniform([batch_size], 0, tf.cast(img_w, tf.float32)), tf.int32)
    cut_ys = tf.cast(tf.random.uniform([batch_size], 0, tf.cast(img_h, tf.float32)), tf.int32)
    lam = np.array([np.random.beta(beta, beta) for _ in range(batch_size)])
    cut_ratios = tf.math.sqrt(1 - tf.convert_to_tensor(lam, dtype=tf.float32))  # cut ratio
    cut_ws = tf.cast(tf.cast(img_w, tf.float32) * cut_ratios, tf.int32)
    cut_hs = tf.cast(tf.cast(img_h, tf.float32) * cut_ratios, tf.int32)
    yas = tf.math.maximum(0, cut_ys - cut_hs // 2)
    ybs = tf.math.minimum(img_h, cut_ys + cut_hs // 2)
    xas = tf.math.maximum(0, cut_xs - cut_ws // 2)
    xbs = tf.math.minimum(img_w, cut_xs + cut_ws // 2)
    # CHOOSE RANDOM IMAGE TO CUTMIX WITH
    js = tf.random.shuffle(tf.range(batch_size, dtype=tf.int32))
    for i in range(batch_size):
        ya, yb, xa, xb, j = yas[i], ybs[i], xas[i], xbs[i], js[i]
        img_org, img_cut = img_batch[i], cut_batch[j]
        # MAKE CUTMIX IMAGE
        img_org_left_middle = img_org[ya:yb, 0:xa, :]
        img_cut = img_cut[ya:yb, xa:xb, :]
        img_org_right_middle = img_org[ya:yb, xb:img_w, :]
        img_middle = tf.concat([img_org_left_middle, img_cut, img_org_right_middle], axis=1)
        img_cutmix = tf.concat([img_org[0:ya, :, :], img_middle, img_org[yb:img_h, :, :]], axis=0)
        imgs.append(img_cutmix)
    img_batch = tf.reshape(tf.stack(imgs), (batch_size, img_h, img_w, 3))
    return img_batch


if __name__ == '__main__':
    pass

