# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 14:06:43 2019

@author: acfba
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import numpy as np
import math
import sklearn.metrics as metrics
from cvxopt import matrix, spdiag, solvers
from tqdm import tqdm

import csv

import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.keras as K
tf.disable_v2_behavior()

import tf_slim as slim

from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt


from preprocessing import inception_preprocessing


import data_import as da
import network_model as nm

                
def main(_):
    g_1 = tf.Graph()
    #### Built Vocab ####
    anno = da.get_captions(Flags.tfrecord_train)

    co_occur, vocab2, vocab, counts = da.preProBuildWordVocab(anno, word_count_threshold=20)

    if not os.path.exists('model/'):
        os.mkdir('model/')

    if not os.path.exists(Flags.train_dir_log):
        os.mkdir(Flags.train_dir_log)
      
    np.save('model/vocab', vocab)
    
    np.save('model/counts', counts)

    n_words = len(vocab)
    maxlen = np.max([x for x in map(lambda x: len(x.decode().split(' ')), anno)])

    weights_vector = counts

    with g_1.as_default():
        sess = tf.Session(graph=g_1)
        K.backend.set_session(session=sess)

        dataset_train = tf.data.TFRecordDataset(Flags.tfrecord_train, num_parallel_reads=4)

        dataset_val = tf.data.TFRecordDataset(Flags.tfrecord_val, num_parallel_reads=4)

        dataset_train = da.read_and_decode(dataset_train, Flags.train_batch_size, 1, Flags.train_dataset_size)

        dataset_val = da.read_and_decode(dataset_val, Flags.val_batch_size, 0, Flags.val_dataset_size)

        train_val_iterator = tf.data.Iterator.from_structure(dataset_train.output_types, dataset_train.output_shapes)
        batch_x, batch_y = train_val_iterator.get_next()

        train_iterator = train_val_iterator.make_initializer(dataset_train)
        val_iterator = train_val_iterator.make_initializer(dataset_val)

        init = tf.lookup.KeyValueTensorInitializer(tf.constant(vocab), tf.constant(np.arange(n_words)))

        table_words = tf.lookup.StaticHashTable(
            init,
            default_value=-1)

        with tf.variable_scope('train', reuse=tf.AUTO_REUSE):            
            text = tf.string_split(batch_y)
            text = tf.sparse_tensor_to_dense(text, default_value=' ')

            is_training = tf.placeholder(tf.bool,shape=())

            total_loss,class_loss, true, pred, centers,cont_loss, sim_loss, distil_loss,cnn_model = nm.apply_model(Flags.train_dataset_size, batch_x, text,
                                                                                                                                    n_words, table_words,Flags.image_size,
                                                                                                                                    Flags.rotate,Flags.zoom,Flags.network,
                                                                                                                                    Flags.distillation,Flags.Temp,
                                                                                                                                    Flags.contrastive,Flags.LAMBDA_contrast,
                                                                                                                                    Flags.triplet,Flags.LAMBDA,
                                                                                                                                    Flags.two_headed,Flags.two_head_size,
                                                                                                                                    is_training)
        reg_loss = tf.losses.get_regularization_loss()

        total_loss += reg_loss

        global_step = tf.Variable(0, trainable=False)

        boundaries = [int(0.5 * Flags.how_many_training_steps * (np.sum(counts) / Flags.train_batch_size)),
                        int(0.75 * Flags.how_many_training_steps * (np.sum(counts) / Flags.train_batch_size))]

        lr_init = [0.00001, 0.000001, 0.0000001]

        lr = tf.train.piecewise_constant(global_step, boundaries,
                                                    lr_init)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        train_op = optimizer.minimize(total_loss, global_step=global_step)

        cnn_model.save_weights(Flags.train_dir_log + '/init')

        saver = tf.train.Saver(max_to_keep=Flags.how_many_training_steps)

        with sess.as_default():

            sess.run(tf.global_variables_initializer())

            tf.tables_initializer().run()

            cnn_model.load_weights(Flags.train_dir_log + '/init')

            train_writer = tf.summary.FileWriter(Flags.train_dir, sess.graph)

            validation_writer = tf.summary.FileWriter(Flags.val_dir, sess.graph)

            for k in range(0, Flags.how_many_training_epochs):
                    print('-------------------------------------------------------')
                    sess.run(train_iterator)
                    steps = 0

                    scores = np.array([])
                    scores_cont = np.array([])
                    scores_simi = np.array([])
                    scores_distil = np.array([])

                    error = np.array([])
                    true_label = np.array([])

                    feat = np.empty([0, int(centers.shape[1])])

                    with slim.queues.QueueRunners(sess):
                        try:
                            with tqdm(total=np.sum(counts)) as pbar:
                                while True:
                                    _, final_loss, gt, err, score, features, score_cont, score_simi, score_distil = sess.run(
                                                                                            [train_op, total_loss, true, pred, class_loss, centers, cont_loss, sim_loss, 
                                                                                            distil_loss],feed_dict={is_training: True, K.backend.learning_phase(): 1})

                                    scores = np.append(scores, score)
                                    scores_cont = np.append(scores_cont, score_cont)
                                    scores_simi = np.append(scores_simi, score_simi)
                                    scores_distil = np.append(scores_distil, score_distil)

                                    error = np.append(error, err)

                                    true_label = np.append(true_label, gt)

                                    feat = np.concatenate([feat, features], axis=0)


                                    pbar.update(Flags.train_batch_size)

                                    print('Epoch %s /%s Step %s /%s: Batch_loss is %f' % (
                                    k, Flags.how_many_training_steps - 1, steps,
                                    (Flags.train_dataset_size // Flags.train_batch_size), score))

                                    steps += 1

                        except tf.errors.OutOfRangeError:
                            saver.save(sess, Flags.train_dir_log + '/model', global_step=k)
                            pass
                    
                    print('Finished Training. BACC %f and Accuracy %f' % (
                    metrics.balanced_accuracy_score(true_label, error),
                    metrics.accuracy_score(true_label, error)))

                    summary = tf.Summary(
                    value=[tf.Summary.Value(tag='losses/Class_Loss', simple_value=np.mean(scores))])

                    train_writer.add_summary(summary, k)

                    summary = tf.Summary(
                        value=[tf.Summary.Value(tag='losses/Cont_Loss', simple_value=np.mean(scores_cont))])

                    train_writer.add_summary(summary, k)

                    summary = tf.Summary(
                        value=[tf.Summary.Value(tag='losses/Simi_Loss', simple_value=np.mean(scores_simi))])

                    train_writer.add_summary(summary, k)

                    summary = tf.Summary(
                        value=[tf.Summary.Value(tag='losses/Distil_Loss', simple_value=np.mean(scores_distil))])

                    train_writer.add_summary(summary, k)

                    summary = tf.Summary(value=[tf.Summary.Value(tag='BACC/Train_BACC',
                                                                simple_value=metrics.balanced_accuracy_score(
                                                                    true_label, error))])

                    train_writer.add_summary(summary, k)

                    summary = tf.Summary(value=[tf.Summary.Value(tag='Accuracy/Train_ACC',
                                                                simple_value=metrics.accuracy_score(true_label,
                                                                                                    error))])
                    train_writer.add_summary(summary, k)


                    sess.run(val_iterator)

                    scores = np.array([])
                    error = np.array([])
                    true_label_val = np.array([])

                    with slim.queues.QueueRunners(sess):
                            try:
                                while True:
                                    val_loss,gt,err = sess.run([class_loss,true, pred],feed_dict={is_training: False,K.backend.learning_phase(): 0})
                                    
                                    scores = np.append(scores, val_loss)

                                    error = np.append(error, err)

                                    true_label_val = np.append(true_label_val, gt)
                            except tf.errors.OutOfRangeError:
                                pass
                    
                    summary = tf.Summary(
                    value=[tf.Summary.Value(tag='losses/Val_Loss', simple_value=np.mean(scores))])

                    validation_writer.add_summary(summary, k)

                    summary = tf.Summary(value=[tf.Summary.Value(tag='BACC/Val_BACC',
                                                             simple_value=metrics.balanced_accuracy_score(
                                                                 true_label_val, error))])

                    validation_writer.add_summary(summary, k)

                    summary = tf.Summary(value=[tf.Summary.Value(tag='Accuracy/Val_ACC',
                                                             simple_value=metrics.accuracy_score(true_label_val,
                                                                                                 error))])

                    validation_writer.add_summary(summary, k)

                    print('Finished validation. BACC %f and Accuracy %f' % (
                            metrics.balanced_accuracy_score(true_label_val, error),
                            metrics.accuracy_score(true_label_val, error)))


            tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
            tsne_results = tsne.fit_transform(feat)

            plt.close()
            fig, ax = plt.subplots(figsize=(16, 10))
            scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], c=true_label, cmap='Accent')
            ax.legend(*scatter.legend_elements(), title='Classes')
            #ax.set_xlim([-12, 12])
            #ax.set_ylim([-12, 12])
            plt.show()

            sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--tfrecord_train',
      type=str,
      default='data/Fold_1/Training/train_full_norm.tfrecords',
      help='Path to folders of train labeled images.'
  )
    parser.add_argument(
      '--tfrecord_val',
      type=str,
      default='data/Fold_1/Validation/val_full_norm.tfrecords',
      help='Path to folders of validation labeled images.'
    )
    parser.add_argument(
        '--train_dir',
        type=str,
        default='model/training',
        help='Place to save summaries and checkpoints.'
    )
    parser.add_argument(
        '--train_dir_log',
        type=str,
        default='model/checkpoints',
        help='Place to save temporary checkpoints.'
    )
    parser.add_argument(
        '--val_dir',
        type=str,
        default='model/validation',
        help='Place to save summaries and checkpoints.'
        
    )
    parser.add_argument(
        '--train_batch_size',
        type=int,
        default=10,
        help='Size of your batch.'
    )
    parser.add_argument(
        '--train_dataset_size',
        type=int,
        default=8500,
        help='Size of your training dataset.'
    )
    parser.add_argument(
        '--val_batch_size',
        type=int,
        default=40,
        help='Size of your batch.'
    )
    parser.add_argument(
        '--val_dataset_size',
        type=int,
        default=2500,
        help='Size of your validation dataset.'
    )
    parser.add_argument(
        '--how_many_training_epochs',
        type=int,
        default=61,
        help='How many epochs.'
    
    )
    parser.add_argument(
        '--network',
        type=str,
        default='densenet169',
        help='Pick the CNN backbone - densenet121,densenet169,resnet50,efficientb1,efficientb3,efficientb4,resnet50,or resnet101.'
    )
    parser.add_argument(
        '--rotate',
        type=bool,
        default=True,
        help='Apply rotation-based transformations on the augmented loss.'
    )
    parser.add_argument(
        '--zoom',
        type=bool,
        default=True,
        help='Apply zoom-based transformations on the augmented loss.'
    )
    parser.add_argument(
        '--distillation',
        type=bool,
        default=True,
        help='Whether to use distillation loss to regularize the model.'
    )
    parser.add_argument(
        '--Temp',
        type=float,
        default=5.0,
        help='Temperature value for distillation loss.'
    )
    parser.add_argument(
        '--contrastive',
        type=bool,
        default=True,
        help='Whether to use contrastive loss to regularize the model.'
    )
    parser.add_argument(
        '--LAMBDA_contrast',
        type=float,
        default=1.0,
        help='Weight of the contrastive loss.'
    )
    parser.add_argument(
        '--triplet',
        type=bool,
        default=False,
        help='Whether to use triplet loss to regularize the model.'
    )
    parser.add_argument(
        '--LAMBDA',
        type=float,
        default=1.0,
        help='Weight of the triplet loss.'
    )
    parser.add_argument(
        '--two_headed',
        type=bool,
        default=True,
        help='Whether to use a two-headed model. Set to true when contrastive and/or triplet loss are used.'
    )
    parser.add_argument(
        '--two_head_size',
        type=int,
        default=256,
        help='Size (number of units) of the second head.'
    )
    Flags, unparsed = parser.parse_known_args()
    tf.app.run(main=main)
