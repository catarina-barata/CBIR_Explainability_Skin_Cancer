from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.keras as K
tf.disable_v2_behavior()

import numpy as np

import data_import as da
import contrastive_loss
import triplet_hard
import triplet_semi


def apply_network_img(network,image_size):
    if network == 'densenet121':
        base_model = K.applications.DenseNet121(include_top=False,
                                                weights="imagenet",
                                                pooling=None)

        cnn_input = K.Input(shape=(image_size, image_size, 3), name='img_cnn')
        x = K.applications.densenet.preprocess_input(cnn_input)
    
    elif network == 'densenet169':
        base_model = K.applications.DenseNet169(include_top=False,
                                                weights="imagenet",
                                                pooling=None)

        cnn_input = K.Input(shape=(image_size, image_size, 3), name='img_cnn')
        x = K.applications.densenet.preprocess_input(cnn_input)

    elif network == 'efficientb1':
        base_model = K.applications.EfficientNetB1(include_top=False,
                                          weights="imagenet",
                                          pooling=None)

        cnn_input = K.Input(shape=(image_size, image_size, 3), name='img_cnn')
        x = K.applications.efficientnet.preprocess_input(cnn_input)
    
    elif network == 'efficientb3':
        base_model = K.applications.EfficientNetB3(include_top=False,
                                          weights="imagenet",
                                          pooling=None)

        cnn_input = K.Input(shape=(image_size, image_size, 3), name='img_cnn')
        x = K.applications.efficientnet.preprocess_input(cnn_input)
    
    elif network == 'efficientb4':
        base_model = K.applications.EfficientNetB4(include_top=False,
                                          weights="imagenet",
                                          pooling=None)

        cnn_input = K.Input(shape=(image_size, image_size, 3), name='img_cnn')
        x = K.applications.efficientnet.preprocess_input(cnn_input)


    elif network == 'resnet50':
        base_model = K.applications.ResNet50V2(include_top=False,
                                               weights="imagenet",
                                               pooling=None)

        cnn_input = K.Input(shape=(image_size, image_size, 3), name='img_cnn')
        x = K.applications.resnet_v2.preprocess_input(cnn_input)
    
    elif network == 'resnet101':
        base_model = K.applications.ResNet50V2(include_top=False,
                                               weights="imagenet",
                                               pooling=None)

        cnn_input = K.Input(shape=(image_size, image_size, 3), name='img_cnn')
        x = K.applications.resnet_v2.preprocess_input(cnn_input)

    x = base_model(x, training=False)

    cnn_output = K.layers.GlobalAveragePooling2D(name='GlobalAvg')(x)

    return K.Model(inputs=cnn_input, outputs=cnn_output, name='CNN')

def apply_model(train_size, data, captions, n_classes, wordtoix,
                             image_size,rotate,zoom, network,
                             distillation, Temp,contrastive,LAMBDA_contrast,
                             triplet,LAMBDA,
                             two_headed,two_head_size,is_training):

    data = tf.cond(is_training, lambda: da.transform_data(data, True, dim=image_size), 
                                lambda: da.transform_data(data, False, dim=image_size))

    batch_size = tf.shape(data)[0]

    def self_aug(imgs):

        data_mod = tf.image.random_flip_left_right(data)

        data_mod = tf.image.random_flip_up_down(data_mod)

        if rotate:
            rot = np.random.randint(1,4)

            data_mod = tf.image.rot90(data_mod,rot)

        if zoom:
            crop_range = np.random.choice(np.linspace(0.8,1,20))

            data_mod = tf.image.central_crop(data_mod,crop_range)

            data_mod = tf.image.resize_image_with_pad(data_mod,image_size,image_size,method = tf.image.ResizeMethod.AREA)

        data_net = tf.concat([data, data_mod], axis=0)

        return data_net

    data = tf.cond(is_training, lambda: self_aug(data), lambda: data)

    CNN = apply_network_img(network)

    CNN.summary()
    
    emb_input = K.Input(shape=(image_size, image_size, 3),name = 'img',tensor = data)

    feat = CNN(emb_input,training = False)

    if two_headed:
        image_embedding = K.layers.Dropout(0.2)(feat)

        image_embedding = K.layers.Dense(two_head_size, activation='relu',name='embedding')(image_embedding)
    else:
        image_embedding = feat

    feat_drop = K.layers.Dropout(0.5)(feat)

    emb = K.layers.Dense(n_classes, activation=None,name='class')(feat_drop)

    model = K.Model(inputs=emb_input,outputs = [emb,image_embedding], name = 'network')

    model.summary()

    logit,img_feat = model(data)
   
    current_caption_ind = wordtoix.lookup(captions)
    
    labels = tf.squeeze(tf.slice(current_caption_ind, [0, 1], [batch_size, 1]), 1)

    onehot = tf.one_hot(labels, n_classes)

    
    def representation (features,logit,labels,training):

        if distillation and training:

            logit, logit_aug = tf.split(logit, 2, 0)

            logit_true = tf.nn.softmax(logit/Temp)
            logit_aug = tf.nn.softmax(logit_aug/Temp)

            logit_true += 1.0e-9
            logit_aug += 1.0e-9

            logit_true = tf.clip_by_value(logit_true, 1e-9, 1.0, name=None)
            logit_aug = tf.clip_by_value(logit_aug, 1e-9, 1.0, name=None)

            logit_true = tf.distributions.Categorical(probs=logit_true)
            logit_aug = tf.distributions.Categorical(probs=logit_aug)

            distillation_loss = tf.distributions.kl_divergence(logit_true,logit_aug,allow_nan_stats= False)

        else:

            distillation_loss = tf.zeros([1,])


        if triplet is None or not training:
            metric_loss = tf.zeros([1,])

        elif triplet == 'triplet_semi' and training:
            image_embedding_similarity = tf.nn.l2_normalize(features,
                                                            axis=1, epsilon=1e-10)        
            labels_triplet = tf.concat([labels,labels],0)
            labels_triplet= tf.ensure_shape(labels_triplet, (None, ))
        

            metric_loss,_ = triplet_semi.triplet_semihard_loss(labels_triplet,image_embedding_similarity, margin=0.2)
            

            metric_loss = tf.cond(tf.is_nan(tf.reduce_mean(metric_loss)),lambda:tf.zeros([1,]),lambda:tf.reduce_mean(metric_loss))

            metric_loss = tf.cond(tf.is_inf(tf.reduce_mean(metric_loss)),lambda:tf.zeros([1,]),lambda:tf.reduce_mean(metric_loss))

        elif triplet == 'triplet_hard' and training:
            image_embedding_similarity = tf.nn.l2_normalize(features,
                                                            axis=1, epsilon=1e-10)
            metric_loss = tf.reduce_mean(triplet_hard.batch_hard(labels, image_embedding_similarity, margin='soft'))

        
        if contrastive and training:
            instance_loss,_,_ = contrastive_loss.add_contrastive_loss(features,
                            hidden_norm=True,
                            temperature=0.07,
                            weights=1.0)

            instance_loss = tf.cond(tf.is_nan(tf.reduce_mean(instance_loss)),lambda:tf.zeros([1,]),lambda:tf.reduce_mean(instance_loss))

            instance_loss = tf.cond(tf.is_inf(tf.reduce_mean(instance_loss)),lambda:tf.zeros([1,]),lambda:tf.reduce_mean(instance_loss))
        else:
            instance_loss = tf.zeros([1, ])
    
        return distillation_loss, metric_loss, instance_loss

    def split_logits(logit):
        logit, logit_aug = tf.split(logit,2,0)
        return logit
    
    distillation_loss, metric_loss, instance_loss = tf.cond(is_training, lambda: representation (img_feat,logit,labels,True),
                                                                         lambda: representation (img_feat,logit,labels,False))
    
    logit = tf.cond(is_training, lambda: split_logits(logit),
                                 lambda: logit)

    gt = tf.argmax(onehot, 1)

    pred = tf.argmax(logit, 1)


    total_loss = tf.losses.softmax_cross_entropy(onehot, logits=logit, reduction=tf.losses.Reduction.NONE)

  
    return tf.reduce_mean(total_loss) + LAMBDA * metric_loss + LAMBDA_contrast*instance_loss + (Temp**2)*tf.reduce_mean(distillation_loss),\
               tf.reduce_mean(total_loss), gt, pred, tf.slice(feat, [0, 0], [batch_size, int(feat.shape[1])]),\
               tf.reduce_mean(instance_loss), tf.reduce_mean(metric_loss), tf.reduce_mean(distillation_loss),model