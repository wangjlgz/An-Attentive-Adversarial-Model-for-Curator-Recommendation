from keras.models import Model

from keras.layers import Dense, Activation, Embedding, Input, concatenate, multiply, add, Flatten, Lambda, dot, Dropout
import keras.layers as layers
from keras import initializers
from keras.models import load_model
from keras.optimizers import Adam
from keras import backend as K
from keras.regularizers import l2, l1
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
import numpy as np
from random import shuffle
import random
import sys
from operator import itemgetter
from scipy.special import expit
import load_data
import utility
import tensorflow as tf
import argparse
from keras import regularizers
from Attention import AttLayer

from keras.backend.tensorflow_backend import set_session


parser = argparse.ArgumentParser()

parser.add_argument('--data', type=str, default='goodreads', help='data set name (Goodreads/Spotify)')
parser.add_argument('--model_name', type=str, default='CuRe', help='model name (default: CuRe)')
parser.add_argument('--epoch_pre_curator', type=int, default=1, help='number of epochs for pretraining the curator preference autoencoder')
parser.add_argument('--epoch_pre_item', type=int, default=1, help='number of epochs for pretraining the item preference autoencoder')
parser.add_argument('--iter', type=int, default=30, help='number of iterations')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=256, help='batch size for training (default: 256)')
parser.add_argument('--emsize', type=int, default=200, help='dimension of latent factors (default: 200)')
parser.add_argument('--neg_item', type=int, default=20, help='negative sampling rate for feedback on items')
parser.add_argument('--neg_curator', type=int, default=20, help='negative sampling rate for feedback on curators')
parser.add_argument('--alpha', type=int, default=0.5, help='weight of the side task')
parser.add_argument('--beta', type=int, default=1, help='weight of the ade')

args = parser.parse_args()


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# set the seed value
seed = args.seed
random.seed(seed)
np.random.seed(seed)

num_user = 0
num_item = 0
num_creator = 0

neg_sample_rate_ui = args.neg_item # negative sampling rate for feedback on curators
neg_sample_rate_uu = args.neg_curator # negative sampling rate for feedback on items

embedding_dimension = args.emsize


class AdversarialJointAutoencoder():
    def __init__(self):

        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, clipnorm=1.)


        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',optimizer=optimizer, metrics=['accuracy'])

        self.uu_encoder = self.build_uu_encoder()
        self.ui_encoder = self.build_ui_encoder()

        self.uu_encoder2 = self.build_uu_encoder()
        self.ui_encoder2 = self.build_ui_encoder()

        self.uu_att = self.build_att()


        user_input_index = Input(shape=(1,), dtype='int32', name = 'user_input_index')
        followee_input = Input(shape=(num_creator,),name='followee_input')
        item_input = Input(shape=(num_item,),name='item_input')

        fff = Dropout(0.5)(followee_input)
        iii = Dropout(0.3)(item_input)

        followee_input_mask = Input(shape=(num_creator,),name='followee_input_mask')
        item_input_mask = Input(shape=(num_item,),name='item_input_mask')


        user_latent = Flatten()(Embedding(input_dim = num_user+num_creator+1, output_dim = embedding_dimension, name = 'user_embedding',input_length=1)(user_input_index))
        

        encoded_uu0 = self.uu_encoder(fff)
        encoded_ui0 = self.ui_encoder(iii)

        encoded_uu2 = self.uu_encoder2(fff)
        
        encoded = self.uu_att([user_latent, encoded_uu0, encoded_ui0])

        self.uu_decoder = self.build_uu_decoder()
        self.ui_decoder = self.build_ui_decoder()

        decoded_uu = self.uu_decoder(concatenate([encoded_uu2,encoded]))
        decoded_ui = self.ui_decoder(encoded)
        

        mask_decoded_uu = multiply([decoded_uu, followee_input_mask])
        mask_decoded_ui = multiply([decoded_ui, item_input_mask])
        # For the adversarial_autoencoder model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator determines validity of the encoding
        validity0 = self.discriminator(encoded_uu0)
        validity1 = self.discriminator(encoded_ui0)
        

        # The adversarial_autoencoder model  (stacked generator and discriminator)
        self.adversarial_autoencoder = Model([user_input_index, followee_input,item_input,followee_input_mask,item_input_mask], [mask_decoded_uu, mask_decoded_ui, validity0, validity1])

        self.adversarial_autoencoder.compile(loss=[custom_loss, custom_loss, 'binary_crossentropy', 'binary_crossentropy'],
            loss_weights=[1,w1,-w2,-w2],
            optimizer=optimizer)

    def build_uu_encoder(self):
        reg = 0
        #---------------------input layer------------------------------
        followee_input = Input(shape=(num_creator,),name='followee_input')
        #---------------------embedding layer--------------------------
        followee_encoded = Dense(embedding_dimension, activation='sigmoid',name='en_uu',kernel_regularizer=regularizers.l2(reg))(followee_input)#0.000001
        followee_encoded = Dropout(0.3)(followee_encoded)

        model = Model(inputs=[followee_input], output = followee_encoded)

        model_uu = build_model_uu()

        model.get_layer('en_uu').set_weights(model_uu.get_layer('en_uu').get_weights())

        return model

    def build_ui_encoder(self):
    
        #---------------------input layer------------------------------
        item_input = Input(shape=(num_item,),name='item_input')
        #---------------------embedding layer--------------------------
        item_encoded = Dense(embedding_dimension, activation='sigmoid',name='en_ui')(item_input)
        item_encoded = Dropout(0.3)(item_encoded)

        model = Model(inputs=[item_input], output = item_encoded)

        model_ui = build_model_ui()

        model.get_layer('en_ui').set_weights(model_ui.get_layer('en_ui').get_weights())

        return model


    def build_discriminator(self):

        input_latent = Input(shape=(embedding_dimension,),name='input_latent')
        
        d1 = Dense(int(embedding_dimension/2), input_dim=embedding_dimension, activation="sigmoid")(input_latent)
        d2 = Dense(int(embedding_dimension/2), activation="sigmoid")(d1)
        pre = Dense(1, activation="sigmoid")(d2)

        return Model(input_latent, pre)


    def build_uu_decoder(self):
        reg = 0
        #---------------------input layer------------------------------
        followee_input = Input(shape=(2*embedding_dimension,),name='followee_input_latent')
        #---------------------embedding layer--------------------------
        followee_decoded = Dense(num_creator, activation='sigmoid',name='dn_uu',kernel_regularizer=regularizers.l2(reg))(followee_input)#0.000001
        
        model = Model(inputs=[followee_input], output = followee_decoded)

        return model

    def build_ui_decoder(self):
        reg = 0
        #---------------------input layer------------------------------
        item_input = Input(shape=(embedding_dimension,),name='item_input_latent')
        #---------------------embedding layer--------------------------
        item_decoded = Dense(num_item, activation='sigmoid',name='en_ui',kernel_regularizer=regularizers.l2(reg))(item_input)

        model = Model(inputs=[item_input], output = item_decoded)

        return model

    def build_att(self):
        reg = 0

        index_latent = Input(shape=(embedding_dimension,),name='index_latent')
        uu_latent = Input(shape=(embedding_dimension,),name='uu_latent')
        ui_latent = Input(shape=(embedding_dimension,),name='ui_latent')

        followee_encoded_ui = concatenate([uu_latent, index_latent])
        item_encoded_ui = concatenate([ui_latent, index_latent])
        
        followee_encoded_ui = Lambda(lambda x: K.expand_dims(x,-2))(followee_encoded_ui)
        item_encoded_ui = Lambda(lambda x: K.expand_dims(x,-2))(item_encoded_ui)

        pre_stack = concatenate([followee_encoded_ui, item_encoded_ui],axis=1)

        encoded_att = AttLayer(50)(pre_stack)

        return Model(inputs=[index_latent, uu_latent, ui_latent], output = encoded_att)


    def train_dis(self, epochs, batch_size=128, sample_interval=50):


        num_train_samples = total_num_user
        # Adversarial ground truths
        label1 = np.ones((batch_size, 1))
        label0 = np.zeros((batch_size, 1))

        dl = []
        da = []
        gl = []
        gmse = []
        gl2 = []
        gl3 = []
        gl4 = []

        epochs = int(num_train_samples/batch_size)+1

        for epoch in range(epochs):
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            # Select a random batch of users
            batch_ids = np.random.randint(0, num_train_samples, batch_size)

            train_b_ui = np.zeros((len(batch_ids),num_item))
            train_b_uu = np.zeros((len(batch_ids),num_creator))

            train_b_u = np.zeros((len(batch_ids),1))


            for j in range(len(batch_ids)):
                idx_b = batch_ids[j]
                obsItem = train_R_ui[idx_b]
                train_b_ui[j][obsItem] =  1.0

                obsCreator = train_R_uu[idx_b]
                train_b_uu[j][obsCreator] =  1.0

                train_b_u[j] = idx_b


            latent_uu = self.uu_encoder.predict(train_b_uu)
            latent_ui = self.ui_encoder.predict(train_b_ui)
            
            tr = np.concatenate((latent_uu, latent_ui), axis=0)
            la = np.concatenate((label0, label1), axis=0)
            # Train the discriminator
            d_loss = self.discriminator.train_on_batch(tr,la)
            
            
            dl.append(d_loss[0])
            da.append(100*d_loss[1])


        print ("%d [D loss: %f, acc: %.2f%%]]" % (epoch, meanL(dl), meanL(da)))

    def train(self, epochs, batch_size=128, sample_interval=50):


        num_train_samples = total_num_user
        # Adversarial ground truths
        label1 = np.ones((batch_size, 1))
        label0 = np.zeros((batch_size, 1))

        dl = []
        da = []
        gl = []
        gmse = []
        gl2 = []
        gl3 = []
        gl4 = []

        for epoch in range(epochs):
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            # Select a random batch of users
            batch_ids = np.random.randint(0, num_train_samples, batch_size)

            train_b_ui = np.zeros((len(batch_ids),num_item))
            train_b_uu = np.zeros((len(batch_ids),num_creator))

            train_b_u = np.zeros((len(batch_ids),1))


            for j in range(len(batch_ids)):
                idx_b = batch_ids[j]
                obsItem = train_R_ui[idx_b]
                train_b_ui[j][obsItem] =  1.0

                obsCreator = train_R_uu[idx_b]
                train_b_uu[j][obsCreator] =  1.0

                train_b_u[j] = idx_b
           
            mask_b_ui = utility.neg_sampling(train_b_ui, range(len(batch_ids)), neg_sample_rate_ui)
            mask_b_uu = utility.neg_sampling(train_b_uu, range(len(batch_ids)), neg_sample_rate_uu)


            latent_uu = self.uu_encoder.predict(train_b_uu)
            latent_ui = self.ui_encoder.predict(train_b_ui)
            
            tr = np.concatenate((latent_uu, latent_ui), axis=0)
            la = np.concatenate((label0, label1), axis=0)
            # Train the discriminator
            d_loss = self.discriminator.train_on_batch(tr,la)
            
            # ---------------------
            #  Train Generator
            # ---------------------
            # Train the generator
            g_loss = self.adversarial_autoencoder.train_on_batch([train_b_u, train_b_uu, train_b_ui, mask_b_uu, mask_b_ui], [train_b_uu, train_b_ui,label0, label1])

            # Plot the progress
            
            dl.append(d_loss[0])
            da.append(100*d_loss[1])
            gl.append(g_loss[0])
            gmse.append(g_loss[1])
            gl2.append(g_loss[2])
            gl3.append(g_loss[3])
            gl4.append(g_loss[4])

            if epoch%(sample_interval)==0:
                print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse1: %f,mse2: %f, d loss1: %f, d loss2: %f]" % (epoch, meanL(dl), meanL(da), meanL(gl), meanL(gmse), meanL(gl2), meanL(gl3), meanL(gl4)))
                dl = []
                da = []
                gl = []
                gmse = []
                gl2 = []
                gl3 = []
                gl4 = []

            if epoch%(sample_interval)==0 and epoch>0:
                self.test_model_all(batch_size, epoch)

            if epoch==epochs-1:
                json_string = self.adversarial_autoencoder.to_json()
                outfile = open(args.model_name+'.json','w')
                outfile.write(json_string)
                self.adversarial_autoencoder.save_weights(args.model_name+'.h5')

                self.test_model_all(batch_size, epoch)

    # model for making prediction        
    def build_pre_model(self):

        reg = 0

        user_input_index = Input(shape=(1,), dtype='int32', name = 'user_input_index')
        followee_input = Input(shape=(num_creator,),name='followee_input')
        item_input = Input(shape=(num_item,),name='item_input')

        user_latent = Flatten()(Embedding(input_dim = num_user+num_creator+1, output_dim = embedding_dimension, name = 'user_embedding',input_length=1)(user_input_index))

        encoded_uu0 = self.uu_encoder(followee_input)
        encoded_ui0 = self.ui_encoder(item_input)

        encoded_uu2 = self.uu_encoder2(followee_input)
        encoded_ui2 = self.ui_encoder2(item_input)

        
        encoded = self.uu_att([user_latent, encoded_uu0, encoded_ui0])

        decoded_uu = self.uu_decoder(concatenate([encoded_uu2,encoded]))
        decoded_ui = self.ui_decoder(encoded)


        output_pre = concatenate([decoded_uu, decoded_ui])
        
        mo = Model([user_input_index, followee_input,item_input], output_pre)
        mo.get_layer('user_embedding').set_weights(self.adversarial_autoencoder.get_layer('user_embedding').get_weights())

        return mo

    def test_model_all(self, batch_size, itr='r'):  # calculate the cost and rmse of testing set in each epoch
    
        num_train_samples = num_user
        index_array = [i for i in range(num_train_samples)]
        batches = _make_batches(num_train_samples, batch_size)

        pre_model = self.build_pre_model()

        for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = index_array[batch_start:batch_end]


                train_b_ui = np.zeros((len(batch_ids),num_item))
                train_b_uu = np.zeros((len(batch_ids),num_creator))
                vali_b_uu = np.zeros((len(batch_ids),num_creator))

                train_b_u = np.zeros((len(batch_ids),1))

                for j in range(len(batch_ids)):
                    idx_b = batch_ids[j]
                    obsItem = train_R_ui[idx_b]
                    train_b_ui[j][obsItem] =  1.0

                    obsCreator = train_R_uu[idx_b]
                    train_b_uu[j][obsCreator] =  1.0
                    vali_b_uu[j][vali_R_uu[idx_b]] =  1.0

                    train_b_u[j] = idx_b

                
                out_all = pre_model.predict([train_b_u, train_b_uu, train_b_ui], batch_size)

                Decodert_uu = out_all[:,:num_creator]
                Decodert_ui = out_all[:,num_creator:]

                if batch_index==0:
                    [precision_uu, recall_uu, f_score_uu, count_uu] = utility.test_model_batch(Decodert_uu, vali_b_uu, train_b_uu)
                else:
                    [precisiont_uu, recallt_uu, f_scoret_uu, countt_uu] = utility.test_model_batch(Decodert_uu, vali_b_uu, train_b_uu)
                    precision_uu += precisiont_uu
                    recall_uu += recallt_uu
                    f_score_uu += f_scoret_uu
                    count_uu += countt_uu

        
        print("user-user", count_uu)
        [precision_uu, recall_uu, f_score_uu, NDCG_uu] = utility.test_model_agg(precision_uu, recall_uu, f_score_uu, count_uu)

        
        return precision_uu, recall_uu, f_score_uu, NDCG_uu

def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.reduce_sum(K.square(y_pred - y_true), axis=1))

def custom_loss2(y_true, y_pred):
    return K.sum(K.square(y_pred - y_true), axis=-1)


def meanL(l):
    return sum(l)*1.0/len(l)

def _make_batches(size, batch_size):
    """Returns a list of batch indices (tuples of indices).
    # Arguments
        size: Integer, total size of the data to slice into batches.
        batch_size: Integer, batch size.
    # Returns
        A list of tuples of array indices.
    """
    num_batches = int(np.ceil(size / float(batch_size)))
    return [(i * batch_size, min(size, (i + 1) * batch_size))
            for i in range(0, num_batches)] 

def build_model_uu():
       
    #---------------------input layer------------------------------
    followee_input = Input(shape=(num_creator,),name='followee_input')
    followee_input_mask = Input(shape=(num_creator,),name='followee_input_mask')
    #---------------------embedding layer--------------------------

    followee_encoded = Dense(embedding_dimension, activation='sigmoid',name='en_uu',kernel_regularizer=regularizers.l2(0.000001))(followee_input)

    followee_decoded = Dense(num_creator, activation='sigmoid',name='de_uu', kernel_regularizer=regularizers.l2(0.000001))(followee_encoded)


    followee_output = multiply([followee_input_mask,followee_decoded])

    
    model = Model(inputs=[followee_input, followee_input_mask], output = followee_output)

    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(optimizer=opt,loss=custom_loss2)

    num_batch_train = int((total_num_user) / float(args.batch_size)) + 1
    for e in range(args.epoch_pre_curator):
        print('Epoch '+str(e)+ ' for pretraining curator preference curator')
        model.fit_generator(train_generator_uu(args.batch_size), 
                  steps_per_epoch = num_batch_train, 
                  epochs = 1,
                  workers = 1,
                  callbacks=[EarlyStopping(monitor='loss', min_delta=0, patience=3, verbose=0, mode='auto'), \
                  ReduceLROnPlateau(monitor='loss', factor=0.2, patience=1, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)]
                  )
    return model



def build_model_ui():
    
    #---------------------input layer------------------------------
    item_input = Input(shape=(num_item,),name='item_input')
    item_input_mask = Input(shape=(num_item,),name='item_input_mask')
    #---------------------embedding layer--------------------------
    item_encoded = Dense(embedding_dimension, activation='sigmoid',name='en_ui')(item_input)

    item_decoded = Dense(num_item, activation='sigmoid',name='de_ui')(item_encoded) #, kernel_regularizer=regularizers.l2(0.0000001)

    item_output = multiply([item_input_mask,item_decoded])
    
    model = Model(inputs=[item_input, item_input_mask], output = item_output)

    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(optimizer=opt,loss=custom_loss2)

    num_batch_train = int((total_num_user) / float(args.batch_size)) + 1
    for e in range(args.epoch_pre_item):
        print('Epoch '+str(e)+ ' for pretraining item preference curator')
        model.fit_generator(train_generator_ui(args.batch_size), 
                  steps_per_epoch = num_batch_train, 
                  epochs = 1,
                  workers = 1,
                  callbacks=[EarlyStopping(monitor='loss', min_delta=0, patience=3, verbose=0, mode='auto'), \
                  ReduceLROnPlateau(monitor='loss', factor=0.2, patience=1, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)]
                  )

    return model

def train_generator_uu(batch_size):
    num_train_samples = total_num_user
    index_array = [i for i in range(num_train_samples)]
    batches = _make_batches(num_train_samples, batch_size)
    while 1:
        np.random.shuffle(index_array)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]

            train_b_uu = np.zeros((len(batch_ids),num_creator))

            for j in range(len(batch_ids)):
                idx_b = batch_ids[j]

                obsCreator = train_R_uu[idx_b]
                train_b_uu[j][obsCreator] =  1.0

           
            mask_b_uu = utility.neg_sampling(train_b_uu, range(len(batch_ids)), neg_sample_rate_uu)

            batch_train_x = [train_b_uu,  mask_b_uu]
            batch_train_y = train_b_uu
            yield (batch_train_x, batch_train_y)


def train_generator_ui(batch_size):
    num_train_samples = total_num_user
    index_array = [i for i in range(num_train_samples)]
    batches = _make_batches(num_train_samples, batch_size)
    while 1:
        np.random.shuffle(index_array)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            train_b_ui = np.zeros((len(batch_ids),num_item))

            for j in range(len(batch_ids)):
                idx_b = batch_ids[j]
                obsItem = train_R_ui[idx_b]
                train_b_ui[j][obsItem] =  1.0

            mask_b_ui = utility.neg_sampling(train_b_ui, range(len(batch_ids)), neg_sample_rate_ui)

            batch_train_x = [train_b_ui, mask_b_ui]
            batch_train_y = train_b_ui
            yield (batch_train_x, batch_train_y)


if __name__ == '__main__':
    w1 = args.alpha
    w2 = args.beta

    train_R_uu, vali_R_uu, train_R_ui, num_user, num_item, num_creator = load_data.load_data_unified(args.data)
    total_num_user = num_user + num_creator

    print('Total Number of Users:', total_num_user)
    print('Number of Users:', num_user)
    print('Number of Curators:', num_creator)
    print('Number of Items:', num_item)

    sample_interval = int((total_num_user)/float(args.batch_size))

    aae = AdversarialJointAutoencoder()

    aae.train(epochs=args.iter*sample_interval, batch_size=args.batch_size, sample_interval=sample_interval)
