from utils.train_utils import *
from utils.Network import *
from utils.data_handling import *
from tensorflow.keras.callbacks import  ModelCheckpoint, CSVLogger, TensorBoard
from sklearn.model_selection import train_test_split
from keras.models import load_model
from os.path import join
import os
import time
import numpy as np
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop



class Trainer(object):

    def __init__(self, args):
        super().__init__()
        self.params = args
        self.model = None
        self.output_size = 45
        self.learning_rate = self.params.learning_rate
        self.batch_size = self.params.train_batch_size
        self.epochs = self.params.epochs
        self.split_ratio = self.params.split_ratio
        self.dropout_prob = self.params.dropout_prob
        self.model_save_dir = self.params.save_dir
        self.trained_model_dir = self.params.trained_model_dir
        if self.trained_model_dir is None:
            self.model_file = join(self.model_save_dir, 'ConvTract.hdf5')
        else: 
            self.model_file = self.trained_model_dir

        self.data_handler = DataHandler(self.params, mode='train')


    def set_model(self, grad_directions):
        self.model = network(grad_directions, self.output_size, self.dropout_prob)
        return

    def train(self):

        # Set data
        print('preprocessing...')
        t0 = time.time()
        data_handler = self.data_handler
        data_handler.resample_dwi()
        grad_directions = 100
        self.indices = get_indices(data_handler.dwi.shape)
        train_index, valid_index,_,_ = train_test_split(self.indices, np.arange(len(self.indices)), test_size=1-self.split_ratio)
        print(f'done\t{time.time()-t0}s\n')
        
        # Set model
        if os.path.exists(self.model_file):
            print('loading model...')
            print(self.model_file)
            self.model = load_model(self.model_file, compile = False)
        else:
            self.set_model(grad_directions)
        self.model.summary()
            
        print(self.model.input_shape, self.model.output_shape)
    
            
        # Set optimizer
        optimizer = Nadam(learning_rate=self.learning_rate)
        loss = keras.losses.MeanSquaredError()
        metric = keras.metrics.MeanAbsoluteError()
        
        self.model.compile(loss=loss, optimizer=optimizer, metrics = metric)
      
        callbacks = []
        callbacks.append(ModelCheckpoint(monitor='val_loss',
                            filepath=self.model_file,
                            save_best_only=False,
                            save_weights_only=False,
                            mode="auto",
                            save_freq=1000))
                             
        callbacks.append(TensorBoard(log_dir=join(self.model_save_dir, 'Tensorboard_logs'),
                             update_freq = 2000))
        
        callbacks.append(CSVLogger(join(self.model_save_dir, 'training.log'), append=True, separator=';'))
                
    
        train_history = self.model.fit(
                generator(train_index, data_handler, self.output_size, self.batch_size),
                epochs=self.epochs,
                verbose=1,
                callbacks=callbacks,
                steps_per_epoch=np.ceil(float(len(train_index))) / float(self.batch_size),
                validation_data=generator(valid_index, data_handler, self.output_size, self.batch_size),
                validation_steps=np.ceil(float(len(valid_index)) / float(self.batch_size))
        )
     
        return
    




