from utils import *
from model import load_model

import cv2
import numpy as np
import pandas as pd
 
batch_size = 32

data_frame = pd.read_csv('data/driving_log.csv', usecols=[0, 1, 2, 3])
data_frame = data_frame.sample(frac=1).reset_index(drop=True)

num_rows_training = int(data_frame.shape[0]*0.9)

training_data = data_frame.loc[0:num_rows_training-1]
validation_data = data_frame.loc[num_rows_training:]

training_generator = get_data_generator(training_data, batch_size=batch_size)
validation_data_generator = get_data_generator(validation_data, batch_size=batch_size)

model = load_model()

samples_per_epoch = (20000//batch_size)*batch_size

model.fit_generator(training_generator, validation_data=validation_data_generator,
                    samples_per_epoch=samples_per_epoch, nb_epoch=3, nb_val_samples=3000)

print("Saving model weights and configuration file.")

model.save_weights('./model.h5')  # always save your weights after training or during training
with open('./model.json', 'w') as outfile:
    outfile.write(model.to_json())