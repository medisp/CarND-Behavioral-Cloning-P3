import os
import csv

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
samples = samples[1:]
samples2=[]
with open('./addtl_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples2.append(line)	
samples.extend(samples2)
print(len(samples))
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
#limit dataset for training
#train_samples, validation_samples1 = train_test_split(samples2, test_size=0.3)
#train_samples, validation_samples1 = train_test_split(samples, test_size=0.2)
#train_samples, validation_samples = train_test_split(validation_samples1, test_size=0.2)
import cv2
import numpy as np
import sklearn
from random import randint

#def image_processing: cv2.cvtColor(cv2.imread(name),cv2.COLOR_BGR2RGB)
from image_utils import *

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
				#name = './addtl_data/IMG/'+batch_sample[0].split('/')[-1] #randint(0,2) to randomly sample left and right cameras	
                name = './data/IMG/'+batch_sample[0].split('/')[-1] #randint(0,2) to randomly sample left and right cameras
                center_image = cv2.cvtColor(cv2.imread(name),cv2.COLOR_BGR2RGB) #cv2.imread(name)#center_image = cv2.cvtColor(transform_image(cv2.imread(name)),cv2.COLOR_BGR2RGB) #cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 80, 320  # Trimmed image format

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D
model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
#model.add(Lambda(lambda x: x/127.5 - 1.,
#        input_shape=(ch, row, col),
 #       output_shape=(ch, row, col)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape = (160,320,3)))		
model.add(Cropping2D(cropping=((60,20),(0,0))))
model.add(Convolution2D(24, 6, 6, subsample=(2,2), border_mode='valid', activation='relu')) # output (100-6-0)/2 47x157x24
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), border_mode='valid', activation='relu')) #output    47-5/2   21x76x36
#model.add(MaxPooling2D())
model.add(Convolution2D(48, 5, 5, subsample=(2,2), border_mode='valid', activation='relu'))   # output 21-5/2   8x36x48
#model.add(Convolution2D(64, 3, 3, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64, 3, 3, subsample=(2,2), border_mode='valid', activation='relu')) # output  9 x 17 x 64
model.add(Convolution2D(128, 3, 3, subsample=(2,2), border_mode='valid', activation='relu')) # output  3x7x128
model.add(Flatten())
#model.add(Dropout(0.5))
model.add(Dense(200))
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
from keras.models import Model
import matplotlib.pyplot as plt

history_object = model.fit_generator(train_generator, samples_per_epoch = len(train_samples), validation_data = validation_generator, nb_val_samples = len(validation_samples), nb_epoch=10, verbose=1)

### print the keys contained in the history object
print(history_object.history.keys())
model.save('model10.h5')
### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
#model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=10)

			
model.save('model10.h5')