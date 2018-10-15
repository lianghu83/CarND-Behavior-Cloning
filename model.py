import os
import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Cropping2D, Lambda
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# Read driving log csv file
samples = []
with open('my_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# Split data
samples = samples[1:]
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Define generator
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                name = 'my_data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
    				 # image augmentation
                center_image_flipped = cv2.flip(center_image, 1)
                center_angle_flipped = center_angle*(-1)
                images.append(center_image_flipped)
                angles.append(center_angle_flipped)
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# Nvidia's network with dropout
model = Sequential()
model.add(Lambda(lambda x: x/255.0-0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
#model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# Train network
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, 
                    samples_per_epoch=len(train_samples),
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples),
                    nb_epoch=2)
"""
# Print the keys contained in the history object
print(history_object.history.keys())

# Plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
"""
# Save model
model.save('model.h5')


























