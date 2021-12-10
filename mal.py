from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
from keras.utils.vis_utils import plot_model
img_width = 64
img_height = 64
x=3
datagen = ImageDataGenerator(rescale=1/255.0, validation_split=0.2)


train_dataset=datagen.flow_from_directory('basedata/Training/',
                                        target_size=(img_width,img_height),
                                        class_mode = 'binary',
                                        batch_size = 32,
                                        subset = 'training')

validation_dataset=datagen.flow_from_directory('basedata/Training/',
                                        target_size=(img_width,img_height),
                                        class_mode = 'binary',
                                        batch_size = 32,
                                        subset = 'validation')

print(train_dataset.class_indices)

##model=tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(64,64,3)),
                                  #tf.keras.layers.MaxPool2D(2,2),
                                  #
                                  #tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
                                  #tf.keras.layers.MaxPool2D(2,2),
                                  #
                                  #tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
                                  #tf.keras.layers.MaxPool2D(2,2),
                                  ##
                                  #tf.keras.layers.Flatten(),
                                  #
                                 # tf.keras.layers.Dense(512,activation='relu'),
                                  #
                                  #tf.keras.layers.Dense(1,activation='sigmoid')
                                 # ])

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(16, (3,3), input_shape = (img_width, img_height, 3), activation='relu'))  #no and size of  filters relu-rectifier linear unit
model.add(tf.keras.layers.MaxPool2D(2,2))
model.add(tf.keras.layers.Dropout(0.1))

model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D(2,2))
model.add(tf.keras.layers.Dropout(0.23))

model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D(2,2))


model.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D(2,2))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Flatten())#  converts multidimensional data to vector 
model.add(tf.keras.layers.Dense(64, activation='relu')) #fully connected layer wher we pass the
model.add(tf.keras.layers.Dropout(0.4)) #droops 40%

model.add(tf.keras.layers.Dense(1, activation='sigmoid')) #sigmoid since binary


model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])
#model.compile(loss='binary_crossentropy',optimizer=RMSprop(lr=0.001),metrics=['accuracy'])    

#model_fit=model.fit(train_dataset,steps_per_epoch=4,epochs=30,validation_data=validation_dataset)      
history = model.fit_generator(generator=train_dataset,
                              steps_per_epoch = len(train_dataset),
                              epochs = x,
                              validation_data = validation_dataset,
                              validation_steps = len(validation_dataset))    
#plot_model(history, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
def plot_learningCurve(history, epoch):
  # Plot training & validation accuracy values
  epoch_range = range(1, epoch+1)
  plt.plot(epoch_range, history.history['accuracy'])
  plt.plot(epoch_range, history.history['val_accuracy'])
  plt.title('Model accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

  # Plot training & validation loss values
  plt.plot(epoch_range, history.history['loss'])
  plt.plot(epoch_range, history.history['val_loss'])
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()
plot_learningCurve(history, x)
model.save('Malaria2.h5')
