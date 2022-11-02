import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.utils import plot_model
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Reshape, Flatten, Input, Lambda

data = np.arange(10).reshape((1,-1))

class Min_Plus_Square(layers.Layer):
    def call(self, x):
        return keras.backend.min(x) + x * x
    
class Sin(layers.Layer):
    def call(self, x):
        return keras.backend.sin(x)

initializer = tf.keras.initializers.Identity()
model =  Sequential() 

model.add(Dense(units=20, input_dim=10, kernel_initializer=initializer, name='dense_75_input'))
model.add(Dropout(rate=0.3))
model.add(Min_Plus_Square())
model.add(Activation('relu'))
model.add(Reshape((2, 10)))
model.add(Flatten())
model.add(Dense(units=10, kernel_initializer=initializer))
model.add(Activation('relu'))
model.add(Sin())
model.add(Dropout(rate=0.1))
model.add(Dense(units=1, kernel_initializer=initializer))
model.add(Activation('sigmoid'))

plot_model(model,show_shapes=True,show_layer_names=True)
model.compile(loss='mean_squared_error', optimizer='sgd')
print(model.predict(data))

