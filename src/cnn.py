import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Flatten, Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
%matplotlib inline



train_path = "../data/train"
test_path = "../data/test"
valid_path = "../data/valid"


train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size = (224,224), classes = ['dr', 'nodr'], batch_size = 225)
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size = (224,224), classes = ['dr', 'nodr'], batch_size = 50)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size = (224,224), classes = ['dr', 'nodr'], batch_size = 100)

def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')


inps = keras.layers.Input(shape=(224, 224, 3), name='image_input')
m = keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet')(inps)
x = keras.layers.Flatten()(m)
x = keras.layers.Dense(1024, activation='relu')(x)
predictions = keras.layers.Dense(2, activation='softmax')(x)
model = keras.Model(inputs=inps, outputs=predictions)

model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

tensor_board = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

model.fit_generator(train_batches, steps_per_epoch = 10, validation_data = valid_batches, validation_steps = 2, epochs = 10, verbose = 2, callbacks=[tensor_board])

model.save("../models/" +"resnet_diabetic_retinopathy"+ ".h5")

predictions = model.predict_generator(test_batches, steps = 4, verbose = 2)

score = model.evaluate_generator(test_batches, steps=4, verbose = 0)

print('Test score:', score[0])
print('Test accuracy:', score[1])
