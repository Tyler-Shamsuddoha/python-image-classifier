
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
from imutils import paths
from tensorflow.keras.losses import categorical_crossentropy #cosine_proximity,
from tensorflow.keras.optimizers import Nadam, Adam
from tensorflow.keras.utils import plot_model
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


import cv2
import numpy as np
import tensorflow as tf
import random
import os
import matplotlib.pyplot as plt
import sklearn


# Set random seeds to ensure the reproducible results
SEED = 309
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

IMG_HEIGHT = 64
IMG_WIDTH = 64
INPUT_SHAPE = (64, 64)
# conv_base = InceptionResNetV2(weights = 'imagenet', include_top = False, input_shape = (IMG_HEIGHT,IMG_WIDTH,3))

train_dir = './Train_data'
test_dir = './test_data'
validation_images_dir = './Validation_Sets'
batch_size = 32
epoch = 100

def construct_model():
    """
    Construct the CNN model.
    ***
        Please add your model implementation here, and don't forget compile the model
        E.g., model.compile(loss='categorical_crossentropy',
                            optimizer='sgd',
                            metrics=['accuracy'])
        NOTE, You must include 'accuracy' in as one of your metrics, which will be used for marking later.
    ***
    :return: model: the initial CNN model
    """

    model = Sequential([
        Conv2D(16, kernel_size=3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D(2, 2),

        Conv2D(32, kernel_size=3, padding='same', activation='relu'),
        MaxPooling2D(2, 2),

        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(2, 2),

        Conv2D(128, 3, padding='same', activation='relu'),
        Flatten(),
        Dense(64, 'relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')
    ])

    model.compile(loss=categorical_crossentropy, optimizer= Nadam(), metrics=['accuracy'])

    return model

    # *************************************
    # Building Model Using Transfer Learning
    # model = Sequential([
    #     conv_base,
    #     Flatten(),
    #     Dense(512, 'relu'),
    #     Dense(3, activation='softmax')
    # ])
    # model.compile(loss=categorical_crossentropy, optimizer= Nadam(), metrics=['accuracy'])
    # return model


def train_model(model):
    """
    Train the CNN model
    ***
        Please add your training implementation here, including pre-processing and training
    ***
    :param model: the initial CNN model
    :return:model:   the trained CNN model
    """

    print("Loading Images...")

    validation_image_generator = ImageDataGenerator(rescale=1./255, rotation_range=40)
    val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                                  directory=validation_images_dir,
                                                                  target_size=INPUT_SHAPE,
                                                                  class_mode='categorical')
    print("Loaded Validation Set Images Successfully\n")

    train_image_generator = ImageDataGenerator(rescale=1./255, zoom_range=0.2, rotation_range=40)
    train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                               directory=train_dir,
                                                               shuffle=True,
                                                               target_size=INPUT_SHAPE,
                                                               class_mode='categorical')
    print("Loaded Training Images Successfully\n")
    print("Starting training....\n")
    model = construct_model()

    filepath = 'model/newmodel111.h5'
    model_checkpoint = ModelCheckpoint(filepath, monitor='val_loss', save_best_only= True, verbose=1, mode = 'min')
    # early_stop = EarlyStopping(filepath, monitor='val_acc', mode='max', patience=5)

    history = model.fit_generator(

        train_data_gen,
        steps_per_epoch= 3748/batch_size,
        epochs= epoch,
        validation_data= val_data_gen,
        validation_steps= 562/batch_size,
        callbacks = [model_checkpoint],
    )

    visualise_results(history)

    return model


def save_model(model):
    """
    Save the keras model for later evaluation
    :param model: the trained CNN model
    :return:
    """
    # ***
    #   Please remove the comment to enable model save.
    #   However, it will overwrite the baseline model we provided.
    # ***
    model.save("model/new_model.h5")
    print("Model Saved Successfully.")


def visualise_results(history):

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    model = train_model(train_dir)
    save_model(model)
