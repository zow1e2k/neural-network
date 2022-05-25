import numpy
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras.datasets import cifar10
from PIL import Image
from glob import glob
from skimage import transform


class Network:
    def analyze_image(self, filepath):
        img = tf.keras.utils.load_img(
            filepath, target_size=(600, 400)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        predictions = self.main_model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(self.class_names[numpy.argmax(score)], 100 * numpy.max(score))
        )
        return self.class_names[numpy.argmax(score)]

    def learn_model(self):
        #seed = 21
        #numpy.random.seed(seed)

        train_ds = tf.keras.utils.image_dataset_from_directory(
            'Z:\\Projects\\Neuron\\resources\\img\\cars_train',
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(600, 400),
            batch_size=32
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            'Z:\\Projects\\Neuron\\resources\\img\\cars_test',
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(600, 400),
            batch_size=32
        )

        class_names = train_ds.class_names
        #print(class_names)

        self.class_names = class_names

        #(x_train, y_train), (x_test, y_test) = tfds.image_classification.Cars196.load_data()

        #x_train = x_train.astype('float32')
        #x_test = x_test.astype('float32')
        #x_train = x_train / 255.0
        #x_test = x_test / 255.0

        # y_train = np_utils.to_categorical(y_train)
        # y_test = np_utils.to_categorical(y_test)
        # num_classes = y_test.shape[1]

        num_classes = 6

        model = tf.keras.Sequential([
            tf.keras.layers.Rescaling(1./255, input_shape=(600, 400, 3)),

            #tf.keras.layers.Conv2D(6, (3, 3), activation='relu'),
            tf.keras.layers.Conv2D(6, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            #tf.keras.layers.Dropout(0.2),
            #tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            #tf.keras.layers.Dropout(0.2),
            #tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            #tf.keras.layers.Dropout(0.2),
            #tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Flatten(),
            #tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Dense(128, kernel_constraint=maxnorm(3), activation='relu'),
            #tf.keras.layers.Dropout(0.2),
            #tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Dense(num_classes, activation='softmax'),
        ])

        model.compile(
            optimizer='adam',
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        model.summary()

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='accuracy',
                min_delta=0.1,
                patience=2,
                verbose=1
            )
        ]

        model.fit(
            train_ds,
            validation_data=val_ds,
            #batch_size=64,
            epochs=25,
            callbacks = callbacks
        )

        # model = Sequential()

        #model.add(Conv2D(32, (3, 3), input_shape=x_train.shape[1:], padding='same'))
        #model.add(Activation('relu'))
        #model.add(Dropout(0.2))
        #model.add(BatchNormalization())

        # model.add(Conv2D(64, (3, 3), padding='same'))
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.2))
        # model.add(BatchNormalization())

        # model.add(Conv2D(64, (3, 3), padding='same'))
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.2))
        # model.add(BatchNormalization())

        # model.add(Conv2D(128, (3, 3), padding='same'))
        # model.add(Activation('relu'))
        # model.add(Dropout(0.2))
        # model.add(BatchNormalization())

        # model.add(Flatten())
        # model.add(Dropout(0.2))

        # model.add(Dense(256, kernel_constraint=maxnorm(3)))
        # model.add(Activation('relu'))
        # model.add(Dropout(0.2))
        # model.add(BatchNormalization())
        # model.add(Dense(128, kernel_constraint=maxnorm(3)))
        # model.add(Activation('relu'))
        # model.add(Dropout(0.2))
        # model.add(BatchNormalization())
        # model.add(Dense(num_classes))
        # model.add(Activation('softmax'))

        # epochs = 25
        # optimizer = 'Adam'

        # model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # print(model.summary())

        # model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=64)

        # scores = model.evaluate(x_test, y_test, verbose=0)
        # print("Accuracy: %.2f%%" % (scores[1] * 100))

        self.main_model = model

        self.save_model()

    def save_model(self):
        self.main_model.save(
            'Z:\\Projects\\Neuron\\resources\\save')

    def load_model(self):
        self.main_model = tf.keras.models.load_model(
            'Z:\\Projects\\Neuron\\resources\\load')
        self.class_names = self.main_model.class_names
        print('Сеть загружена')
