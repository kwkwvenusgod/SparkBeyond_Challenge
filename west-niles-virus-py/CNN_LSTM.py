
from keras.models import Sequential
from keras.layers import Dense, Merge, regularizers, LSTM, TimeDistributed, GlobalAveragePooling2D
from keras.layers.embeddings import Embedding
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import Adamax
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D


class CNN_LSTM:
    def __init__(self, kernel_size, input_shape):
        self._model = None
        model = Sequential()
        # define CNN model
        model.add(
            TimeDistributed(Conv2D(filters=64, kernel_size=kernel_size, padding='same', activation='relu'),
                            input_shape=input_shape))
        model.add(TimeDistributed(MaxPooling2D((1, 2))))
        model.add(TimeDistributed(GlobalAveragePooling2D()))
        # define LSTM model
        model.add(LSTM(units=100, dropout=0.3))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # model.fit(train_x, train_y, epochs=5, batch_size=32)
        self._model = model
        print model.summary()

    def fit_generator(self, generator, epochs, steps_per_epoch, class_weight=None):
        self._model.fit_generator(generator = generator, epochs=epochs, steps_per_epoch=steps_per_epoch, class_weight=class_weight)

    def predict_generator(self, generator, steps):
        y_pred = self._model.predict_generator(generator=generator, steps=steps)
        return y_pred

    def save(self, file_path):
        self._model.save(file_path)

    def load(self, file_path):
        self._model.load_weights(filepath=file_path, by_name=True)
