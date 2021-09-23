from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import Sequential
from ModelOptions import ModelOptions


class ModelCreator:
    @staticmethod
    def _model1(model: Sequential, input_shape: (int, int, int), drop_conv, drop_fc) -> Sequential:
        model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(drop_conv))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(drop_conv))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dropout(drop_fc))
        model.add(Dense(units=128, activation='relu'))
        model.add(Dense(units=1, activation='sigmoid'))
        return model

    def __init__(self):
        self._MODEL_CONFIGURATIONS: {} = {
            'model1': self._model1,
        }

    # create CNN model with specified drop_chance
    def create_model(self, options: ModelOptions, type_model: str = 'model1') -> keras.models.Sequential:
        assert type_model in self._MODEL_CONFIGURATIONS.keys(), 'Assertion error, not defined type model.'
        # create model
        model = keras.models.Sequential(name=options.name)
        # define model layers
        model = self._MODEL_CONFIGURATIONS[type_model](
            model=model,
            input_shape=options.input_shape,
            drop_conv=options.drop_conv,
            drop_fc=options.drop_fc,
        )
        # compile model
        model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.001, momentum=0.9),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model
