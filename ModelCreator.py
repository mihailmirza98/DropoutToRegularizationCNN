from tensorflow import keras
from ModelOptions import ModelOptions
from typing import Collection, Any
from collections import namedtuple


# creator of CNN models
class ModelCreator:
    # create CNN model with specified drop_chance
    @staticmethod
    def create_model(options: ModelOptions) -> keras.models.Sequential:
        """ Create CNN model consisting of three convolutional layers and two-layers fully connected ANN."""
        builder = ModelBuilder(options.name, input_shape=options.input_shape)
        builder.add_conv_block(filters=[32], drop_rate=options.drop_fc, input_layer=True)
        builder.add_conv_block(filters=[64], drop_rate=options.drop_fc)
        builder.add_conv_block(filters=[128], drop_rate=options.drop_fc)
        builder.add_fc_block(units=[128], drop_rate=options.drop_conv, output_layer=True)
        builder.compile_SGD()
        builder.model.summary()
        return builder.model


# builder of CNN models
class ModelBuilder:
    # parameters of convolutional layer
    _CONV_PARAMS = namedtuple('convolution_parameters', ['filter', 'activation', 'padding', 'kernel'])
    # parameters of fully connected layer
    _FC_PARAMS = namedtuple('fully_connected_params', ['units', 'activation'])

    @property
    def model(self) -> keras.models.Sequential:
        return self._model

    # preprocessing collection of objects (casting to specified length
    @staticmethod
    def _collection_preprocess(obj: Collection[Any], conversion_type: type, excepted_len: int) -> tuple:
        """ Converts obj with conversion_type in tuple with len excepted_len."""
        obj = [obj] if type(obj) is conversion_type else obj
        residue = obj[0:excepted_len]
        extension = [obj[-1]] * (excepted_len - len(obj))
        return tuple(residue + extension)

    def __init__(self, model_name: str, input_shape: tuple[int, int, int]) -> None:
        """ Initialize sequential model"""
        self._model: keras.models.Sequential = keras.models.Sequential(name=model_name)
        self._input_shape: tuple[int, int, int] = input_shape

    # add sequence of convolutional layers with specified drop chances and pooling layer to model
    def add_conv_block(self, filters: Collection[int],
                       num_conv: int = 1,
                       drop_rate: float = 0.0,
                       activation: str = "relu",
                       padding: str = "same",
                       conv_kernel: tuple[int, int] = (3, 3),
                       pool_kernel: tuple[int, int] = (2, 2),
                       input_layer: bool = False) -> None:
        """ Add a sequence of convolutional layers, with specified parameters ending with a pooling layer to model. """
        # convert filters to tuple with len num_conv
        filters = self._collection_preprocess(obj=filters, conversion_type=int, excepted_len=num_conv)
        # add sequence of convolutional layers
        for num_layer in range(num_conv):
            # shaping convolutional layer params
            params = self._CONV_PARAMS(filter=filters[num_layer], activation=activation, padding=padding, kernel=conv_kernel)._asdict()
            if num_layer != 0 or not input_layer:
                self._model.add(keras.layers.Dropout(rate=drop_rate))
            else:
                params['input_shape'] = self._input_shape
            self._model.add(keras.layers.Conv2D(**params))
        self._model.add(keras.layers.MaxPooling2D(pool_size=pool_kernel))

    # add sequence of fully connected layers with specified drop chances to model
    def add_fc_block(self, units: Collection[int],
                     num_fc: int = 1,
                     drop_rate: float = 0.0,
                     activation: str = "relu",
                     output_layer: bool = False) -> None:
        """ Add a sequence of fully connected layers, with specified parameters to model. """
        # convert filters to tuple with len num_fc
        units = self._collection_preprocess(obj=units, conversion_type=int, excepted_len=num_fc)
        self._model.add(keras.layers.Flatten())
        # add sequence of fully connected layers
        for num_layer in range(num_fc):
            self._model.add(keras.layers.Dropout(rate=drop_rate))
            # shaping fully connected  layer params
            params = self._FC_PARAMS(units=units[num_layer], activation=activation)._asdict()
            self._model.add(keras.layers.Dense(**params))
        if output_layer:
            self._model.add(keras.layers.Dense(units=1, activation='sigmoid'))

    # compile model with Sequential Gradient Descent optimizer
    def compile_SGD(self, learning_rate: float = 0.001,
                    momentum: float = 0.9,
                    loss: str = "binary_crossentropy",
                    metrics: Collection[str] = tuple(["accuracy"])) -> None:
        """ Compile model with Sequential Gradient Descent optimizer."""
        self._model.compile(optimizer=keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum),
                            loss=loss,
                            metrics=metrics)
