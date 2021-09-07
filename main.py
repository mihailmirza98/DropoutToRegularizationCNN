import os
import tensorflow as tf
from tensorflow import keras
from ModelOptions import ModelOptions
from VisualisationFunctions import save_models_to_png, save_model_to_png, save_model_to_xlsx


# create CNN model with specified drop_chance
def create_model(options: ModelOptions):
    # create model
    model = keras.models.Sequential(name=options.name)
    model.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation='relu',
                                  input_shape=options.input_shape))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Dropout(options.drop_conv))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Dropout(options.drop_conv))
    model.add(keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(options.drop_fc))
    model.add(keras.layers.Dense(units=128, activation='relu'))
    model.add(keras.layers.Dense(units=1, activation='sigmoid'))
    # compile model
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.001, momentum=0.9),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# create data iterators for model
def create_data_iterators(target_size: (int, int), dataset_location: str):
    # create data generator
    data_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255.0)
    # prepare iterator
    train_iterator = data_generator.flow_from_directory(dataset_location + '/train/',
                                                        class_mode='binary',
                                                        target_size=target_size)
    test_iterator = data_generator.flow_from_directory(dataset_location + '/test/',
                                                       class_mode='binary',
                                                       target_size=target_size)
    return train_iterator, test_iterator


with tf.device("/GPU:0"):
    # specify params
    epochs = 2
    target_size = (200, 200)
    input_shape = target_size + (3,)
    # specify types of models
    models_options_set = (
        ModelOptions(name='model1', drop_conv=0.0, drop_fc=0.0, input_shape=input_shape),
        ModelOptions(name='model2', drop_conv=0.2, drop_fc=0.2, input_shape=input_shape),
        ModelOptions(name='model3', drop_conv=0.2, drop_fc=0.5, input_shape=input_shape),
        ModelOptions(name='model4', drop_conv=0.5, drop_fc=0.2, input_shape=input_shape),
        ModelOptions(name='model5', drop_conv=0.5, drop_fc=0.5, input_shape=input_shape),
    )
    # prepare iterators
    dataset_location = os.curdir + '/cats_vs_dogs_dataset'
    train_iterator, test_iterator = create_data_iterators(dataset_location=dataset_location, target_size=target_size)
    # test different types of models
    for model_options in models_options_set:
        # create model
        model = create_model(options=model_options)
        model.summary()
        # fit model
        model_options.history = model.fit(train_iterator, validation_data=test_iterator, epochs=epochs).history
        # save model history to csv file
        save_model_to_xlsx(history=model_options.history, file_name=model_options.get_common_xlsx_file_name(),
                           model_name=model_options.name)
        # save model history to png file
        save_model_to_png(history=model_options.history, file_name=model_options.get_model_png_file_name())
    # compare models and save results to png file
    save_models_to_png(models_options_set=models_options_set, file_name=ModelOptions.get_common_png_file_name())
