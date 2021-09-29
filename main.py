import os
import tensorflow as tf
from tensorflow.keras import preprocessing
from ModelOptions import ModelOptions
from ModelCreator import ModelCreator
from VisualisationFunctions import save_models_to_png, save_model_to_png, save_model_to_xlsx


# create data iterators for model
def create_data_iterators(target_size: tuple[int, int], dataset_location: str):
    """ Create iterators for train and test data sets."""
    # create data generator
    data_generator = preprocessing.image.ImageDataGenerator(rescale=1.0 / 255.0)
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
    epochs = 40
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
    # create directory to save visualisation file
    ModelOptions.create_data_directory()
    # create model creator
    creator = ModelCreator()
    # test different types of models
    for model_options in models_options_set:
        # create model
        model = creator.create_model(options=model_options)
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
