import os
import pandas as pd
from matplotlib import pyplot as plt


# save model history to csv file
def save_model_to_xlsx(history: {}, file_name: str, model_name: str):
    assert file_name.endswith('.xlsx'), 'File format error. Need to specify xlsx file.'
    # create dataframe
    df = pd.DataFrame(history)
    # save dataframe to csv file
    mode = 'a' if os.path.exists(file_name) else 'w'
    with pd.ExcelWriter(file_name, engine='openpyxl', mode=mode) as writer:
        df.to_excel(writer, sheet_name=model_name, index=True)


# save model history to png file
def save_model_to_png(history: {}, file_name: str):
    assert file_name.endswith('.png'), 'File format error. Need to specify png file.'
    plt.subplots_adjust(hspace=0.5)
    # plot loss
    plt.subplot(2, 1, 1)
    plt.title('Loss', fontsize=24)
    plt.plot(history['loss'], label='train')
    plt.plot(history['val_loss'], label='test')
    plt.legend()
    # plot accuracy
    plt.subplot(2, 1, 2)
    plt.title('Accuracy', fontsize=24)
    plt.plot(history['accuracy'], label='train')
    plt.plot(history['val_accuracy'], label='test')
    plt.legend()
    # save plot to file
    plt.savefig(file_name)
    plt.close()


# save results for all models to png file
def save_models_to_png(models_options_set: {}, file_name: str):
    assert file_name.endswith('.png'), 'File format error. Need to specify png file.'
    plt.subplots_adjust(hspace=0.4)
    for model_options in models_options_set:
        # plot train loss
        plt.subplot(2, 2, 1)
        plt.title('Train loss', fontsize=24)
        plt.plot(model_options.history['loss'], label=model_options.name)
        # plot train accuracy
        plt.subplot(2, 2, 2)
        plt.title('Train accuracy', fontsize=24)
        plt.plot(model_options.history['accuracy'], label=model_options.name)
        # plot test loss
        plt.subplot(2, 2, 3)
        plt.title('Test loss', fontsize=24)
        plt.plot(model_options.history['val_loss'], label=model_options.name)
        # plot test accuracy
        plt.subplot(2, 2, 4)
        plt.title('Test accuracy', fontsize=24)
        plt.plot(model_options.history['val_accuracy'], label=model_options.name)
    plt.savefig(file_name)
    plt.close()