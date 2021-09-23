import os


# Options of CNN model
class ModelOptions:
    _common_file_name: str = 'models_summary'
    _data_directory: str = 'data/'

    def __init__(self, name: str, drop_conv: float, drop_fc: float, input_shape: (int, int, int)):
        self.name: str = name
        self.drop_conv: float = drop_conv
        self.drop_fc: float = drop_fc
        self.input_shape: (int, int, int) = input_shape
        self.history: {} = None

    def get_model_png_file_name(self):
        return self.__class__._data_directory + self.name + '.png'

    @classmethod
    def get_common_xlsx_file_name(cls):
        return cls._data_directory + cls._common_file_name + '.xlsx'

    @classmethod
    def get_common_png_file_name(cls):
        return cls._data_directory + cls._common_file_name + '.png'

    @classmethod
    def create_data_directory(cls):
        directory = os.curdir + '/' + cls._data_directory
        if not os.path.isdir(directory):
            os.mkdir(directory)
