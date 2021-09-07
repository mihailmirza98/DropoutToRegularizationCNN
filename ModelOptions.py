# Options of CNN model
class ModelOptions:
    _common_file_name: str = 'models_summary'
    _data_directory = 'data/'

    def __init__(self, name: str, drop_conv: float, drop_fc: float, input_shape: (int, int, int)):
        self.name = name
        self.drop_conv = drop_conv
        self.drop_fc = drop_fc
        self.input_shape = input_shape
        self.history: {} = None

    def get_model_png_file_name(self):
        return self.__class__._data_directory + self.name + '.png'

    @classmethod
    def get_common_xlsx_file_name(cls):
        return cls._data_directory + cls._common_file_name + '.xlsx'

    @classmethod
    def get_common_png_file_name(cls):
        return cls._data_directory + cls._common_file_name + '.png'
